"""Cross-map generalization benchmark for CAST-MARL.

Protocol
--------
The archived CAST-MARL checkpoint is trained on the plain terrain
(train_env=terrain_plain, seed 42).  This script deploys that same
checkpoint zero-shot to the urban and mountainous terrain types, and
re-evaluates it on the plain terrain as an in-distribution reference.
For comparison, the executable A* planner is run under the same
obstacle-grid protocol on each target terrain; A* re-plans from the
per-episode occupancy grid and therefore does not require training, but
it also does not transfer any learned representation.

Metrics are computed by the unchanged simulator step function, matching
the manuscript protocol: coverage, conflict rate, reward, path length.
"""

from __future__ import annotations

import csv
import heapq
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch


PACKAGE = Path(__file__).resolve().parents[2]
SOURCE = PACKAGE / "experimental_runs" / "reproducibility" / "cast_marl_experiment_source.py"
spec = importlib.util.spec_from_file_location("cast_marl_source", SOURCE)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
assert spec.loader is not None
spec.loader.exec_module(mod)

mod.TERRAIN_MANIFEST_PATH = str(PACKAGE / "experimental_runs" / "reproducibility" / "missing_terrain_manifest.json")
mod._TERRAIN_CACHE.clear()

DATASET = [
    str(PACKAGE / "dataset" / "dataset1_users35.npz"),
    str(PACKAGE / "dataset" / "dataset1_users45.npz"),
    str(PACKAGE / "dataset" / "dataset1_users60.npz"),
]
OUT = PACKAGE / "experimental_runs" / "reproducibility" / "cross_map"
OUT.mkdir(parents=True, exist_ok=True)

CHECKPOINT = PACKAGE / "experimental_runs" / "localized_uav_best.pth"
SEEDS = [42, 43, 44, 45, 46]
EPISODES_PER_SEED = 12
NUM_UAVS = 6
MAX_STEPS = 35
GRID_N = 20
INFLATION = 135.0
MAX_SPEED = 150.0
ALT_TARGET = 120.0


# ---- grid planner helpers (same as classical_planner_benchmarks.py) ----
def cell_xy(cell):
    return np.array([(cell[0] + 0.5) * mod.AREA_X / GRID_N, (cell[1] + 0.5) * mod.AREA_Y / GRID_N], dtype=np.float32)


def xy_cell(xy):
    return (int(np.clip(xy[0] / mod.AREA_X * GRID_N, 0, GRID_N - 1)), int(np.clip(xy[1] / mod.AREA_Y * GRID_N, 0, GRID_N - 1)))


def neighbours(cell):
    x, y = cell
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
        q = (x + dx, y + dy)
        if 0 <= q[0] < GRID_N and 0 <= q[1] < GRID_N:
            yield q


def segment_cells(a, b):
    n = max(abs(a[0] - b[0]), abs(a[1] - b[1]), 1)
    return [(round(a[0] + (b[0] - a[0]) * i / n), round(a[1] + (b[1] - a[1]) * i / n)) for i in range(n + 1)]


def occupancy(env):
    occ = np.zeros((GRID_N, GRID_N), dtype=bool)
    for gx in range(GRID_N):
        for gy in range(GRID_N):
            p = cell_xy((gx, gy))
            for obs in env.obstacles:
                x0, y0, _, x1, y1, _ = obs
                if (x0 - INFLATION <= p[0] <= x1 + INFLATION and y0 - INFLATION <= p[1] <= y1 + INFLATION):
                    occ[gx, gy] = True
                    break
    return occ


def nearest_free(cell, occ):
    if not occ[cell]:
        return cell
    for radius in range(1, GRID_N):
        for x in range(max(0, cell[0] - radius), min(GRID_N, cell[0] + radius + 1)):
            for y in range(max(0, cell[1] - radius), min(GRID_N, cell[1] + radius + 1)):
                if not occ[x, y]:
                    return (x, y)
    return cell


def valid_edge(a, b, occ):
    if occ[b] or any(occ[c] for c in segment_cells(a, b)):
        return False
    return True


def astar(start, goal, occ):
    start, goal = nearest_free(start, occ), nearest_free(goal, occ)
    if start == goal:
        return [start]

    def h(a):
        return math.hypot(a[0] - goal[0], a[1] - goal[1])

    open_set = [(h(start), 0.0, 0, start)]
    parent, gscore = {}, {start: 0.0}
    serial = 0
    while open_set:
        _, g, t, cur = heapq.heappop(open_set)
        if cur == goal:
            path = [cur]
            while cur in parent:
                cur = parent[cur]
                path.append(cur)
            return list(reversed(path))
        for nxt in neighbours(cur):
            if not valid_edge(cur, nxt, occ):
                continue
            ng = g + math.hypot(nxt[0] - cur[0], nxt[1] - cur[1])
            if ng < gscore.get(nxt, float("inf")):
                gscore[nxt] = ng
                parent[nxt] = cur
                serial += 1
                heapq.heappush(open_set, (ng + h(nxt), ng, t + 1, nxt))
    return [start]


def target_for(env, i):
    support = env._assigned_uncovered(i)
    if len(support):
        return support[int(np.argmin(np.linalg.norm(support[:, :2] - env.uav_pos[i, :2], axis=1)))][:2]
    return env.region_centers[i, :2]


def astar_action(env, occ):
    actions = np.zeros((env.num_uavs, 3), dtype=np.float32)
    for i in range(env.num_uavs):
        s = xy_cell(env.uav_pos[i, :2])
        g = xy_cell(target_for(env, i))
        path = astar(s, g, occ)
        cur = xy_cell(env.uav_pos[i, :2])
        idx = min(1, len(path) - 1)
        if path[0] != cur:
            idx = 0
        goal_xy = cell_xy(path[idx])
        vec = goal_xy - env.uav_pos[i, :2]
        norm = np.linalg.norm(vec)
        if norm < 1e-6:
            vec = np.asarray(target_for(env, i), dtype=np.float32) - env.uav_pos[i, :2]
            norm = np.linalg.norm(vec)
        actions[i, :2] = vec / max(norm, 1e-6)
        actions[i, 2] = np.clip((ALT_TARGET - env.uav_pos[i, 2]) / 18.0, -1.0, 1.0)
    return actions


def run_astar_episode(seed, dataset, env_type, obstacle_ratio):
    mod.set_seed(seed)
    env = mod.MultiUAVCoverageEnv(
        dataset, NUM_UAVS, env_type=env_type, max_steps=MAX_STEPS, max_speed=MAX_SPEED,
        obstacle_ratio=obstacle_ratio, use_failure_module=False, region_reassign_interval=4, num_users=None,
    )
    env.reset()
    occ = occupancy(env)
    total_reward = 0.0
    for _ in range(MAX_STEPS):
        _, reward, done, info = env.step(astar_action(env, occ))
        total_reward += float(reward)
        if done:
            break
    return {
        "coverage": info["coverage_rate"] * 100.0,
        "conflict_rate": info["conflict_rate"],
        "reward": total_reward,
        "path_length": info["path_length"],
    }


def run_marl_episode(seed, dataset, env_type, obstacle_ratio, model, config, device):
    mod.set_seed(seed)
    env = mod.MultiUAVCoverageEnv(
        dataset, NUM_UAVS, env_type=env_type, max_steps=MAX_STEPS, max_speed=MAX_SPEED,
        obstacle_ratio=obstacle_ratio,
        high_conflict_threshold=config.high_conflict_threshold,
        high_conflict_penalty=config.high_conflict_penalty,
        reward_gain_coef=config.reward_gain_coef,
        reward_coverage_coef=config.reward_coverage_coef,
        reward_path_penalty=config.reward_path_penalty,
        reward_conflict_penalty=config.reward_conflict_penalty,
        reward_redundancy_penalty=config.reward_redundancy_penalty,
        reward_obstacle_penalty=config.reward_obstacle_penalty,
        assignment_bonus_coef=config.assignment_bonus_coef,
        frontier_bonus_coef=config.frontier_bonus_coef,
        stagnation_penalty_coef=config.stagnation_penalty_coef,
        region_reassign_interval=config.region_reassign_interval,
        use_region_assignment=config.use_region_assignment,
        region_balance_strength=config.region_balance_strength,
        region_global_mix=config.region_global_mix,
        region_support_radius_scale=config.region_support_radius_scale,
        heuristic_view_radius_scale=config.heuristic_view_radius_scale,
        assignment_decay_cover=config.assignment_decay_cover,
        assignment_decay_time=config.assignment_decay_time,
        frontier_decay_cover=config.frontier_decay_cover,
        frontier_decay_time=config.frontier_decay_time,
        late_stage_visible_boost=config.late_stage_visible_boost,
        fail_prob_base=config.fail_prob_base,
        fail_prob_load_scale=config.fail_prob_load_scale,
        fail_reward_recovery_coef=config.fail_reward_recovery_coef,
        fail_energy_fair_coef=config.fail_energy_fair_coef,
        use_failure_module=config.use_failure_module,
        fail_trigger_min_step=config.fail_trigger_min_step,
        fail_trigger_max_step=config.fail_trigger_max_step,
        fail_max_count=config.fail_max_count,
        fail_fixed_count=config.fail_fixed_count,
        fail_load_aware=config.fail_load_aware,
        fail_reassign_on_event=config.fail_reassign_on_event,
        num_users=None,
    )
    obs = env.reset()
    total_reward = 0.0
    for _ in range(MAX_STEPS):
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            action, _, _, _ = model.act(obs_t, deterministic=True)
            action = action.squeeze(0).cpu().numpy()
        obs, reward, done, info = env.step(action)
        total_reward += float(reward)
        if done:
            break
    return {
        "coverage": info["coverage_rate"] * 100.0,
        "conflict_rate": info["conflict_rate"],
        "reward": total_reward,
        "path_length": info["path_length"],
    }


def main():
    dataset = mod.ScenarioDataset(DATASET)
    device = torch.device("cpu")

    # Load archived plain-trained checkpoint
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    cfg_dict = ckpt["config"]
    if not isinstance(cfg_dict, dict):
        cfg_dict = {k: getattr(cfg_dict, k) for k in dir(cfg_dict) if not k.startswith("_")}
    config = mod.ExperimentConfig(**{k: v for k, v in cfg_dict.items() if k in mod.ExperimentConfig.__dataclass_fields__})
    state_dim = mod.MultiUAVCoverageEnv(dataset, NUM_UAVS, env_type="terrain_plain", max_steps=MAX_STEPS,
                                        max_speed=MAX_SPEED, obstacle_ratio=0.12, use_failure_module=False,
                                        region_reassign_interval=4, num_users=None).state_dim
    model = mod.TmarlActorCritic(state_dim, config, variant="tmarl")
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()

    # Training map: terrain_plain, obstacle_ratio 0.08 (1 obstacle).
    # Target maps use increasing obstacle densities so the zero-shot
    # deployment crosses genuinely different maps (2-6 obstacles).
    targets = [
        ("terrain_plain", "plain", 0.08),
        ("terrain_urban", "urban", 0.20),
        ("terrain_mountain", "mountain", 0.16),
        ("terrain_mountain", "mountain-dense", 0.30),
    ]
    rows = []
    for env_type, label, obstacle_ratio in targets:
        for seed in SEEDS:
            for ep in range(EPISODES_PER_SEED):
                m = run_marl_episode(seed * 1000 + ep, dataset, env_type, obstacle_ratio, model, config, device)
                rows.append({"target_map": label, "method": "CAST-MARL (plain-trained)", "seed": seed, "episode": ep, **m})
                a = run_astar_episode(seed * 1000 + ep, dataset, env_type, obstacle_ratio)
                rows.append({"target_map": label, "method": "A*", "seed": seed, "episode": ep, **a})
            print(label, "seed", seed, "done", flush=True)

    with (OUT / "CrossMap_ByEpisode.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    summary = []
    for target_map in ["plain", "urban", "mountain", "mountain-dense"]:
        for method in ["CAST-MARL (plain-trained)", "A*"]:
            g = [r for r in rows if r["target_map"] == target_map and r["method"] == method]
            summary.append({
                "target_map": target_map,
                "method": method,
                **{f"{k}_mean": float(np.mean([x[k] for x in g])) for k in ("coverage", "conflict_rate", "reward", "path_length")},
                **{f"{k}_std": float(np.std([x[k] for x in g], ddof=1)) for k in ("coverage", "conflict_rate", "reward", "path_length")},
                "n": len(g),
            })
    with (OUT / "CrossMap_Summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    (OUT / "CrossMap_Protocol.json").write_text(json.dumps({
        "checkpoint": str(CHECKPOINT.name),
        "train_env": "terrain_plain",
        "target_maps": ["plain (r=0.08)", "urban (r=0.20)", "mountain (r=0.16)", "mountain-dense (r=0.30)"],
        "methods": ["CAST-MARL (plain-trained) zero-shot", "A* re-planning"],
        "seeds": SEEDS,
        "episodes_per_seed": EPISODES_PER_SEED,
        "num_uavs": NUM_UAVS,
        "max_steps": MAX_STEPS,
        "metrics": "unchanged simulator step metrics",
    }, indent=2), encoding="utf-8")
    print("CROSS-MAP DONE")


if __name__ == "__main__":
    main()
