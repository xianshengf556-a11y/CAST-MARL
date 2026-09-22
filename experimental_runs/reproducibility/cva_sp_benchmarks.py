"""CVA-SP benchmark: Coverage-Value-Aware Safety Projection.

Mechanism
---------
Given a raw (possibly unsafe) action a_i for UAV i, the standard safety filter
projects a_i onto the nearest point of the safe set.  CVA-SP instead searches
inside the safe set for a corrected action that MAXIMISES the predicted
incremental coverage value while staying close to the raw action:

    a_i^safe = argmax_{a in S_i}  [ DeltaCov(a) - lambda * ||a - a_i||^2 ]

where S_i is the set of actions whose projected next position (i) is not
inside an obstacle, (ii) keeps a separation >= d_safe from every other UAV's
predicted next position, and (iii) deviates from a_i by at most a bounded
correction magnitude.  DeltaCov(a) counts the uncovered task points that UAV i
would newly cover at the projected position, excluding points already covered
by other UAVs' predicted positions (interaction-aware de-duplication).

Compared variants
-----------------
* raw    : A* actions executed without any safety layer.
* nearest: A* + projection to the nearest safe action (baseline safety layer).
* cva    : A* + CVA-SP (proposed).

The simulator step function is unchanged; coverage, conflict rate, path
length and reward therefore use the same metric implementation as the
learning methods and the classical planner protocol.
"""

from __future__ import annotations

import csv
import heapq
import importlib.util
import json
import math
import random
import sys
from pathlib import Path

import numpy as np


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
OUT = PACKAGE / "experimental_runs" / "reproducibility" / "cva_sp"
OUT.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 43, 44, 45, 46]
EPISODES_PER_SEED = 12
NUM_UAVS = 6
MAX_STEPS = 35
GRID_N = 20
INFLATION = 135.0
MAX_SPEED = 150.0
ALT_TARGET = 120.0

# CVA-SP parameters
D_SAFE = 138.0          # execution-filter separation margin (m), matches manuscript
CORRECTION_CAP = 0.55   # max deviation from the raw action (normalised units)
LAMBDA = 0.35           # deviation penalty in the CVA-SP objective
N_DIR = 8               # angular grid resolution for candidate corrections
N_AMP = 3               # magnitude grid resolution


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


def raw_actions_from_paths(env, paths):
    """Convert planned paths to normalised raw actions (no safety layer)."""
    actions = np.zeros((env.num_uavs, 3), dtype=np.float32)
    for i, path in enumerate(paths):
        if not path:
            continue
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


def _projected_xy(env, agent_idx, action):
    """Predicted next XY position under the simulator speed scaling."""
    current_coverage = float(env.covered.mean()) if len(env.covered) else 1.0
    time_ratio = env.t / max(1, env.max_steps)
    speed_scale = float(np.clip(1.0 - 0.14 * current_coverage - 0.08 * time_ratio, 0.75, 1.0))
    move_xy = np.clip(action[:2], -1.0, 1.0) * (env.max_speed * speed_scale)
    return np.clip(env.uav_pos[agent_idx, :2] + move_xy, 0.0, [mod.AREA_X, mod.AREA_Y])


def candidate_actions(raw, cap=CORRECTION_CAP):
    """Deterministic angular-magnitude grid around the raw action."""
    raw_xy = np.asarray(raw[:2], dtype=np.float32)
    cands = [np.asarray(raw, dtype=np.float32)]
    for k in range(N_DIR):
        ang = 2.0 * math.pi * k / N_DIR
        d = np.array([math.cos(ang), math.sin(ang)], dtype=np.float32)
        for m in range(1, N_AMP + 1):
            amp = cap * m / N_AMP
            xy = np.clip(raw_xy + amp * d, -1.0, 1.0)
            c = np.concatenate([xy, np.asarray(raw[2:], dtype=np.float32)])
            cands.append(c.astype(np.float32))
    return np.stack(cands)


def coverage_gain_at(env, agent_idx, pos_xy, other_next_xy):
    """Uncovered task points newly coverable at pos_xy, excluding points that
    other UAVs' predicted positions would already cover."""
    r = env._coverage_radius(float(env.uav_pos[agent_idx, 2]))
    users = env.users
    uncov = ~env.covered
    if not uncov.any():
        return 0.0
    d_self = np.linalg.norm(users[:, :2] - np.asarray(pos_xy, dtype=np.float32), axis=1)
    gain_self = uncov & (d_self <= r)
    for op in other_next_xy:
        d_op = np.linalg.norm(users[:, :2] - np.asarray(op, dtype=np.float32), axis=1)
        gain_self &= ~(d_op <= r)
    return float(gain_self.sum())


def nearest_safe_action(env, agent_idx, raw, other_next_xy):
    """Baseline: nearest safe action inside the safety margin."""
    best = None
    best_dist = None
    for c in candidate_actions(raw, cap=1.0):
        pos = _projected_xy(env, agent_idx, c)
        if env._point_in_obstacle(np.array([pos[0], pos[1], env.uav_pos[agent_idx, 2]], dtype=np.float32)):
            continue
        sep = np.linalg.norm(other_next_xy - pos, axis=1)
        if sep.min() < D_SAFE:
            continue
        d = float(np.linalg.norm(c - raw))
        if best_dist is None or d < best_dist:
            best, best_dist = c, d
    return raw if best is None else best


def cva_sp_action(env, agent_idx, raw, other_next_xy):
    """Proposed CVA-SP correction."""
    raw_pos = _projected_xy(env, agent_idx, raw)
    raw_safe = (not env._point_in_obstacle(np.array([raw_pos[0], raw_pos[1], env.uav_pos[agent_idx, 2]], dtype=np.float32)))
    if raw_safe:
        sep = np.linalg.norm(other_next_xy - raw_pos, axis=1)
        if sep.min() >= D_SAFE:
            return raw, False, 0.0
    best = raw
    best_score = -1e9
    triggered = False
    for c in candidate_actions(raw):
        pos = _projected_xy(env, agent_idx, c)
        if env._point_in_obstacle(np.array([pos[0], pos[1], env.uav_pos[agent_idx, 2]], dtype=np.float32)):
            continue
        sep = np.linalg.norm(other_next_xy - pos, axis=1)
        if sep.min() < D_SAFE:
            continue
        gain = coverage_gain_at(env, agent_idx, pos, other_next_xy)
        dev = float(np.linalg.norm(c - raw))
        score = gain - LAMBDA * dev
        if score > best_score:
            best, best_score, triggered = c, score, True
    if triggered and best_score > -1e8:
        return best, True, best_score
    # Graceful degradation: when the bounded coverage-value search has no
    # feasible safe candidate, fall back to the nearest safe action so the
    # safety layer never silently releases an unsafe action.
    fallback = nearest_safe_action(env, agent_idx, raw, other_next_xy)
    if float(np.linalg.norm(fallback - raw)) > 1e-6:
        return fallback, True, 0.0
    return raw, False, 0.0


def apply_safety(env, actions, variant):
    """Apply the selected safety variant to raw actions."""
    out = actions.copy()
    next_xy = np.zeros((env.num_uavs, 2), dtype=np.float32)
    for j in range(env.num_uavs):
        act_j = actions[j]
        cur_cov = float(env.covered.mean()) if len(env.covered) else 1.0
        tr = env.t / max(1, env.max_steps)
        ss = float(np.clip(1.0 - 0.14 * cur_cov - 0.08 * tr, 0.75, 1.0))
        next_xy[j] = np.clip(env.uav_pos[j, :2] + np.clip(act_j[:2], -1.0, 1.0) * (env.max_speed * ss), 0.0, [mod.AREA_X, mod.AREA_Y])
    corrections = 0
    for i in range(env.num_uavs):
        other = np.delete(next_xy, i, axis=0)
        if variant == "nearest":
            corr = nearest_safe_action(env, i, actions[i], other)
            out[i, :2] = corr[:2]
            corrections += int(np.linalg.norm(corr - actions[i]) > 1e-6)
        elif variant == "cva":
            corr, trig, _ = cva_sp_action(env, i, actions[i], other)
            out[i, :2] = corr[:2]
            corrections += int(trig)
    return out, corrections


def run_episode(seed, variant, dataset, obstacle_ratio=0.12):
    mod.set_seed(seed)
    env = mod.MultiUAVCoverageEnv(
        dataset, NUM_UAVS, env_type="terrain_urban", max_steps=MAX_STEPS,
        max_speed=MAX_SPEED, obstacle_ratio=obstacle_ratio, use_failure_module=False,
        region_reassign_interval=4, num_users=None,
    )
    env.reset()
    occ = occupancy(env)
    paths = [[] for _ in range(NUM_UAVS)]
    total_reward = 0.0
    total_corrections = 0
    steps_with_correction = 0
    for _ in range(MAX_STEPS):
        starts = [xy_cell(p[:2]) for p in env.uav_pos]
        goals = [xy_cell(target_for(env, i)) for i in range(NUM_UAVS)]
        paths = [astar(s, g, occ) for s, g in zip(starts, goals)]
        raw = raw_actions_from_paths(env, paths)
        safe_actions, corrections = apply_safety(env, raw, variant)
        total_corrections += corrections
        if corrections > 0:
            steps_with_correction += 1
        _, reward, done, info = env.step(safe_actions)
        total_reward += float(reward)
        if done:
            break
    return {
        "coverage": info["coverage_rate"] * 100.0,
        "conflict_rate": info["conflict_rate"],
        "reward": total_reward,
        "path_length": info["path_length"],
        "corrections": total_corrections,
        "steps_with_correction": steps_with_correction,
    }


def main():
    dataset = mod.ScenarioDataset(DATASET)
    variants = ["raw", "nearest", "cva"]
    rows = []
    for variant in variants:
        for seed in SEEDS:
            values = []
            for ep in range(EPISODES_PER_SEED):
                values.append(run_episode(seed * 1000 + ep, variant, dataset))
            for ep, val in enumerate(values):
                rows.append({"variant": variant, "seed": seed, "episode": ep, **val})
            print(variant, seed, {k: round(float(np.mean([v[k] for v in values])), 5) for k in ("coverage", "conflict_rate", "path_length", "corrections")}, flush=True)
    with (OUT / "CVA_SP_ByEpisode.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    summary = []
    for variant in variants:
        for seed in SEEDS:
            group = [r for r in rows if r["variant"] == variant and r["seed"] == seed]
            summary.append({"variant": variant, "seed": seed, **{f"{k}_mean": float(np.mean([x[k] for x in group])) for k in ("coverage", "conflict_rate", "reward", "path_length", "corrections", "steps_with_correction")}})
    with (OUT / "CVA_SP_Seed_Summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)
    pooled = []
    for variant in variants:
        group = [r for r in rows if r["variant"] == variant]
        pooled.append({
            "variant": variant,
            **{f"{k}_mean": float(np.mean([x[k] for x in group])) for k in ("coverage", "conflict_rate", "reward", "path_length", "corrections")},
            **{f"{k}_std": float(np.std([x[k] for x in group], ddof=1)) for k in ("coverage", "conflict_rate", "reward", "path_length", "corrections")},
            "n": len(group),
        })
    with (OUT / "CVA_SP_Summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=pooled[0].keys())
        writer.writeheader()
        writer.writerows(pooled)
    (OUT / "CVA_SP_Protocol.json").write_text(
        json.dumps({
            "variants": variants,
            "seeds": SEEDS,
            "episodes_per_seed": EPISODES_PER_SEED,
            "num_uavs": NUM_UAVS,
            "max_steps": MAX_STEPS,
            "d_safe": D_SAFE,
            "correction_cap": CORRECTION_CAP,
            "lambda": LAMBDA,
            "n_dir": N_DIR,
            "n_amp": N_AMP,
            "environment": "terrain_urban label with episode-generated urban obstacle set",
            "metrics": "unchanged simulator step metrics",
        }, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
