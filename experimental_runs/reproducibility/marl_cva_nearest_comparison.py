# -*- coding: utf-8 -*-
"""CVA-SP vs nearest-point projection on a trained MARL checkpoint.

The main CAST-MARL experiments use the learned soft safety inside the actor.
Here we attach an external hard projection layer (the same one evaluated on
A*) on top of the same trained plain checkpoint and compare the nearest-point
and CVA-SP projection objectives under the unchanged cross-map test protocol.
This directly tests whether the correction objective of Eq. (9) changes
behaviour when the raw action comes from a learned policy rather than from a
coverage-oriented planner.
"""

from __future__ import annotations

import csv
import importlib.util
import json
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

SPEC2 = importlib.util.spec_from_file_location(
    "cva_sp_bench",
    Path(__file__).resolve().parent / "cva_sp_benchmarks.py",
)
cva = importlib.util.module_from_spec(SPEC2)
sys.modules[SPEC2.name] = cva
assert SPEC2.loader is not None
SPEC2.loader.exec_module(cva)

CHECKPOINT = PACKAGE / "experimental_runs" / "localized_uav_best.pth"
OUT = PACKAGE / "experimental_runs" / "reproducibility" / "cross_map"
SEEDS = [42, 43, 44, 45, 46]
EPISODES_PER_SEED = 12
MAX_STEPS = 35


def load_checkpoint():
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    cfg_dict = ckpt["config"]
    if not isinstance(cfg_dict, dict):
        cfg_dict = {k: getattr(cfg_dict, k) for k in dir(cfg_dict)
                    if not k.startswith("_")}
    config = mod.ExperimentConfig(**{
        k: v for k, v in cfg_dict.items()
        if k in mod.ExperimentConfig.__dataclass_fields__})
    dataset = mod.ScenarioDataset(cva.DATASET)
    state_dim = mod.MultiUAVCoverageEnv(
        dataset, cva.NUM_UAVS, env_type="terrain_plain",
        max_steps=cva.MAX_STEPS, max_speed=cva.MAX_SPEED,
        obstacle_ratio=0.12, use_failure_module=False,
        region_reassign_interval=4, num_users=None).state_dim
    model = mod.TmarlActorCritic(state_dim, config, variant="tmarl")
    model.load_state_dict(ckpt["state_dict"])
    return model, config, dataset


def next_xy(env, actions):
    out = np.zeros((env.num_uavs, 2), dtype=np.float32)
    for j in range(env.num_uavs):
        act_j = actions[j]
        cur_cov = float(env.covered.mean()) if len(env.covered) else 1.0
        tr = env.t / max(1, env.max_steps)
        ss = float(np.clip(1.0 - 0.14 * cur_cov - 0.08 * tr, 0.75, 1.0))
        out[j] = np.clip(
            env.uav_pos[j, :2]
            + np.clip(act_j[:2], -1.0, 1.0) * (env.max_speed * ss),
            0.0,
            [mod.AREA_X, mod.AREA_Y],
        )
    return out


def run_episode(seed, dataset, env_type, obstacle_ratio, model, config,
                device, projection):
    mod.set_seed(seed)
    env = mod.MultiUAVCoverageEnv(
        dataset, cva.NUM_UAVS, env_type=env_type,
        max_steps=MAX_STEPS, max_speed=cva.MAX_SPEED,
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
    n_proj = 0
    for _ in range(MAX_STEPS):
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32,
                                 device=device).unsqueeze(0)
            action_t, _, _, _ = model.act(obs_t, deterministic=True)
            raw = action_t.squeeze(0).cpu().numpy()
        nxy = next_xy(env, raw)
        safe = raw.copy()
        for i in range(env.num_uavs):
            other = np.delete(nxy, i, axis=0)
            if projection == "nearest":
                corr = cva.nearest_safe_action(env, i, raw[i], other)
                safe[i, :2] = corr[:2]
                n_proj += int(np.linalg.norm(corr - raw[i]) > 1e-6)
            elif projection == "cva":
                corr, trig, _ = cva.cva_sp_action(env, i, raw[i], other)
                safe[i, :2] = corr[:2]
                n_proj += int(trig)
        obs, reward, done, info = env.step(safe)
        total_reward += float(reward)
        if done:
            break
    return {
        "coverage": info["coverage_rate"] * 100.0,
        "conflict_rate": info["conflict_rate"],
        "reward": total_reward,
        "path_length": info["path_length"],
        "projections": n_proj,
    }


def main():
    device = torch.device("cpu")
    model, config, dataset = load_checkpoint()
    model.to(device)
    model.eval()
    targets = [
        ("terrain_plain", "plain", 0.08),
        ("terrain_urban", "urban", 0.20),
        ("terrain_mountain", "mountain", 0.16),
        ("terrain_mountain", "mountain-dense", 0.30),
    ]
    all_rows = []
    for projection in ["nearest", "cva"]:
        for env_type, label, ratio in targets:
            for seed in SEEDS:
                for ep in range(EPISODES_PER_SEED):
                    r = run_episode(seed * 1000 + ep, dataset, env_type,
                                    ratio, model, config, device, projection)
                    all_rows.append({"projection": projection,
                                     "target_map": label, "seed": seed,
                                     "episode": ep, **r})
                print(projection, label, seed, "done", flush=True)
    with (OUT / "Marl_CvaNearest_ByEpisode.csv").open(
            "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    summary = []
    for projection in ["nearest", "cva"]:
        for label in ["plain", "urban", "mountain", "mountain-dense"]:
            g = [r for r in all_rows if r["projection"] == projection
                 and r["target_map"] == label]
            summary.append({
                "projection": projection,
                "target_map": label,
                **{f"{k}_mean": round(float(np.mean([x[k] for x in g])), 4)
                   for k in ("coverage", "conflict_rate", "reward",
                             "path_length", "projections")},
                **{f"{k}_std": round(float(np.std([x[k] for x in g], ddof=1)), 4)
                   for k in ("coverage", "conflict_rate", "reward",
                             "path_length", "projections")},
                "n": len(g),
            })
    with (OUT / "Marl_CvaNearest_Summary.csv").open(
            "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    (OUT / "Marl_CvaNearest_Protocol.json").write_text(json.dumps({
        "checkpoint": str(CHECKPOINT.name),
        "base_train_env": "terrain_plain (seed 42)",
        "projections": ["nearest", "cva"],
        "test_set": "unchanged cross-map protocol",
        "note": "external hard projection attached after the learned actor",
    }, indent=2), encoding="utf-8")
    print("MARL CVA vs NEAREST DONE")


if __name__ == "__main__":
    main()
