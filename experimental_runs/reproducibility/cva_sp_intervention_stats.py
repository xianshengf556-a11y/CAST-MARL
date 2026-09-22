# -*- coding: utf-8 -*-
"""Intervention-level statistics for CVA-SP vs nearest-point projection.

For every step in which the safety filter actually corrects at least one UAV,
record per-corrected-UAV:
  * correction distance in normalised action units,
  * incremental coverage retained by the corrected action (uncovered task
    points that would newly fall inside the sensing footprint, excluding
    points coverable by other UAVs),
  * number of feasible candidate actions available to the CVA-SP search,
  * whether the CVA-SP search fell back to the nearest safe action because no
    feasible candidate existed.

The experiment reuses the executable A* protocol from cva_sp_benchmarks.py
(5 seeds x 12 episodes, 6 UAVs, 35 steps, unchanged simulator metrics).
"""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np


SPEC = importlib.util.spec_from_file_location(
    "cva_sp_bench",
    Path(__file__).resolve().parent / "cva_sp_benchmarks.py",
)
cva = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = cva
assert SPEC.loader is not None
SPEC.loader.exec_module(cva)

mod = cva.mod


def _other_next_xy(env, actions):
    next_xy = np.zeros((env.num_uavs, 2), dtype=np.float32)
    for j in range(env.num_uavs):
        act_j = actions[j]
        cur_cov = float(env.covered.mean()) if len(env.covered) else 1.0
        tr = env.t / max(1, env.max_steps)
        ss = float(np.clip(1.0 - 0.14 * cur_cov - 0.08 * tr, 0.75, 1.0))
        next_xy[j] = np.clip(
            env.uav_pos[j, :2]
            + np.clip(act_j[:2], -1.0, 1.0) * (env.max_speed * ss),
            0.0,
            [mod.AREA_X, mod.AREA_Y],
        )
    return next_xy


def intervention_run(seed, dataset, obstacle_ratio=0.12, variant="nearest"):
    """Run one episode with a single filter, recording intervention rows for
    both filters evaluated on the same raw actions and the same state."""
    mod.set_seed(seed)
    env = mod.MultiUAVCoverageEnv(
        dataset, cva.NUM_UAVS, env_type="terrain_urban",
        max_steps=cva.MAX_STEPS, max_speed=cva.MAX_SPEED,
        obstacle_ratio=obstacle_ratio, use_failure_module=False,
        region_reassign_interval=4, num_users=None,
    )
    env.reset()
    occ = cva.occupancy(env)

    rows = []
    for _ in range(cva.MAX_STEPS):
        starts = [cva.xy_cell(p[:2]) for p in env.uav_pos]
        goals = [cva.xy_cell(cva.target_for(env, i)) for i in range(cva.NUM_UAVS)]
        paths = [cva.astar(s, g, occ) for s, g in zip(starts, goals)]
        raw = cva.raw_actions_from_paths(env, paths)

        next_xy = _other_next_xy(env, raw)
        for i in range(cva.NUM_UAVS):
            other = np.delete(next_xy, i, axis=0)
            a_raw = raw[i]
            raw_pos = cva._projected_xy(env, i, a_raw)
            raw_safe = not env._point_in_obstacle(
                np.array([raw_pos[0], raw_pos[1], env.uav_pos[i, 2]],
                         dtype=np.float32))
            raw_ok = raw_safe and np.linalg.norm(other - raw_pos, axis=1).min() >= cva.D_SAFE

            corr_n = cva.nearest_safe_action(env, i, a_raw, other)
            corr_c, trig_c, _ = cva.cva_sp_action(env, i, a_raw, other)

            # feasible candidate count for CVA-SP bounded search
            n_feasible = 0
            for cand in cva.candidate_actions(a_raw):
                pos = cva._projected_xy(env, i, cand)
                if env._point_in_obstacle(
                        np.array([pos[0], pos[1], env.uav_pos[i, 2]],
                                 dtype=np.float32)):
                    continue
                if np.linalg.norm(other - pos, axis=1).min() < cva.D_SAFE:
                    continue
                n_feasible += 1

            dev_n = float(np.linalg.norm(corr_n - a_raw))
            dev_c = float(np.linalg.norm(corr_c - a_raw))
            gain_n = cva.coverage_gain_at(
                env, i, cva._projected_xy(env, i, corr_n), other)
            gain_c = cva.coverage_gain_at(
                env, i, cva._projected_xy(env, i, corr_c), other)

            corrected_n = dev_n > 1e-6
            corrected_c = trig_c and dev_c > 1e-6
            # fallback: CVA-SP search found no feasible candidate but the
            # nearest-point fallback still corrected the action
            fallback = corrected_c and n_feasible == 0
            if corrected_n or corrected_c:
                rows.append({
                    "seed": seed,
                    "step": env.t,
                    "uav": i,
                    "raw_ok": int(raw_ok),
                    "n_feasible": n_feasible,
                    "dev_nearest": round(dev_n, 4),
                    "dev_cva": round(dev_c, 4),
                    "gain_nearest": gain_n,
                    "gain_cva": gain_c,
                    "corrected_nearest": int(corrected_n),
                    "corrected_cva": int(corrected_c),
                    "fallback_cva": int(fallback),
                    "delta_dev_cva_minus_nearest": round(dev_c - dev_n, 4),
                    "delta_gain_cva_minus_nearest": gain_c - gain_n,
                })

        _, _, done, _ = env.step(cva.apply_safety(env, raw, variant)[0])
        if done:
            break
    return rows


def main():
    dataset = mod.ScenarioDataset(cva.DATASET)
    all_rows = []
    for seed in cva.SEEDS:
        for ep in range(cva.EPISODES_PER_SEED):
            all_rows.extend(intervention_run(seed * 1000 + ep, dataset))
        print("seed", seed, "rows so far", len(all_rows), flush=True)

    out = cva.OUT
    with (out / "CVA_SP_Intervention_ByStep.csv").open("w", newline="",
                                                        encoding="utf-8") as f:
        fields = ["seed", "step", "uav", "raw_ok", "n_feasible",
                  "dev_nearest", "dev_cva", "gain_nearest", "gain_cva",
                  "corrected_nearest", "corrected_cva", "fallback_cva",
                  "delta_dev_cva_minus_nearest", "delta_gain_cva_minus_nearest"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)

    n = len(all_rows)
    if n == 0:
        print("no interventions recorded")
        return
    keys = ["dev_nearest", "dev_cva", "gain_nearest", "gain_cva",
            "delta_gain_cva_minus_nearest"]
    summary = {"n_interventions": n,
               "n_fallback_cva": int(sum(r["fallback_cva"] for r in all_rows))}
    for k in keys:
        vals = np.array([r[k] for r in all_rows], dtype=float)
        summary[f"{k}_mean"] = round(float(vals.mean()), 4)
        summary[f"{k}_std"] = round(float(vals.std(ddof=1)), 4)
    gain_diff = np.array([r["delta_gain_cva_minus_nearest"] for r in all_rows],
                         dtype=float)
    summary["gain_cva_gt_nearest_share"] = round(
        float((gain_diff > 0).mean()), 4)
    summary["gain_cva_eq_nearest_share"] = round(
        float((gain_diff == 0).mean()), 4)
    summary["gain_cva_lt_nearest_share"] = round(
        float((gain_diff < 0).mean()), 4)
    with (out / "CVA_SP_Intervention_Summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
