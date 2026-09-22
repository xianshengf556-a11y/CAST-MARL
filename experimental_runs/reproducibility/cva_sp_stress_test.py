# -*- coding: utf-8 -*-
"""Non-saturated stress test for CVA-SP vs nearest-point projection.

The main CVA-SP comparison runs under the executable A* protocol on the
urban map at obstacle ratio 0.12, where coverage saturates near 99.5% and the
two rules rarely differ in retained coverage.  Here we rerun the same
intervention-level comparison under a denser obstacle layout (ratio 0.30,
mountain-dense level) so that coverage is no longer saturated and the
correction objective has room to differentiate candidates.

Protocol: 5 seeds x 12 episodes, 6 UAVs, 35 steps, unchanged simulator
metrics; both filters evaluated on the identical raw actions and states.
"""

from __future__ import annotations

import csv
import importlib.util
import json
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


def run(seed, dataset, obstacle_ratio):
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
        goals = [cva.xy_cell(cva.target_for(env, i))
                 for i in range(cva.NUM_UAVS)]
        paths = [cva.astar(s, g, occ) for s, g in zip(starts, goals)]
        raw = cva.raw_actions_from_paths(env, paths)
        next_xy = _other_next_xy(env, raw)
        for i in range(cva.NUM_UAVS):
            other = np.delete(next_xy, i, axis=0)
            a_raw = raw[i]
            corr_n = cva.nearest_safe_action(env, i, a_raw, other)
            corr_c, trig_c, _ = cva.cva_sp_action(env, i, a_raw, other)
            dev_n = float(np.linalg.norm(corr_n - a_raw))
            dev_c = float(np.linalg.norm(corr_c - a_raw))
            gain_n = cva.coverage_gain_at(
                env, i, cva._projected_xy(env, i, corr_n), other)
            gain_c = cva.coverage_gain_at(
                env, i, cva._projected_xy(env, i, corr_c), other)
            if dev_n > 1e-6 or (trig_c and dev_c > 1e-6):
                rows.append({
                    "seed": seed,
                    "step": env.t,
                    "uav": i,
                    "dev_nearest": round(dev_n, 4),
                    "dev_cva": round(dev_c, 4),
                    "gain_nearest": gain_n,
                    "gain_cva": gain_c,
                    "delta_gain": gain_c - gain_n,
                })
        _, _, done, _ = env.step(cva.apply_safety(env, raw, "nearest")[0])
        if done:
            break
    return rows


def main():
    dataset = mod.ScenarioDataset(cva.DATASET)
    ratio = 0.30
    all_rows = []
    for seed in cva.SEEDS:
        for ep in range(cva.EPISODES_PER_SEED):
            all_rows.extend(run(seed * 1000 + ep, dataset, ratio))
        print("seed", seed, "rows", len(all_rows), flush=True)
    out = cva.OUT
    with (out / "CVA_SP_Stress_ByStep.csv").open("w", newline="",
                                                 encoding="utf-8") as f:
        fields = list(all_rows[0].keys())
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_rows)
    n = len(all_rows)
    g = np.array([r["delta_gain"] for r in all_rows], dtype=float)
    summary = {
        "obstacle_ratio": ratio,
        "n_interventions": n,
        "coverage_gain_nearest_mean": round(float(np.mean(
            [r["gain_nearest"] for r in all_rows])), 4),
        "coverage_gain_cva_mean": round(float(np.mean(
            [r["gain_cva"] for r in all_rows])), 4),
        "delta_gain_mean": round(float(g.mean()), 4),
        "cva_gt_share": round(float((g > 0).mean()), 4),
        "cva_eq_share": round(float((g == 0).mean()), 4),
        "cva_lt_share": round(float((g < 0).mean()), 4),
        "dev_nearest_mean": round(float(np.mean(
            [r["dev_nearest"] for r in all_rows])), 4),
        "dev_cva_mean": round(float(np.mean(
            [r["dev_cva"] for r in all_rows])), 4),
    }
    with (out / "CVA_SP_Stress_Summary.json").open("w",
                                                   encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
