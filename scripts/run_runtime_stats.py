# -*- coding: utf-8 -*-
"""Full distribution of the CVA-SP filter decision time.

The manuscript reported mean and p95 only.  Because the distribution is
strongly right-skewed (rare empty-feasible-set fallbacks), the mean sits ABOVE
the 95th percentile, which reads as an error unless the median and the tail
are reported.  This script re-runs the archived timing protocol and reports
median, p99, max, and the share of decisions above several thresholds, plus
the integer count of the slow tail.

Protocol is copied from the archived cva_sp_runtime.py so the numbers are
directly comparable with CVA_SP_Runtime.json.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

REPRO = (Path(r"E:\从D盘搬迁\科研项目整理\03_无人机路径规划\05_IEEEAccess2026_重投"
              r"\CAST-MARL_IEEE_Access_R1\experiments\experimental_runs\reproducibility"))
OUT = Path(r"E:\从D盘搬迁\科研项目整理\03_无人机路径规划\05_IEEEAccess2026_重投"
           r"\CAST-MARL_IEEE_Access_R1\results\runtime")
OUT.mkdir(parents=True, exist_ok=True)

SPEC = importlib.util.spec_from_file_location(
    "cva_sp_bench", REPRO / "cva_sp_benchmarks.py")
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
            0.0, [mod.AREA_X, mod.AREA_Y])
    return next_xy


def run_timing(dataset, seeds, episodes_per_seed):
    """Identical to the archived run_timing, but also records per-decision
    values so the tail can be characterised."""
    t_nearest, t_cva, t_plan = [], [], []
    for seed in seeds:
        for ep in range(episodes_per_seed):
            mod.set_seed(seed * 1000 + ep)
            env = mod.MultiUAVCoverageEnv(
                dataset, cva.NUM_UAVS, env_type="terrain_urban",
                max_steps=cva.MAX_STEPS, max_speed=cva.MAX_SPEED,
                obstacle_ratio=0.12, use_failure_module=False,
                region_reassign_interval=4, num_users=None)
            env.reset()
            occ = cva.occupancy(env)
            for _ in range(cva.MAX_STEPS):
                t0 = time.perf_counter()
                starts = [cva.xy_cell(p[:2]) for p in env.uav_pos]
                goals = [cva.xy_cell(cva.target_for(env, i))
                         for i in range(cva.NUM_UAVS)]
                paths = [cva.astar(s, g, occ) for s, g in zip(starts, goals)]
                raw = cva.raw_actions_from_paths(env, paths)
                next_xy = _other_next_xy(env, raw)
                t1 = time.perf_counter()
                t_plan.append((t1 - t0) * 1000.0)
                for i in range(cva.NUM_UAVS):
                    other = np.delete(next_xy, i, axis=0)
                    t2 = time.perf_counter()
                    cva.nearest_safe_action(env, i, raw[i], other)
                    t3 = time.perf_counter()
                    t_nearest.append((t3 - t2) * 1000.0)
                    t2b = time.perf_counter()
                    cva.cva_sp_action(env, i, raw[i], other)
                    t3b = time.perf_counter()
                    t_cva.append((t3b - t2b) * 1000.0)
                _, _, done, _ = env.step(
                    cva.apply_safety(env, raw, "nearest")[0])
                if done:
                    break
    return np.asarray(t_nearest), np.asarray(t_cva), np.asarray(t_plan)


def full_summary(v):
    if v.size == 0:
        return {"n": 0}
    d = {
        "n": int(v.size),
        "mean_ms": round(float(v.mean()), 4),
        "std_ms": round(float(v.std(ddof=1)), 4),
        "min_ms": round(float(v.min()), 4),
        "p50_ms": round(float(np.percentile(v, 50)), 4),
        "p90_ms": round(float(np.percentile(v, 90)), 4),
        "p95_ms": round(float(np.percentile(v, 95)), 4),
        "p99_ms": round(float(np.percentile(v, 99)), 4),
        "p999_ms": round(float(np.percentile(v, 99.9)), 4),
        "max_ms": round(float(v.max()), 4),
    }
    for thr in (0.2, 0.5, 1.0, 5.0, 20.0):
        k = int((v > thr).sum())
        d[f"n_gt_{thr}ms"] = k
        d[f"share_gt_{thr}ms"] = round(k / v.size, 6)
    # how much of the total time the slow tail accounts for
    d["tail_share_of_total_time_gt_0.5ms"] = round(
        float(v[v > 0.5].sum() / v.sum()), 6) if v.sum() > 0 else 0.0
    return d


def main():
    dataset = mod.ScenarioDataset(cva.DATASET)
    t_n, t_c, t_p = run_timing(dataset, cva.SEEDS, cva.EPISODES_PER_SEED)

    result = {
        "filter_decision_nearest_ms": full_summary(t_n),
        "filter_decision_cva_ms": full_summary(t_c),
        "planning_step_full_ms": full_summary(t_p),
        "num_uavs": cva.NUM_UAVS,
        "max_steps": cva.MAX_STEPS,
        "seeds": cva.SEEDS,
        "episodes_per_seed": cva.EPISODES_PER_SEED,
        "n_dir": cva.N_DIR,
        "n_amp": cva.N_AMP,
        "environment": "terrain_urban, obstacle_ratio 0.12",
        "platform": "Windows desktop, single-threaded CPython",
        "note": ("Timing is indicative of filter overhead only; absolute "
                 "values depend on hardware and implementation. The CVA-SP "
                 "distribution is right-skewed because a small number of "
                 "decisions fall back to nearest-point projection after an "
                 "empty feasible candidate set."),
        "recorded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (OUT / "CVA_SP_Runtime_Full.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    np.save(OUT / "cva_ms.npy", t_c)
    np.save(OUT / "nearest_ms.npy", t_n)
    print(json.dumps(result, indent=2))
    print("RUNTIME STATS DONE")


if __name__ == "__main__":
    main()
