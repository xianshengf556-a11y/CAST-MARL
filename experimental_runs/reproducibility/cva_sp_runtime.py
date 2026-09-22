# -*- coding: utf-8 -*-
"""Per-step runtime of the nearest-point and CVA-SP filters.

Measures wall-clock time for a single filter decision (one UAV, one step)
under the executable A* protocol, plus the full per-step planning loop.
Reported as mean / p95 over 5 seeds x 12 episodes.
"""

from __future__ import annotations

import importlib.util
import json
import statistics
import sys
import time
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


def run_timing(dataset, seeds, episodes_per_seed):
    t_nearest = []
    t_cva = []
    t_plan = []
    for seed in seeds:
        for ep in range(episodes_per_seed):
            mod.set_seed(seed * 1000 + ep)
            env = mod.MultiUAVCoverageEnv(
                dataset, cva.NUM_UAVS, env_type="terrain_urban",
                max_steps=cva.MAX_STEPS, max_speed=cva.MAX_SPEED,
                obstacle_ratio=0.12, use_failure_module=False,
                region_reassign_interval=4, num_users=None,
            )
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
    return t_nearest, t_cva, t_plan


def summarize(vals):
    if not vals:
        return {"n": 0}
    return {
        "n": len(vals),
        "mean_ms": round(float(np.mean(vals)), 3),
        "std_ms": round(float(np.std(vals, ddof=1)), 3),
        "p50_ms": round(float(np.percentile(vals, 50)), 3),
        "p95_ms": round(float(np.percentile(vals, 95)), 3),
    }


def main():
    dataset = mod.ScenarioDataset(cva.DATASET)
    t_n, t_c, t_p = run_timing(dataset, cva.SEEDS, cva.EPISODES_PER_SEED)
    result = {
        "filter_decision_nearest_ms": summarize(t_n),
        "filter_decision_cva_ms": summarize(t_c),
        "planning_step_full_ms": summarize(t_p),
        "num_uavs": cva.NUM_UAVS,
        "max_steps": cva.MAX_STEPS,
        "n_dir": cva.N_DIR,
        "n_amp": cva.N_AMP,
        "platform": "Windows desktop, single-threaded CPython",
        "note": "timing is indicative of filter overhead only; absolute values "
                "depend on hardware and implementation.",
    }
    out = cva.OUT
    (out / "CVA_SP_Runtime.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
