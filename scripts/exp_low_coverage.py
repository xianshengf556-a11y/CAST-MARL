# -*- coding: utf-8 -*-
"""Low-coverage regime test for CVA-SP.

MOTIVATION
----------
Under the archived executable A* protocol (6 UAVs, 35 decision steps) coverage
saturates at ~99.5%, so all admissible correction candidates cover almost the
same set of remaining cells and the coverage-aware term of the CVA-SP objective
has almost no room to differentiate candidates.  This script therefore
tests a less saturated regime, in which the
difference between the nearest-point rule and CVA-SP would be observable in
practice rather than only statistically.

DESIGN
------
The A* protocol of cva_sp_benchmarks.py is reproduced unchanged (same 20x20
occupancy grid, 135 m inflation, same target rule, same simulator step function)
while the coverage pressure is reduced along two axes:
fewer vehicles relative to the fixed 5000x5000 m workspace, and a shorter
decision horizon.  The archived configuration (6 UAVs, 35 steps) is kept as the
saturated reference point.

For every configuration the three archived variants are evaluated on identical
episodes (same seeds and episode indices): raw A* (no safety layer), A* with
nearest-point projection, and A* with CVA-SP.  Because every variant sees the
same episode, the nearest-versus-CVA-SP difference is paired, and we report it
with a paired t-test so the effect size is visible next to the p-value.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bundle_paths as bp                              # noqa: E402

REV = bp.ROOT

_spec = importlib.util.spec_from_file_location("fb", HERE / "exp_fallback_threshold.py")
fb = importlib.util.module_from_spec(_spec)
sys.modules["fb"] = fb
assert _spec.loader is not None
_spec.loader.exec_module(fb)

mod = fb.mod
cva = fb.cva

OUT = bp.RESULTS / "low_coverage"
OUT.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 43, 44, 45, 46]
EPISODES_PER_SEED = 12
MAX_SPEED = 150.0

# (num_uavs, max_steps, label) -- decreasing coverage pressure
CONFIGS = [
    (6, 35, "6 UAVs / 35 steps (archived, saturated)"),
    (4, 20, "4 UAVs / 20 steps"),
    (3, 15, "3 UAVs / 15 steps"),
    (2, 12, "2 UAVs / 12 steps"),
]
VARIANTS = ["raw", "nearest", "cva"]


def run_episode(seed, dataset, num_uavs, max_steps, variant, d_safe=138.0, lam=0.35):
    """Same protocol as exp_fallback_threshold.run_episode, with the horizon
    and the fleet size exposed."""
    mod.set_seed(seed)
    env = mod.MultiUAVCoverageEnv(
        dataset, num_uavs, env_type="terrain_urban", max_steps=max_steps,
        max_speed=MAX_SPEED, obstacle_ratio=0.12, use_failure_module=False,
        region_reassign_interval=4, num_users=None)
    env.reset()
    occ = cva.occupancy(env)
    corrections = 0
    for _ in range(max_steps):
        starts = [cva.xy_cell(p[:2]) for p in env.uav_pos]
        goals = [cva.xy_cell(cva.target_for(env, i)) for i in range(num_uavs)]
        paths = [cva.astar(s, g, occ) for s, g in zip(starts, goals)]
        raw = cva.raw_actions_from_paths(env, paths)
        next_xy = fb._next_xy_all(env, raw)
        out = raw.copy()
        for i in range(num_uavs):
            other = np.delete(next_xy, i, axis=0)
            if variant == "raw":
                out[i, :2] = raw[i, :2]
            elif variant == "nearest":
                corr = fb.nearest_safe_action_p(env, i, raw[i], other, d_safe)
                out[i, :2] = corr[:2]
                corrections += int(np.linalg.norm(corr - raw[i]) > 1e-6)
            else:
                corr, trig, _s, _st, _nf = fb.cva_sp_action_p(
                    env, i, raw[i], other, d_safe, lam)
                out[i, :2] = corr[:2]
                corrections += int(trig)
        _, reward, done, info = env.step(out)
        if done:
            break
    return {
        "coverage": float(info["coverage_rate"]) * 100.0,
        "conflict_rate": float(info["conflict_rate"]),
        "path_length": float(info["path_length"]),
        "corrections": corrections,
    }


def main():
    dataset = mod.ScenarioDataset(cva.DATASET)
    rows = []
    for num_uavs, max_steps, label in CONFIGS:
        per_variant = {}
        for variant in VARIANTS:
            vals = []
            for seed in SEEDS:
                for ep in range(EPISODES_PER_SEED):
                    vals.append(run_episode(seed * 1000 + ep, dataset,
                                            num_uavs, max_steps, variant))
            per_variant[variant] = vals
            cov = np.array([v["coverage"] for v in vals])
            conf = np.array([v["conflict_rate"] for v in vals])
            path = np.array([v["path_length"] for v in vals])
            corr = np.array([v["corrections"] for v in vals])
            rows.append({
                "config": label, "num_uavs": num_uavs, "max_steps": max_steps,
                "variant": variant, "episodes": len(vals),
                "coverage_mean": round(float(cov.mean()), 4),
                "coverage_std": round(float(cov.std(ddof=1)), 4),
                "conflict_mean": round(float(conf.mean()), 8),
                "conflict_events_total": int(round(float(conf.sum()) * max_steps
                                                   * (num_uavs * (num_uavs - 1) // 2))),
                "path_mean": round(float(path.mean()), 2),
                "corrections_mean": round(float(corr.mean()), 3),
            })
            print("%-38s %-8s cov=%6.3f +- %5.3f  conf=%.6f  path=%9.1f  corr=%.2f"
                  % (label, variant, cov.mean(), cov.std(ddof=1), conf.mean(),
                     path.mean(), corr.mean()), flush=True)

        # paired nearest vs cva on the same episodes
        n_cov = np.array([v["coverage"] for v in per_variant["nearest"]])
        c_cov = np.array([v["coverage"] for v in per_variant["cva"]])
        n_path = np.array([v["path_length"] for v in per_variant["nearest"]])
        c_path = np.array([v["path_length"] for v in per_variant["cva"]])
        t_cov, p_cov = stats.ttest_rel(c_cov, n_cov)
        t_path, p_path = stats.ttest_rel(c_path, n_path)
        d_cov = c_cov - n_cov
        print("    paired CVA-SP - nearest: coverage diff %+.4f pp (t=%.3f, p=%.3g), "
              "path diff %+.2f m (t=%.3f, p=%.3g)"
              % (d_cov.mean(), t_cov, p_cov, (c_path - n_path).mean(),
                 t_path, p_path), flush=True)
        rows.append({
            "config": label, "num_uavs": num_uavs, "max_steps": max_steps,
            "variant": "PAIRED_cva_minus_nearest", "episodes": len(d_cov),
            "coverage_mean": round(float(d_cov.mean()), 4),
            "coverage_std": round(float(d_cov.std(ddof=1)), 4),
            "conflict_mean": "", "conflict_events_total": "",
            "path_mean": round(float((c_path - n_path).mean()), 2),
            "corrections_mean": "",
            "paired_t_coverage": round(float(t_cov), 4),
            "paired_p_coverage": float(p_cov),
            "paired_t_path": round(float(t_path), 4),
            "paired_p_path": float(p_path),
        })

    with (OUT / "LowCoverage_Results.csv").open("w", newline="",
                                                encoding="utf-8") as f:
        keys = sorted({k for r in rows for k in r})
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    (OUT / "LowCoverage_Protocol.json").write_text(json.dumps({
        "configs": [{"num_uavs": n, "max_steps": s, "label": l}
                    for n, s, l in CONFIGS],
        "variants": VARIANTS,
        "seeds": SEEDS,
        "episodes_per_seed": EPISODES_PER_SEED,
        "environment": "terrain_urban label, episode-generated obstacles, ratio 0.12",
        "d_safe": 138.0, "lambda_d": 0.35,
        "purpose": ("reduce coverage pressure along two axes "
                    "(fleet size relative to a fixed workspace, and "
                    "horizon length) so that the coverage-aware correction term "
                    "has room to differentiate admissible candidates"),
    }, indent=2), encoding="utf-8")
    print("\nwrote", OUT / "LowCoverage_Results.csv")
    print("LOW-COVERAGE EXPERIMENT DONE")


if __name__ == "__main__":
    main()
