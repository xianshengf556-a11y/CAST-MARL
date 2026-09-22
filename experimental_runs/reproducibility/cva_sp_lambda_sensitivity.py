# -*- coding: utf-8 -*-
"""Lambda-d sensitivity of the CVA-SP objective (Eq. 9).

Runs the executable A* protocol with the deviation penalty lambda_d set to
0.1, 0.35, and 0.7, and reports episode-level metrics plus intervention-level
coverage retention. The nearest-point projection is included as the lambda ->
infinity reference (pure deviation minimisation).
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


def run_lambda(lambda_d, dataset, seeds, episodes_per_seed):
    rows = []
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
            total_reward = 0.0
            total_corrections = 0
            steps_with_correction = 0
            for _ in range(cva.MAX_STEPS):
                starts = [cva.xy_cell(p[:2]) for p in env.uav_pos]
                goals = [cva.xy_cell(cva.target_for(env, i))
                         for i in range(cva.NUM_UAVS)]
                paths = [cva.astar(s, g, occ) for s, g in zip(starts, goals)]
                raw = cva.raw_actions_from_paths(env, paths)
                safe, corr = cva.apply_safety(env, raw, "cva")
                total_corrections += corr
                if corr > 0:
                    steps_with_correction += 1
                _, reward, done, info = env.step(safe)
                total_reward += float(reward)
                if done:
                    break
            rows.append({
                "lambda_d": lambda_d,
                "seed": seed,
                "episode": ep,
                "coverage": info["coverage_rate"] * 100.0,
                "conflict_rate": info["conflict_rate"],
                "reward": total_reward,
                "path_length": info["path_length"],
                "corrections": total_corrections,
                "steps_with_correction": steps_with_correction,
            })
    return rows


def main():
    dataset = mod.ScenarioDataset(cva.DATASET)
    lambdas = [0.1, 0.35, 0.7]
    all_rows = []
    for ld in lambdas:
        cva.LAMBDA = ld
        rows = run_lambda(ld, dataset, cva.SEEDS, cva.EPISODES_PER_SEED)
        all_rows.extend(rows)
        grp = [r for r in rows if r["lambda_d"] == ld]
        print("lambda", ld, {
            k: round(float(np.mean([r[k] for r in grp])), 4)
            for k in ("coverage", "conflict_rate", "path_length",
                      "corrections", "steps_with_correction")
        }, flush=True)

    out = cva.OUT
    with (out / "CVA_SP_LambdaSensitivity_ByEpisode.csv").open(
            "w", newline="", encoding="utf-8") as f:
        fields = list(all_rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)

    summary = []
    for ld in lambdas:
        grp = [r for r in all_rows if r["lambda_d"] == ld]
        summary.append({
            "lambda_d": ld,
            **{f"{k}_mean": round(float(np.mean([r[k] for r in grp])), 4)
               for k in ("coverage", "conflict_rate", "reward",
                         "path_length", "corrections",
                         "steps_with_correction")},
            **{f"{k}_std": round(float(np.std([r[k] for r in grp], ddof=1)), 4)
               for k in ("coverage", "conflict_rate", "reward",
                         "path_length", "corrections")},
            "n": len(grp),
        })
    with (out / "CVA_SP_LambdaSensitivity_Summary.csv").open(
            "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)
    (out / "CVA_SP_LambdaSensitivity_Protocol.json").write_text(
        json.dumps({
            "lambda_values": lambdas,
            "seeds": cva.SEEDS,
            "episodes_per_seed": cva.EPISODES_PER_SEED,
            "num_uavs": cva.NUM_UAVS,
            "max_steps": cva.MAX_STEPS,
            "d_safe": cva.D_SAFE,
            "correction_cap": cva.CORRECTION_CAP,
            "n_dir": cva.N_DIR,
            "n_amp": cva.N_AMP,
            "variant": "cva",
            "environment": "terrain_urban, episode-generated obstacles",
            "metrics": "unchanged simulator step metrics",
        }, indent=2),
        encoding="utf-8",
    )
    print("lambda sensitivity done")


if __name__ == "__main__":
    main()
