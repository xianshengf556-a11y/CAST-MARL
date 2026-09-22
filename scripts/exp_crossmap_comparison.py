# -*- coding: utf-8 -*-
"""Cross-map comparison of planners and safety layers.

This script compares A* + CVA-SP, CAST-MARL (zero-shot), and A* (no safety layer)
on the cross-map test set, so that the effect of the planner can be separated
from the effect of the safety layer.  It also evaluates the greedy planners on
unseen maps, which the earlier tables reported only on the fixed urban map.

This script evaluates, on the four cross-map conditions (plain, urban, mountain,
mountain-dense) and on identical episodes:
  * A* with no safety layer, A* + nearest-point, A* + CVA-SP
  * cluster-, nearest-, and frontier-greedy
and prints the archived zero-shot CAST-MARL row for the same condition so the
three-way comparison is visible in one place.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

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

OUT = bp.RESULTS / "crossmap"
OUT.mkdir(parents=True, exist_ok=True)

# label -> (env_type, obstacle ratio), matching the cross-map protocol
TERRAINS = [
    ("plain", "terrain_plain", 0.08),
    ("urban", "terrain_urban", 0.20),
    ("mountain", "terrain_mountain", 0.16),
    ("mountain-dense", "terrain_mountain", 0.30),
]
SEEDS = [42, 43, 44, 45, 46]
EPISODES_PER_SEED = 12
NUM_UAVS = 6
EVAL_SEED_BASE = 90000
GREEDY = [("Cluster-Greedy", "cluster_greedy"),
          ("Nearest-Greedy", "nearest_greedy"),
          ("Frontier-Greedy", "frontier_greedy")]


def greedy_config(ratio: int | float):
    cfg = mod.ExperimentConfig()
    cfg.obstacle_ratio = float(ratio)
    cfg.device = "cuda" if mod.torch.cuda.is_available() else "cpu"
    return cfg


def main():
    dataset = mod.ScenarioDataset(cva.DATASET)
    rows = []

    # ---- A* protocol: raw / nearest / cva -------------------------------
    for label, env_type, ratio in TERRAINS:
        for variant in ("raw", "nearest", "cva"):
            cov, conf, path = [], [], []
            for seed in SEEDS:
                for ep in range(EPISODES_PER_SEED):
                    r = fb.run_episode(seed * 1000 + ep, dataset, env_type,
                                       ratio, NUM_UAVS, variant=variant)
                    cov.append(r["episode_coverage"])
                    conf.append(r["episode_conflict_sim"])
                    path.append(r["episode_path"])
            cov = np.asarray(cov); conf = np.asarray(conf); path = np.asarray(path)
            name = {"raw": "A* (no safety layer)",
                    "nearest": "A* + nearest-point",
                    "cva": "A* + CVA-SP"}[variant]
            rows.append({
                "map": label, "family": "A* protocol", "method": name,
                "episodes": len(cov),
                "coverage_mean": round(float(cov.mean()), 4),
                "coverage_std": round(float(cov.std(ddof=1)), 4),
                "conflict_mean": round(float(conf.mean()), 8),
                "conflict_events_total": int(round(float(conf.sum()) * 35 * 15)),
                "path_mean": round(float(path.mean()), 2),
                "path_std": round(float(path.std(ddof=1)), 2),
            })
            print("%-15s %-22s cov=%6.3f +- %5.3f  conf=%.6f  path=%9.1f"
                  % (label, name, cov.mean(), cov.std(ddof=1), conf.mean(),
                     path.mean()), flush=True)

    # ---- greedy planners on the same maps -------------------------------
    for label, env_type, ratio in TERRAINS:
        cfg = greedy_config(ratio)
        for display, key in GREEDY:
            res = mod.evaluate_controller(
                key, dataset, cfg, model=None, env_type=env_type,
                num_uavs=NUM_UAVS, episodes=EPISODES_PER_SEED * len(SEEDS),
                eval_seed=EVAL_SEED_BASE + int(ratio * 1000))
            samples = [float(x) for x in res["eval_conflict_samples"]]
            events = int(round(sum(samples) * 35 * 15))
            rows.append({
                "map": label, "family": "greedy", "method": display,
                "episodes": len(samples),
                "coverage_mean": round(float(res["coverage"]), 4),
                "coverage_std": round(float(np.std(res["eval_coverage_samples"],
                                                   ddof=1)), 4),
                "conflict_mean": round(float(np.mean(samples)), 8),
                "conflict_events_total": events,
                "path_mean": round(float(res["path_length"]), 2),
                "path_std": 0.0,
            })
            print("%-15s %-22s cov=%6.3f  conf=%.6f  path=%9.1f"
                  % (label, display, res["coverage"], float(np.mean(samples)),
                     res["path_length"]), flush=True)

    keys = list(rows[0].keys())
    with (OUT / "CrossMap_Comparison.csv").open("w", newline="",
                                                encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    (OUT / "CrossMap_Comparison_Protocol.json").write_text(json.dumps({
        "terrains": [{"label": l, "env_type": e, "obstacle_ratio": r}
                     for l, e, r in TERRAINS],
        "num_uavs": NUM_UAVS, "seeds": SEEDS,
        "episodes_per_seed": EPISODES_PER_SEED,
        "astar_protocol": "cva_sp_benchmarks (20x20 grid, 135 m inflation)",
        "greedy_protocol": "evaluate_controller(model=None, baseline_name=...)",
        "note": ("the archived zero-shot CAST-MARL rows for the same conditions "
                 "are reported in the manuscript and in the archived cross-map "
                 "records"),
    }, indent=2), encoding="utf-8")
    print("\nwrote", OUT / "CrossMap_Comparison.csv")


if __name__ == "__main__":
    main()
