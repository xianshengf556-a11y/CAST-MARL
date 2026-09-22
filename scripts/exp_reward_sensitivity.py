# -*- coding: utf-8 -*-
"""Reward-coefficient sensitivity for the coverage-safety trade-off.

The manuscript reports every reward coefficient; this script adds
a sensitivity analysis.  It script varies the two coefficients that govern the
coverage--safety trade-off --- the newly-covered gain beta_c and the conflicting
pair penalty beta_q --- one at a time around their archived values, retrains
CAST-MARL on the urban terrain with the archived schedule, and evaluates every
setting on the unchanged four-terrain cross-map test set with a fixed
evaluation seed.

Usage (one setting per process, so the sweep parallelises):
  python exp_reward_sensitivity.py --setting beta_c_lo --outdir <dir>
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REV = HERE.parent
REPRO = bp.REPRO
ARCHIVE = bp.ARCHIVE_SUMMARIES
DATASET = bp.DATASETS

_spec = importlib.util.spec_from_file_location(
    "cms", REPRO / "cast_marl_experiment_source.py")
mod = importlib.util.module_from_spec(_spec)
sys.modules["cms"] = mod
assert _spec.loader is not None
_spec.loader.exec_module(mod)
mod.TERRAIN_MANIFEST_PATH = str(REPRO / "missing_terrain_manifest.json")
mod._TERRAIN_CACHE.clear()

TERRAIN = "urban"                       # archived: 120 iterations
ITERS = 120
SEED = 42
EVAL_SEEDS = [42, 43, 44, 45, 46]
EVAL_EPS = 12
EVAL_SEED = 90000 + SEED * 100 + 2

# (label, beta_c, beta_q, beta_c_prime)
SETTINGS = [
    ("baseline", 6.2, 0.45, 3.35),
    ("beta_c_lo", 3.1, 0.45, 1.675),
    ("beta_c_hi", 12.4, 0.45, 6.70),
    ("beta_q_lo", 6.2, 0.225, 3.35),
    ("beta_q_hi", 6.2, 0.90, 3.35),
]

TEST_MAPS = [("plain", "terrain_plain", 0.08),
             ("urban", "terrain_urban", 0.20),
             ("mountain", "terrain_mountain", 0.16),
             ("mountain-dense", "terrain_mountain", 0.30)]


def archived_config():
    p = ARCHIVE / ("terrain_%s_summary.json" % TERRAIN)
    cfgd = json.load(open(p, encoding="utf-8"))["config"]
    fields = mod.ExperimentConfig.__dataclass_fields__
    cfgd = {k: v for k, v in cfgd.items() if k in fields}
    cfgd["dataset_paths"] = list(DATASET)
    cfgd["seed"] = SEED
    cfgd["episodes"] = ITERS
    cfgd["device"] = "cuda" if mod.torch.cuda.is_available() else "cpu"
    return mod.ExperimentConfig(**cfgd)


def make_env(config, dataset, env_type, ratio):
    kw = {k: getattr(config, k) for k in (
        "high_conflict_threshold", "high_conflict_penalty", "reward_gain_coef",
        "reward_coverage_coef", "reward_path_penalty", "reward_conflict_penalty",
        "reward_redundancy_penalty", "reward_obstacle_penalty",
        "assignment_bonus_coef", "frontier_bonus_coef", "stagnation_penalty_coef",
        "region_reassign_interval", "use_region_assignment",
        "region_balance_strength", "region_global_mix",
        "region_support_radius_scale", "heuristic_view_radius_scale",
        "assignment_decay_cover", "assignment_decay_time",
        "frontier_decay_cover", "frontier_decay_time",
        "late_stage_visible_boost", "fail_prob_base", "fail_prob_load_scale",
        "fail_reward_recovery_coef", "fail_energy_fair_coef", "use_failure_module",
        "fail_trigger_min_step", "fail_trigger_max_step", "fail_max_count",
        "fail_fixed_count", "fail_load_aware", "fail_reassign_on_event",
        "camera_fov_deg", "obstacle_height_min", "obstacle_height_max",
        "dynamic_altitude_ratio", "safety_distance")}
    return mod.MultiUAVCoverageEnv(
        dataset=dataset, num_uavs=6, env_type=env_type,
        max_steps=config.max_steps, max_speed=config.max_speed,
        obstacle_ratio=ratio, num_users=None, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    label = args.setting
    row = next((s for s in SETTINGS if s[0] == label), None)
    if row is None:
        raise SystemExit("unknown setting %s" % label)
    _, bc, bq, bcp = row

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = archived_config()
    cfg.reward_gain_coef = bc
    cfg.reward_coverage_coef = bcp
    cfg.reward_conflict_penalty = bq
    dataset = mod.ScenarioDataset(cfg.dataset_paths)

    print("setting=%s beta_c=%.2f beta_c'=%.2f beta_q=%.2f iters=%d"
          % (label, bc, bcp, bq, ITERS), flush=True)
    mod.set_seed(cfg.seed)
    run = mod.train_variant("tmarl", cfg, dataset, str(out / "train"),
                            stage_label="reward-sensitivity")
    if run.get("best_state") is not None:
        run["model"].load_state_dict(run["best_state"])

    rows = []
    for name, env_type, ratio in TEST_MAPS:
        env_cfg = cfg
        env_cfg.obstacle_ratio = ratio
        res = mod.evaluate_controller(
            "CAST-MARL", dataset, env_cfg, model=run["model"],
            env_type=env_type, num_uavs=6, episodes=EVAL_EPS * len(EVAL_SEEDS),
            eval_seed=EVAL_SEED)
        samples = [float(x) for x in res["eval_conflict_samples"]]
        events = int(round(sum(samples) * 35 * 15))
        r = {"setting": label, "beta_c": bc, "beta_c_prime": bcp, "beta_q": bq,
             "map": name,
             "coverage_mean": round(float(res["coverage"]), 4),
             "coverage_std": round(float(np.std(res["eval_coverage_samples"],
                                                ddof=1)), 4),
             "conflict_mean": round(float(np.mean(samples)), 8),
             "conflict_events_total": events,
             "reward_mean": round(float(res["reward"]), 4),
             "path_mean": round(float(res["path_length"]), 2)}
        rows.append(r)
        print("  %-14s cov=%6.3f conf=%.6f reward=%6.2f"
              % (name, r["coverage_mean"], r["conflict_mean"], r["reward_mean"]),
              flush=True)

    with (out / ("reward_sensitivity_%s.csv" % label)).open(
            "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("DONE", label, flush=True)


if __name__ == "__main__":
    main()
