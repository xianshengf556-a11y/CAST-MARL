# -*- coding: utf-8 -*-
"""Controlled re-run of the terrain benchmark and the component ablation.

WHY THIS SCRIPT EXISTS
----------------------
The archived terrain tables (paper Table 2 and Table 4) were produced by
run_train_pipeline(), whose evaluation calls hardcode the evaluation
environment:

    line 5829  CAST-MARL  -> evaluate_controller(...)                    # env_type defaults to "simple"
    lines 5830-5833  MAPPO/MADDPG/QMIX/PPO -> env_type="simple"
    line 5851  ALL ablation variants       -> env_type="dynamic_users"

and the ablation evaluation passes no `eval_seed`, so it is not even
deterministic.  The tables are therefore labelled "terrain-wise" while the
evaluation environment is the same for every row, and the ablation was not
evaluated on terrain at all.

This script fixes the protocol:
  * training env  = the named terrain (unchanged, same hyperparameters)
  * evaluation env = the SAME named terrain (config.train_env)
  * evaluation is seeded with a fixed, documented eval_seed
  * episodes = 20 for every method and every variant (more data than the
    archived 20/12 split, to give the conflict statistics a chance)
  * per-episode conflict SAMPLES are stored so integer pair-event counts can
    be recovered exactly from the per-episode records

Everything else (network, reward, PPO settings, per-terrain hyperparameters)
is taken verbatim from the archived summary.json of the corresponding terrain,
so the ONLY change relative to the archive is the evaluation protocol.

ONE (terrain, seed) PER PROCESS so the run can be parallelised.

Usage:
  python run_terrain_controlled.py --terrain plain --seed 42 --outdir <dir>
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

REPRO = (Path(r"E:\从D盘搬迁\科研项目整理\03_无人机路径规划\05_IEEEAccess2026_重投"
              r"\CAST-MARL_IEEE_Access_R1\experiments\experimental_runs\reproducibility"))
ARCHIVE_SUMMARIES = Path(
    r"E:\从D盘搬迁\科研项目整理\03_无人机路径规划\无人机路径规划\无人机路径规划"
    r"\terrain_assets_central_20260416\summaries")

# Local dataset paths (the archived config points at /root/autodl-tmp/...).
_DS = (r"C:\Users\2460566145\Documents\无人机\_source_extract\无人机路径规划"
       r"\无人机部署\dataset")
LOCAL_DATASET = [os.path.join(_DS, f) for f in
                 ("dataset1_users35.npz", "dataset1_users45.npz",
                  "dataset1_users60.npz")]
for _p in LOCAL_DATASET:
    if not os.path.exists(_p):
        raise SystemExit("dataset missing: %s" % _p)

EVAL_EPISODES = 20
MAX_STEPS = 35
NUM_PAIRS = 15                       # C(6,2) at the evaluated fleet size

_spec = importlib.util.spec_from_file_location(
    "cms", REPRO / "cast_marl_experiment_source.py")
mod = importlib.util.module_from_spec(_spec)
sys.modules["cms"] = mod
assert _spec.loader is not None
_spec.loader.exec_module(mod)

MAIN_VARIANTS = [("tmarl", "CAST-MARL"), ("mappo", "MAPPO"),
                 ("ppo", "PPO")]          # trained via train_variant


def archived_config(terrain: str, seed: int, device: str, out_root: str):
    """Load the archived per-terrain config verbatim, overriding only seed,
    dataset paths, device and output root."""
    p = ARCHIVE_SUMMARIES / ("terrain_%s_summary.json" % terrain)
    cfgd = json.load(open(p, encoding="utf-8"))["config"]
    fields = mod.ExperimentConfig.__dataclass_fields__
    cfgd = {k: v for k, v in cfgd.items() if k in fields}
    cfgd["dataset_paths"] = list(LOCAL_DATASET)
    cfgd["seed"] = seed
    cfgd["device"] = device
    if "output_root" in fields:
        cfgd["output_root"] = out_root
    return mod.ExperimentConfig(**cfgd)


def train_all(config, dataset, output_dir: str):
    """Mirror run_train_pipeline's training sequence exactly (L5800-5820),
    but keep the run objects so we can evaluate with our own protocol."""
    mod.set_seed(config.seed)
    runs = {}
    t0 = time.time()
    runs["tmarl"] = mod.train_variant("tmarl", config, dataset, output_dir,
                                      stage_label="main")
    print("  trained tmarl  %.1fs" % (time.time() - t0), flush=True)
    runs["mappo"] = mod.train_variant("mappo", config, dataset, output_dir,
                                      stage_label="main")
    runs["maddpg"] = mod.train_maddpg_variant(config, dataset, output_dir,
                                              stage_label="main")
    runs["qmix"] = mod.train_qmix_variant(config, dataset, output_dir,
                                          stage_label="main")
    runs["ppo"] = mod.train_variant("ppo", config, dataset, output_dir,
                                    stage_label="main")
    for variant_name, display in mod.MAINLINE_ABLATION_VARIANTS:
        runs[variant_name] = mod.train_variant(variant_name, config, dataset,
                                               output_dir,
                                               stage_label="ablation")
        print("  trained %-14s %.1fs" % (variant_name, time.time() - t0),
              flush=True)
    for name, run in runs.items():
        if run.get("best_state") is not None:
            run["model"].load_state_dict(run["best_state"])
    return runs, time.time() - t0


def conflict_event_check(samples):
    """Recover integer pair-event counts k from per-episode rates k/(t*15).

    Returns (events_list, all_integral).
    """
    ev, ok = [], True
    for s in samples:
        k = float(s) * MAX_STEPS * NUM_PAIRS
        kr = round(k)
        if abs(k - kr) > 1e-6:
            ok = False
        ev.append(int(kr))
    return ev, ok


def evaluate(name, display, cfg, dataset, run, eval_seed):
    res = mod.evaluate_controller(
        display, dataset, cfg, model=run["model"], env_type=cfg.train_env,
        num_uavs=cfg.train_num_uavs, episodes=EVAL_EPISODES,
        eval_seed=eval_seed)
    samples = [float(x) for x in res["eval_conflict_samples"]]
    events, integral = conflict_event_check(samples)
    cov = [float(x) for x in res["eval_coverage_samples"]]
    return {
        "method": display,
        "coverage_mean": float(np.mean(cov)),
        "coverage_std": float(np.std(cov, ddof=1)),
        "coverage_samples": cov,
        "conflict_rate_mean": float(np.mean(samples)),
        "conflict_rate_std": float(np.std(samples, ddof=1)),
        "conflict_samples": samples,
        "conflict_events": events,
        "conflict_events_total": int(sum(events)),
        "episodes_with_conflict": int(sum(1 for e in events if e > 0)),
        "max_events_single_episode": int(max(events)) if events else 0,
        "event_counts_integral": bool(integral),
        "path_length_mean": float(res["path_length"]),
        "reward_mean": float(res["reward"]),
        "task_time_mean": float(res["task_time"]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--terrain", required=True,
                    choices=["plain", "urban", "mountain"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--eval-seed-base", type=int, default=90000)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny run that exercises every code path, for validation only")
    args = ap.parse_args()

    global EVAL_EPISODES

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if mod.torch.cuda.is_available() else "cpu"
    cfg = archived_config(args.terrain, args.seed, device, str(out))
    if args.smoke:
        cfg.episodes = 2
        EVAL_EPISODES = 2
        print("SMOKE MODE: episodes=2, eval_episodes=2", flush=True)
    # deterministic eval seed (never use hash(): it is salted per process)
    terrain_offset = {"plain": 1, "urban": 2, "mountain": 3}[args.terrain]
    eval_seed = args.eval_seed_base + args.seed * 100 + terrain_offset

    print("=" * 74, flush=True)
    print("terrain=%s seed=%d device=%s train_env=%s eval_env=%s "
          "eval_seed=%d episodes=%d ppo_iters=%d"
          % (args.terrain, args.seed, device, cfg.train_env, cfg.train_env,
             eval_seed, EVAL_EPISODES, cfg.episodes), flush=True)
    print("=" * 74, flush=True)

    dataset = mod.ScenarioDataset(cfg.dataset_paths)
    t_start = time.time()
    try:
        runs, train_s = train_all(cfg, dataset, str(out / "train_artifacts"))

        bench = []
        for name, display in MAIN_VARIANTS + [("maddpg", "MADDPG"),
                                              ("qmix", "QMIX")]:
            r = evaluate(name, display, cfg, dataset, runs[name], eval_seed)
            bench.append(r)
            print("  EVAL %-10s cov=%.3f conflict=%.6f events=%d/%d eps"
                  % (display, r["coverage_mean"], r["conflict_rate_mean"],
                     r["episodes_with_conflict"], EVAL_EPISODES), flush=True)

        abl = []
        for variant_name, display in mod.MAINLINE_ABLATION_VARIANTS:
            r = evaluate(variant_name, display, cfg, dataset,
                         runs[variant_name], eval_seed)
            abl.append(r)
            print("  ABL  %-16s cov=%.3f conflict=%.6f events=%d/%d eps"
                  % (display, r["coverage_mean"], r["conflict_rate_mean"],
                     r["episodes_with_conflict"], EVAL_EPISODES), flush=True)

        payload = {
            "terrain": args.terrain,
            "seed": args.seed,
            "train_env": cfg.train_env,
            "eval_env": cfg.train_env,          # <-- the fix
            "eval_seed": eval_seed,
            "eval_episodes": EVAL_EPISODES,
            "max_steps": MAX_STEPS,
            "num_uavs": cfg.train_num_uavs,
            "ppo_iterations": cfg.episodes,
            "batch_episodes": cfg.batch_episodes,
            "obstacle_ratio": cfg.obstacle_ratio,
            "terrain_clearance": cfg.terrain_clearance,
            "terrain_height_cap": cfg.terrain_height_cap,
            "train_seconds": round(train_s, 1),
            "wall_seconds": round(time.time() - t_start, 1),
            "benchmark": bench,
            "ablation": abl,
            "protocol_note": (
                "Evaluation environment is the training terrain for every row "
                "(the archived tables evaluated all rows on 'simple' and the "
                "ablation on 'dynamic_users'). Evaluation is seeded with a "
                "fixed eval_seed, so the protocol is reproducible."),
        }
        tag = "%s_seed%d" % (args.terrain, args.seed)
        (out / ("controlled_%s.json" % tag)).write_text(
            json.dumps(payload, indent=2), encoding="utf-8")
        print("WROTE controlled_%s.json  (%.1f min total)"
              % (tag, (time.time() - t_start) / 60.0), flush=True)
        print("CONTROLLED RUN DONE", flush=True)
    except Exception:
        traceback.print_exc()
        (out / ("FAILED_%s_seed%d.txt" % (args.terrain, args.seed))).write_text(
            traceback.format_exc(), encoding="utf-8")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
