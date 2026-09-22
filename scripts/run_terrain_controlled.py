# -*- coding: utf-8 -*-
"""Controlled re-run of the terrain benchmark and the component ablation.

PROTOCOL
--------
This driver applies the evaluation protocol documented in the paper: training
and evaluation use the same named terrain, the evaluation seed is fixed and
recorded, five training seeds are used, one model is trained per process so
that every model starts from the same seeded state, and per-episode conflict
samples are exported so that integer pair-event counts can be recovered.

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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bundle_paths as bp                              # noqa: E402

REPRO = bp.REPRO
ARCHIVE_SUMMARIES = bp.ARCHIVE_SUMMARIES
LOCAL_DATASET = bp.DATASETS
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

# Bundle-local terrain-manifest path, absent by design: the 3-D terrain asset
# used by the original April 2026 runs is not part of the released artifacts,
# so the terrain label drives obstacle generation over a flat surface.  Set
# explicitly so that training and evaluation see identical geometry.
mod.TERRAIN_MANIFEST_PATH = str(REPRO / "missing_terrain_manifest.json")
mod._TERRAIN_CACHE.clear()

# The five compared methods and the four component variants.  A job may train
# all of them, one half, or a single model, so that a many-core machine is fully
# used.
MAIN_LIST = [("tmarl", "CAST-MARL"), ("mappo", "MAPPO"),
             ("maddpg", "MADDPG"), ("qmix", "QMIX"), ("ppo", "PPO")]
ABL_LIST = list(mod.MAINLINE_ABLATION_VARIANTS)
MAIN_NAMES = {n for n, _ in MAIN_LIST}


def select_models(group):
    """Return (main_models, ablation_models) selected by --group."""
    if group == "all":
        return list(MAIN_LIST), list(ABL_LIST)
    if group == "main":
        return list(MAIN_LIST), []
    if group == "ablation":
        return [], list(ABL_LIST)
    m = [x for x in MAIN_LIST if x[0] == group]
    if m:
        return m, []
    a = [x for x in ABL_LIST if x[0] == group]
    if a:
        return [], a
    raise SystemExit("unknown --group value: %s" % group)


def train_one(name, config, dataset, output_dir):
    if name == "maddpg":
        return mod.train_maddpg_variant(config, dataset, output_dir,
                                        stage_label="main")
    if name == "qmix":
        return mod.train_qmix_variant(config, dataset, output_dir,
                                      stage_label="main")
    return mod.train_variant(name, config, dataset, output_dir,
                             stage_label=("main" if name in MAIN_NAMES
                                          else "ablation"))


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


def train_all(config, dataset, output_dir, main_models, abl_models):
    """Train the selected models.

    set_seed is called before the first model, and because a job normally
    trains a single model it means every model starts from the same seeded
    state.  The archived pipeline trained all variants in one process, so each
    variant inherited a different RNG offset from the variants trained before
    it; that offset is a training-noise confound in a component ablation, and
    splitting the runs removes it.
    """
    mod.set_seed(config.seed)
    runs = {}
    t0 = time.time()
    for name, _display in list(main_models) + list(abl_models):
        runs[name] = train_one(name, config, dataset, output_dir)
        print("  trained %-16s %7.1fs" % (name, time.time() - t0), flush=True)
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
    ap.add_argument("--group", default="all",
                    help="which models this process trains: 'all', 'main' (the "
                         "five compared methods), 'ablation' (the four variants), "
                         "or a single model name such as tmarl or no_attention")
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
        main_models, abl_models = select_models(args.group)
        runs, train_s = train_all(cfg, dataset, str(out / "train_artifacts"),
                                  main_models, abl_models)

        bench = []
        for name, display in main_models:
            r = evaluate(name, display, cfg, dataset, runs[name], eval_seed)
            bench.append(r)
            print("  EVAL %-10s cov=%.3f conflict=%.6f events=%d/%d eps"
                  % (display, r["coverage_mean"], r["conflict_rate_mean"],
                     r["episodes_with_conflict"], EVAL_EPISODES), flush=True)

        abl = []
        for name, display in abl_models:
            r = evaluate(name, display, cfg, dataset, runs[name], eval_seed)
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
        group_suffix = "" if args.group == "all" else "_" + args.group
        (out / ("controlled_%s%s.json" % (tag, group_suffix))).write_text(
            json.dumps(payload, indent=2), encoding="utf-8")
        print("WROTE controlled_%s%s.json  (%.1f min total)"
              % (tag, group_suffix, (time.time() - t_start) / 60.0), flush=True)
        print("CONTROLLED RUN DONE", flush=True)
    except Exception:
        traceback.print_exc()
        (out / ("FAILED_%s_seed%d.txt" % (args.terrain, args.seed))).write_text(
            traceback.format_exc(), encoding="utf-8")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
