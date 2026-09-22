# -*- coding: utf-8 -*-
"""R1-3 / R3-4 fair-budget control: single-terrain continued training.

Reviewer concern
----------------
The archived multi-domain run continued from the single-terrain plain
checkpoint for 160 PPO iterations on a 4-terrain mix.  The single-terrain
reference checkpoint had only 80 iterations.  The reported coverage gain is
therefore confounded by the larger training budget.

This script removes the confound: it starts from the SAME checkpoint
(localized_uav_best.pth) and trains for the SAME 160 PPO iterations, same
learning rate, same seed, but with a SINGLE terrain in the sampling mix.
Network, reward and evaluation protocol are byte-identical code paths.

Evaluation always uses the full 4-terrain cross-map test set.

Usage:  python run_control_single_terrain_160.py --seed 2026
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bundle_paths as bp                              # noqa: E402

REPRO = bp.REPRO

spec = importlib.util.spec_from_file_location("md", REPRO / "multi_domain_train.py")
md = importlib.util.module_from_spec(spec)
sys.modules["md"] = md
assert spec.loader is not None
spec.loader.exec_module(md)

import numpy as np
import torch

RESULTS = bp.RESULTS / "control_single_terrain_160"

# Which single terrain the control trains on.  The paper's single-terrain
# reference checkpoint was trained on plain, so plain is the matched control.
CONTROL_TERRAIN = ("terrain_plain", "plain", 0.08)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--iters", type=int, default=160)
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device, "| seed:", args.seed, "| iters:", args.iters,
          flush=True)

    model, config, dataset = md.load_checkpoint()

    # --- training: same code path, single-terrain mix, same budget ---
    full_mix = md.TERRAIN_MIX
    md.TERRAIN_MIX = [CONTROL_TERRAIN]
    md.EPISODES_PER_TERRAIN = args.iters
    md.TOTAL_EPISODES = args.iters
    md.TRAIN_SEED = args.seed
    model.to(device)
    try:
        model, train_log, train_time = md.train_multidomain(
            model, config, dataset, device)
    finally:
        md.TERRAIN_MIX = full_mix          # restore before evaluation

    ckpt = RESULTS / f"control_single_terrain_{CONTROL_TERRAIN[1]}_{args.iters}iters_seed{args.seed}.pth"
    torch.save({
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "config": {k: getattr(config, k)
                   for k in md.mod.ExperimentConfig.__dataclass_fields__},
        "variant": "tmarl",
        "train_tag": f"single-terrain control ({CONTROL_TERRAIN[1]}), {args.iters} iters",
        "seed": args.seed,
    }, ckpt)
    print("saved", ckpt, flush=True)

    # --- evaluation on the FULL 4-terrain cross-map test set ---
    rows = md.evaluate(model, config, dataset, device)
    for r in rows:
        r["seed_trained"] = args.seed
    summary = md.summarize(rows, f"single-terrain-control-{args.iters}")

    with (RESULTS / f"Control_SingleTerrain_{args.iters}iters_ByEpisode.csv").open(
            "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with (RESULTS / f"Control_SingleTerrain_{args.iters}iters_Summary.csv").open(
            "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    (RESULTS / f"Control_SingleTerrain_{args.iters}iters_Protocol.json").write_text(
        json.dumps({
            "control": "single-terrain continued training at matched budget",
            "start_checkpoint": "localized_uav_best.pth (same as multi-domain run)",
            "terrain_mix": [CONTROL_TERRAIN[1]],
            "ppo_iterations": args.iters,
            "batch_episodes": md.BATCH_EPISODES,
            "env_episodes_total": args.iters * md.BATCH_EPISODES,
            "lr": md.OVERRIDE_LR,
            "seed": args.seed,
            "train_seconds": round(train_time, 1),
            "evaluation": {
                "seeds": md.EVAL_SEEDS,
                "episodes_per_seed": md.EVAL_EPISODES_PER_SEED,
                "protocol": "unchanged cross-map test set (4 terrains)",
            },
            "purpose": "disentangle domain diversity from training budget "
                       "(R1 concern 3 / R3 concern 4)",
        }, indent=2), encoding="utf-8")

    print("SUMMARY:", flush=True)
    for s in summary:
        print(s, flush=True)
    print("CONTROL RUN DONE", flush=True)


if __name__ == "__main__":
    main()
