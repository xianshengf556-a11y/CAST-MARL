# -*- coding: utf-8 -*-
"""Validate the reproduction harness against an archived, published number.

Target: the manuscript's Table 8 row "Raw A* (no filter)", which is the A*
protocol with NO safety layer, evaluated over seeds 42-46 x 12 episodes:
    coverage     99.505 +/- 1.161  %
    conflict     0.01025 +/- 0.01928
    path length  20884.7 +/- 4264.7 m

If this script reproduces those values, then the instrumented episode loop in
exp_fallback_threshold.py faithfully reproduces the archived protocol and can
be trusted for the experiments reported in the revision.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "expfb", HERE / "exp_fallback_threshold.py")
expfb = importlib.util.module_from_spec(_spec)
sys.modules["expfb"] = expfb
assert _spec.loader is not None
_spec.loader.exec_module(expfb)

mod = expfb.mod
cva = expfb.cva

TARGET = {
    "coverage": (99.505, 1.161),
    "conflict": (0.01025, 0.01928),
    "path": (20884.7, 4264.7),
}


def main():
    dataset = mod.ScenarioDataset(cva.DATASET)
    cov, conf, path, events, pairsteps = [], [], [], 0, 0
    diffs = []
    for seed in expfb.SEEDS:
        for ep in range(expfb.EPISODES_PER_SEED):
            r = expfb.run_episode(seed * 1000 + ep, dataset,
                                  "terrain_urban", 0.12, 6, variant="raw")
            cov.append(r["episode_coverage"])
            conf.append(r["episode_conflict_sim"])
            path.append(r["episode_path"])
            events += r["conflict_events_120"]
            pairsteps += r["pair_steps"]
            # self-check of the trajectory-based recomputation
            diffs.append(abs(r["episode_conflict_sim"]
                             - r["episode_conflict_recomputed_120"]))

    cov = np.asarray(cov); conf = np.asarray(conf); path = np.asarray(path)
    got = {
        "coverage": (cov.mean(), cov.std(ddof=1)),
        "conflict": (conf.mean(), conf.std(ddof=1)),
        "path": (path.mean(), path.std(ddof=1)),
    }
    print("n episodes =", len(cov))
    print("%-10s %18s %18s" % ("metric", "reproduced", "manuscript"))
    ok = True
    for k, (m, s) in TARGET.items():
        gm, gs = got[k]
        d = abs(gm - m) / max(abs(m), 1e-12) * 100.0
        flag = "OK " if d < 1.0 else "DIFF"
        if d >= 1.0:
            ok = False
        print("%-10s %10.5f +- %-6.5f %8.5f +- %-6.5f  %s (%.3f%%)"
              % (k, gm, gs, m, s, flag, d))
    print("pooled conflict events @120m = %d over %d pair-steps = %.8f"
          % (events, pairsteps, events / pairsteps))
    md = max(diffs)
    print("trajectory recomputation self-check: max |sim - recomputed| = %.3e -> %s"
          % (md, "PASSED" if md < 1e-9 else "FAILED"))
    print("HARNESS VALIDATION:", "PASSED" if ok else "MISMATCH - investigate")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
