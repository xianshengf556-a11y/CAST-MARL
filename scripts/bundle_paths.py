# -*- coding: utf-8 -*-
"""Portable path resolution for the CAST-MARL resubmission bundle.

The bundle layout is::

    <bundle>/repo/              experiment source + benchmark modules
    <bundle>/scripts/           these drivers (this file lives here)
    <bundle>/archived_configs/  archived per-terrain summary.json (configs)
    <bundle>/dataset/           dataset1_users*.npz
    <bundle>/results/           all outputs

Every path can be overridden with an environment variable, and CAST_BUNDLE_ROOT
relocates the whole bundle (used when the bundle is unpacked somewhere else,
e.g. on a rented Linux server::

    export CAST_BUNDLE_ROOT=/root/autodl-tmp/cast_marl_rerun
"""
from __future__ import annotations

import os
from pathlib import Path

_here = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("CAST_BUNDLE_ROOT", str(_here.parent))).resolve()

REPO = Path(os.environ.get(
    "CAST_REPRO_DIR",
    str(ROOT / "experimental_runs" / "reproducibility")))
REPRO = REPO                      # alias: the drivers import bp.REPRO
ARCHIVE_SUMMARIES = Path(
    os.environ.get("CAST_ARCHIVE_SUMMARIES", str(ROOT / "archived_configs")))
DATASET_DIR = Path(os.environ.get("CAST_DATASET_DIR", str(ROOT / "dataset")))
RESULTS = Path(os.environ.get("CAST_RESULTS_DIR", str(ROOT / "results")))

DATASETS = [
    str(DATASET_DIR / "dataset1_users35.npz"),
    str(DATASET_DIR / "dataset1_users45.npz"),
    str(DATASET_DIR / "dataset1_users60.npz"),
]

SOURCE_PY = REPO / "cast_marl_experiment_source.py"
CVA_PY = REPO / "cva_sp_benchmarks.py"

RESULTS.mkdir(parents=True, exist_ok=True)


def report() -> str:
    return ("bundle paths:\n"
            "  ROOT               = %s\n"
            "  REPO               = %s\n"
            "  ARCHIVE_SUMMARIES  = %s\n"
            "  DATASET_DIR        = %s\n"
            "  RESULTS            = %s" % (ROOT, REPO, ARCHIVE_SUMMARIES,
                                           DATASET_DIR, RESULTS))


if __name__ == "__main__":
    print(report())
    for label, p in (("source", SOURCE_PY), ("cva", CVA_PY)):
        print("  %-7s exists=%s" % (label, p.exists()))
    for d in DATASETS:
        print("  dataset exists=%s  %s" % (os.path.exists(d), d))
    for t in ("plain", "urban", "mountain"):
        p = ARCHIVE_SUMMARIES / ("terrain_%s_summary.json" % t)
        print("  config %-9s exists=%s" % (t, p.exists()))
