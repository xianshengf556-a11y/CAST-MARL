# -*- coding: utf-8 -*-
"""Aggregate the controlled terrain reruns into paper-ready tables.

Input : controlled_<terrain>_seed<seed>.json files written by
        run_terrain_controlled.py
Output: CSV tables with
          * mean +/- std across seeds for coverage / path length / reward
          * conflict rate reported BOTH as a rate and as integer pair-event
            counts, with an exact Poisson (Garwood) 95% confidence interval
          * the number of evaluation episodes that contained any conflict at
            all -- this is the quantity that shows why the original ablation
            conflict column needs integer event counts to be interpretable

Usage:
  python collect_controlled_results.py --indir <dir> --outdir <dir>
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from collections import defaultdict

import numpy as np
from scipy import stats

MAX_STEPS = 35
NUM_PAIRS = 15
PAIR_STEPS_PER_EPISODE = MAX_STEPS * NUM_PAIRS        # 525

BENCH_ORDER = ["CAST-MARL", "MAPPO", "MADDPG", "QMIX", "PPO"]
ABL_ORDER = ["CAST-MARL", "w/o Transformer", "w/o Attention",
             "w/o CTDE", "w/o Safety Layer"]


def poisson_ci(k, n, alpha=0.05):
    """Exact (Garwood) confidence interval for a Poisson rate k/n."""
    if n <= 0:
        return (0.0, 0.0)
    lo = 0.0 if k == 0 else stats.chi2.ppf(alpha / 2, 2 * k) / 2.0
    hi = stats.chi2.ppf(1 - alpha / 2, 2 * k + 2) / 2.0
    return (lo / n, hi / n)


def ms(vals):
    a = np.asarray(vals, dtype=float)
    if a.size == 0:
        return (float("nan"), float("nan"))
    if a.size == 1:
        return (float(a[0]), 0.0)
    return (float(a.mean()), float(a.std(ddof=1)))


def load(indir):
    files = sorted(glob.glob(os.path.join(indir, "controlled_*.json")))
    runs = []
    for f in files:
        try:
            runs.append(json.load(open(f, encoding="utf-8")))
        except Exception as exc:                      # noqa: BLE001
            print("skip %s: %s" % (f, exc))
    return runs


def aggregate(runs, key):
    """terrain -> method -> per-seed records"""
    out = defaultdict(lambda: defaultdict(list))
    for r in runs:
        for rec in r.get(key, []):
            out[r["terrain"]][rec["method"]].append(
                {"seed": r["seed"], "eval_seed": r["eval_seed"], **rec})
    return out


def build_table(agg, terrains, order, variant_key, fallback_agg=None):
    """Build a paper table.

    `fallback_agg` supplies records absent from `agg`.  Used for the ablation
    table: with --group the ablation process trains only the four component
    variants, so the "Full" (CAST-MARL) row is taken from the benchmark records
    of the same (terrain, seed) runs.
    """
    rows = []
    for terrain in terrains:
        per_method = agg.get(terrain, {})
        for method in order:
            recs = per_method.get(method)
            used_fallback = False
            if not recs and fallback_agg is not None:
                recs = fallback_agg.get(terrain, {}).get(method)
                used_fallback = bool(recs)
            if not recs:
                continue
            cov_m, cov_s = ms([x["coverage_mean"] for x in recs])
            path_m, path_s = ms([x["path_length_mean"] for x in recs])
            rew_m, rew_s = ms([x["reward_mean"] for x in recs])
            rate_m, rate_s = ms([x["conflict_rate_mean"] for x in recs])
            ev_tot = int(sum(x["conflict_events_total"] for x in recs))
            eps_tot = sum(len(x["conflict_events"]) for x in recs)
            eps_conf = int(sum(x["episodes_with_conflict"] for x in recs))
            max_single = int(max(x["max_events_single_episode"] for x in recs))
            pair_steps = eps_tot * PAIR_STEPS_PER_EPISODE
            lo, hi = poisson_ci(ev_tot, pair_steps)
            integral = all(x["event_counts_integral"] for x in recs)
            rows.append({
                "terrain": terrain,
                "variant" if variant_key else "method":
                    method,
                "n_seeds": len(recs),
                "coverage_mean_pct": round(cov_m, 4),
                "coverage_std_pct": round(cov_s, 4),
                "path_length_mean_m": round(path_m, 2),
                "path_length_std_m": round(path_s, 2),
                "reward_mean": round(rew_m, 4),
                "conflict_rate_mean": round(rate_m, 8),
                "conflict_rate_std": round(rate_s, 8),
                "conflict_events_total": ev_tot,
                "pair_steps_total": pair_steps,
                "conflict_rate_pooled": round(ev_tot / pair_steps, 8),
                "conflict_rate_ci95_low": round(lo, 8),
                "conflict_rate_ci95_high": round(hi, 8),
                "eval_episodes_total": eps_tot,
                "episodes_with_any_conflict": eps_conf,
                "share_episodes_with_conflict": round(eps_conf / eps_tot, 5),
                "max_events_in_one_episode": max_single,
                "event_counts_integral": integral,
            })
    return rows


def write_csv(path, rows):
    if not rows:
        print("no rows for", path)
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote", path, "(%d rows)" % len(rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    runs = load(args.indir)
    if not runs:
        raise SystemExit("no controlled_*.json found in %s" % args.indir)
    terrains = sorted({r["terrain"] for r in runs})
    print("loaded %d runs | terrains=%s | seeds=%s"
          % (len(runs), terrains,
             sorted({r["seed"] for r in runs})))
    missing = [t for t in ("plain", "urban", "mountain") if t not in terrains]
    if missing:
        print("WARNING: missing terrains:", missing)

    bench_agg = aggregate(runs, "benchmark")
    abl_agg = aggregate(runs, "ablation")

    bench_rows = build_table(bench_agg, terrains, BENCH_ORDER, False)
    abl_rows = build_table(abl_agg, terrains, ABL_ORDER, True,
                           fallback_agg=bench_agg)

    write_csv(os.path.join(args.outdir, "Table_benchmark_controlled.csv"),
              bench_rows)
    write_csv(os.path.join(args.outdir, "Table_ablation_controlled.csv"),
              abl_rows)

    # long per-seed table, for supplementary material / the public repo
    long_rows = []
    for r in runs:
        for key in ("benchmark", "ablation"):
            for rec in r.get(key, []):
                long_rows.append({
                    "kind": key, "terrain": r["terrain"], "seed": r["seed"],
                    "eval_env": r["eval_env"], "eval_seed": r["eval_seed"],
                    "method": rec["method"],
                    "coverage_mean": round(rec["coverage_mean"], 5),
                    "conflict_rate_mean": round(rec["conflict_rate_mean"], 8),
                    "conflict_events_total": rec["conflict_events_total"],
                    "episodes_with_conflict": rec["episodes_with_conflict"],
                    "path_length_mean": round(rec["path_length_mean"], 3),
                    "reward_mean": round(rec["reward_mean"], 5),
                })
    write_csv(os.path.join(args.outdir, "PerSeed_long.csv"), long_rows)

    # human-readable console summary of the conflict evidence
    print("\nCONFLICT EVIDENCE: integer pair-event counts per condition")
    print("%-9s %-18s %8s %10s %12s %14s"
          % ("terrain", "variant", "events", "eps", "eps_with", "rate[95% CI]"))
    for row in abl_rows:
        print("%-9s %-18s %8d %10d %12d   %.6f [%.6f, %.6f]"
              % (row["terrain"], row["method"], row["conflict_events_total"],
                 row["eval_episodes_total"], row["episodes_with_any_conflict"],
                 row["conflict_rate_pooled"], row["conflict_rate_ci95_low"],
                 row["conflict_rate_ci95_high"]))
    print("\nALL INTEGRAL:", all(r["event_counts_integral"]
                                 for r in bench_rows + abl_rows))
    print("COLLECT DONE")


if __name__ == "__main__":
    main()
