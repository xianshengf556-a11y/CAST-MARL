# -*- coding: utf-8 -*-
"""Correction-scale sweep for CVA-SP.

HYPOTHESIS
----------
CVA-SP shows no measurable coverage benefit in either the saturated or the
low-coverage regime.  The proposed mechanism is a scale mismatch: the bounded
correction ball has radius rho_x = rho_a * v_max * dt = 0.55 * 150 = 82.5 m,
whereas the sensing footprint has radius 612-918 m.  A displacement an order of
magnitude smaller than the footprint almost never changes the discrete set of
task cells inside it, so the coverage term is nearly constant across candidates
and CVA-SP degenerates to a tie-breaker (the intervention statistics show it
retains strictly more coverage in only 1.6% of corrections).

TEST
----
Sweep the normalised correction cap rho_a over {0.55 (archived), 1.0, 1.5, 2.0}
in a saturated and a low-coverage protocol, and for each setting report
  * episode-level coverage / path for the nearest-point rule and for CVA-SP,
    paired on identical episodes, and
  * the intervention-level share of corrections in which CVA-SP retains
    strictly more incremental coverage than the nearest-point rule.

If the scale mismatch is the cause, the intervention-level share should rise
with rho_a, and coverage should start to separate once the correction reach
becomes comparable to the footprint scale.

NOTE  rho_a >= 1.0 saturates against the action box [-1,1] for raw actions near
the boundary, so the effective reach grows sub-linearly; the sweep is therefore
a diagnostic of the mechanism, not a proposal to deploy a larger correction.
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

OUT = bp.RESULTS / "correction_scale"
OUT.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 43, 44, 45, 46]
EPISODES_PER_SEED = 12
MAX_SPEED = 150.0
D_SAFE = 138.0
LAM = 0.35

SCENARIOS = [
    (6, 35, "saturated (6 UAVs / 35 steps)"),
    (3, 15, "low coverage (3 UAVs / 15 steps)"),
]
CAPS = [0.55, 1.0, 1.5, 2.0]


def cva_sp_cap(env, i, raw, other, d_safe, lam, cap):
    """cva.cva_sp_action with the correction cap exposed."""
    raw_pos = fb._projected_xy(env, i, raw)
    if not fb._blocked(env, i, raw_pos):
        if np.linalg.norm(other - raw_pos, axis=1).min() >= d_safe:
            return raw, False, 0.0
    best, best_score, triggered = raw, -1e9, False
    for c in cva.candidate_actions(raw, cap=cap):
        pos = fb._projected_xy(env, i, c)
        if fb._blocked(env, i, pos):
            continue
        if np.linalg.norm(other - pos, axis=1).min() < d_safe:
            continue
        gain = cva.coverage_gain_at(env, i, pos, other)
        score = gain - lam * float(np.linalg.norm(c - raw))
        if score > best_score:
            best, best_score, triggered = c, score, True
    if triggered and best_score > -1e8:
        return best, True, best_score
    return fb.nearest_safe_action_p(env, i, raw, other, d_safe), True, 0.0


def run_episode(seed, dataset, num_uavs, max_steps, variant, cap):
    """variant: 'nearest' or 'cva'.  When 'cva', also records the
    intervention-level comparison against the nearest-point rule on the same
    states."""
    mod.set_seed(seed)
    env = mod.MultiUAVCoverageEnv(
        dataset, num_uavs, env_type="terrain_urban", max_steps=max_steps,
        max_speed=MAX_SPEED, obstacle_ratio=0.12, use_failure_module=False,
        region_reassign_interval=4, num_users=None)
    env.reset()
    occ = cva.occupancy(env)
    corrections = 0
    interv = []
    for _ in range(max_steps):
        starts = [cva.xy_cell(p[:2]) for p in env.uav_pos]
        goals = [cva.xy_cell(cva.target_for(env, i)) for i in range(num_uavs)]
        paths = [cva.astar(s, g, occ) for s, g in zip(starts, goals)]
        raw = cva.raw_actions_from_paths(env, paths)
        next_xy = fb._next_xy_all(env, raw)
        out = raw.copy()
        for i in range(num_uavs):
            other = np.delete(next_xy, i, axis=0)
            if variant == "nearest":
                corr = fb.nearest_safe_action_p(env, i, raw[i], other, D_SAFE)
                out[i, :2] = corr[:2]
                corrections += int(np.linalg.norm(corr - raw[i]) > 1e-6)
                continue
            corr, trig, _s = cva_sp_cap(env, i, raw[i], other, D_SAFE, LAM, cap)
            out[i, :2] = corr[:2]
            corrections += int(trig)
            raw_pos = fb._projected_xy(env, i, raw[i])
            safe = (not fb._blocked(env, i, raw_pos)) and (
                np.linalg.norm(other - raw_pos, axis=1).min() >= D_SAFE)
            if not safe:
                near = fb.nearest_safe_action_p(env, i, raw[i], other, D_SAFE)
                if (np.linalg.norm(near - raw[i]) > 1e-6
                        and np.linalg.norm(corr - raw[i]) > 1e-6):
                    g_n = cva.coverage_gain_at(
                        env, i, fb._projected_xy(env, i, near), other)
                    g_c = cva.coverage_gain_at(
                        env, i, fb._projected_xy(env, i, corr), other)
                    interv.append((float(g_c), float(g_n)))
        _, _r, done, info = env.step(out)
        if done:
            break
    return {
        "coverage": float(info["coverage_rate"]) * 100.0,
        "conflict_rate": float(info["conflict_rate"]),
        "path_length": float(info["path_length"]),
        "corrections": corrections,
    }, interv


def main():
    dataset = mod.ScenarioDataset(cva.DATASET)
    rows = []
    for num_uavs, max_steps, label in SCENARIOS:
        for cap in CAPS:
            n_cov, c_cov, n_path, c_path = [], [], [], []
            all_interv = []
            for seed in SEEDS:
                for ep in range(EPISODES_PER_SEED):
                    s = seed * 1000 + ep
                    rn, _ = run_episode(s, dataset, num_uavs, max_steps, "nearest", cap)
                    rc, iv = run_episode(s, dataset, num_uavs, max_steps, "cva", cap)
                    n_cov.append(rn["coverage"]); c_cov.append(rc["coverage"])
                    n_path.append(rn["path_length"]); c_path.append(rc["path_length"])
                    all_interv.extend(iv)
            n_cov = np.array(n_cov); c_cov = np.array(c_cov)
            n_path = np.array(n_path); c_path = np.array(c_path)
            if np.allclose(c_cov - n_cov, 0):
                t_cov = p_cov = float("nan")
            else:
                t_cov, p_cov = stats.ttest_rel(c_cov, n_cov)
            t_path, p_path = stats.ttest_rel(c_path, n_path)
            if all_interv:
                g_c = np.array([x[0] for x in all_interv])
                g_n = np.array([x[1] for x in all_interv])
                gt = float((g_c > g_n + 1e-12).mean())
                eq = float((np.abs(g_c - g_n) <= 1e-12).mean())
                lt = float((g_c < g_n - 1e-12).mean())
                dg = float((g_c - g_n).mean())
            else:
                gt = eq = lt = dg = float("nan")
            reach = cap * MAX_SPEED
            rows.append({
                "scenario": label, "rho_a": cap,
                "correction_reach_m": reach,
                "nearest_coverage": round(float(n_cov.mean()), 4),
                "cva_coverage": round(float(c_cov.mean()), 4),
                "coverage_diff": round(float((c_cov - n_cov).mean()), 4),
                "coverage_p": p_cov,
                "nearest_path": round(float(n_path.mean()), 1),
                "cva_path": round(float(c_path.mean()), 1),
                "path_diff": round(float((c_path - n_path).mean()), 2),
                "path_p": p_path,
                "corrections_compared": len(all_interv),
                "cva_strictly_more_coverage_share": round(gt, 4),
                "equal_share": round(eq, 4),
                "cva_less_share": round(lt, 4),
                "mean_coverage_advantage": round(dg, 4),
            })
            print("%-32s rho_a=%.2f reach=%5.0fm  cov %.3f vs %.3f (d=%+.3f p=%.3g)  "
                  "interv n=%4d  cva>near %.1f%%  dgain=%+.4f"
                  % (label, cap, reach, float(n_cov.mean()), float(c_cov.mean()),
                     float((c_cov - n_cov).mean()), p_cov, len(all_interv),
                     100 * gt, dg), flush=True)

    keys = list(rows[0].keys())
    with (OUT / "CorrectionScale_Results.csv").open("w", newline="",
                                                    encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    (OUT / "CorrectionScale_Protocol.json").write_text(json.dumps({
        "scenarios": [{"num_uavs": n, "max_steps": s, "label": l}
                      for n, s, l in SCENARIOS],
        "rho_a_values": CAPS, "d_safe": D_SAFE, "lambda_d": LAM,
        "seeds": SEEDS, "episodes_per_seed": EPISODES_PER_SEED,
        "sensing_radius_m": "612-918 (altitude dependent)",
        "hypothesis": ("the archived correction reach (82.5 m at rho_a=0.55) is an "
                       "order of magnitude below the sensing radius, so the "
                       "coverage term cannot differentiate candidates"),
    }, indent=2), encoding="utf-8")
    print("\nwrote", OUT / "CorrectionScale_Results.csv")


if __name__ == "__main__":
    main()
