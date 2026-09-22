# -*- coding: utf-8 -*-
"""Empty-feasible-set statistics and threshold sweeps.

This script characterizes two things.  First, how often the execution filter
finds no admissible correction at all, across terrain conditions and fleet
sizes, and whether such failures occur in consecutive steps for a single
vehicle.  Second, how the separation margin of the filter and the
conflict-reporting threshold affect the reported behavior.

DESIGN NOTES (important for defensibility)
------------------------------------------
1. The executable A* protocol of cva_sp_benchmarks.py is reproduced exactly
   (same 20x20 occupancy grid, 135 m inflation, same A* target rule, same
   simulator step function).

2. `cva_sp_action` is re-implemented here with instrumentation but with
   IDENTICAL decision logic.  It distinguishes FOUR outcomes per UAV-step:
     accepted          - raw action already admissible, filter does not act
     cva               - CVA-SP found a feasible candidate and corrected
     fallback_nearest  - CVA feasible candidate set EMPTY -> nearest-point used
     raw_released      - CVA set empty AND nearest-point also empty -> the raw
                         (possibly unsafe) action is released unchanged
   The last outcome is the one that matters for safety, and it is not reported
   anywhere in the current manuscript.

3. d_conflict is swept WITHOUT touching the simulator: the module constant
   COLLISION_DISTANCE also feeds reward/risk terms, so changing it would
   confound the comparison.  Instead the conflict rate is recomputed
   post-hoc from env.trajectory, and the recomputation is validated against
   the simulator's own info["conflict_rate"] at 120 m as a self-check.

4. d_safe is swept only inside the filter copy, leaving the simulator alone.

Usage:
  python exp_fallback_threshold.py --mode fallback
  python exp_fallback_threshold.py --mode threshold
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bundle_paths as bp                              # noqa: E402

REPRO = bp.REPRO
OUT = bp.RESULTS / "fallback_threshold"
OUT.mkdir(parents=True, exist_ok=True)

_spec = importlib.util.spec_from_file_location("cva", REPRO / "cva_sp_benchmarks.py")
cva = importlib.util.module_from_spec(_spec)
sys.modules["cva"] = cva
assert _spec.loader is not None
_spec.loader.exec_module(cva)
mod = cva.mod

# scenario label -> (env_type label, obstacle ratio), matching the cross-map protocol
SCENARIOS = [
    ("plain", "terrain_plain", 0.08),
    ("urban", "terrain_urban", 0.20),
    ("mountain", "terrain_mountain", 0.16),
    ("mountain-dense", "terrain_mountain", 0.30),
]
SEEDS = [42, 43, 44, 45, 46]
EPISODES_PER_SEED = 12
MAX_STEPS = 35
MAX_SPEED = 150.0


# --------------------------------------------------------------------------
# parametrised, instrumented copies of the archived filter functions
# --------------------------------------------------------------------------
def _projected_xy(env, i, action):
    cur_cov = float(env.covered.mean()) if len(env.covered) else 1.0
    tr = env.t / max(1, env.max_steps)
    ss = float(np.clip(1.0 - 0.14 * cur_cov - 0.08 * tr, 0.75, 1.0))
    move = np.clip(action[:2], -1.0, 1.0) * (env.max_speed * ss)
    return np.clip(env.uav_pos[i, :2] + move, 0.0, [mod.AREA_X, mod.AREA_Y])


def _next_xy_all(env, actions):
    out = np.zeros((env.num_uavs, 2), dtype=np.float32)
    for j in range(env.num_uavs):
        out[j] = _projected_xy(env, j, actions[j])
    return out


def _blocked(env, i, pos):
    return bool(env._point_in_obstacle(
        np.array([pos[0], pos[1], env.uav_pos[i, 2]], dtype=np.float32)))


def nearest_safe_action_p(env, i, raw, other, d_safe):
    """Verbatim copy of cva.nearest_safe_action with d_safe exposed."""
    best, best_dist = None, None
    for c in cva.candidate_actions(raw, cap=1.0):
        pos = _projected_xy(env, i, c)
        if _blocked(env, i, pos):
            continue
        if np.linalg.norm(other - pos, axis=1).min() < d_safe:
            continue
        d = float(np.linalg.norm(c - raw))
        if best_dist is None or d < best_dist:
            best, best_dist = c, d
    return raw if best is None else best


def cva_sp_action_p(env, i, raw, other, d_safe, lam):
    """Verbatim copy of cva.cva_sp_action with d_safe/lambda exposed and a
    four-way outcome label added."""
    raw_pos = _projected_xy(env, i, raw)
    if not _blocked(env, i, raw_pos):
        if np.linalg.norm(other - raw_pos, axis=1).min() >= d_safe:
            return raw, False, 0.0, "accepted", 0

    best, best_score, triggered = raw, -1e9, False
    n_feasible = 0
    for c in cva.candidate_actions(raw):
        pos = _projected_xy(env, i, c)
        if _blocked(env, i, pos):
            continue
        if np.linalg.norm(other - pos, axis=1).min() < d_safe:
            continue
        n_feasible += 1
        gain = cva.coverage_gain_at(env, i, pos, other)
        dev = float(np.linalg.norm(c - raw))
        score = gain - lam * dev
        if score > best_score:
            best, best_score, triggered = c, score, True

    if triggered and best_score > -1e8:
        return best, True, best_score, "cva", n_feasible

    # CVA feasible candidate set is EMPTY -> nearest-point fallback
    fallback = nearest_safe_action_p(env, i, raw, other, d_safe)
    if float(np.linalg.norm(fallback - raw)) > 1e-6:
        return fallback, True, 0.0, "fallback_nearest", 0
    # both sets empty -> the raw action is released unchanged
    return raw, False, 0.0, "raw_released", 0


# --------------------------------------------------------------------------
# post-hoc conflict rate (no simulator modification)
# --------------------------------------------------------------------------
def conflict_rate_from_trajectory(traj, num_uavs, d_conflict):
    """traj: list of (num_uavs, 3) states.

    traj[0] is the state appended by reset(); traj[k] is the state after step
    k.  The simulator counts conflicts only inside step() (post-step states)
    and normalises by the step count t, so the post-step slice traj[1:] is used
    and the denominator is len(traj) - 1.
    """
    pairs = num_uavs * (num_uavs - 1) // 2
    events = 0
    steps = max(0, len(traj) - 1)
    for pos in traj[1:]:
        p = pos[:, :3]                      # 3-D, matching the simulator
        # use the simulator's own distance kernel so the boundary comparison
        # at exactly d_conflict is bit-identical to info["conflict_rate"]
        d = mod.pairwise_dist(p, p)
        iu = np.triu_indices(num_uavs, 1)
        events += int((d[iu] < d_conflict).sum())
    return events, steps, (events / max(1, steps * pairs))


# --------------------------------------------------------------------------
# one episode
# --------------------------------------------------------------------------
def run_episode(seed, dataset, env_type, obstacle_ratio, num_uavs,
                variant="cva", d_safe=138.0, lam=0.35):
    mod.set_seed(seed)
    env = mod.MultiUAVCoverageEnv(
        dataset, num_uavs, env_type=env_type, max_steps=MAX_STEPS,
        max_speed=MAX_SPEED, obstacle_ratio=obstacle_ratio,
        use_failure_module=False, region_reassign_interval=4, num_users=None)
    env.reset()
    occ = cva.occupancy(env)

    status_counts = {"accepted": 0, "cva": 0,
                     "fallback_nearest": 0, "raw_released": 0}
    corrections = 0
    # per-UAV streak tracking for consecutive empty feasible sets
    streak = [0] * num_uavs
    max_streak = 0
    n_streaks_ge2 = 0
    n_streak_breaks = 0
    uav_step_records = []

    for _ in range(MAX_STEPS):
        starts = [cva.xy_cell(p[:2]) for p in env.uav_pos]
        goals = [cva.xy_cell(cva.target_for(env, i)) for i in range(num_uavs)]
        paths = [cva.astar(s, g, occ) for s, g in zip(starts, goals)]
        raw = cva.raw_actions_from_paths(env, paths)
        next_xy = _next_xy_all(env, raw)

        out = raw.copy()
        for i in range(num_uavs):
            other = np.delete(next_xy, i, axis=0)
            if variant == "raw":
                out[i, :2] = raw[i, :2]
                continue
            if variant == "nearest":
                corr = nearest_safe_action_p(env, i, raw[i], other, d_safe)
                out[i, :2] = corr[:2]
                corrections += int(np.linalg.norm(corr - raw[i]) > 1e-6)
                continue
            corr, trig, _score, status, n_feas = cva_sp_action_p(
                env, i, raw[i], other, d_safe, lam)
            out[i, :2] = corr[:2]
            corrections += int(trig)
            status_counts[status] += 1
            empty = status in ("fallback_nearest", "raw_released")
            if empty:
                streak[i] += 1
                if streak[i] == 2:
                    n_streaks_ge2 += 1
                max_streak = max(max_streak, streak[i])
            else:
                if streak[i] >= 1:
                    n_streak_breaks += 1
                streak[i] = 0
            uav_step_records.append({
                "uav": i, "status": status, "n_feasible": n_feas,
                "empty": int(empty), "streak": streak[i],
            })

        _, reward, done, info = env.step(out)
        if done:
            break

    events120, steps, cr120 = conflict_rate_from_trajectory(
        env.trajectory, num_uavs, 120.0)
    return {
        "episode_coverage": info["coverage_rate"] * 100.0,
        "episode_conflict_sim": info["conflict_rate"],
        "episode_conflict_recomputed_120": cr120,
        "conflict_events_120": events120,
        "pair_steps": steps * (num_uavs * (num_uavs - 1) // 2),
        "episode_path": info["path_length"],
        "corrections": corrections,
        "uav_steps": num_uavs * MAX_STEPS,
        "accepted": status_counts["accepted"],
        "cva": status_counts["cva"],
        "fallback_nearest": status_counts["fallback_nearest"],
        "raw_released": status_counts["raw_released"],
        "max_empty_streak": max_streak,
        "n_empty_streaks_ge2": n_streaks_ge2,
        "trajectory": env.trajectory,
    }


# --------------------------------------------------------------------------
# modes
# --------------------------------------------------------------------------
def mode_fallback(dataset):
    """R3-9: empty-set fallback frequency across scenarios and fleet sizes."""
    rows = []
    recs = []
    for label, env_type, ratio in SCENARIOS:
        for num_uavs in (4, 6, 8):
            for seed in SEEDS:
                for ep in range(EPISODES_PER_SEED):
                    r = run_episode(seed * 1000 + ep, dataset, env_type,
                                    ratio, num_uavs)
                    recs.append(r["trajectory"])
                    r.pop("trajectory")
                    rows.append({"scenario": label, "num_uavs": num_uavs,
                                 "seed": seed, "episode": ep, **r})
            sub = [x for x in rows if x["scenario"] == label
                   and x["num_uavs"] == num_uavs]
            tot_steps = sum(x["uav_steps"] for x in sub)
            fb = sum(x["fallback_nearest"] for x in sub)
            rr = sum(x["raw_released"] for x in sub)
            print("%-15s uavs=%d  episodes=%d  uav-steps=%d  "
                  "CVA-empty=%d (%.4f%%)  near-fallback=%d  "
                  "RAW-RELEASED=%d  maxstreak=%d  streaks>=2=%d"
                  % (label, num_uavs, len(sub), tot_steps, fb + rr,
                     100.0 * (fb + rr) / max(1, tot_steps), fb, rr,
                     max(x["max_empty_streak"] for x in sub),
                     sum(x["n_empty_streaks_ge2"] for x in sub)), flush=True)

    with (OUT / "Fallback_ByEpisode.csv").open("w", newline="",
                                               encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    summary = []
    for label, _env, _r in SCENARIOS:
        for num_uavs in (4, 6, 8):
            sub = [x for x in rows if x["scenario"] == label
                   and x["num_uavs"] == num_uavs]
            tot_steps = sum(x["uav_steps"] for x in sub)
            fb = sum(x["fallback_nearest"] for x in sub)
            rr = sum(x["raw_released"] for x in sub)
            summary.append({
                "scenario": label,
                "num_uavs": num_uavs,
                "episodes": len(sub),
                "uav_steps": tot_steps,
                "cva_empty_total": fb + rr,
                "cva_empty_rate_pct": round(100.0 * (fb + rr) / max(1, tot_steps), 5),
                "fallback_nearest": fb,
                "fallback_nearest_rate_pct": round(100.0 * fb / max(1, tot_steps), 5),
                "raw_released": rr,
                "raw_released_rate_pct": round(100.0 * rr / max(1, tot_steps), 5),
                "episodes_with_any_raw_release": sum(
                    1 for x in sub if x["raw_released"] > 0),
                "max_consecutive_empty_steps": max(
                    x["max_empty_streak"] for x in sub),
                "n_consecutive_empty_runs_ge2": sum(
                    x["n_empty_streaks_ge2"] for x in sub),
                "mean_conflict_rate": round(
                    float(np.mean([x["episode_conflict_sim"] for x in sub])), 8),
                "conflict_events_120_total": sum(
                    x["conflict_events_120"] for x in sub),
                "mean_coverage": round(
                    float(np.mean([x["episode_coverage"] for x in sub])), 4),
            })
    with (OUT / "Fallback_Summary.csv").open("w", newline="",
                                             encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    # self-check: recomputed 120 m conflict rate must match the simulator
    diffs = [abs(x["episode_conflict_sim"] - x["episode_conflict_recomputed_120"])
             for x in rows]
    print("SELF-CHECK max |sim - recomputed| conflict rate @120m = %.3e"
          % max(diffs), flush=True)
    (OUT / "Fallback_Protocol.json").write_text(json.dumps({
        "scenarios": [{"label": l, "env_type": e, "obstacle_ratio": r}
                      for l, e, r in SCENARIOS],
        "fleet_sizes": [4, 6, 8],
        "seeds": SEEDS,
        "episodes_per_seed": EPISODES_PER_SEED,
        "max_steps": MAX_STEPS,
        "d_safe": 138.0, "lambda": 0.35,
        "self_check_max_abs_diff_conflict_rate": max(diffs),
    }, indent=2), encoding="utf-8")
    print("FALLBACK MODE DONE", flush=True)


def mode_threshold(dataset):
    """R2-9: sweep d_safe (filter margin) and d_conflict (reporting threshold)."""
    label, env_type, ratio = "urban", "terrain_urban", 0.20
    num_uavs = 6

    d_safe_rows = []
    for d_safe in (110.0, 120.0, 130.0, 138.0, 150.0, 165.0):
        vals = []
        for seed in SEEDS:
            for ep in range(EPISODES_PER_SEED):
                r = run_episode(seed * 1000 + ep, dataset, env_type, ratio,
                                num_uavs, d_safe=d_safe)
                vals.append(r)
        tot_steps = sum(v["uav_steps"] for v in vals)
        d_safe_rows.append({
            "d_safe_m": d_safe,
            "episodes": len(vals),
            "coverage_mean": round(float(np.mean([v["episode_coverage"] for v in vals])), 4),
            "coverage_std": round(float(np.std([v["episode_coverage"] for v in vals], ddof=1)), 4),
            "path_mean": round(float(np.mean([v["episode_path"] for v in vals])), 2),
            "path_std": round(float(np.std([v["episode_path"] for v in vals], ddof=1)), 2),
            "conflict_sim_mean": round(float(np.mean([v["episode_conflict_sim"] for v in vals])), 8),
            "conflict_events_120": sum(v["conflict_events_120"] for v in vals),
            "corrections_mean": round(float(np.mean([v["corrections"] for v in vals])), 3),
            "cva_empty_rate_pct": round(100.0 * sum(v["fallback_nearest"] + v["raw_released"] for v in vals) / max(1, tot_steps), 5),
            "raw_released_total": sum(v["raw_released"] for v in vals),
        })
        print(d_safe_rows[-1], flush=True)
    with (OUT / "Threshold_DSafe_Sweep.csv").open("w", newline="",
                                                  encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(d_safe_rows[0].keys()))
        w.writeheader()
        w.writerows(d_safe_rows)

    # d_conflict sweep: recomputed post-hoc on the SAME trajectories (d_safe=138)
    print("collecting trajectories for the d_conflict sweep...", flush=True)
    trajs = []
    for seed in SEEDS:
        for ep in range(EPISODES_PER_SEED):
            r = run_episode(seed * 1000 + ep, dataset, env_type, ratio,
                            num_uavs, d_safe=138.0)
            trajs.append(r.pop("trajectory"))
            r["_seed_ep"] = (seed, ep)
    d_conf_rows = []
    for d_conf in (100.0, 110.0, 120.0, 130.0, 138.0, 150.0):
        ev = 0
        ps = 0
        for tr in trajs:
            e, s, _ = conflict_rate_from_trajectory(tr, num_uavs, d_conf)
            ev += e
            ps += s * (num_uavs * (num_uavs - 1) // 2)
        d_conf_rows.append({
            "d_conflict_m": d_conf,
            "episodes": len(trajs),
            "conflict_events_total": ev,
            "pair_steps": ps,
            "conflict_rate": round(ev / max(1, ps), 8),
        })
        print(d_conf_rows[-1], flush=True)
    with (OUT / "Threshold_DConflict_Sweep.csv").open("w", newline="",
                                                      encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(d_conf_rows[0].keys()))
        w.writeheader()
        w.writerows(d_conf_rows)

    (OUT / "Threshold_Protocol.json").write_text(json.dumps({
        "scenario": label, "env_type": env_type, "obstacle_ratio": ratio,
        "num_uavs": num_uavs, "seeds": SEEDS,
        "episodes_per_seed": EPISODES_PER_SEED,
        "note": ("d_conflict is swept by recomputing pairwise-distance events "
                 "from the recorded trajectory, so the simulator (and its "
                 "reward/risk terms) is never modified. d_safe is swept only "
                 "inside the filter."),
    }, indent=2), encoding="utf-8")
    print("THRESHOLD MODE DONE", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["fallback", "threshold"], required=True)
    args = ap.parse_args()
    dataset = mod.ScenarioDataset(cva.DATASET)
    if args.mode == "fallback":
        mode_fallback(dataset)
    else:
        mode_threshold(dataset)


if __name__ == "__main__":
    main()
