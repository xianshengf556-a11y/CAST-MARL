# Result records

Every number reported in the paper and in the response document is produced by a
CSV or JSON file in this tree. Nothing here is hand-edited: the files are written
by the drivers in `scripts/` and by the aggregation step of `run_all.sh`.

| directory | contents |
|---|---|
| `controlled/` | Per-seed records of the controlled rerun (`PerSeed_long.csv`, one row per trained model) and the two aggregated tables behind Table 2 and Table 4. |
| `latex/` | The same tables rendered as LaTeX, plus a plain-text digest. |
| `fallback_threshold/` | Empty-feasible-set statistics by scenario and fleet size, and the `d_safe` / `d_conflict` sweeps. |
| `runtime/` | Distribution of the filter decision time (machine-dependent; re-measure on your own hardware). |
| `control_single_terrain_160/` | Matched-budget single-terrain control: summary, per-episode records, and the protocol used. |
| `crossmap/` | Planner and safety-layer comparison on the four unseen maps, plus the greedy baselines. |
| `correction_scale/` | Correction-reach sweep (`rho_a` = 0.55, 1.0, 1.5, 2.0). |
| `low_coverage/` | Reduced-coverage regime test (fewer UAVs, shorter horizon). |
| `reward_sensitivity/` | Reward-coefficient sensitivity for `beta_c` and `beta_q`. |
| `archived_records/` | Summary records of the original archived runs, kept for provenance of the pre-correction tables. |

Columns are documented in the docstring of the script that writes each file, and
the conflict columns are integer pair-event counts together with their exact
Poisson 95% confidence intervals.
