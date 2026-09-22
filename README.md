# CAST-MARL — reproducible artifacts

**English** | [中文说明](README.zh-CN.md)

Code, configuration and per-seed records for the manuscript

> **CAST-MARL: An Engineering Framework for Safety-Constrained Cooperative
> Multi-UAV Coverage Planning**
> Suhui Feng, Lin Li, and Qizhi Zhang
> *IEEE Access*

This repository supports independent verification of the paper. It contains the
complete experiment source, the exact configuration of every reported run, the
per-seed and per-episode records behind each table, and a one-command driver that
reproduces the runs from scratch.

---

## Layout

```
experimental_runs/
  reproducibility/     experiment source and benchmark modules
  *.pth                the two archived checkpoints
scripts/               drivers that produce every reported table
  run_terrain_controlled.py         controlled benchmark + component ablation
  collect_controlled_results.py     aggregation into the paper tables
  exp_fallback_threshold.py         empty-feasible-set statistics + threshold sweeps
  run_runtime_stats.py              filter decision-time distribution
  exp_low_coverage.py               low-coverage regime test
  exp_correction_scale.py           correction-reach sweep
  exp_crossmap_comparison.py        planner and safety-layer comparison on unseen maps
  exp_reward_sensitivity.py         reward-coefficient sensitivity
  run_control_single_terrain_160.py matched training-budget control
archived_configs/      archived per-terrain configuration (used verbatim)
dataset/               task datasets (0.5 MB)
results/               outputs of the drivers, including the paper tables
run_all.sh             one-command reproduction
```

The single-purpose analyses (`exp_*.py`, plus the budget control) are run
directly and write one CSV each; they are the evidence behind the corresponding
sections of the paper and of the response document.

## Environment

Python 3.8+ with `numpy`, `scipy`, `matplotlib`, `torch`.

**A GPU is not required.** The model has 444k parameters and peaks at about
99 MB of GPU memory, so the workload is simulation-bound: the number of CPU cores
matters far more than the accelerator. All results in the paper were produced on
a CPU-only container.

```bash
pip install -r requirements.txt
python scripts/bundle_paths.py     # sanity check: prints every resolved path
```

## Reproduction

```bash
bash run_all.sh                    # all stages, workers = nproc
STAGES=0 bash run_all.sh           # only the protocol self-check (about 2 min)
WORKERS=16 bash run_all.sh         # explicit worker count
```

| Stage | What it does |
|---|---|
| 0 | Environment check plus a protocol self-check against a published number |
| 1 | Terrain benchmark and component ablation (5 seeds) |
| 2 | Empty-feasible-set statistics and threshold sweeps |
| 3 | Filter runtime distribution (must run on an idle machine) |
| 4 | Aggregation into the paper tables |

Stage 1 is the heavy part: 3 terrains x 5 seeds x 9 models = 135 training jobs.
Each job trains one model in its own process, so every model starts from the same
seeded state and a many-core machine is fully used.

## Protocol used by the drivers

The controlled driver `scripts/run_terrain_controlled.py` applies the following
protocol, which the scripts record in the output JSON of every run:

* training and evaluation use the **same named terrain**;
* the evaluation seed is **fixed and recorded**;
* **five training seeds** are used per condition;
* each model is trained in its **own process**, so all models start from the same
  seeded state;
* conflict statistics are exported as **per-episode samples**, from which integer
  pair-event counts are recovered.

## Validation

`scripts/_validate_protocol.py` re-runs the protocol behind the paper's
*Raw A\* (no filter)* reference row and compares coverage, conflict rate and path
length against the published values. It reproduces them to five significant
figures:

```
coverage  99.50529 +- 1.16123         vs published  99.505 +- 1.161        (0.000%)
conflict  0.01025 +- 0.01928          vs published  0.01025 +- 0.01928     (0.035%)
path      20884.66659 +- 4264.67166   vs published  20884.7 +- 4264.7      (0.000%)
trajectory recomputation self-check: max |sim - recomputed| = 0.000e+00   PASSED
```

If this does not pass, the later stages should not be trusted.

## Ablation variants

| Variant | Backbone | Parameters |
|---|---|---|
| Full model (CAST-MARL) | interaction-aware transformer | 444,118 |
| w/o Attention | transformer backbone, multi-head attention replaced by parameter-free mean aggregation over the agent axis | 444,118 |
| w/o Transformer | shared two-layer MLP | 54,279 |

## Data and metrics

Every reported quantity comes from the unchanged simulator step function.
Coverage is the fraction of task cells observed at least once; the conflict rate
counts unordered vehicle pairs whose separation falls below the reporting
threshold, normalized by the pair-step opportunities; path length accumulates
Euclidean motion. Conflict statistics are reported as **integer pair-event counts
with exact Poisson confidence intervals**, because at these magnitudes a single
event moves the reported rate by about 2.4e-4.

## License

MIT (see `LICENSE`). Please contact the corresponding author before reusing the
terrain scene assets.
