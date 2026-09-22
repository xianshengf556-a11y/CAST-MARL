# CAST-MARL — reproducible artifacts

Code, configuration and per-seed records for the manuscript

> **CAST-MARL: An Engineering Framework for Safety-Constrained Cooperative
> Multi-UAV Coverage Planning**
> Suhui Feng, Lin Li, and Qizhi Zhang
> Manuscript Access-2026-39960, *IEEE Access* (resubmission)

This repository is the independent-verification artifact for the paper. It
contains the complete experiment source, the exact configuration of every
reported run, the per-seed and per-episode records behind each table, and a
one-command driver that reproduces the runs from scratch.

---

## Why this repository exists

The original submission shipped the experiment code as a ZIP attachment. A
reviewer asked for a public repository instead, because a pipeline of this size
cannot be verified from an attachment. This repository is the result, and it also
documents two corrections that were made while preparing the resubmission.

## Layout

```
experimental_runs/
  reproducibility/     experiment source and benchmark modules
  *.pth                the two archived checkpoints
scripts/               drivers that produce every reported table
archived_configs/      archived per-terrain configuration (used verbatim)
dataset/               task datasets (0.5 MB)
results/               outputs of the drivers, including the paper tables
run_all.sh             one-command reproduction
```

## Environment

Python 3.8+ with `numpy`, `scipy`, `matplotlib`, `torch`.

**A GPU is not required.** The model has 444k parameters and peaks at about
99 MB of GPU memory, so the workload is simulation-bound: the number of CPU
cores matters far more than the accelerator. All results in the paper were
reproduced on a CPU-only container.

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
| 1 | Controlled terrain benchmark and component ablation (5 seeds) |
| 2 | Empty-feasible-set statistics and threshold sweeps |
| 3 | Filter runtime distribution (must run on an idle machine) |
| 4 | Aggregation into the paper tables |

Stage 1 is the heavy part: 3 terrains x 5 seeds x 9 models = 135 training jobs.
Each job trains one model in its own process, so every model starts from the same
seeded state and a many-core machine is fully used.

### Stage 0 is a real check

`scripts/_validate_protocol.py` re-runs the protocol behind the paper's
*Raw A\* (no filter)* row and compares coverage, conflict rate and path length
against the published values. It reproduces them to five significant figures:

```
coverage  99.50529 +- 1.16123   vs published  99.505 +- 1.161    OK (0.000%)
conflict  0.01025 +- 0.01928    vs published  0.01025 +- 0.01928 OK (0.035%)
path      20884.66659 +- 4264.67166  vs published 20884.7 +- 4264.7 OK (0.000%)
trajectory recomputation self-check: max |sim - recomputed| = 0.000e+00  PASSED
```

If this does not pass, the later stages should not be trusted.

## Two corrections carried by this code

1. **Evaluation configuration.** In the original script, the evaluation calls
   behind the terrain benchmark and the component ablation did not use the
   terrain label those tables report: every method in the benchmark was
   evaluated in the same `simple` environment and every ablation variant in the
   `dynamic_users` environment, so the terrain label reached those tables only
   through the trained weights. The same calls passed no evaluation seed. The
   controlled driver fixes this: training and evaluation use the named terrain,
   the evaluation seed is fixed and recorded, and five training seeds are used.

2. **Attention ablation.** The original attention variant did not keep the
   transformer backbone while removing only the attention operation, so the two
   ablation rows did not isolate separate factors. The variant now keeps the
   four-stream encoder, the two pre-norm blocks, the feed-forward sublayers and
   the full 444,118-parameter budget, and replaces only multi-head attention
   with a parameter-free mean over the agent axis. Verified parameter counts:
   full model 444,118 · w/o Attention 444,118 · w/o Transformer 54,279.

## Data and metrics

Every reported quantity comes from the unchanged simulator step function.
Coverage is the fraction of task cells observed at least once; the conflict rate
counts unordered vehicle pairs whose separation falls below the reporting
threshold, normalized by the pair-step opportunities; path length accumulates
Euclidean motion. Conflict statistics are reported as **integer pair-event
counts with exact Poisson confidence intervals**, because at these magnitudes a
single event moves the reported rate by about 2.4e-4.

## Reporting conventions used in the paper

Several results in the paper are negative and are reported as such rather than
smoothed over. In particular:

* CVA-SP, the coverage-aware correction objective, shows **no measurable
  coverage gain** over the nearest-point rule in either the saturated protocol
  or deliberately reduced-coverage regimes. The mechanism is a scale mismatch:
  the bounded correction reach is 82.5 m against a sensing footprint of
  612--918 m, so the coverage term is nearly constant across candidates.
* A matched single-terrain control at the same training budget reaches
  essentially the same coverage as the multi-domain model, so the coverage gain
  previously attributed to domain diversity is a **training-budget effect**.
* When no admissible correction exists, the filter executes the raw action
  unchanged. This occurred in 115 of 151,200 vehicle-step decisions, in one run
  of 33 consecutive steps, and should be read as a limit of per-vehicle
  sequential filtering.

## License

MIT (see `LICENSE`). Please confirm with the corresponding author before reusing
the terrain scene assets.
