#!/usr/bin/env bash
# ============================================================================
#  CAST-MARL — IEEE Access resubmission: full experiment driver
# ============================================================================
#  Usage (on the server, from the bundle root):
#
#      bash run_all.sh                 # defaults: 6 workers, all 3 terrains, 5 seeds
#      WORKERS=16 bash run_all.sh      # more cores -> faster
#      STAGES=1 bash run_all.sh        # run only stage 1
#      TERRAINS="plain" SEEDS="42" bash run_all.sh
#
#  Stages
#    0  environment + protocol self-check   (fast, must pass)
#    1  controlled terrain rerun + ablation (the heavy part, parallel)
#    2  empty-set fallback + threshold sweep (parallel with stage 1 is fine)
#    3  filter runtime distribution         (MUST run alone: timing data)
#    4  aggregate into paper-ready CSV tables
#
#  Stage 3 is deliberately isolated: running it while the CPU is busy would
#  produce meaningless timings.
# ============================================================================
set -uo pipefail

BUNDLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CAST_BUNDLE_ROOT="$BUNDLE_ROOT"

WORKERS="${WORKERS:-$(nproc 2>/dev/null || echo 6)}"
TERRAINS="${TERRAINS:-plain urban mountain}"
SEEDS="${SEEDS:-42 43 44 45 46}"
MODELS="${MODELS:-tmarl mappo maddpg qmix ppo no_transformer no_attention no_ctde tmarl_no_safety}"
STAGES="${STAGES:-0 1 2 3 4}"
PY="${PY:-python}"
OUT="${OUT:-$BUNDLE_ROOT/results/controlled}"

# One process per core: without this, every job would spawn 8 OpenMP threads
# and the parallel batch would thrash.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8

mkdir -p "$OUT" "$BUNDLE_ROOT/logs"
LOGS="$BUNDLE_ROOT/logs"

say() { echo "[$(date +%H:%M:%S)] $*"; }
stage_enabled() { [[ " $STAGES " == *" $1 "* ]]; }

say "bundle root : $BUNDLE_ROOT"
say "python      : $($PY -c 'import sys;print(sys.version.split()[0])' 2>/dev/null || echo MISSING)"
say "workers     : $WORKERS"
say "terrains    : $TERRAINS"
say "seeds       : $SEEDS"
say "stages      : $STAGES"
say "output      : $OUT"

# ---------------------------------------------------------------- stage 0 ---
if stage_enabled 0; then
  say "=== stage 0: environment + protocol self-check ==="
  $PY scripts/bundle_paths.py || { say "FATAL: bundle paths broken"; exit 1; }
  $PY scripts/_validate_protocol.py 2>&1 | tail -n 12
  say "stage 0 done (the harness must reproduce the published Raw-A* row)"
fi

# ---------------------------------------------------------------- stage 1 ---
if stage_enabled 1; then
  say "=== stage 1: controlled terrain rerun (parallel, WORKERS=$WORKERS) ==="
  jobs="$LOGS/jobs.txt"
  : > "$jobs"
  for t in $TERRAINS; do
    for s in $SEEDS; do
      for g in $MODELS; do printf '%s %s %s\n' "$t" "$s" "$g" >> "$jobs"; done
    done
  done
  say "queued $(wc -l < "$jobs") (terrain, seed, model) jobs;"
  say "  one model per job, so every model starts from the same seeded state"

  run_one() {
    local terrain="$1" seed="$2" group="$3"
    local log="$LOGS/controlled_${terrain}_seed${seed}_${group}.log"
    local t0=$SECONDS
    if $PY scripts/run_terrain_controlled.py --terrain "$terrain" \
         --seed "$seed" --group "$group" --outdir "$OUT" > "$log" 2>&1; then
      echo "[ ok  ] $terrain seed=$seed $group  ($((SECONDS - t0))s)"
    else
      echo "[FAIL ] $terrain seed=$seed $group  (see $log)"
    fi
  }
  export -f run_one
  export PY OUT LOGS

  # shellcheck disable=SC2016
  xargs -P "$WORKERS" -L 1 -a "$jobs" bash -c 'run_one "$0" "$1" "$2"'
  say "stage 1 done: $(ls -1 "$OUT"/controlled_*.json 2>/dev/null | wc -l) result files"
fi

# ---------------------------------------------------------------- stage 2 ---
if stage_enabled 2; then
  say "=== stage 2: empty-set fallback + threshold sweeps ==="
  $PY scripts/exp_fallback_threshold.py --mode fallback  > "$LOGS/fallback.log"  2>&1 \
    && say "fallback mode ok  -> $LOGS/fallback.log" \
    || say "fallback mode FAILED -> $LOGS/fallback.log"
  $PY scripts/exp_fallback_threshold.py --mode threshold > "$LOGS/threshold.log" 2>&1 \
    && say "threshold mode ok -> $LOGS/threshold.log" \
    || say "threshold mode FAILED -> $LOGS/threshold.log"
fi

# ---------------------------------------------------------------- stage 3 ---
if stage_enabled 3; then
  say "=== stage 3: filter runtime distribution (exclusive: timing data) ==="
  say "waiting for other work to settle before measuring..."
  sleep 20
  $PY scripts/run_runtime_stats.py > "$LOGS/runtime.log" 2>&1 \
    && say "runtime stats ok -> $LOGS/runtime.log" \
    || say "runtime stats FAILED -> $LOGS/runtime.log"
fi

# ---------------------------------------------------------------- stage 4 ---
if stage_enabled 4; then
  say "=== stage 4: aggregate results ==="
  $PY scripts/collect_controlled_results.py --indir "$OUT" \
      --outdir "$BUNDLE_ROOT/results/tables" 2>&1 | tee "$LOGS/collect.log"
  say "stage 4 done -> $BUNDLE_ROOT/results/tables"
fi

say "ALL REQUESTED STAGES FINISHED"
say "download these back:  results/tables/  results/controlled/  results/fallback_threshold/  results/runtime/  logs/"
