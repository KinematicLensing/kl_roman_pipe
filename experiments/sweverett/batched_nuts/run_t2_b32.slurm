#!/bin/bash
#SBATCH -J kl_t2_b32
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 06:00:00
#SBATCH -A JPL-PUB
#SBATCH -o runs/%x_%j.log

# T2: batched B=32 (128 lanes) at production-matched 300 first-pass draws
# -- measures the B=16 -> B=32 sampling-scaling factor, decision input 2.
# The 128-lane program is a new shape, so this job pays its one-time
# compile; JAX_LOG_COMPILES lines let us subtract it from the scaling
# number. If sampling OOMs at 128 lanes the fallback (KLPIPE_N_FITS=24)
# is a separate manual submission -- an OOM here is itself the HBM-ceiling
# measurement. Afterward the job retries sub-gate fits at 1000 draws
# (production escalation analog) so no second queue round-trip is needed.
# sbatch from the repo root so the log lands in runs/.

set -o pipefail
REPO=$STOCKYARD/repos/kl_roman_pipe
cd $REPO || { echo "[job] FATAL: cannot cd to $REPO"; exit 1; }
source experiments/sweverett/vista_kit/env_vista.sh
[ -n "${KLPIPE_PYTHON:-}" ] || { echo "[job] FATAL: KLPIPE_PYTHON unset"; exit 1; }

export APPTAINERENV_JAX_LOG_COMPILES=1
echo "[job] jax cache entries before: $(ls $SCRATCH/jax_cache 2>/dev/null | wc -l)"

if [ ! -f runs/cosmos25_batched_b32/manifest.parquet ]; then
    echo "[job] expanding cosmos25_batched_b32"
    $KLPIPE_PYTHON -m kl_pipe.ensemble expand \
        configs/ensembles/cosmos25_batched_b32.yaml
fi

mkdir -p runs/demo_v5_b32
KLPIPE_RUN_DIR=runs/cosmos25_batched_b32 KLPIPE_N_FITS=32 \
KLPIPE_INIT=map_window KLPIPE_NUTS_WARMUP=200 KLPIPE_NUTS_SAMPLES=300 \
KLPIPE_DEMO_OUT=runs/demo_v5_b32 \
    $KLPIPE_PYTHON -u experiments/sweverett/batched_nuts/batched_nuts_demo.py \
    2>&1 | tee runs/demo_v5_b32/demo.log

echo "[job] gate analysis (no matched solos in this bank; parity is T1b's job)"
$KLPIPE_PYTHON experiments/sweverett/batched_nuts/analyze_demo.py \
    runs/demo_v5_b32 runs/cosmos25_ab_bb32gr32/results \
    2>&1 | tee runs/demo_v5_b32/gates.log || true

# T2r: retry the failing fits at 1000 draws. More than half the bank
# failing means the recipe is refuted at B=32 -- leave that for a human
# read instead of burning node-hours on a mass retry.
FAILS=$(awk '$NF == "FAIL" {print $1}' runs/demo_v5_b32/gates.log | paste -sd, -)
N_FAILS=$(awk '$NF == "FAIL"' runs/demo_v5_b32/gates.log | wc -l | tr -d ' ')
echo "[job] gate failures: ${N_FAILS} (${FAILS:-none})"
if [ -n "$FAILS" ] && [ "$N_FAILS" -le 16 ]; then
    mkdir -p runs/demo_v5_b32_retry
    KLPIPE_RUN_DIR=runs/cosmos25_batched_b32 KLPIPE_FIT_IDS=$FAILS \
    KLPIPE_INIT=map_window KLPIPE_NUTS_WARMUP=200 KLPIPE_NUTS_SAMPLES=1000 \
    KLPIPE_DEMO_SEED=20260803 KLPIPE_DEMO_OUT=runs/demo_v5_b32_retry \
        $KLPIPE_PYTHON -u experiments/sweverett/batched_nuts/batched_nuts_demo.py \
        2>&1 | tee runs/demo_v5_b32_retry/demo.log
    $KLPIPE_PYTHON experiments/sweverett/batched_nuts/analyze_demo.py \
        runs/demo_v5_b32_retry runs/cosmos25_ab_bb32gr32/results \
        2>&1 | tee runs/demo_v5_b32_retry/gates.log || true
elif [ "$N_FAILS" -gt 16 ]; then
    echo "[job] SKIPPING retry: ${N_FAILS}/32 failed, recipe likely refuted at B=32"
fi

echo "[job] jax cache entries after: $(ls $SCRATCH/jax_cache 2>/dev/null | wc -l)"
echo "[job] done"
