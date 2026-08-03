#!/bin/bash
#SBATCH -J kl_t1b_mapwindow
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 03:00:00
#SBATCH -A JPL-PUB
#SBATCH -o runs/%x_%j.log

# T1b: batched B=16, MAP init + 200-step dense window adaptation
# (map_window) -- the closest blackjax analog of production's Laplace-init
# + adapt-mass recipe. Decision input 1 of the adopt/stop rule. The 08-02
# idev attempt hit the 2 h wall mid-sampling with no output; the batch
# queue removes that failure mode (SLURM charges actual runtime, not the
# requested wall). Warm-cache projection ~91 min; wall 3 h covers a cold
# recompile of every phase. sbatch from the repo root so the log lands in
# runs/.

set -o pipefail
REPO=$STOCKYARD/repos/kl_roman_pipe
cd $REPO || { echo "[job] FATAL: cannot cd to $REPO"; exit 1; }
source experiments/sweverett/vista_kit/env_vista.sh
[ -n "${KLPIPE_PYTHON:-}" ] || { echo "[job] FATAL: KLPIPE_PYTHON unset"; exit 1; }

# log every XLA compile with its wall time so warm-cache runs are
# distinguishable from cold ones in the timing writeup
export APPTAINERENV_JAX_LOG_COMPILES=1
echo "[job] jax cache entries before: $(ls $SCRATCH/jax_cache 2>/dev/null | wc -l)"

mkdir -p runs/demo_v4b_mapwindow
KLPIPE_INIT=map_window KLPIPE_NUTS_WARMUP=200 KLPIPE_NUTS_SAMPLES=600 \
KLPIPE_DEMO_OUT=runs/demo_v4b_mapwindow \
    $KLPIPE_PYTHON -u experiments/sweverett/batched_nuts/batched_nuts_demo.py \
    2>&1 | tee runs/demo_v4b_mapwindow/demo.log

echo "[job] jax cache entries after: $(ls $SCRATCH/jax_cache 2>/dev/null | wc -l)"
echo "[job] gate + parity analysis"
$KLPIPE_PYTHON experiments/sweverett/batched_nuts/analyze_demo.py \
    runs/demo_v4b_mapwindow runs/cosmos25_ab_bb32gr32/results || true
echo "[job] done"
