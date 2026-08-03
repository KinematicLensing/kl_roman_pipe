#!/bin/bash
#SBATCH -J kl_t4_packed_solo
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 03:00:00
#SBATCH -A JPL-PUB
#SBATCH -o runs/%x_%j.log

# T4 (optional): finish the remaining 12 bb32gr32 fits with the production
# packed configuration (4 workers under MPS, same as the stamp-A/B
# campaign) -- measures the real packed-solo fits/node-hr on the CURRENT
# config, the denominator every speedup proposal gets compared against.
# Also completes the bank so any future strategy trial has 16/16 solo
# references. sbatch from the repo root so the log lands in runs/.

set -o pipefail
REPO=$STOCKYARD/repos/kl_roman_pipe
cd $REPO || { echo "[job] FATAL: cannot cd to $REPO"; exit 1; }
source experiments/sweverett/vista_kit/env_vista.sh
[ -n "${KLPIPE_PYTHON:-}" ] || { echo "[job] FATAL: KLPIPE_PYTHON unset"; exit 1; }

export APPTAINERENV_JAX_LOG_COMPILES=1
# parent must never hold HBM; dispatch slices MEM_FRACTION per worker
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# CUDA MPS so packed workers' kernels overlap instead of time-slicing
export CUDA_MPS_PIPE_DIRECTORY="$SCRATCH/mps_pipe_${SLURM_JOB_ID}"
export CUDA_MPS_LOG_DIRECTORY="$SCRATCH/mps_log_${SLURM_JOB_ID}"
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
nvidia-cuda-mps-control -d 2>/dev/null && echo "[job] MPS started" \
    || echo "[job] MPS unavailable, continuing without it"

$KLPIPE_PYTHON -m kl_pipe.ensemble reclaim --run-dir runs/cosmos25_ab_bb32gr32 || true
T0=$(date +%s)
$KLPIPE_PYTHON -m kl_pipe.ensemble run \
    --run-dir runs/cosmos25_ab_bb32gr32 --workers 4
T1=$(date +%s)
echo "[job] packed wall: $((T1 - T0)) s for this pass"

$KLPIPE_PYTHON -m kl_pipe.ensemble status --run-dir runs/cosmos25_ab_bb32gr32 || true
$KLPIPE_PYTHON - <<'EOF'
import glob
import pandas as pd
fs = sorted(glob.glob('runs/cosmos25_ab_bb32gr32/results/*.parquet'))
rows = pd.concat([pd.read_parquet(p) for p in fs])
cols = ['fit_id', 'fit_wallclock_s', 'precond_wallclock_s', 'num_steps_total',
        'max_rhat', 'min_ess', 'n_attempts', 'status']
print(rows[[c for c in cols if c in rows.columns]].to_string())
print(f'{len(rows)} fits done; median wallclock '
      f'{rows.fit_wallclock_s.median() / 60:.1f} min')
EOF
echo "[job] done"
