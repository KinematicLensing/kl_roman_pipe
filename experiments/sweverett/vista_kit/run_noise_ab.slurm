#!/bin/bash
#SBATCH -J kl_noise_ab
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 04:00:00
#SBATCH -A JPL-PUB
#SBATCH -o kl_noise_ab_%j.log

# Noise-model A/B on GH200: the poisson arm (physical published-depth
# backgrounds + source shot noise) of the same 8-galaxy COSMOS25 bank the
# stamp A/B's bb32gr32 arm already fit with matched_filter noise. 16 fits.
# Self-contained and idempotent, same pattern as run_stamp_ab.slurm:
# expands if needed, reclaims stale claims, packs 4 workers, collates,
# prints a per-fit summary including the realized snr_effective columns.
# Safe to resubmit; a second job finishes leftovers. sbatch from
# $SCRATCH/kl_runs so the job log lands there.

# no set -u: Lmod's module function evals code with unset variables and
# aborts silently under nounset
set -o pipefail
REPO=$STOCKYARD/repos/kl_roman_pipe
RUNS=$SCRATCH/kl_runs
ARMS="cosmos25_noise_ab_poisson"
cd $REPO || { echo "[job] FATAL: cannot cd to $REPO"; exit 1; }
source experiments/sweverett/vista_kit/env_vista.sh
[ -n "${KLPIPE_PYTHON:-}" ] || { echo "[job] FATAL: KLPIPE_PYTHON unset after env_vista.sh"; exit 1; }

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_MPS_PIPE_DIRECTORY="$SCRATCH/mps_pipe_${SLURM_JOB_ID}"
export CUDA_MPS_LOG_DIRECTORY="$SCRATCH/mps_log_${SLURM_JOB_ID}"
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
nvidia-cuda-mps-control -d 2>/dev/null && echo "[job] MPS started" \
    || echo "[job] MPS unavailable, continuing without it"

echo "[job] expand + reclaim"
for arm in $ARMS; do
    if [ ! -f $RUNS/$arm/manifest.parquet ]; then
        $KLPIPE_PYTHON -m kl_pipe.ensemble expand \
            configs/ensembles/$arm.yaml --runs-dir $RUNS
    fi
    $KLPIPE_PYTHON -m kl_pipe.ensemble reclaim --run-dir $RUNS/$arm || true
done

echo "[job] packed run, 4 workers"
for arm in $ARMS; do
    $KLPIPE_PYTHON -m kl_pipe.ensemble run --run-dir $RUNS/$arm --workers 4
done

echo "[job] collate + summary"
for arm in $ARMS; do
    $KLPIPE_PYTHON -m kl_pipe.ensemble collate --run-dir $RUNS/$arm || true
    $KLPIPE_PYTHON -m kl_pipe.ensemble status --run-dir $RUNS/$arm || true
done
$KLPIPE_PYTHON - <<'EOF'
import glob, os
import pandas as pd
for arm in ['cosmos25_noise_ab_poisson']:
    fs = glob.glob(os.environ['SCRATCH'] + f'/kl_runs/{arm}/results/*.parquet')
    print(f'\n=== {arm}: {len(fs)} fits done ===')
    if not fs:
        continue
    rows = pd.concat([pd.read_parquet(p) for p in fs])
    cols = ['fit_id', 'fit_wallclock_s', 'num_steps_total', 'max_rhat',
            'min_ess', 'divergence_rate', 'n_attempts', 'noise_model',
            'snr_effective_F129', 'snr_effective_F158',
            'snr_effective_line_total']
    have = [c for c in cols if c in rows.columns]
    print(rows[have].to_string())
    if 'fit_wallclock_s' in rows.columns:
        print(f'median fit wallclock: {rows.fit_wallclock_s.median()/60:.1f} min')
EOF
echo "[job] done"
