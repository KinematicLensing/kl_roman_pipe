#!/bin/bash
#SBATCH -J kl_t3b_solos
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -A JPL-PUB
#SBATCH -o runs/%x_%j.log

# T3b: remaining 2 solo production baselines on the bb32gr32 bank
# (worker claims the next unclaimed fits; 93b40b3f/dcbcabdc are done).
# JAX_LOG_COMPILES settles whether the 9.5 min/fit solo number includes
# each galaxy's compile or rode a warm $SCRATCH/jax_cache -- the solo
# tradespace row depends on it. sbatch from the repo root so the log
# lands in runs/.

set -o pipefail
REPO=$STOCKYARD/repos/kl_roman_pipe
cd $REPO || { echo "[job] FATAL: cannot cd to $REPO"; exit 1; }
source experiments/sweverett/vista_kit/env_vista.sh
[ -n "${KLPIPE_PYTHON:-}" ] || { echo "[job] FATAL: KLPIPE_PYTHON unset"; exit 1; }

export APPTAINERENV_JAX_LOG_COMPILES=1
echo "[job] jax cache entries before: $(ls $SCRATCH/jax_cache 2>/dev/null | wc -l)"

$KLPIPE_PYTHON -m kl_pipe.ensemble worker \
    --run-dir runs/cosmos25_ab_bb32gr32 --label solo_b --max-fits 2 \
    2>&1 | tee runs/cosmos25_ab_bb32gr32/solo_b.log

echo "[job] jax cache entries after: $(ls $SCRATCH/jax_cache 2>/dev/null | wc -l)"
echo "[job] solo results so far"
$KLPIPE_PYTHON - <<'EOF'
import glob
import pandas as pd
fs = sorted(glob.glob('runs/cosmos25_ab_bb32gr32/results/*.parquet'))
rows = pd.concat([pd.read_parquet(p) for p in fs])
cols = ['fit_id', 'fit_wallclock_s', 'precond_wallclock_s',
        'max_rhat', 'min_ess', 'n_attempts', 'status']
print(rows[[c for c in cols if c in rows.columns]].to_string())
print(f'median fit wallclock: {rows.fit_wallclock_s.median() / 60:.1f} min')
EOF
echo "[job] done"
