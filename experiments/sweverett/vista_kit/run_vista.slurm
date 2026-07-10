#!/bin/bash
#-----------------------------------------------------------------------------
# Vista GH200 benchmark: kl_pipe grism posterior matrix (fp64 + fp32 passes)
#
# Submit from the vista_kit directory on Vista:
#   sbatch run_vista.slurm
#
# TACC Vista notes:
# - GH nodes: 1x GH200 (96 GB HBM3) + 72-core Grace ARM host per node.
# - Queues (check `sinfo -o "%P %l %D"` or the Vista user guide for current
#   names/limits): "gh" = production GPU queue (1 SU/node-hr),
#   "gh-dev" = dev/debug queue (shorter max walltime, faster scheduling).
#   Use the dev queue for the first smoke run.
# - Containers: module tacc-apptainer; NGC images run with --nv.
#-----------------------------------------------------------------------------
#SBATCH -J klpipe-gh200-bench
#SBATCH -o klpipe-gh200-bench.%j.out
#SBATCH -e klpipe-gh200-bench.%j.err
#SBATCH -N 1                       # 1 node = 1 GH200
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -p gh-dev                  # dev queue for smoke runs; switch to "gh" for production
#SBATCH -A JPL-SPHEREx             # TACC charge code (user sweveret)

set -euo pipefail

#-----------------------------------------------------------------------------
# paths (sweveret Vista layout: $STOCKYARD=/work/09102/sweveret)
#-----------------------------------------------------------------------------
REPO="$STOCKYARD/repos/kl_roman_pipe"            # repo clone location on Vista
KIT="$REPO/experiments/sweverett/vista_kit"
TAG="26.06-py3"                                  # NGC JAX tag (must match provision_vista.sh)
CONTAINER="$SCRATCH/containers/jax_${TAG}.sif"   # HOT copy on fast per-machine $SCRATCH (staged from $WORK master)
PIPDIR="$SCRATCH/klpipe_pipdeps"                 # HOT copy of the pip sidecar
RESULTS="$STOCKYARD/klpipe_bench_results"        # persistent results dir

mkdir -p "$RESULTS"

# module init scripts are not 'set -u' clean (apptainer bash-completion
# references unbound vars); disable nounset just around module.
set +u; module load tacc-apptainer; set -u

# Stage the persistent $WORK master -> fast $SCRATCH (idempotent; single-job
# safe). For a large job ARRAY, run stage_vista.sh ONCE before sbatch instead
# of relying on this line, so array tasks don't each trigger a copy.
bash "$KIT/stage_vista.sh" "$TAG"

STAMP=$(date +%Y%m%d_%H%M%S)
HOSTTAG=$(hostname -s)

# Container python + our extra pip deps + the repo itself.
# JAX/CUDA come from the NGC image; do NOT pip-install jax in the container.
APPTAINER_RUN="apptainer exec --nv \
  --bind $REPO:$REPO \
  --bind $RESULTS:$RESULTS \
  --bind $PIPDIR:$PIPDIR \
  --env PYTHONPATH=$REPO:$PIPDIR \
  --env XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 \
  $CONTAINER"

echo "=== container + GPU sanity ==="
$APPTAINER_RUN python -c "
import jax, jaxlib, socket
print('host:', socket.gethostname())
print('jax', jax.__version__, 'jaxlib', jaxlib.__version__)
print('devices:', jax.devices())
assert any('cuda' in str(d).lower() or 'gpu' in str(d).lower() for d in jax.devices()), 'NO GPU VISIBLE'
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
assert jnp.zeros(1).dtype == jnp.float64, 'x64 not working'
print('x64 OK')
"

#-----------------------------------------------------------------------------
# pass 1: fp64 (full matrix)
#-----------------------------------------------------------------------------
echo "=== fp64 pass ==="
cd "$KIT"
$APPTAINER_RUN python "$KIT/bench_matrix.py" \
  --x64 1 --nreps 30 \
  --out "$KIT/results_vista_${STAMP}_${HOSTTAG}_fp64.json"

#-----------------------------------------------------------------------------
# pass 2: fp32 (full matrix; section e inside each pass also does the
# cross-precision pixel-delta comparison via subprocesses)
#-----------------------------------------------------------------------------
echo "=== fp32 pass ==="
$APPTAINER_RUN python "$KIT/bench_matrix.py" \
  --x64 0 --nreps 30 \
  --out "$KIT/results_vista_${STAMP}_${HOSTTAG}_fp32.json"

#-----------------------------------------------------------------------------
# collect
#-----------------------------------------------------------------------------
cp -v "$KIT"/results_vista_${STAMP}_*.json "$RESULTS/"
# keep the section-e child artifacts too (pixel-delta npys + child jsons)
cp -v "$KIT"/results_vista_${STAMP}_*child* "$RESULTS/" 2>/dev/null || true

echo "=== done; results in $RESULTS ==="
ls -la "$RESULTS" | grep "$STAMP"
