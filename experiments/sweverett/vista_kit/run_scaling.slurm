#!/bin/bash
#-----------------------------------------------------------------------------
# Vista GH200 scaling run: sections g (galaxy-batch throughput), h (Laplace
# preconditioner cost), i (full-fit throughput). fp64 only -- these are
# throughput/cost measurements, not precision comparisons.
#
# Submit from the vista_kit directory on Vista:
#   sbatch run_scaling.slurm
#
# Companion to run_vista.slurm (the a-e precision matrix). Both run on a
# compute node (apptainer no-ops on login nodes).
#-----------------------------------------------------------------------------
#SBATCH -J klpipe-gh200-scaling
#SBATCH -o klpipe-gh200-scaling.%j.out
#SBATCH -e klpipe-gh200-scaling.%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -p gh-dev                  # dev queue; switch to "gh" for production
#SBATCH -A JPL-SPHEREx

set -euo pipefail

REPO="$STOCKYARD/repos/kl_roman_pipe"
KIT="$REPO/experiments/sweverett/vista_kit"
TAG="26.06-py3"
CONTAINER="$SCRATCH/containers/jax_${TAG}.sif"
PIPDIR="$SCRATCH/klpipe_pipdeps"
RESULTS="$STOCKYARD/klpipe_bench_results"

mkdir -p "$RESULTS"

# module init scripts are not 'set -u' clean; disable nounset around module.
set +u; module load tacc-apptainer; set -u

# stage the persistent $WORK master -> fast $SCRATCH (idempotent, single-job safe)
bash "$KIT/stage_vista.sh" "$TAG"

STAMP=$(date +%Y%m%d_%H%M%S)
HOSTTAG=$(hostname -s)

APPTAINER_RUN="apptainer exec --nv \
  --bind $REPO:$REPO \
  --bind $RESULTS:$RESULTS \
  --bind $PIPDIR:$PIPDIR \
  --env PYTHONPATH=$REPO:$PIPDIR \
  --env XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 \
  $CONTAINER"

echo "=== scaling run: sections g,h,i (fp64), configs Q,P ==="
cd "$KIT"
$APPTAINER_RUN python "$KIT/bench_matrix.py" \
  --sections g,h,i --configs Q,P --x64 1 \
  --batch-sizes 1,2,4,8,16,32,64,128,256,512,1024,2048 \
  --precond-starts 1,4 \
  --fit-samples 1000 --fit-warmup 500 --fit-chains 4 \
  --out "$KIT/results_scaling_${STAMP}_${HOSTTAG}.json"

cp -v "$KIT"/results_scaling_${STAMP}_*.json "$RESULTS/"
echo "=== done; results in $RESULTS ==="
