#!/bin/bash
#-----------------------------------------------------------------------------
# Vista GH200 GPU-packing benchmark: section j.
#
# Companion to run_stampede3_pack.slurm (the CPU side of the platform
# decision). A single fit underutilizes the GH200 ~5-7x (section g), so N
# independent fit processes sharing the GPU may recover most of that with
# zero code refactor. Children split HBM via XLA_PYTHON_CLIENT_MEM_FRACTION
# = 0.85/N; if CUDA MPS is available the script starts it so kernels from
# different processes overlap instead of time-slicing.
#
# Reference serial numbers (fd hessian, warm cache): P = 178 s/fit = 20.2
# fits/node-hr. If N=4 packing gets P to ~60-80 fits/node-hr, GPU is back
# in the game vs the SPR node.
#
# Submit from the vista_kit directory on Vista:
#   sbatch run_pack_vista.slurm
#-----------------------------------------------------------------------------
#SBATCH -J klpipe-gh200-pack
#SBATCH -o klpipe-gh200-pack.%j.out
#SBATCH -e klpipe-gh200-pack.%j.err
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
CACHE="$SCRATCH/jaxcache"

mkdir -p "$RESULTS" "$CACHE"

# module init scripts are not 'set -u' clean; disable nounset around module.
set +u; module load tacc-apptainer; set -u

# stage the persistent $WORK master -> fast $SCRATCH (idempotent)
bash "$KIT/stage_vista.sh" "$TAG"

# CUDA MPS (optional): lets kernels from the N children overlap on the GPU.
# Started on the HOST (outside the container); pipe dir is bind-mounted in.
MPS_ON=0
if command -v nvidia-cuda-mps-control >/dev/null 2>&1; then
    export CUDA_MPS_PIPE_DIRECTORY="$SCRATCH/mps_pipe"
    export CUDA_MPS_LOG_DIRECTORY="$SCRATCH/mps_log"
    mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
    if nvidia-cuda-mps-control -d; then
        MPS_ON=1
        echo "=== CUDA MPS daemon started ==="
    else
        echo "=== CUDA MPS failed to start; children will time-slice ==="
    fi
else
    echo "=== nvidia-cuda-mps-control not found; children will time-slice ==="
fi

STAMP=$(date +%Y%m%d_%H%M%S)
HOSTTAG=$(hostname -s)

APPTAINER_RUN="apptainer exec --nv \
  --bind $REPO:$REPO \
  --bind $RESULTS:$RESULTS \
  --bind $PIPDIR:$PIPDIR \
  --bind $SCRATCH:$SCRATCH \
  --env PYTHONPATH=$REPO:$PIPDIR \
  $CONTAINER"

echo "=== GPU packing: section j, config P, N = 8,16,32 (MPS=$MPS_ON) ==="
cd "$KIT"
# 2026-07-12 run measured N=1,2,4,8: 10.1 / 20.6 / 30.7 / 42.7 fits/node-hr
# (not saturated). This sweep extends the curve: N=8 repeated as the
# run-to-run anchor, then 16 and 32. --pack-vary-data: each child gets
# distinct mock data -> per-galaxy recompile included, as production pays.
$APPTAINER_RUN python "$KIT/bench_matrix.py" \
  --sections j --pack-configs P --pack-workers 8,16,32 \
  --pack-device gpu --pack-mem-fraction 0.85 --pack-stagger-s 0.5 \
  --pack-vary-data --pack-vary-seeds --x64 1 \
  --fit-samples 1000 --fit-warmup 500 --fit-chains 4 \
  --jax-cache-dir "$CACHE" \
  --out "$KIT/results_ghpack_${STAMP}_${HOSTTAG}_P.json"

# stop MPS if we started it
if [[ "$MPS_ON" == "1" ]]; then
    echo quit | nvidia-cuda-mps-control || true
fi

cp -v "$KIT"/results_ghpack_${STAMP}_*.json "$RESULTS/"
echo "=== done; results in $RESULTS ==="
