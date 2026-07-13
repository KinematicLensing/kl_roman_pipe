#!/bin/bash
#-----------------------------------------------------------------------------
# Stampede3 SPR CPU-node packing benchmark: section j.
#
# THE GPU-vs-CPU platform decider for the paper ensembles. A GH200 runs our
# fits serially (underutilized ~5-7x); an SPR node has 112 cores and our
# per-fit memory footprint is small, so 112 core-pinned parallel fits may
# beat the GPU node on fits/node-hour -- and fits/SU. This measures it:
#
#   sweep 1 (Q, cheap): pack widths 1, 56, 112 -> contention curve
#   sweep 2 (P, paper): pack widths 1, 112     -> headline fits/node-hr
#
# Compare against GH200: P = 20.2 fits/node-hr serial (warm cache,
# results_vista_20260711_233536), Q = 71.
#
# Prereqs (once, login node is fine):
#   bash provision_stampede3.sh          # venv on $WORK
#   repo cloned at $STOCKYARD/repos/kl_roman_pipe (same layout as Vista)
#
# Submit from the vista_kit directory:
#   sbatch run_stampede3_pack.slurm
#
# Stampede3 notes:
# - SPR nodes: 112 cores Sapphire Rapids, 128 GB HBM2e. Queues: "spr"
#   (production), "skx-dev" (debug; 48-core Skylake -- adjust pack widths).
# - MEMORY is the risk at N=112: ~128 GB / 112 procs ~ 1.1 GB per JAX
#   process. Section j records per-child peak RSS; if children die at 112,
#   read the RSS at N=56 and resubmit with widths that fit.
# - Lonestar6 variant: -p normal, 128 cores/node -> widths 1,64,128; run
#   provision_stampede3.sh there too (same x86 wheels).
# - Verify the charge code: Vista uses JPL-SPHEREx; your Stampede3
#   allocation name may differ (check `sacct -X` or the TACC portal).
#-----------------------------------------------------------------------------
#SBATCH -J klpipe-spr-pack
#SBATCH -o klpipe-spr-pack.%j.out
#SBATCH -e klpipe-spr-pack.%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 112                      # bind the batch step to ALL cores: with the
                                    # default -c 1 the step gets a 1-core
                                    # affinity mask (nproc=1) even on an
                                    # exclusive node (observed 2026-07-12)
#SBATCH -t 10:00:00
#SBATCH -p spr
#SBATCH -A JPL-SPHEREx

set -euo pipefail

REPO="$STOCKYARD/repos/kl_roman_pipe"
KIT="$REPO/experiments/sweverett/vista_kit"
VENV="$WORK/klpipe_cpu_venv"
RESULTS="$STOCKYARD/klpipe_bench_results"
CACHE="$SCRATCH/jaxcache_cpu"          # persistent-compile-cache; children share

mkdir -p "$RESULTS" "$CACHE"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
export PYTHONPATH="$REPO"

STAMP=$(date +%Y%m%d_%H%M%S)
HOSTTAG=$(hostname -s)
cd "$KIT"

# thread headroom: 112 jax processes need thousands of threads; raise the
# soft per-user process limit to the hard cap (section j also does this
# in-process, but do it here too so the numbers appear in the log)
ulimit -u "$(ulimit -Hu)" 2>/dev/null || true
echo "=== node: $(hostname) / $(nproc) cores visible / ulimit -u $(ulimit -u) ==="
python -c "import os, jax; print('jax', jax.__version__, jax.devices(), 'os.cpu_count', os.cpu_count())"

# sweep 1: Q contention curve (N=1 also warms the shared compile cache)
python "$KIT/bench_matrix.py" \
  --sections j --pack-configs Q --pack-workers 1,56,112 \
  --pack-device cpu --pack-stagger-s 0.5 --x64 1 \
  --fit-samples 1000 --fit-warmup 500 --fit-chains 4 \
  --jax-cache-dir "$CACHE" \
  --out "$KIT/results_spr_${STAMP}_${HOSTTAG}_packQ.json"

# sweep 2: P headline (the paper config). --pack-vary-data gives each child
# distinct mock data: data is baked into the jitted posterior as constants,
# so this includes the per-galaxy recompile cost production actually pays.
python "$KIT/bench_matrix.py" \
  --sections j --pack-configs P --pack-workers 1,56,112 \
  --pack-device cpu --pack-stagger-s 0.5 --pack-vary-data --pack-vary-seeds \
  --x64 1 \
  --fit-samples 1000 --fit-warmup 500 --fit-chains 4 \
  --jax-cache-dir "$CACHE" \
  --out "$KIT/results_spr_${STAMP}_${HOSTTAG}_packP.json"

cp -v "$KIT"/results_spr_${STAMP}_*.json "$RESULTS/"
echo "=== done; results in $RESULTS ==="
