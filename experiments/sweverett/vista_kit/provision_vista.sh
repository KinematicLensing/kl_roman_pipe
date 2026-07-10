#!/bin/bash
#-----------------------------------------------------------------------------
# provision_vista.sh -- idempotently set up the GH200 benchmark environment.
#
# Assembles the three layers the kit needs (see SETUP.md "Why a container"):
#   1. the NGC JAX container (.sif)  -- CUDA/cuDNN/jaxlib for Hopper
#   2. a pip --target sidecar        -- numpyro astropy pyyaml
#   3. (repo is cloned separately to $STOCKYARD/repos/kl_roman_pipe)
#
# Durable artifacts live on $WORK (persistent), NOT $SCRATCH (purged after
# ~10 days of no access). Safe to re-run any time: each step is skipped when
# already satisfied, so this doubles as "repair after a purge / new session".
#
# Usage:
#   module-free; run on a Vista login node.
#   bash provision_vista.sh              # default tag 26.06-py3
#   bash provision_vista.sh 26.06-py3    # pin a specific NGC JAX tag
#-----------------------------------------------------------------------------
set -euo pipefail

TAG="${1:-26.06-py3}"                        # NGC JAX tag (nvcr.io/nvidia/jax)
CONTAINER_DIR="$WORK/containers"
CONTAINER="$CONTAINER_DIR/jax_${TAG}.sif"
PIPDIR="$WORK/klpipe_pipdeps"
REPO="$STOCKYARD/repos/kl_roman_pipe"
RESULTS="$STOCKYARD/klpipe_bench_results"

module load tacc-apptainer

# 1. container: pull once, keep on $WORK. ~8.5 GB; multi-arch -> arm64 on Vista.
mkdir -p "$CONTAINER_DIR"
if [[ -f "$CONTAINER" ]]; then
  echo "[provision] container present: $CONTAINER"
else
  echo "[provision] pulling docker://nvcr.io/nvidia/jax:${TAG} (several minutes) ..."
  apptainer pull "$CONTAINER" "docker://nvcr.io/nvidia/jax:${TAG}"
fi

# 2. pip sidecar: install once. Check by importing from the target dir; if the
#    import succeeds the deps are already there. NEVER let pip touch jax/jaxlib
#    (those come from the image) -- these three are pure-python / small builds.
mkdir -p "$PIPDIR"
if apptainer exec "$CONTAINER" python -c \
     "import sys; sys.path.insert(0, '$PIPDIR'); import numpyro, astropy, yaml" 2>/dev/null; then
  echo "[provision] pip deps present: $PIPDIR"
else
  echo "[provision] installing numpyro astropy pyyaml -> $PIPDIR ..."
  apptainer exec "$CONTAINER" \
    python -m pip install --target "$PIPDIR" numpyro astropy pyyaml
fi

mkdir -p "$RESULTS"

# 3. import-only sanity (no GPU needed here; run the --nv device check on a
#    gh-dev node -- see SETUP.md step 4).
echo "[provision] import sanity ..."
apptainer exec \
  --bind "$REPO:$REPO" \
  --env "PYTHONPATH=$REPO:$PIPDIR" \
  "$CONTAINER" python -c "import numpyro, astropy, yaml, kl_pipe.source; print('imports OK')"

cat <<EOF

[provision] done. Persistent locations (survive \$SCRATCH purges):
  CONTAINER = $CONTAINER
  PIPDIR    = $PIPDIR
  REPO      = $REPO
  RESULTS   = $RESULTS

Next: on a gh-dev node, run the GPU device check (SETUP.md step 4) then the
micro-run, or 'sbatch run_vista.slurm' for the full matrix. For interactive
runs, this shell function wraps the exec incantation:

  klrun() { apptainer exec --nv \\
    --bind $REPO:$REPO --bind $RESULTS:$RESULTS --bind $PIPDIR:$PIPDIR \\
    --env PYTHONPATH=$REPO:$PIPDIR \\
    --env XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 \\
    $CONTAINER "\$@"; }
  # e.g.: klrun python bench_matrix.py --sections a --configs Q --nreps 5
EOF
