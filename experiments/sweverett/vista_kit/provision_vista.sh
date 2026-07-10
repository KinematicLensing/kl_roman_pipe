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
# Usage: run INSIDE an idev session (compute node). Apptainer is a NO-OP on
# Vista LOGIN nodes -- it prints a "do not run on login nodes" banner and does
# nothing, so a login-node run silently pulls/installs NOTHING (and reports
# false success). Grab a node first: idev -p gh-dev -N 1 -n 1 -t 01:00:00
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

# module init scripts are not 'set -u' clean (apptainer bash-completion
# references unbound vars); disable nounset just around module.
set +u; module load tacc-apptainer; set -u

# 1. container: pull once, keep on $WORK. ~8.5 GB; multi-arch -> arm64 on Vista.
# Cache on $SCRATCH -- the default $HOME/.apptainer/cache blows the small home
# quota. Pull NON-INTERACTIVELY (redirect to a log): apptainer's progress bar
# panics on this multi-layer image ("index out of range in ProgressComplete");
# no tty -> no progress bar -> no panic.
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-$SCRATCH/apptainer_cache}"
mkdir -p "$CONTAINER_DIR" "$APPTAINER_CACHEDIR"
if [[ -f "$CONTAINER" ]]; then
  echo "[provision] container present: $CONTAINER"
else
  LOG="$CONTAINER_DIR/pull_${TAG}.log"
  echo "[provision] pulling docker://nvcr.io/nvidia/jax:${TAG} (several minutes; log: $LOG) ..."
  apptainer pull "$CONTAINER" "docker://nvcr.io/nvidia/jax:${TAG}" > "$LOG" 2>&1
fi

# 2. pip sidecar: install once. Check by importing from the target dir; if the
#    import succeeds the deps are already there. NEVER let pip touch jax/jaxlib
#    (those come from the image) -- these three are pure-python / small builds.
mkdir -p "$PIPDIR"
# -B binds $PIPDIR into the container: pip --target writes from INSIDE the
# container, so the target must be explicitly bind-mounted (don't rely on
# TACC auto-bind).
if apptainer exec -B "$PIPDIR:$PIPDIR" "$CONTAINER" python -c \
     "import sys; sys.path.insert(0, '$PIPDIR'); import numpyro, astropy, yaml" 2>/dev/null; then
  echo "[provision] pip deps present: $PIPDIR"
else
  echo "[provision] installing numpyro astropy pyyaml -> $PIPDIR ..."
  apptainer exec -B "$PIPDIR:$PIPDIR" "$CONTAINER" \
    python -m pip install --no-cache-dir --target "$PIPDIR" numpyro astropy pyyaml
fi

mkdir -p "$RESULTS"

# NOTE: no sanity 'apptainer exec' here -- running containers (compute) on
# login nodes is against TACC policy. Only the internet-bound downloads above
# (pull + pip) run on the login node. Do the GPU device + import checks inside
# an idev session or the batch job (SETUP.md step 4).

cat <<EOF

[provision] done. Persistent locations (survive \$SCRATCH purges):
  CONTAINER = $CONTAINER
  PIPDIR    = $PIPDIR
  REPO      = $REPO
  RESULTS   = $RESULTS

Next: grab a compute node -- 'idev -p gh-dev -N 1 -n 1 -t 01:00:00' -- then run
the GPU device check (SETUP.md step 4) + micro-run THERE, or 'sbatch
run_vista.slurm' for the full matrix. Do NOT run the container on the login
node. For interactive runs (on a compute node), this wraps the exec incantation:

  klrun() { apptainer exec --nv \\
    --bind $REPO:$REPO --bind $RESULTS:$RESULTS --bind $PIPDIR:$PIPDIR \\
    --env PYTHONPATH=$REPO:$PIPDIR \\
    --env XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 \\
    $CONTAINER "\$@"; }
  # e.g.: klrun python bench_matrix.py --sections a --configs Q --nreps 5
EOF
