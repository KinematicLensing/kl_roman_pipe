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

# apptainer is a no-op on Vista login nodes -- fail loudly (run inside idev).
case "$(hostname -s)" in
  login*) echo "ERROR: on login node $(hostname -s) -- apptainer no-ops here. Run inside 'idev -p gh-dev -N 1 -n 1 -t 01:00:00'." >&2; exit 1 ;;
esac

TAG="${1:-26.06-py3}"                        # NGC JAX tag (nvcr.io/nvidia/jax)
CONTAINER_DIR="$WORK/containers"
CONTAINER="$CONTAINER_DIR/jax_${TAG}.sif"
PIPDIR="$WORK/klpipe_pipdeps"
FFTWDIR="$WORK/fftw3"
FFTW_VERSION="3.3.10"
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

# 2a. FFTW: galsim's one C dependency. PyPI has NO aarch64 galsim wheel, so
#     pip builds galsim from source inside the container (compiler + Eigen
#     auto-download both work there); only libfftw3 is missing. Build it once
#     into $WORK with the CONTAINER's toolchain (ABI consistency) and point
#     the galsim build at it via FFTW_DIR. galsim's setup records the fftw
#     runtime path, so no LD_LIBRARY_PATH is needed at run time -- but the
#     import check below verifies that loudly.
if [[ -f "$FFTWDIR/lib/libfftw3.so" ]]; then
  echo "[provision] fftw present: $FFTWDIR"
else
  echo "[provision] building fftw-$FFTW_VERSION -> $FFTWDIR (few minutes) ..."
  FFTW_TMP="$(mktemp -d)"
  mkdir -p "$FFTWDIR"
  # -f: fail on HTTP errors instead of piping an error page into tar
  if ! curl -fsSL "https://fftw.org/pub/fftw/fftw-${FFTW_VERSION}.tar.gz" \
       -o "$FFTW_TMP/fftw.tar.gz"; then
    echo "[provision] primary fftw URL failed; trying mirror ..."
    curl -fsSL "https://www.fftw.org/fftw-${FFTW_VERSION}.tar.gz" \
      -o "$FFTW_TMP/fftw.tar.gz" || {
      echo "ERROR: could not download fftw-$FFTW_VERSION" >&2; exit 1; }
  fi
  tar xzf "$FFTW_TMP/fftw.tar.gz" -C "$FFTW_TMP" || {
    echo "ERROR: fftw tarball extraction failed" >&2; exit 1; }
  # run each build stage explicitly so a failure names its stage and shows
  # the log tail (set -e would otherwise abort this script silently)
  # CC/CXX pinned to the container's gcc: the HOST module environment leaks
  # into apptainer exec (TACC binds /home1/apps and exports the nvhpc CC),
  # and configure otherwise picks the host's nvc, which cannot create
  # container-runnable executables. See the env-leakage note in SETUP.md.
  if ! apptainer exec -B "$FFTW_TMP:$FFTW_TMP" -B "$FFTWDIR:$FFTWDIR" \
      --env CC=gcc --env CXX=g++ "$CONTAINER" \
      bash -c "set -e; cd $FFTW_TMP/fftw-$FFTW_VERSION && \
               ./configure --enable-shared --disable-fortran --disable-doc \
                           --prefix=$FFTWDIR > $FFTWDIR/build.log 2>&1 && \
               make -j8 >> $FFTWDIR/build.log 2>&1 && \
               make install >> $FFTWDIR/build.log 2>&1"; then
    echo "ERROR: fftw build failed. Last 30 lines of $FFTWDIR/build.log:" >&2
    tail -30 "$FFTWDIR/build.log" >&2 || true
    exit 1
  fi
  rm -rf "$FFTW_TMP"
  [[ -f "$FFTWDIR/lib/libfftw3.so" ]] || {
    echo "ERROR: fftw build reported success but $FFTWDIR/lib/libfftw3.so" >&2
    echo "is missing -- see $FFTWDIR/build.log" >&2; exit 1; }
  echo "[provision] fftw built: $FFTWDIR"
fi

# 2. pip sidecar: install once. Check by importing from the target dir; if the
#    import succeeds the deps are already there. NEVER let pip touch jax/jaxlib
#    (those come from the image) -- pure-python or wheels, EXCEPT galsim,
#    which builds from source against the fftw above (~10-20 min compile,
#    one-time). pandas/pyarrow feed the ensemble parquet manifests/results;
#    corner feeds run diagnostics.
SIDECAR_PKGS="numpyro astropy pyyaml matplotlib galsim pandas pyarrow corner"
mkdir -p "$PIPDIR"
# -B binds $PIPDIR into the container: pip --target writes from INSIDE the
# container, so the target must be explicitly bind-mounted (don't rely on
# TACC auto-bind).
if apptainer exec -B "$PIPDIR:$PIPDIR" -B "$FFTWDIR:$FFTWDIR" "$CONTAINER" python -c \
     "import sys; sys.path.insert(0, '$PIPDIR'); import numpyro, astropy, yaml, matplotlib, galsim, pandas, pyarrow, corner" 2>/dev/null; then
  echo "[provision] pip deps present: $PIPDIR"
else
  echo "[provision] installing $SIDECAR_PKGS -> $PIPDIR (galsim compiles from"
  echo "[provision] source, ~10-20 min one-time) ..."
  # CC/CXX pinned for the same host-env-leakage reason as the fftw build
  apptainer exec -B "$PIPDIR:$PIPDIR" -B "$FFTWDIR:$FFTWDIR" \
    --env FFTW_DIR="$FFTWDIR" --env CC=gcc --env CXX=g++ "$CONTAINER" \
    python -m pip install --no-cache-dir --target "$PIPDIR" $SIDECAR_PKGS
  # numpyro drags in a jax/jaxlib/CUDA-plugin set that mismatches (and would
  # shadow) the container's GPU jax; strip it so only the container's is used.
  echo "[provision] removing pip-dragged jax/CUDA from the sidecar ..."
  rm -rf "$PIPDIR"/jax* "$PIPDIR"/nvidia*
  # loud post-install check: galsim must import AND find libfftw3 at runtime
  apptainer exec -B "$PIPDIR:$PIPDIR" -B "$FFTWDIR:$FFTWDIR" "$CONTAINER" \
    python -c "import sys; sys.path.insert(0, '$PIPDIR'); import galsim; \
print('[provision] galsim', galsim.__version__, 'imports OK')" || {
    echo "ERROR: galsim import failed after install. If the error is a" >&2
    echo "missing libfftw3.so, bind $FFTWDIR and prepend" >&2
    echo "$FFTWDIR/lib to LD_LIBRARY_PATH in your exec command." >&2
    exit 1
  }
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
    --bind $FFTWDIR:$FFTWDIR \\
    --env PYTHONPATH=$REPO:$PIPDIR \\
    --env XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 \\
    $CONTAINER "\$@"; }
  # e.g.: klrun python bench_matrix.py --sections a --configs Q --nreps 5
EOF
