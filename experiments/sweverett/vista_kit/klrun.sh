#!/bin/bash
#-----------------------------------------------------------------------------
# klrun.sh -- run a command inside the kl_pipe GPU container on a COMPUTE node.
#
# Apptainer is a NO-OP on Vista LOGIN nodes (prints a "do not run on login
# nodes" banner and exits without executing). Run this only inside idev or a
# batch job. Copy-paste friendly: one command, no heredocs or line breaks.
#
# Usage:
#   bash klrun.sh python sanity_check.py
#   bash klrun.sh python bench_matrix.py --sections a --configs Q --nreps 5
#
# Env overrides: KLPIPE_TAG (container tag, default 26.06-py3).
#-----------------------------------------------------------------------------
set -euo pipefail
# apptainer is a no-op on Vista login nodes -- fail loudly instead of silently
# doing nothing (run this inside idev or sbatch).
case "$(hostname -s)" in
  login*) echo "ERROR: on login node $(hostname -s) -- apptainer no-ops here. Use 'idev -p gh-dev ...' or sbatch." >&2; exit 1 ;;
esac
# module init scripts are not 'set -u' clean; disable nounset around module.
set +u; module load tacc-apptainer; set -u

TAG="${KLPIPE_TAG:-26.06-py3}"
REPO="$STOCKYARD/repos/kl_roman_pipe"
RESULTS="$STOCKYARD/klpipe_bench_results"

# prefer the fast $SCRATCH staged copy; fall back to the $WORK master.
CONTAINER="$SCRATCH/containers/jax_${TAG}.sif"
[[ -f "$CONTAINER" ]] || CONTAINER="$WORK/containers/jax_${TAG}.sif"
PIPDIR="$SCRATCH/klpipe_pipdeps"
[[ -d "$PIPDIR" ]] || PIPDIR="$WORK/klpipe_pipdeps"

[[ -f "$CONTAINER" ]] || { echo "ERROR: container not found: $CONTAINER -- run 'bash provision_vista.sh' in idev first" >&2; exit 1; }
mkdir -p "$RESULTS"

exec apptainer exec --nv \
  --bind "$REPO:$REPO" \
  --bind "$RESULTS:$RESULTS" \
  --bind "$PIPDIR:$PIPDIR" \
  --env "PYTHONPATH=$REPO:$PIPDIR" \
  --env "LD_PRELOAD=$PIPDIR/galsim/libfftw3.so.3" \
  --env XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 \
  "$CONTAINER" "$@"
