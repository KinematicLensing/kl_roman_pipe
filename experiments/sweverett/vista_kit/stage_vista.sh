#!/bin/bash
#-----------------------------------------------------------------------------
# stage_vista.sh -- copy the persistent $WORK master artifacts to the fast,
# per-machine $SCRATCH filesystem, where jobs should do their I/O.
#
# Why (not clobbering -- contention): the .sif is read-only, so any number of
# concurrent fits reading it is safe. But $WORK (Stockyard) is ONE global
# filesystem shared across all TACC machines and is not built for job I/O.
# Hundreds/thousands of concurrent fits reading an 8.5 GB image + a
# many-small-file pip sidecar from Stockyard hammers a center-wide resource.
# $SCRATCH is the high-performance, per-machine filesystem for exactly this.
#
# Model: master copy persists on $WORK (survives $SCRATCH purges); this script
# mirrors it to $SCRATCH at a fixed, expected path. run_vista.slurm points
# jobs at the $SCRATCH copy.
#
# Idempotent + atomic. Run ONCE before launching a large job array (do NOT let
# each array task copy -- that is a copy storm). A single smoke job can rely on
# run_vista.slurm's auto-stage prologue.
#
# Usage: bash stage_vista.sh [tag]        # default 26.06-py3
#-----------------------------------------------------------------------------
set -euo pipefail

TAG="${1:-26.06-py3}"
SRC_SIF="$WORK/containers/jax_${TAG}.sif"
DST_DIR="$SCRATCH/containers"
DST_SIF="$DST_DIR/jax_${TAG}.sif"
SRC_PIP="$WORK/klpipe_pipdeps"
DST_PIP="$SCRATCH/klpipe_pipdeps"

[[ -f "$SRC_SIF" ]] || { echo "ERROR: master image missing: $SRC_SIF -- run provision_vista.sh first" >&2; exit 1; }
[[ -d "$SRC_PIP" ]] || { echo "ERROR: master pipdeps missing: $SRC_PIP -- run provision_vista.sh first" >&2; exit 1; }

mkdir -p "$DST_DIR"

# container: copy only if absent or size-mismatched; write to temp then mv
# (atomic rename) so a concurrent reader never sees a partial file.
if [[ -f "$DST_SIF" && "$(stat -c%s "$SRC_SIF")" == "$(stat -c%s "$DST_SIF")" ]]; then
  echo "[stage] container already staged: $DST_SIF"
else
  echo "[stage] copying container -> $DST_SIF ..."
  cp -f "$SRC_SIF" "$DST_SIF.tmp.$$"
  mv -f "$DST_SIF.tmp.$$" "$DST_SIF"
fi

# pip sidecar: mirror the directory (rsync if present, else cp -r).
echo "[stage] syncing pip sidecar -> $DST_PIP ..."
if command -v rsync >/dev/null 2>&1; then
  mkdir -p "$DST_PIP"
  rsync -a --delete "$SRC_PIP/" "$DST_PIP/"
else
  rm -rf "$DST_PIP"
  cp -r "$SRC_PIP" "$DST_PIP"
fi

echo "[stage] done. Jobs should use:"
echo "  CONTAINER=$DST_SIF"
echo "  PIPDIR=$DST_PIP"
