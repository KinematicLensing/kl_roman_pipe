#!/bin/bash
# pull_vista_run.sh -- refresh a local copy of a vista ensemble run and rebuild
# its dashboard.
#
#   bash scripts/pull_vista_run.sh <remote_run_dir> <local_run_dir> [job_id ...]
#
# Pulls results/, status/, provenance/ and the manifest (not chains or mocks;
# add PULL_CHAINS=1 for chains/), the GPU logs and job logs of the listed
# SLURM job ids from $SCRATCH/gpu_speedups, then collates and writes
# <local_run_dir>/diagnostics/dashboard.html and opens it. Rides the user's
# open ControlMaster socket (VISTA_HOST, default sweveret@login1.vista.tacc.utexas.edu).
set -euo pipefail
[ $# -ge 2 ] || { echo "usage: $0 <remote_run_dir> <local_run_dir> [job_id ...]" >&2; exit 1; }
REMOTE=$1; LOCAL=$2; shift 2
HOST=${VISTA_HOST:-sweveret@login1.vista.tacc.utexas.edu}
SCRATCH_REMOTE=${VISTA_SCRATCH:-/scratch/09102/sweveret}
mkdir -p "$LOCAL"
EXCL=(--exclude mocks); [ "${PULL_CHAINS:-0}" = 1 ] || EXCL+=(--exclude chains)
rsync -az "${EXCL[@]}" "$HOST:$REMOTE/" "$LOCAL/"
for j in "$@"; do
  scp -q "$HOST:$SCRATCH_REMOTE/gpu_speedups/gpu_${j}*.csv" "$LOCAL/" 2>/dev/null || echo "no gpu log for $j"
  scp -q "$HOST:$SCRATCH_REMOTE/gpu_speedups/*_${j}.out" "$LOCAL/" 2>/dev/null || echo "no job log for $j"
done
# a fit can finish between the results and status transfers; pull results once more
rsync -az "$HOST:$REMOTE/results/" "$LOCAL/results/"
PY=${KLPIPE_PYTHON:-"conda run -n klpipe --no-capture-output python"}
$PY -m kl_pipe.ensemble collate --run-dir "$LOCAL" | grep -E "collated|wrote" || true
$PY -m kl_pipe.ensemble dashboard --run-dir "$LOCAL" --open
