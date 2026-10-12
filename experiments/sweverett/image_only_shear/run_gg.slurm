#!/bin/bash
# image-only shear arm on one Vista gg (Grace CPU) node: 36 workers x 4 cores, tasks strided over workers.
# Usage: sbatch --export=ARM=pinned run_gg.slurm   (resubmitting resumes: finished fits are skipped)
#SBATCH -J io_shear
#SBATCH -p gg
#SBATCH -A JPL-PUB
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 04:00:00
#SBATCH -o io_shear-%j.out
#SBATCH -e io_shear-%j.err
set -euo pipefail

: "${ARM:?set ARM=pinned or ARM=sampled}"
export KLPIPE_REPO="$STOCKYARD/repos/kl_roman_pipe_xu23"
source "$KLPIPE_REPO/experiments/sweverett/vista_kit/env_vista.sh"
export JAX_PLATFORMS=cpu
export KLPIPE_CPU_DEVICES=4

SCRIPT="$KLPIPE_REPO/experiments/sweverett/image_only_shear/run_image_only.py"
OUT="$SCRATCH/image_only_shear/out"
LOGS="$SCRATCH/image_only_shear/logs/$SLURM_JOB_ID"
mkdir -p "$OUT" "$LOGS"
N_WORKERS=36
N_TASKS=$($KLPIPE_PYTHON "$SCRIPT" list --arm "$ARM" 2>/dev/null | tail -1)
echo "arm $ARM: $N_TASKS tasks over $N_WORKERS workers, out $OUT"

for w in $(seq 0 $((N_WORKERS - 1))); do
  idx=$(seq -s, "$w" "$N_WORKERS" $((N_TASKS - 1)))
  taskset -c $((4 * w))-$((4 * w + 3)) \
    $KLPIPE_PYTHON "$SCRIPT" run --arm "$ARM" --tasks "$idx" --out "$OUT" \
    > "$LOGS/w$w.out" 2> "$LOGS/w$w.err" &
done
wait
echo "done: $(ls "$OUT/$ARM"/*.json 2>/dev/null | wc -l) results, $(ls "$OUT/$ARM"/*.FAILED 2>/dev/null | wc -l) failed"
