#!/bin/bash
# image-only shear arm on Vista gg (Grace CPU, 0.33 SU/node-hr): array of nodes x 36 workers x 4 cores; tasks in
# descending image SNR, interleaved over nodes then workers.
# Usage: sbatch -a 0-1 --export=ALL,ARM=pinned,MIN_SNR=47[,MAX_SNR=224,PAS=pa0:pa45] run_gg.slurm   (resubmitting resumes: finished fits skipped)
#SBATCH -J io_shear
#SBATCH -p gg
#SBATCH -A JPL-PUB
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 04:00:00
#SBATCH -o io_shear-%A_%a.out
#SBATCH -e io_shear-%A_%a.err
set -euo pipefail

: "${ARM:?set ARM=pinned or ARM=sampled}"
export KLPIPE_REPO="$STOCKYARD/repos/kl_roman_pipe_xu23"
# module init scripts are not set -u clean
set +u; source "$KLPIPE_REPO/experiments/sweverett/vista_kit/env_vista.sh"; set -u
export JAX_PLATFORMS=cpu
export KLPIPE_CPU_DEVICES=4

SCRIPT="$KLPIPE_REPO/experiments/sweverett/image_only_shear/run_image_only.py"
OUT="$SCRATCH/image_only_shear/out"
LOGS="$SCRATCH/image_only_shear/logs/$SLURM_ARRAY_JOB_ID"
mkdir -p "$OUT" "$LOGS"
N_WORKERS=36
N_NODES=$SLURM_ARRAY_TASK_COUNT
NODE=$SLURM_ARRAY_TASK_ID
ORDER=$($KLPIPE_PYTHON "$SCRIPT" order --arm "$ARM" --min-snr "${MIN_SNR:-0}" --max-snr "${MAX_SNR:-inf}" ${PAS:+--pas "${PAS//:/,}"} ${MAX_REP:+--max-rep "$MAX_REP"} | grep -E "^[0-9,]+$" | tail -1 || true)
[[ "$ORDER" =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo "task order not an index list: '$ORDER'" >&2; exit 1; }
IFS=, read -ra ALL <<< "$ORDER"
echo "arm $ARM node $NODE/$N_NODES: ${#ALL[@]} tasks (min SNR ${MIN_SNR:-0}) over $N_NODES x $N_WORKERS workers, out $OUT"

for w in $(seq 0 $((N_WORKERS - 1))); do
  mine=()
  for ((k = NODE * N_WORKERS + w; k < ${#ALL[@]}; k += N_NODES * N_WORKERS)); do mine+=("${ALL[k]}"); done
  [[ ${#mine[@]} -gt 0 ]] || continue
  idx=$(IFS=,; echo "${mine[*]}")
  taskset -c $((4 * w))-$((4 * w + 3)) \
    $KLPIPE_PYTHON "$SCRIPT" run --arm "$ARM" --tasks "$idx" --out "$OUT" \
    > "$LOGS/n${NODE}_w$w.out" 2> "$LOGS/n${NODE}_w$w.err" &
done
wait
echo "done: $(ls "$OUT/$ARM"/*.json 2>/dev/null | wc -l) results, $(ls "$OUT/$ARM"/*.FAILED 2>/dev/null | wc -l) failed"
