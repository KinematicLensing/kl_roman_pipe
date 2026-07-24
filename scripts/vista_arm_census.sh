#!/bin/bash
# vista_arm_census.sh -- GH200 sampler-cost census over the paired
# flagship2_shear_dev arms (frozen300 / adapt400 / adapt800).
#
# Run INSIDE an idev session (compute node), from anywhere:
#
#   idev -p gh-dev -N 1 -n 1 -t 02:00:00
#   bash $STOCKYARD/repos/kl_roman_pipe/scripts/vista_arm_census.sh
#
# Knobs (env vars):
#   MAX_FITS=4                          fits per arm (default 2)
#   ARMS="frozen300 adapt400"           subset of arms (default all three);
#                                       useful for short queue windows --
#                                       adapt800 may cost 30-40 min/fit
#
# Results land in $SCRATCH/kl_runs/; a tarball with the collated tables and
# run logs is written to $HOME/kl_arm_census.tgz for handing back. Re-running
# after a partial pass is fine; the tarball is rebuilt from completed arms.

# no set -e/-u: TACC lmod + the provision/env scripts are not clean under
# them and were killing the arm loop silently; outcomes checked explicitly

REPO=$STOCKYARD/repos/kl_roman_pipe
MAX_FITS=${MAX_FITS:-2}
ARMS=${ARMS:-"frozen300 adapt400 adapt800"}
RUNS=$SCRATCH/kl_runs

cd $REPO || { echo "[census] ERROR: cannot cd $REPO"; exit 1; }
bash experiments/sweverett/vista_kit/provision_vista.sh
source experiments/sweverett/vista_kit/env_vista.sh
if [ -z "${KLPIPE_PYTHON:-}" ]; then
  echo "[census] ERROR: KLPIPE_PYTHON not set after sourcing env_vista.sh"
  exit 1
fi
echo "[census] arms: $ARMS | max_fits: $MAX_FITS | runs dir: $RUNS"

for ARM in $ARMS; do
  echo "=== arm $ARM (max_fits=$MAX_FITS) ==="
  $KLPIPE_PYTHON -m kl_pipe.ensemble expand \
      configs/ensembles/flagship2_shear_dev_${ARM}.yaml --runs-dir $RUNS \
      || { echo "[census] expand failed for $ARM"; exit 1; }
  ( time $KLPIPE_PYTHON -m kl_pipe.ensemble run \
      --run-dir $RUNS/flagship2_shear_dev_${ARM} --max-fits $MAX_FITS ) \
      2>&1 | tee $RUNS/${ARM}_run.log
  $KLPIPE_PYTHON -m kl_pipe.ensemble collate \
      --run-dir $RUNS/flagship2_shear_dev_${ARM} \
      || { echo "[census] collate failed for $ARM"; exit 1; }
  $KLPIPE_PYTHON -m kl_pipe.ensemble status \
      --run-dir $RUNS/flagship2_shear_dev_${ARM} 2>&1 | tee $RUNS/${ARM}_status.log
done

tar czf $HOME/kl_arm_census.tgz \
    $RUNS/flagship2_shear_dev_*/*_collated.parquet \
    $RUNS/*_run.log $RUNS/*_status.log 2>/dev/null
echo "[census] complete -> $HOME/kl_arm_census.tgz"
