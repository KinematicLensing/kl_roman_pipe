#!/bin/bash
# run_arm_census.sh -- GH200 sampler-cost census over the three paired
# flagship2_shear_dev arms (frozen300 / adapt400 / adapt800).
#
# Run INSIDE an idev session (compute node), from anywhere:
#
#   idev -p gh-dev -N 1 -n 1 -t 02:00:00
#   bash $STOCKYARD/repos/kl_roman_pipe/scripts/vista_arm_census.sh
#
# Optional: MAX_FITS=4 bash .../run_arm_census.sh   (default 2; bump if the
# queue window allows -- adapt800 may cost 30-40 min/fit on first compile).
#
# Results land in $SCRATCH/kl_runs/; a tarball with the collated tables and
# run logs is written to $HOME/kl_arm_census.tgz for handing back.

set -euo pipefail

REPO=$STOCKYARD/repos/kl_roman_pipe
MAX_FITS=${MAX_FITS:-2}
RUNS=$SCRATCH/kl_runs

cd $REPO
bash experiments/sweverett/vista_kit/provision_vista.sh
source experiments/sweverett/vista_kit/env_vista.sh

for ARM in frozen300 adapt400 adapt800; do
  echo "=== arm $ARM (max_fits=$MAX_FITS) ==="
  $KLPIPE_PYTHON -m kl_pipe.ensemble expand \
      configs/ensembles/flagship2_shear_dev_${ARM}.yaml --runs-dir $RUNS
  ( time $KLPIPE_PYTHON -m kl_pipe.ensemble run \
      --run-dir $RUNS/flagship2_shear_dev_${ARM} --max-fits $MAX_FITS ) \
      2>&1 | tee $RUNS/${ARM}_run.log
  $KLPIPE_PYTHON -m kl_pipe.ensemble collate \
      --run-dir $RUNS/flagship2_shear_dev_${ARM}
  $KLPIPE_PYTHON -m kl_pipe.ensemble status \
      --run-dir $RUNS/flagship2_shear_dev_${ARM} 2>&1 | tee $RUNS/${ARM}_status.log
done

tar czf $HOME/kl_arm_census.tgz \
    $RUNS/flagship2_shear_dev_*/*_collated.parquet \
    $RUNS/*_run.log $RUNS/*_status.log
echo "census complete -> $HOME/kl_arm_census.tgz"
