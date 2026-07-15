# env_vista.sh -- source this on Vista to define the containerized launcher:
#
#   source $STOCKYARD/repos/kl_roman_pipe/experiments/sweverett/vista_kit/env_vista.sh
#
# Defines KLPIPE_PYTHON, used two ways:
#   - directly: $KLPIPE_PYTHON -m kl_pipe.ensemble <cmd> ...  (compute nodes only)
#   - by 'kl_pipe.ensemble slurm', which bakes it into the emitted submit.slurm
# Sourcing on a login node is fine (defining the variable is harmless);
# EXECUTING $KLPIPE_PYTHON only works inside idev or a batch job.
# Paths match provision_vista.sh; fftw comes via the sidecar-embedded
# libfftw3.so.3 LD_PRELOAD (never LD_LIBRARY_PATH).

module load tacc-apptainer 2>/dev/null || true

export KLPIPE_PYTHON="apptainer exec --nv \
  -B $STOCKYARD/repos/kl_roman_pipe -B $WORK/klpipe_pipdeps -B $SCRATCH \
  --env PYTHONPATH=$STOCKYARD/repos/kl_roman_pipe:$WORK/klpipe_pipdeps \
  --env LD_PRELOAD=$WORK/klpipe_pipdeps/galsim/libfftw3.so.3 \
  --env JAX_COMPILATION_CACHE_DIR=$SCRATCH/jax_cache \
  $WORK/containers/jax_26.06-py3.sif python"

echo "[env_vista] KLPIPE_PYTHON set (jax_26.06-py3.sif; exec on compute nodes only)"
