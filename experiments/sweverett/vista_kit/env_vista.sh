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

# Optional release JAX stack: KLPIPE_JAX_RELEASE=<version> (e.g. 0.11.1) puts
# $WORK/klpipe_jax_release_<version> (pip --target of jax[cuda13]==<version>,
# installed inside the container on a compute node) ahead of the container's
# own JAX and hides /opt/jax + /opt/jaxlibs behind empty binds. Required for
# any float32 work: the container nightly cannot create single-precision
# cuFFT plans. Unset = container JAX (float64 only).

module load tacc-apptainer 2>/dev/null || true

if [ -n "${KLPIPE_JAX_RELEASE:-}" ]; then
  _JAXREL="$WORK/klpipe_jax_release_${KLPIPE_JAX_RELEASE}"
  [ -d "$_JAXREL/jax" ] || echo "[env_vista] WARNING: $_JAXREL missing; see SETUP.md release-stack section"
  _EMPTY="$SCRATCH/empty_dir"; mkdir -p "$_EMPTY"
  _JAX_BINDS="-B $WORK -B $_EMPTY:/opt/jax -B $_EMPTY:/opt/jaxlibs"
  _PYPATH="$STOCKYARD/repos/kl_roman_pipe:$_JAXREL:$WORK/klpipe_pipdeps"
  _CACHE="$SCRATCH/jax_cache_rel_${KLPIPE_JAX_RELEASE}"
  _STACK="release jax ${KLPIPE_JAX_RELEASE}"
else
  _JAX_BINDS=""
  _PYPATH="$STOCKYARD/repos/kl_roman_pipe:$WORK/klpipe_pipdeps"
  _CACHE="$SCRATCH/jax_cache"
  _STACK="container jax"
fi

export KLPIPE_PYTHON="apptainer exec --nv \
  -B $STOCKYARD/repos/kl_roman_pipe -B $WORK/klpipe_pipdeps -B $SCRATCH $_JAX_BINDS \
  --env PYTHONPATH=$_PYPATH \
  --env LD_PRELOAD=$WORK/klpipe_pipdeps/galsim/libfftw3.so.3 \
  --env JAX_COMPILATION_CACHE_DIR=$_CACHE \
  $WORK/containers/jax_26.06-py3.sif python"

echo "[env_vista] KLPIPE_PYTHON set (jax_26.06-py3.sif, $_STACK; exec on compute nodes only)"
