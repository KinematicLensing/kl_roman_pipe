#!/bin/bash
#-----------------------------------------------------------------------------
# One-shot: build the CPU venv for the bench kit on Stampede3 (x86_64).
#
# No container needed (unlike Vista/aarch64): official x86 CPU jax wheels
# install directly. galsim intentionally absent -- the kit ships an exact
# analytic PSF shim (psf_numpy.py, parity-checked vs galsim).
#
# Run ONCE on a Stampede3 login node (pip is fine there; only heavy compute
# is banned):
#   bash provision_stampede3.sh            # venv at $WORK/klpipe_cpu_venv
#   bash provision_stampede3.sh /path/venv # custom location
#
# Idempotent: re-running upgrades in place.
#-----------------------------------------------------------------------------
set -euo pipefail

VENV="${1:-$WORK/klpipe_cpu_venv}"

# TACC python module: need >= 3.10 for current jax. Fail loudly, not softly.
set +u; module load python3 2>/dev/null || true; set -u
PYBIN=$(command -v python3)
PYOK=$($PYBIN -c 'import sys; print(int(sys.version_info >= (3, 10)))')
if [[ "$PYOK" != "1" ]]; then
    echo "ERROR: python3 at $PYBIN is $($PYBIN -V); jax needs >= 3.10." >&2
    echo "Try 'module spider python3' for a newer module, or install" >&2
    echo "miniforge on \$WORK and re-run with it first on PATH." >&2
    exit 1
fi

echo "=== venv at $VENV (python: $PYBIN, $($PYBIN -V)) ==="
$PYBIN -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install --upgrade pip

# CPU jax + the kl_pipe inference deps the kit imports. Mirrors the Vista
# pip sidecar (SETUP.md sec 3) minus the container-provided stack.
pip install --no-cache-dir jax numpyro scipy astropy pyyaml matplotlib

echo "=== sanity ==="
# taskset -c 0: the XLA CPU client sizes its thread pool from the affinity
# mask; unpinned on a 112-core login node it wants 100+ threads and login
# ulimits kill it (pthread_create EAGAIN -> SIGABRT, observed 2026-07-12)
PIN=""
command -v taskset >/dev/null 2>&1 && PIN="taskset -c 0"
if $PIN python - <<'EOF'
import jax, numpyro, astropy, yaml, matplotlib, scipy
print('jax', jax.__version__, jax.devices())
assert jax.default_backend() == 'cpu'
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
assert jnp.zeros(1).dtype == jnp.float64, 'x64 not working'
print('venv sanity OK')
EOF
then :; else
    echo "WARNING: sanity check crashed. If the trace shows pthread_create" >&2
    echo "failure, that is the login-node thread limit, NOT a broken venv;" >&2
    echo "re-run the check on a compute node (idev). Install itself succeeded." >&2
fi

echo "=== done; activate with: source $VENV/bin/activate ==="
