"""Float32 robustness of the analytic grism arithmetic.

The pipeline runs in float64 by default and in float32 under ``KLPIPE_FP32``.
Precision is process-global, so these tests do not flip the mode; they feed
float32 arrays through the pure functions (JAX keeps input dtypes under x64)
and compare against the same computation in float64. Budgets are 10x the
float32 error measured on the production-shaped inputs below on 2026-09-03
(rounded up to one significant figure); the previous formulations fail them
by 1-3 orders of magnitude, so a regression to naive differencing of large
numbers is caught here rather than in a slice tier under KLPIPE_FP32.
"""

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from kl_pipe.constants import C_KMS
from kl_pipe.dispersion import (
    _normal_cdf_antiderivative,
    disperse_line_analytic,
    gaussian_tent_profile,
    line_dispersion_offsets,
)

# jitted once per dtype so the module stays at ~1 s of wall time
_disperse = jax.jit(disperse_line_analytic, static_argnums=3)
_tent = jax.jit(gaussian_tent_profile)
_naive_psi = jax.jit(_normal_cdf_antiderivative)

# a third of the production grism roll (32x32 at oversample 3, halfwidth 31)
# so the module runs in ~1 s; the errors are per element, so they carry over.
# sigma_s spans the 150 km/s prior ceiling at z ~ 1.9.
ROWS, COLS, HALFWIDTH = 32, 32, 15

# float32 error of the shipped forms measured on these inputs, then x10 and
# rounded up: tent profile 1.0e-6 of peak (naive Psi differencing 1.4e-5);
# line image 4.1e-7 of peak; xi 3.1e-7 fine px (absolute-wavelength form
# 3.6e-4); sigma_s 1.6e-7 relative
PROFILE_BUDGET = 1e-5
IMAGE_BUDGET = 5e-6
XI_BUDGET = 4e-6
SIGMA_S_BUDGET = 2e-6


@pytest.fixture(scope='module')
def spaxel_fields():
    rng = np.random.default_rng(0)
    xi = rng.uniform(-6.0, 6.0, (ROWS, COLS))
    sigma_s = rng.uniform(1.0, 2.6, (ROWS, COLS))
    amp = rng.uniform(0.0, 1.0, (ROWS, COLS))
    return xi, sigma_s, amp


def _as(x, dtype):
    return jnp.asarray(np.asarray(x), dtype=dtype)


def test_float32_inputs_stay_float32():
    # the comparison below is only meaningful if float32 inputs are not
    # silently promoted; x64 mode keeps input dtypes
    out = _disperse(
        jnp.ones((4, 8), jnp.float32),
        jnp.zeros((4, 8), jnp.float32),
        jnp.ones((4, 8), jnp.float32),
        2,
    )
    assert out.dtype == jnp.float32


def test_tent_profile_float32_precision(spaxel_fields):
    xi, sigma_s, _ = spaxel_fields
    taps = np.arange(-HALFWIDTH, HALFWIDTH + 1, dtype=float)
    u = taps[None, None, :] - xi[..., None]
    ref = np.asarray(
        gaussian_tent_profile(_as(u, jnp.float64), _as(sigma_s, jnp.float64)[..., None])
    )
    got = np.asarray(
        gaussian_tent_profile(_as(u, jnp.float32), _as(sigma_s, jnp.float32)[..., None])
    )
    peak = np.abs(ref).max()
    stable_err = np.abs(got - ref).max() / peak
    assert stable_err < PROFILE_BUDGET
    # the naive second difference of Psi loses digits at large |z|
    z = _as(u, jnp.float32) / _as(sigma_s, jnp.float32)[..., None]
    inv = 1.0 / _as(sigma_s, jnp.float32)[..., None]
    naive = _as(sigma_s, jnp.float32)[..., None] * (
        _naive_psi(z + inv) - 2.0 * _naive_psi(z) + _naive_psi(z - inv)
    )
    naive_err = np.abs(np.asarray(naive) - ref).max() / peak
    assert naive_err > 5 * stable_err


def test_line_image_float32_precision(spaxel_fields):
    xi, sigma_s, amp = spaxel_fields
    args64 = [_as(a, jnp.float64) for a in (amp, xi, sigma_s)]
    args32 = [_as(a, jnp.float32) for a in (amp, xi, sigma_s)]
    ref = np.asarray(_disperse(*args64, HALFWIDTH))
    got = np.asarray(_disperse(*args32, HALFWIDTH))
    assert np.abs(got - ref).max() / np.abs(ref).max() < IMAGE_BUDGET


def test_dispersion_offsets_float32_precision():
    rng = np.random.default_rng(1)
    v_los = rng.uniform(-220.0, 220.0, (ROWS, COLS)) + 10.0
    lam_rest, z, dispersion, oversample, sigma_kms = 656.28, 1.0, 1.1, 3, 50.0
    lam_sys = lam_rest * (1.0 + z)
    lam_ref = lam_sys  # production: reference at the systemic line centre

    def offsets(dtype):
        xi, sigma_s = line_dispersion_offsets(
            jnp.asarray(lam_sys, dtype),
            lam_ref,
            _as(v_los, dtype),
            jnp.asarray(sigma_kms, dtype),
            dispersion,
            oversample,
        )
        return np.asarray(xi), np.asarray(sigma_s)

    xi64, s64 = offsets(jnp.float64)
    xi32, s32 = offsets(jnp.float32)
    assert np.abs(xi32 - xi64).max() < XI_BUDGET
    assert np.abs(s32 - s64).max() / np.abs(s64).max() < SIGMA_S_BUDGET
    # the absolute-wavelength form rounds ~1300 nm to 1.2e-4 nm before differencing
    lam_obs32 = jnp.asarray(lam_sys, jnp.float32) * (
        1.0 + _as(v_los, jnp.float32) / C_KMS
    )
    xi_abs32 = np.asarray(
        (lam_obs32 - jnp.asarray(lam_ref, jnp.float32)) / dispersion * oversample
    )
    assert np.abs(xi_abs32 - xi64).max() > XI_BUDGET


def test_gradient_matches_finite_difference(spaxel_fields):
    """Autodiff of the line dispersal agrees with central differences (float64)."""
    xi, sigma_s, amp = spaxel_fields
    args = [_as(a, jnp.float64) for a in (amp, xi, sigma_s)]
    rng = np.random.default_rng(2)
    w = jnp.asarray(rng.normal(size=(ROWS, COLS)))
    direction = [jnp.asarray(rng.normal(size=(ROWS, COLS))) for _ in args]

    @jax.jit
    def loss(a, x, s):
        return jnp.sum(w * disperse_line_analytic(a, x, s, HALFWIDTH))

    grads = jax.jit(jax.grad(loss, argnums=(0, 1, 2)))(*args)
    directional = sum(float(jnp.sum(g * d)) for g, d in zip(grads, direction))
    eps = 1e-6
    plus = float(loss(*[a + eps * d for a, d in zip(args, direction)]))
    minus = float(loss(*[a - eps * d for a, d in zip(args, direction)]))
    fd = (plus - minus) / (2 * eps)
    # central differences at eps=1e-6 on an O(1e2) loss: ~1e-7 relative truncation + round-off
    assert abs(directional - fd) / abs(fd) < 1e-6
