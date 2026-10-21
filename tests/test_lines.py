"""Tests for the line-wavelength registry and air <-> vacuum conversion."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from kl_pipe.lines import LINE_LAMBDAS, air_to_vacuum, vacuum_to_air

# SDSS vacuum line list [A] (classic.sdss.org/dr7/algorithms/linestable)
VACUUM_REF_A = {
    'Hgamma': 4341.68,
    'Hbeta': 4862.68,
    'Halpha': 6564.61,
    'OII3726': 3727.092,
    'OII3728': 3729.875,
    'OIII4959': 4960.295,
    'OIII5007': 5008.240,
    'NII6548': 6549.86,
    'NII6584': 6585.27,
    'SII6717': 6718.29,
    'SII6731': 6732.67,
}

# NIST ASD standard-air wavelengths [A] of the same transitions
AIR_REF_A = {
    'Hgamma': 4340.47,
    'Hbeta': 4861.33,
    'Halpha': 6562.80,
    'OII3726': 3726.03,
    'OII3728': 3728.82,
    'OIII4959': 4958.91,
    'OIII5007': 5006.84,
    'NII6548': 6548.05,
    'NII6584': 6583.45,
    'SII6717': 6716.44,
    'SII6731': 6730.82,
}

# 0.005 nm = 0.05 A: half the 0.01 nm registry precision; SDSS vs Morton-
# converted NIST values agree to <= 0.0011 nm (measured 2026-09-25)
WAVELENGTH_TOL_NM = 0.005

# round-trip is an exact fixed-point inverse; float64 error measured 1.1e-13 nm
ROUND_TRIP_TOL_NM = 1e-6


@pytest.mark.parametrize('name', sorted(VACUUM_REF_A))
def test_registry_matches_vacuum_reference(name):
    assert LINE_LAMBDAS[name] == pytest.approx(
        VACUUM_REF_A[name] / 10.0, abs=WAVELENGTH_TOL_NM
    )


def test_blended_oii_is_doublet_mean():
    mean = 0.5 * (LINE_LAMBDAS['OII3726'] + LINE_LAMBDAS['OII3728'])
    assert LINE_LAMBDAS['OII3727'] == pytest.approx(mean, abs=WAVELENGTH_TOL_NM)


@pytest.mark.parametrize('name', sorted(AIR_REF_A))
def test_air_to_vacuum_matches_reference(name):
    lam_vac = float(air_to_vacuum(AIR_REF_A[name] / 10.0))
    assert lam_vac == pytest.approx(VACUUM_REF_A[name] / 10.0, abs=WAVELENGTH_TOL_NM)


@pytest.mark.parametrize('name', sorted(AIR_REF_A))
def test_registry_is_not_air(name):
    # vacuum - air is 0.10-0.18 nm across 370-680 nm (n - 1 ~ 2.8e-4)
    diff = LINE_LAMBDAS[name] - AIR_REF_A[name] / 10.0
    assert 0.09 < diff < 0.20, f"{name}: vac - air = {diff} nm"


def test_halpha_vacuum_air_offset():
    diff = LINE_LAMBDAS['Halpha'] - 656.28
    assert diff == pytest.approx(0.181, abs=WAVELENGTH_TOL_NM)


def test_round_trip():
    lam = np.linspace(300.0, 1000.0, 7001)
    back = np.asarray(air_to_vacuum(vacuum_to_air(lam)))
    assert np.max(np.abs(back - lam)) < ROUND_TRIP_TOL_NM
    back = np.asarray(vacuum_to_air(air_to_vacuum(lam)))
    assert np.max(np.abs(back - lam)) < ROUND_TRIP_TOL_NM


def test_air_shorter_than_vacuum():
    lam = np.linspace(200.0, 2500.0, 101)
    assert np.all(np.asarray(vacuum_to_air(lam)) < lam)


def test_jit_matches_eager():
    lam = jnp.array([486.268, 656.461, 900.0])
    eager = vacuum_to_air(lam)
    jitted = jax.jit(vacuum_to_air)(lam)
    np.testing.assert_allclose(np.asarray(jitted), np.asarray(eager), rtol=1e-14)
    back = jax.jit(air_to_vacuum)(jitted)
    np.testing.assert_allclose(np.asarray(back), np.asarray(lam), atol=1e-9)


def test_grad_finite():
    g = jax.grad(lambda x: air_to_vacuum(x))(656.28)
    # d lambda_vac / d lambda_air = n + lambda dn/dlambda ~ 1.00028
    assert float(g) == pytest.approx(1.00028, abs=1e-4)


@pytest.mark.parametrize('func', [air_to_vacuum, vacuum_to_air])
@pytest.mark.parametrize('bad', [121.567, np.array([500.0, 150.0]), np.nan, -1.0])
def test_invalid_concrete_input_raises(func, bad):
    with pytest.raises(ValueError):
        func(bad)
