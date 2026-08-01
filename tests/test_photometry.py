"""
Unit tests for kl_pipe/photometry.py: unit-conversion constants and helpers.

The constants are literal pins (same rationale as the survey pins in
tests/test_surveys_roman.py: everything downstream references them
symbolically, so nothing else fails if one is silently edited).
"""

import numpy as np
import pytest

from kl_pipe.photometry import (
    AB_MAG_UJY_PIVOT,
    C_A_PER_S,
    CGS_FNU_TO_UJY,
    CGS_TO_F17,
    EXP_R50_OVER_RSCALE,
    HALPHA_REST_A,
    UJY_TO_CGS,
    ab_mag_to_ujy,
    fnu_to_flambda,
    powerlaw_fnu,
    ujy_to_ab_mag,
)


class TestConstants:
    def test_literal_pins(self):
        # AB system: m_AB = 23.9 corresponds to exactly 1 uJy (3631 Jy zero
        # point); f_nu[cgs] = 1e-29 * uJy and its inverse; line fluxes are
        # carried in 1e-17 erg/s/cm2
        assert AB_MAG_UJY_PIVOT == 23.9
        assert UJY_TO_CGS == 1e-29
        assert CGS_FNU_TO_UJY == 1e29
        assert UJY_TO_CGS * CGS_FNU_TO_UJY == pytest.approx(1.0, rel=1e-15)
        assert CGS_TO_F17 == 1e17
        # c in Angstrom/s at the 4-digit precision the adapters use
        assert C_A_PER_S == 2.998e18
        # Halpha air rest wavelength; kl_pipe/lines.py carries 656.28 nm
        assert HALPHA_REST_A == 6562.8
        # exponential disk: r50 = 1.678 * scale length
        assert EXP_R50_OVER_RSCALE == 1.678

    def test_lines_registry_agrees_on_halpha(self):
        from kl_pipe.lines import LINE_LAMBDAS

        assert LINE_LAMBDAS['Halpha'] * 10.0 == pytest.approx(HALPHA_REST_A)


class TestAbMagConversions:
    def test_pivot_is_one_ujy(self):
        assert ab_mag_to_ujy(23.9) == pytest.approx(1.0, rel=1e-12)
        assert ujy_to_ab_mag(1.0) == pytest.approx(23.9, rel=1e-12)

    def test_round_trip(self):
        mags = np.linspace(18.0, 28.0, 11)
        np.testing.assert_allclose(ujy_to_ab_mag(ab_mag_to_ujy(mags)), mags, rtol=1e-12)

    def test_five_magnitudes_is_factor_100(self):
        assert ab_mag_to_ujy(18.9) / ab_mag_to_ujy(23.9) == pytest.approx(
            100.0, rel=1e-12
        )

    def test_nonpositive_flux_raises(self):
        with pytest.raises(ValueError, match='positive'):
            ujy_to_ab_mag(0.0)
        with pytest.raises(ValueError, match='positive'):
            ujy_to_ab_mag(np.array([1.0, -0.5]))


class TestFnuToFlambda:
    def test_unit_chain(self):
        # f_lambda = f_nu * c / lambda^2 checked against a hand-computed
        # value: f_nu = 1e-29 (1 uJy) at 10000 A ->
        # 1e-29 * 2.998e18 / 1e8 = 2.998e-19 erg/cm2/s/A
        assert fnu_to_flambda(1e-29, 1.0e4) == pytest.approx(2.998e-19, rel=1e-12)

    def test_vectorized(self):
        f_nu = np.array([1e-29, 2e-29])
        lam = np.array([1.0e4, 2.0e4])
        expected = f_nu * C_A_PER_S / lam**2
        np.testing.assert_allclose(fnu_to_flambda(f_nu, lam), expected, rtol=1e-15)


class TestPowerlawFnu:
    def test_recovers_pure_power_law(self):
        # a power-law SED through the two pivots is reproduced exactly at
        # any wavelength, interpolated or extrapolated
        alpha = -1.3
        lam_b, lam_r = 1.1e4, 1.5e4
        f_b = 5.0
        f_r = f_b * (lam_r / lam_b) ** alpha
        lam = np.array([1.0e4, 1.2e4, 1.5e4, 1.8e4])
        expected = f_b * (lam / lam_b) ** alpha
        got = powerlaw_fnu(np.array([f_b]), lam_b, np.array([f_r]), lam_r, lam)
        np.testing.assert_allclose(got, expected, rtol=1e-12)

    def test_flat_sed_is_flat(self):
        got = powerlaw_fnu(
            np.array([2.0]), 1.1e4, np.array([2.0]), 1.5e4, np.array([1.3e4])
        )
        assert got[0] == pytest.approx(2.0, rel=1e-12)

    def test_nonpositive_photometry_raises(self):
        with pytest.raises(ValueError, match='positive photometry'):
            powerlaw_fnu(
                np.array([0.0]), 1.1e4, np.array([1.0]), 1.5e4, np.array([1.3e4])
            )
        with pytest.raises(ValueError, match='positive photometry'):
            powerlaw_fnu(
                np.array([1.0]), 1.1e4, np.array([-2.0]), 1.5e4, np.array([1.3e4])
            )
