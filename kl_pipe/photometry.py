"""
Photometric unit conversions and helpers.

Pure numpy functions and unit constants shared by catalog adapters, the
ensemble population machinery, and anyone assigning physical fluxes to a
data vector (simulations, tutorials, priors). The model and likelihood
layers are unit-agnostic; these helpers convert published quantities
(AB magnitudes, flux densities, depths) into the units a given data
vector carries. Survey-specific numbers (depths, flux limits, band
wavelengths) live in ``kl_pipe.surveys``.
"""

from __future__ import annotations

from typing import Union

import numpy as np

ArrayLike = Union[float, np.ndarray]

# AB magnitude <-> microjansky: m_AB = 23.9 - 2.5 log10(f_nu / uJy)
AB_MAG_UJY_PIVOT = 23.9

# f_nu[erg/cm2/s/Hz] = 1e-29 * uJy, and its inverse
UJY_TO_CGS = 1e-29
CGS_FNU_TO_UJY = 1e29

# integrated line flux unit for scene parameters: Halpha.flux is carried in
# 1e-17 erg/s/cm2 so catalog truths land O(10-1000) (the selected-sample
# median is ~1e-15 cgs); the continuum flux_per_nm inherits the same unit
# per nm through the EW division
CGS_TO_F17 = 1e17

# speed of light [Angstrom / s]; converts f_nu [erg/cm2/s/Hz] to
# f_lambda = f_nu * c / lambda^2 [erg/cm2/s/A]
C_A_PER_S = 2.998e18

# Halpha rest wavelength [Angstrom] (air; standard line-list value).
# kl_pipe/lines.py carries the same air value as 656.28 nm.
HALPHA_REST_A = 6562.8

# exponential-disk half-light-to-scale-length ratio, r50 = 1.678 * rscale
EXP_R50_OVER_RSCALE = 1.678


def ab_mag_to_ujy(mag_ab: ArrayLike) -> ArrayLike:
    """Convert an AB magnitude to a flux density in microjansky.

    f_nu [uJy] = 10 ** ((23.9 - m_AB) / 2.5).

    Parameters
    ----------
    mag_ab : float or np.ndarray
        AB magnitude(s).

    Returns
    -------
    float or np.ndarray
        Flux density in uJy.
    """
    return 10.0 ** ((AB_MAG_UJY_PIVOT - np.asarray(mag_ab, dtype=np.float64)) / 2.5)


def ujy_to_ab_mag(f_ujy: ArrayLike) -> ArrayLike:
    """Convert a flux density in microjansky to an AB magnitude.

    m_AB = 23.9 - 2.5 log10(f_nu / uJy). Raises on non-positive flux
    (a magnitude is undefined there).

    Parameters
    ----------
    f_ujy : float or np.ndarray
        Flux density in uJy; must be strictly positive.

    Returns
    -------
    float or np.ndarray
        AB magnitude(s).
    """
    f_ujy = np.asarray(f_ujy, dtype=np.float64)
    if np.any(f_ujy <= 0):
        raise ValueError("ujy_to_ab_mag requires strictly positive flux")
    return AB_MAG_UJY_PIVOT - 2.5 * np.log10(f_ujy)


def fnu_to_flambda(f_nu_cgs: ArrayLike, lambda_a: ArrayLike) -> ArrayLike:
    """Convert a frequency flux density to a wavelength flux density.

    f_lambda = f_nu * c / lambda^2:
    [erg/cm2/s/Hz] * [A/s] / [A^2] = [erg/cm2/s/A].

    Parameters
    ----------
    f_nu_cgs : float or np.ndarray
        Flux density in erg/cm2/s/Hz.
    lambda_a : float or np.ndarray
        Wavelength in Angstrom.

    Returns
    -------
    float or np.ndarray
        Flux density in erg/cm2/s/A.
    """
    lambda_a = np.asarray(lambda_a, dtype=np.float64)
    return np.asarray(f_nu_cgs, dtype=np.float64) * C_A_PER_S / lambda_a**2


def powerlaw_fnu(
    f_blue: np.ndarray,
    lam_blue: float,
    f_red: np.ndarray,
    lam_red: float,
    lam: np.ndarray,
) -> np.ndarray:
    """f_nu at lam from a power law through two pivot measurements.

    Interpolates (or mildly extrapolates) a local power-law SED between two
    measured pivots: log-log interpolation, the same local power-law
    continuity assumption as the continuum-at-lambda_obs gap interpolation.
    """
    if np.any(f_blue <= 0) or np.any(f_red <= 0):
        raise ValueError("power-law flux interpolation requires positive photometry")
    alpha = np.log(f_red / f_blue) / np.log(lam_red / lam_blue)
    return f_blue * (np.asarray(lam, dtype=np.float64) / lam_blue) ** alpha
