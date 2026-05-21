"""Regression test for the slitless-grism spectral resolution.

Renders a near-delta-function source (tiny intensity scale + zero rotation +
small vel_dispersion regularizer) through KLModel.render_grism and asserts
that R_measured = lambda_obs / FWHM_nm agrees with the Roman spec
R = 461 * lambda_um within +/-5%.

Post-LSF-refactor, the slitless spectral resolution is produced entirely by
the PSF-per-slice + dispersion geometry; no separate sigma_inst term. This
test guards against accidental regressions of that geometry (e.g. a stray
broadening term, a wrong dispersion solution, or a PSF FWHM drift). See
docs/plans/phase2_lsf_refactor.md and experiments/sweverett/lsf_gate_test/
for the empirical study that drove the refactor.
"""

from __future__ import annotations

import os

import jax

jax.config.update('jax_enable_x64', True)

import numpy as np
import pytest
import galsim
from scipy.optimize import curve_fit

from kl_pipe.parameters import ImagePars
from kl_pipe.velocity import CenteredVelocityModel
from kl_pipe.intensity import InclinedExponentialModel
from kl_pipe.model import KLModel
from kl_pipe.spectral import SpectralConfig, SpectralModel, halpha_line, HALPHA
from kl_pipe.dispersion import build_grism_pars_for_line
from kl_pipe.observation import build_grism_obs


# Roman grism resolving power at lambda (slitless spec): R = 461 * lambda_um.
def _roman_R_spec(lam_nm: float) -> float:
    return 461.0 * lam_nm / 1000.0


def _gaussian_1d(x, amp, mu, sigma, baseline):
    return baseline + amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def _fit_gaussian(profile_1d):
    """Fit a 1D Gaussian; return (amp, mu_pix, sigma_pix, baseline)."""
    x = np.arange(len(profile_1d), dtype=float)
    y = np.asarray(profile_1d, dtype=float)
    baseline_init = float(np.median(y))
    amp_init = float(np.max(y) - baseline_init)
    mu_init = float(np.argmax(y))
    half = baseline_init + 0.5 * amp_init
    above = np.where(y >= half)[0]
    if len(above) >= 2:
        sigma_init = (above[-1] - above[0]) / 2.355
    else:
        sigma_init = 1.0
    sigma_init = max(sigma_init, 0.5)
    popt, _ = curve_fit(
        _gaussian_1d,
        x,
        y,
        p0=[amp_init, mu_init, sigma_init, baseline_init],
        maxfev=10000,
    )
    return popt


def test_psf_dispersion_resolution_matches_roman_spec():
    """R_measured / R_spec must be within +/-5% for a point-source line."""
    image_pars = ImagePars(shape=(32, 96), pixel_scale=0.11, indexing='ij')

    vel_model = CenteredVelocityModel()
    int_model = InclinedExponentialModel()
    spec_config = SpectralConfig(lines=(halpha_line(),), spectral_oversample=5)
    spec_model = SpectralModel(spec_config, int_model, vel_model)
    kl_model = KLModel(
        vel_model,
        int_model,
        shared_pars={'cosi', 'theta_int', 'g1', 'g2'},
        spectral_model=spec_model,
    )

    z = 0.98
    grism_pars = build_grism_pars_for_line(
        HALPHA.lambda_rest,
        redshift=z,
        image_pars=image_pars,
        dispersion=1.1,
    )

    psf = galsim.Gaussian(fwhm=0.18)
    grism_obs = build_grism_obs(grism_pars, z=z, psf=psf, oversample=5)

    pars = {
        # geometry / shared
        'cosi': 1.0,
        'theta_int': 0.0,
        'g1': 0.0,
        'g2': 0.0,
        # velocity
        'v0': 0.0,
        'vcirc': 0.0,
        'vel_rscale': 0.5,
        # intensity (near-delta in space)
        'flux': 100.0,
        'int_rscale': 0.001,
        'int_h_over_r': 0.1,
        'int_x0': 0.0,
        'int_y0': 0.0,
        # spectral
        'z': z,
        'vel_dispersion': 1.0,  # numerical regularizer; sub-dominant
        'Ha_flux': 100.0,
        'Ha_cont': 0.0,
    }
    theta = kl_model.pars2theta(pars)
    dispersed = np.asarray(kl_model.render_grism(theta, grism_obs))

    # extract row slice along dispersion direction (dispersion_angle=0 -> columns)
    row_idx = int(np.argmax(dispersed.sum(axis=1)))
    profile = dispersed[row_idx, :]

    _, _, sigma_pix, _ = _fit_gaussian(profile)
    fwhm_pix = 2.355 * abs(sigma_pix)
    fwhm_nm = fwhm_pix * grism_obs.grism_pars.dispersion
    lam_obs = HALPHA.lambda_rest * (1.0 + z)
    R_measured = lam_obs / fwhm_nm
    R_spec = _roman_R_spec(lam_obs)

    ratio = R_measured / R_spec
    assert 0.95 <= ratio <= 1.05, (
        f"R_measured / R_spec = {ratio:.4f} outside +/-5% tolerance "
        f"(R_measured={R_measured:.2f}, R_spec={R_spec:.2f}). "
        f"FWHM_nm={fwhm_nm:.3f}, lambda_obs={lam_obs:.2f}. "
        f"The PSF+dispersion geometry has drifted from Roman spec; see "
        f"docs/plans/phase2_lsf_refactor.md."
    )
