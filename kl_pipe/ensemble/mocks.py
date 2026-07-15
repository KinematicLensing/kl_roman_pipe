"""
Per-fit on-the-fly mock observation construction.

The mock is a deterministic function of (truth, observing config, spec SNR
knobs, noise_seed): the model renders its own truth datavector (guaranteeing
fit == truth self-consistency) and Gaussian noise is added at the
matched-filter variance ``||I||^2 / SNR^2``. Broadband channels use the whole
image (``noise.add_intensity_noise`` convention); grism channels normalize on
the emission LINE only (``noise.grism_line_noise``: ``var =
||I_line||^2 / line_snr^2``), so the labeled ``line_snr`` is the line SNR, not
the continuum-dominated whole-stamp SNR. Construction mirrors the flagship
builders (tests/test_flagship.py, experiments/sweverett/vista_kit/tasks_vista.py).

Per-channel noise seeds are derived from the manifest row's ``noise_seed``
via a SeedSequence spawn, so a single integer in the manifest reproduces the
whole multi-channel realization.

PSFs are galsim.Gaussian in v1 (the complex WFI grism PSF is a known pending
gap). On hosts without galsim (e.g. Vista NGC containers), the vista_kit
GaussianPSFShim mechanism applies; this module raises a clear ImportError
rather than silently substituting.
"""

from __future__ import annotations

from typing import Dict, NamedTuple, TYPE_CHECKING

import numpy as np
import jax.numpy as jnp

from kl_pipe.dispersion import GrismPars, build_grism_pars_for_line
from kl_pipe.ensemble.scene import build_source_model, scene_priors
from kl_pipe.lines import LINE_LAMBDAS
from kl_pipe.noise import grism_line_noise
from kl_pipe.observation import build_grism_obs, build_image_obs
from kl_pipe.parameters import ImagePars
from kl_pipe.render import RenderConfig

if TYPE_CHECKING:
    from kl_pipe.ensemble.spec import EnsembleSpec, ObservingConfig
    from kl_pipe.priors import PriorDict
    from kl_pipe.source import SourceModel


class FitInputs(NamedTuple):
    """Everything the sampler needs for one fit."""

    source: 'SourceModel'
    priors: 'PriorDict'
    image_obs: Dict[str, object]
    grism_obs: Dict[str, object]
    truth: Dict[str, float]


def _build_gaussian_psf(fwhm_arcsec: float):
    try:
        import galsim
    except ImportError as err:
        raise ImportError(
            "galsim is required to build the mock PSF. On hosts without "
            "galsim (e.g. Vista NGC containers) install the vista_kit galsim "
            "stub + GaussianPSFShim before importing kl_pipe (see "
            "experiments/sweverett/vista_kit/psf_numpy.py)."
        ) from err
    return galsim.Gaussian(fwhm=fwhm_arcsec)


def _channel_seeds(noise_seed: int, n_channels: int) -> np.ndarray:
    ss = np.random.SeedSequence(int(noise_seed))
    return ss.generate_state(n_channels, dtype=np.uint32)


def _noisy(data_true: np.ndarray, snr: float, seed: int):
    """Matched-filter-variance Gaussian noise: var = ||I||^2 / SNR^2."""
    var = float(np.sum(data_true**2)) / snr**2
    if var <= 0:
        raise ValueError("mock datavector has zero power; cannot set noise variance")
    rng = np.random.default_rng(int(seed))
    noisy = data_true + rng.normal(0.0, np.sqrt(var), size=data_true.shape)
    return noisy, var


def _wcs_with_pc(shape, pixel_scale: float, rotation_radians: float):
    from astropy.wcs import WCS

    n_row, n_col = shape
    c = float(np.cos(rotation_radians))
    s = float(np.sin(rotation_radians))
    wcs = WCS(naxis=2)
    wcs.wcs.pc = np.array([[c, -s], [s, c]])
    wcs.wcs.cdelt = np.array([pixel_scale, pixel_scale])
    wcs.wcs.crpix = np.array([n_col / 2, n_row / 2])
    wcs.wcs.crval = np.array([0.0, 0.0])
    wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    wcs.wcs.cunit = ['arcsec', 'arcsec']
    wcs.pixel_shape = (n_col, n_row)
    wcs.wcs.set()
    return wcs


def _make_band_obs(source, truth, psf, image_pars, band, snr, seed, oversample):
    int_model = source.broadband_models[band]
    obs_clean = build_image_obs(
        image_pars,
        psf=psf,
        render_config=RenderConfig(oversample=oversample),
        int_model=int_model,
        broadband_key=band,
    )
    data_true = np.asarray(source.render_broadband(truth, obs_clean, band))
    data_noisy, var = _noisy(data_true, snr, seed)
    return build_image_obs(
        image_pars,
        psf=psf,
        render_config=RenderConfig(oversample=oversample),
        data=jnp.asarray(data_noisy),
        variance=var,
        int_model=int_model,
        broadband_key=band,
    )


def _grism_pars_for_roll(
    config: 'ObservingConfig', z: float, roll_deg: float, single_roll: bool
) -> GrismPars:
    shape = (config.stamp_grism_pix, config.stamp_grism_pix)
    if single_roll and roll_deg == 0.0:
        # flagship-Q path: plain pixel-scale ImagePars, no WCS
        image_pars = ImagePars(
            shape=shape, pixel_scale=config.pixel_scale_arcsec, indexing='ij'
        )
        return build_grism_pars_for_line(
            LINE_LAMBDAS['Halpha'],
            redshift=z,
            image_pars=image_pars,
            dispersion=config.grism_dispersion_nm_per_pix,
        )
    # multi-roll path: roll encoded as a rotated WCS (production convention)
    wcs = _wcs_with_pc(shape, config.pixel_scale_arcsec, np.deg2rad(roll_deg))
    image_pars = ImagePars(shape=shape, wcs=wcs, indexing='ij')
    return GrismPars(
        image_pars=image_pars,
        dispersion=config.grism_dispersion_nm_per_pix,
        lambda_ref=LINE_LAMBDAS['Halpha'] * (1.0 + z),
        dispersion_angle_detector=0.0,
    )


def _make_roll_obs(
    source, truth, psf, config, z, roll_deg, single_roll, line_snr, seed, oversample
):
    grism_pars = _grism_pars_for_roll(config, z, roll_deg, single_roll)
    # render truth at the SAME oversample as the fit obs below (mirrors
    # _make_band_obs); otherwise the grism truth datavector is generated at the
    # RenderConfig default oversample=1 while the fit runs at config.oversample,
    # silently breaking grism fit==truth self-consistency
    obs_clean = build_grism_obs(
        grism_pars,
        z=z,
        psf=psf,
        render_config=RenderConfig(oversample=oversample),
    )
    data_true = np.asarray(source.render_grism(truth, obs_clean))
    # line-only render (continuum amplitudes zeroed) sets the noise level so the
    # labeled SNR is the emission-LINE matched-filter SNR, not the whole
    # (continuum-dominated) dispersed stamp; the continuum is still rendered
    # into the data below and marginalized in the fit
    line_truth = {
        k: (0.0 if k.endswith('.cont.flux_per_nm') else v) for k, v in truth.items()
    }
    line_true = np.asarray(source.render_grism(line_truth, obs_clean))
    data_noisy, var = grism_line_noise(data_true, line_true, line_snr, seed)
    return build_grism_obs(
        grism_pars,
        z=z,
        psf=psf,
        render_config=RenderConfig(oversample=oversample),
        data=jnp.asarray(data_noisy),
        variance=var,
    )


def build_fit_inputs(
    truth: Dict[str, float],
    noise_seed: int,
    spec: 'EnsembleSpec',
    config: 'ObservingConfig',
    *,
    broadband_snr: float,
    line_snr: float,
) -> FitInputs:
    """
    Build the per-fit source model, priors, and noisy mock observations.

    Parameters
    ----------
    truth : dict
        Fully-resolved dotted truth from the manifest row.
    noise_seed : int
        The row's noise seed; per-channel seeds derive from it.
    spec : EnsembleSpec
        Population distributions (for the fit priors).
    config : ObservingConfig
        Structural instrument setup.
    broadband_snr, line_snr : float
        Per-fit SNR values from the manifest row (the manifest, not the
        spec, is the source of truth -- line_snr varies per row on a
        config-sweep axis). ``line_snr`` is the emission-line matched-filter
        SNR (see kl_pipe.noise.grism_line_noise).

    Returns
    -------
    FitInputs
        (source, priors, image_obs, grism_obs, truth).
    """
    if broadband_snr <= 0 or line_snr <= 0:
        raise ValueError(
            f"SNR values must be positive, got ({broadband_snr}, {line_snr})"
        )
    source = build_source_model(config)
    priors = scene_priors(truth, config, spec)

    sampled = set(priors.sampled_names) | set(priors.fixed_names)
    missing = sampled - set(truth)
    if missing:
        raise ValueError(
            f"truth is missing parameters the fit declares: {sorted(missing)}"
        )

    z = truth['z']
    seeds = _channel_seeds(noise_seed, len(config.bands) + len(config.grism_rolls_deg))

    image_pars = ImagePars(
        shape=(config.stamp_broadband_pix, config.stamp_broadband_pix),
        pixel_scale=config.pixel_scale_arcsec,
        indexing='ij',
    )
    image_obs = {}
    for i, band in enumerate(config.bands):
        psf = _build_gaussian_psf(config.band_psf_fwhm[band])
        image_obs[band] = _make_band_obs(
            source,
            truth,
            psf,
            image_pars,
            band,
            broadband_snr,
            seeds[i],
            config.oversample,
        )

    grism_psf = _build_gaussian_psf(config.grism_psf_fwhm)
    single_roll = len(config.grism_rolls_deg) == 1
    grism_obs = {}
    for j, roll in enumerate(config.grism_rolls_deg):
        grism_obs[f'roll{j}'] = _make_roll_obs(
            source,
            truth,
            grism_psf,
            config,
            z,
            roll,
            single_roll,
            line_snr,
            seeds[len(config.bands) + j],
            config.oversample,
        )

    return FitInputs(
        source=source,
        priors=priors,
        image_obs=image_obs,
        grism_obs=grism_obs,
        truth=truth,
    )
