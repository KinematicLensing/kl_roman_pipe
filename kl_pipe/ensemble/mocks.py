"""
Per-fit on-the-fly mock observation construction.

The mock is a deterministic function of (truth, observation config, per-fit
SNR labels, noise_seed): the model renders its own truth datavector
(guaranteeing fit == truth self-consistency) and Gaussian noise is added at
the matched-filter variance ``||I||^2 / SNR^2``. Broadband channels use the
whole image (``noise.add_intensity_noise`` convention) with one SNR label
per band (catalog mode: the galaxy's own published-depth matched-filter SNR;
sampled mode: the spec scalar shared by all bands); grism channels normalize
on the emission LINE only (``noise.grism_line_noise``: ``var =
||I_line||^2 / line_snr^2``), so the labeled ``line_snr`` is the line SNR, not
the continuum-dominated whole-stamp SNR. Construction mirrors the flagship
builders (tests/test_flagship.py, experiments/sweverett/vista_kit/tasks_vista.py).

Per-channel noise seeds are derived from the manifest row's ``noise_seed``
via a SeedSequence spawn, so a single integer in the manifest reproduces the
whole multi-channel realization.

PSFs are built per channel from the observation config's ``PSFSpec``:
``gaussian`` (galsim.Gaussian at the configured FWHM) or ``roman_wfi``
(monochromatic ``galsim.roman.getPSF``; broadband at the band's effective
wavelength, grism at the observed Halpha wavelength). Roman grism kernels
are rendered at a stamp size pinned to the ensemble's largest observed
wavelength so the PSF array shape is constant across z. When the PSFSpec
loosens the fit-kernel ``folding_threshold``, mock/truth data is still
rendered through the tighter ``mock_folding_threshold`` kernels (default:
galsim's 5e-3) -- the mock is always at least as accurate as the fit model. On hosts without
galsim (e.g. Vista NGC containers), the vista_kit GaussianPSFShim mechanism
applies to GAUSSIAN configs only; roman_wfi needs the real galsim (the
pupil-plane and aberration data ship inside the galsim package itself, so
any working install suffices). This module raises a clear ImportError
rather than silently substituting.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, NamedTuple, Optional, Tuple, TYPE_CHECKING

import numpy as np
import jax.numpy as jnp

from kl_pipe.dispersion import GrismPars, build_grism_pars_for_line
from kl_pipe.ensemble.scene import (
    build_source_model,
    scene_priors,
    scene_truth_defaults,
)
from kl_pipe.lines import LINE_LAMBDAS
from kl_pipe.noise import (
    add_map_noise,
    grism_line_noise,
    matched_filter_snr,
    physical_variance_map,
)
from kl_pipe.observation import build_grism_obs, build_image_obs
from kl_pipe.parameters import ImagePars
from kl_pipe.photometry import CGS_TO_F17, EXP_R50_OVER_RSCALE, HALPHA_REST_A
from kl_pipe.render import RenderConfig
from kl_pipe.surveys.roman import (
    ELECTRONS_PER_UJY,
    F_LIM_PER_PASS_CGS,
    F_LIM_REF_LAMBDA_A,
    F_LIM_REF_R50_ARCSEC,
    GRISM_REF_LINE_WIDTH_KMS,
    band_sigma_bkg_ujy,
    grism_electrons_per_f17_per_pass,
    grism_sigma_bkg_per_pass,
)

if TYPE_CHECKING:
    from kl_pipe.ensemble.spec import EnsembleSpec, ObservationConfig, PSFSpec
    from kl_pipe.priors import PriorDict
    from kl_pipe.source import SourceModel


class FitInputs(NamedTuple):
    """Everything the sampler needs for one fit."""

    source: 'SourceModel'
    priors: 'PriorDict'
    image_obs: Dict[str, object]
    grism_obs: Dict[str, object]
    truth: Dict[str, float]
    # realized matched-filter SNR of each channel's noiseless truth against
    # the actual mock variance (kl_pipe.noise.matched_filter_snr): one entry
    # per band, per grism roll ('line_roll<j>'), plus the roll coadd
    # ('line_total'). Under noise_model='matched_filter' these equal the
    # labels exactly; under 'poisson' they are the realized depth including
    # shot noise (the snr_effective results columns).
    snr_effective: Optional[Dict[str, float]] = None


def _import_galsim():
    try:
        import galsim
    except ImportError as err:
        raise ImportError(
            "galsim is required to build the mock PSF. On hosts without "
            "galsim (e.g. Vista NGC containers) install the vista_kit galsim "
            "stub + GaussianPSFShim before importing kl_pipe (see "
            "experiments/sweverett/vista_kit/psf_numpy.py)."
        ) from err
    return galsim


def _build_gaussian_psf(fwhm_arcsec: float):
    galsim = _import_galsim()
    return galsim.Gaussian(fwhm=fwhm_arcsec)


# process-level cache: getPSF is expensive (~0.5 s) and the worker builds
# several obs per fit (bands + rolls) plus the pinned-size reference PSF, so
# repeated wavelengths must not re-run the optics computation. Bounded FIFO:
# a long-lived worker claims many fits at distinct redshifts, so an unbounded
# cache would accumulate one GSObject per fit; within-fit reuse only needs a
# handful of entries.
_ROMAN_PSF_CACHE: Dict[
    Tuple[int, int, float, Optional[str], Optional[float]], object
] = {}
_ROMAN_PSF_CACHE_MAX = 16


def _get_roman_wfi_psf(
    sca: int,
    pupil_bin: int,
    wavelength_nm: float,
    bandpass: Optional[str],
    folding_threshold: Optional[float] = None,
):
    """Monochromatic Roman WFI PSF via galsim.roman.getPSF, cached.

    Cache key rounds the wavelength to 0.1 nm -- far below any PSF-size
    scale of interest -- so per-fit continuous redshifts still hit the
    cache across rolls within a fit.

    Parameters
    ----------
    sca : int
        WFI sensor chip assembly (1-18).
    pupil_bin : int
        Pupil-plane image binning (galsim.roman.getPSF pupil_bin).
    wavelength_nm : float
        Wavelength in nm (galsim.roman.getPSF takes nm).
    bandpass : str or None
        galsim.roman bandpass key selecting the pupil-plane configuration
        (F184 is the only long-wavelength band); None uses the
        short-wavelength pupil image.
    folding_threshold : float, optional
        GSParams folding_threshold applied to the returned PSF; controls
        the rendered kernel stamp size (and hence the padded convolution
        FFT). None keeps GalSim's default (5e-3).
    """
    _import_galsim()
    import galsim
    from galsim import roman

    key = (
        int(sca),
        int(pupil_bin),
        round(float(wavelength_nm), 1),
        bandpass,
        folding_threshold,
    )
    if key not in _ROMAN_PSF_CACHE:
        while len(_ROMAN_PSF_CACHE) >= _ROMAN_PSF_CACHE_MAX:
            _ROMAN_PSF_CACHE.pop(next(iter(_ROMAN_PSF_CACHE)))
        psf = roman.getPSF(
            SCA=key[0], bandpass=bandpass, wavelength=key[2], pupil_bin=key[1]
        )
        if folding_threshold is not None:
            psf = psf.withGSParams(galsim.GSParams(folding_threshold=folding_threshold))
        _ROMAN_PSF_CACHE[key] = psf
    return _ROMAN_PSF_CACHE[key]


# official Roman WFI filter names -> galsim.roman bandpass keys. galsim
# kept the legacy WFIRST names (H158, Z087, ...); the official filter set is
# F062-F213 (roman-docs.stsci.edu), so configs may use either.
_OFFICIAL_TO_GALSIM_BAND = {
    'F062': 'R062',
    'F087': 'Z087',
    'F106': 'Y106',
    'F129': 'J129',
    'F146': 'W146',
    'F158': 'H158',
    'F184': 'F184',
    'F213': 'K213',
}


def _galsim_roman_band(band: str) -> str:
    """Resolve an official Roman WFI filter name or a galsim legacy key to
    the galsim.roman bandpass key."""
    if band in _OFFICIAL_TO_GALSIM_BAND:
        return _OFFICIAL_TO_GALSIM_BAND[band]
    if band in _OFFICIAL_TO_GALSIM_BAND.values():
        return band
    raise ValueError(
        f"band '{band}' is not a Roman WFI filter; official names: "
        f"{sorted(_OFFICIAL_TO_GALSIM_BAND)} (galsim legacy keys "
        f"{sorted(_OFFICIAL_TO_GALSIM_BAND.values())} also accepted)"
    )


@lru_cache(maxsize=None)
def _band_effective_wavelength_nm(band: str) -> float:
    """Effective wavelength (nm) of a Roman WFI bandpass (official or
    galsim legacy name)."""
    _import_galsim()
    from galsim import roman

    key = _galsim_roman_band(band)
    return float(roman.getBandpasses()[key].effective_wavelength)


def _spec_folding_threshold(
    psf_spec: 'PSFSpec', mock: bool, z: Optional[float] = None
) -> Optional[float]:
    """Folding threshold for the requested role: fit kernels resolve
    ``folding_threshold``/``folding_threshold_tiers`` at the scene redshift
    ``z``, truth-render kernels use ``mock_folding_threshold`` (all None =
    galsim default; mock validated at least as accurate at every z).

    ``z`` is consulted only for a tiered fit schedule; a scalar/None fit
    threshold ignores it, and a tiered schedule with ``z is None`` raises
    loudly (rather than silently picking a tier)."""
    if mock:
        return psf_spec.mock_folding_threshold
    return psf_spec.resolve_fit_folding_threshold(z)


def _build_band_psf(
    psf_spec: 'PSFSpec', band: str, z: Optional[float] = None, mock: bool = False
):
    """Broadband PSF for one band from its PSFSpec.

    mock=True selects the truth-render kernel fidelity
    (``mock_folding_threshold``); mock=False the fit-kernel fidelity, whose
    folding_threshold may depend on the scene redshift ``z`` (tier schedule).
    """
    if psf_spec.psf_type == 'gaussian':
        return _build_gaussian_psf(psf_spec.fwhm_arcsec)
    if psf_spec.psf_type == 'roman_wfi':
        # monochromatic at the band's effective wavelength; the bandpass
        # name selects the matching pupil-plane configuration
        return _get_roman_wfi_psf(
            psf_spec.sca,
            psf_spec.pupil_bin,
            _band_effective_wavelength_nm(band),
            bandpass=_galsim_roman_band(band),
            folding_threshold=_spec_folding_threshold(psf_spec, mock, z),
        )
    raise NotImplementedError(f"psf type '{psf_spec.psf_type}' has no mock PSF builder")


def _build_grism_psf(psf_spec: 'PSFSpec', z: float, mock: bool = False):
    """Grism PSF from its PSFSpec, at the observed Halpha wavelength.

    mock=True selects the truth-render kernel fidelity
    (``mock_folding_threshold``); mock=False the fit-kernel fidelity, whose
    folding_threshold may depend on the scene redshift ``z`` (tier schedule).
    """
    if psf_spec.psf_type == 'gaussian':
        return _build_gaussian_psf(psf_spec.fwhm_arcsec)
    if psf_spec.psf_type == 'roman_wfi':
        # monochromatic at the observed line wavelength; bandpass=None uses
        # the short-wavelength pupil-plane image (galsim.roman does not
        # model the grism element's own pupil)
        return _get_roman_wfi_psf(
            psf_spec.sca,
            psf_spec.pupil_bin,
            LINE_LAMBDAS['Halpha'] * (1.0 + z),
            bandpass=None,
            folding_threshold=_spec_folding_threshold(psf_spec, mock, z),
        )
    raise NotImplementedError(f"psf type '{psf_spec.psf_type}' has no mock PSF builder")


def _grism_psf_kernel_size(
    config: 'ObservationConfig', spec: 'EnsembleSpec', mock: bool = False
) -> Optional[int]:
    """Pinned grism PSF kernel stamp size (fine pixels), constant across z.
    mock=True sizes the truth-render kernel (mock_folding_threshold fidelity).

    The roman_wfi grism PSF is monochromatic at the observed Halpha
    wavelength, so its GalSim good image size grows with z. Rendering every
    fit's kernel at the good size of the ensemble's LARGEST observed
    wavelength keeps the PSF array shape (and the padded FFT shape)
    constant across the ensemble. Smaller-z kernels are simply drawn on the
    larger stamp and stay unit-normalized. Gaussian grism PSFs are
    z-independent, so no pinning applies (returns None).
    """
    if config.grism_psf.psf_type == 'gaussian':
        return None
    if spec.catalog_population is not None:
        # catalog populations: the selection z_range caps the observed
        # wavelength across the ensemble
        z_max = spec.catalog_population.z_range[1]
    else:
        z_draw = spec.draw.get('z')
        if z_draw is None:
            raise ValueError(
                "a roman_wfi grism psf requires z in population.draw so the "
                "kernel size can be pinned at the ensemble's largest observed "
                "wavelength"
            )
        if z_draw.dist != 'uniform':
            raise NotImplementedError(
                f"grism kernel-size pinning knows the z range for uniform "
                f"draws only, got dist '{z_draw.dist}'"
            )
        z_max = z_draw.params['high']
    psf_max = _build_grism_psf(config.grism_psf, z_max, mock=mock)
    fine_ps = config.pixel_scale_arcsec / spec.render_oversample
    size = int(psf_max.getGoodImageSize(fine_ps))
    if size % 2 == 0:
        size += 1
    return size


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


def _psf_l2_norm(psf, pixel_scale: float) -> float:
    """L2 norm of the unit-flux PSF image at the survey pixel scale.

    ||K||_2 of the point-source template the published imaging depths refer
    to, drawn from the same PSF object the mock renders with (pixel
    integration included via galsim drawImage). The default stamp truncates
    far-wing flux at the folding threshold; that costs the L2 norm almost
    nothing (wings are tiny per pixel), but a badly truncated draw means
    the wrong PSF object and raises.
    """
    image = psf.drawImage(scale=pixel_scale)
    total = float(image.array.sum())
    if not 0.95 <= total <= 1.02:
        raise ValueError(
            f"unit-flux PSF image sums to {total:.4f}; the point-source "
            f"template is badly truncated or unnormalized"
        )
    return float(np.sqrt(np.sum(np.asarray(image.array, dtype=np.float64) ** 2)))


def _make_band_obs(
    source,
    truth,
    psf_mock,
    psf_fit,
    image_pars,
    band,
    snr,
    seed,
    oversample,
    flux_unit=None,
    noise_model='matched_filter',
):
    int_model = source.broadband_models[band]
    # truth data through the mock-fidelity kernel; the fit obs carries the
    # fit-fidelity kernel (identical objects unless the PSFSpec splits them)
    obs_clean = build_image_obs(
        image_pars,
        psf=psf_mock,
        render_config=RenderConfig(oversample=oversample),
        int_model=int_model,
        broadband_key=band,
    )
    data_true = np.asarray(source.render_broadband(truth, obs_clean, band))
    if noise_model == 'poisson':
        # flat background anchored to the published point-source depth
        # through this config's own PSF, plus the galaxy's shot noise; the
        # labeled snr is NOT used to set the noise here -- it stays the
        # selection/plot axis while snr_effective reports the realized depth
        if band not in ELECTRONS_PER_UJY:
            raise KeyError(
                f"band '{band}' has no electrons_per_ujy conversion; "
                f"known bands: {sorted(ELECTRONS_PER_UJY)}"
            )
        sigma_bkg = band_sigma_bkg_ujy(
            band, _psf_l2_norm(psf_mock, image_pars.pixel_scale)
        )
        var = physical_variance_map(data_true, sigma_bkg, ELECTRONS_PER_UJY[band])
        data_noisy = add_map_noise(data_true, var, seed)
        variance = jnp.asarray(var)
    else:
        data_noisy, var = _noisy(data_true, snr, seed)
        variance = var
    obs = build_image_obs(
        image_pars,
        psf=psf_fit,
        render_config=RenderConfig(oversample=oversample),
        data=jnp.asarray(data_noisy),
        variance=variance,
        int_model=int_model,
        broadband_key=band,
        flux_unit=flux_unit,
    )
    return obs, matched_filter_snr(data_true, var)


def _grism_pars_for_roll(
    config: 'ObservationConfig', z: float, roll_deg: float, single_roll: bool
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


# reference-template L2 norms keyed by the grism-render-relevant config
# fields (see _grism_reference_line_norm); a long-lived worker reuses one
# entry per observation config
_GRISM_REF_NORM_CACHE: Dict[tuple, float] = {}


def _grism_reference_line_norm(config: 'ObservationConfig', oversample: int) -> float:
    """L2 norm of the survey reference source's dispersed line template.

    The published grism limit refers to an extended reference source, not a
    point source, so the background anchor requires rendering that reference
    through this config's own grism model: a round face-on exponential disk
    of half-light radius F_LIM_REF_R50_ARCSEC, zero rotation, line width
    GRISM_REF_LINE_WIDTH_KMS, Halpha observed at F_LIM_REF_LAMBDA_A, at the
    PER-PASS flux limit (see the surveys.roman background block for why and
    for the stated kinematics systematic). Rendered once per config on the
    first roll: the scene is circularly symmetric (face-on, centered, no
    rotation, zero shear), so every roll's template has the same norm.
    """
    key = (
        config.grism_psf,
        config.stamp_grism_pix,
        config.pixel_scale_arcsec,
        config.grism_dispersion_nm_per_pix,
        len(config.grism_rolls_deg) == 1,
        config.grism_rolls_deg[0],
        oversample,
    )
    if key in _GRISM_REF_NORM_CACHE:
        return _GRISM_REF_NORM_CACHE[key]

    z_ref = F_LIM_REF_LAMBDA_A / HALPHA_REST_A - 1.0
    truth = scene_truth_defaults(config, {})
    truth.update(
        {
            'z': z_ref,
            'cosi': 1.0,
            'theta_int': 0.0,
            'g1': 0.0,
            'g2': 0.0,
            'vel.vcirc': 0.0,
            'vel.v0': 0.0,
            'Halpha.flux': F_LIM_PER_PASS_CGS * CGS_TO_F17,
            'Halpha.rscale': F_LIM_REF_R50_ARCSEC / EXP_R50_OVER_RSCALE,
            'Halpha.dispersion': GRISM_REF_LINE_WIDTH_KMS,
            'Halpha.cont.flux_per_nm': 0.0,
        }
    )
    source = build_source_model(config)
    grism_pars = _grism_pars_for_roll(
        config, z_ref, config.grism_rolls_deg[0], len(config.grism_rolls_deg) == 1
    )
    obs_clean = build_grism_obs(
        grism_pars,
        z=z_ref,
        psf=_build_grism_psf(config.grism_psf, z_ref, mock=True),
        render_config=RenderConfig(oversample=oversample),
    )
    line_template = np.asarray(source.render_grism(truth, obs_clean))
    norm = float(np.sqrt(np.sum(line_template**2)))
    if norm <= 0:
        raise ValueError("grism reference template has zero power")
    _GRISM_REF_NORM_CACHE[key] = norm
    return norm


def _make_roll_obs(
    source,
    truth,
    psf_mock,
    psf_fit,
    config,
    z,
    roll_deg,
    single_roll,
    line_snr,
    seed,
    oversample,
    kernel_size_mock,
    kernel_size_fit,
    flux_unit=None,
    noise_model='matched_filter',
    sigma_bkg=None,
    electrons_per_f17=None,
):
    grism_pars = _grism_pars_for_roll(config, z, roll_deg, single_roll)
    # render truth at the SAME oversample as the fit obs below (mirrors
    # _make_band_obs); otherwise the grism truth datavector is generated at the
    # RenderConfig default oversample=1 while the fit runs at the spec's
    # render_oversample,
    # silently breaking grism fit==truth self-consistency. The truth data goes
    # through the mock-fidelity kernel; the fit obs carries the fit-fidelity
    # kernel (identical unless the PSFSpec splits them).
    obs_clean = build_grism_obs(
        grism_pars,
        z=z,
        psf=psf_mock,
        render_config=RenderConfig(oversample=oversample),
        psf_kernel_size=kernel_size_mock,
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
    if noise_model == 'poisson':
        # flat per-roll background anchored to the per-pass published limit
        # (sigma_bkg from the rendered reference source), plus shot noise from
        # the FULL dispersed truth -- the continuum contributes photons even
        # though only the line defines the labeled SNR
        var = physical_variance_map(data_true, sigma_bkg, electrons_per_f17)
        data_noisy = add_map_noise(data_true, var, seed)
        variance = jnp.asarray(var)
    else:
        data_noisy, var = grism_line_noise(data_true, line_true, line_snr, seed)
        variance = var
    obs = build_grism_obs(
        grism_pars,
        z=z,
        psf=psf_fit,
        render_config=RenderConfig(oversample=oversample),
        data=jnp.asarray(data_noisy),
        variance=variance,
        psf_kernel_size=kernel_size_fit,
        flux_unit=flux_unit,
    )
    # snr_effective normalizes on the LINE template (the labeled convention)
    # against the full realized variance
    return obs, matched_filter_snr(line_true, var)


def build_fit_inputs(
    truth: Dict[str, float],
    noise_seed: int,
    spec: 'EnsembleSpec',
    config: 'ObservationConfig',
    *,
    band_snrs: Dict[str, float],
    line_snr: float,
    row: Optional[Dict] = None,
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
    config : ObservationConfig
        Structural instrument setup.
    band_snrs : dict
        Per-band matched-filter SNR from the manifest row (always read from
        the manifest, never the spec: one shared scalar per fit in sampled
        mode, per-galaxy published-depth values in catalog mode).
        Must cover exactly the config's bands.
    line_snr : float
        Per-fit emission-line matched-filter SNR (see
        kl_pipe.noise.grism_line_noise); varies per row on a config-sweep
        axis and per galaxy in catalog mode.
    row : dict or pd.Series, optional
        The manifest row; required for catalog populations (their vcirc
        prior reads the row's pop.prior_vcirc_* columns).

    Returns
    -------
    FitInputs
        (source, priors, image_obs, grism_obs, truth).
    """
    if set(band_snrs) != set(config.bands):
        raise ValueError(
            f"band_snrs keys {sorted(band_snrs)} must match the config's "
            f"bands {sorted(config.bands)}"
        )
    bad_snrs = {k: v for k, v in band_snrs.items() if v <= 0}
    if bad_snrs or line_snr <= 0:
        raise ValueError(
            f"SNR values must be positive, got bands {bad_snrs or 'ok'}, "
            f"line {line_snr}"
        )
    # catalog mode with the bulge paint on: broadband bands are BulgeDiskModel
    # with this galaxy's fixed Sersic index; sampled mode and the disk-only
    # catalog twin (paint.bulge: false): single-disk bands (bulge_nsersic None)
    is_catalog = spec.catalog_population is not None
    if is_catalog and row is None:
        raise ValueError(
            "catalog-mode fits require the manifest row (pop.prior_vcirc_* "
            "and, with the bulge paint, pop.bulge_nsersic columns); pass row"
        )
    if config.noise_model == 'poisson' and not is_catalog:
        raise ValueError(
            "noise_model 'poisson' requires a catalog population: shot noise "
            "converts physical fluxes to electrons, and sampled-mode scenes "
            "carry no physical flux units"
        )
    use_bulge = is_catalog and spec.catalog_population.paint_bulge
    bulge_nsersic = float(row['pop.bulge_nsersic']) if use_bulge else None
    source = build_source_model(
        config,
        bulge_nsersic=bulge_nsersic,
        sample_bulge_nsersic=use_bulge and spec.sample_bulge_nsersic,
    )
    priors = scene_priors(truth, config, spec, row=row)

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
    snr_effective: Dict[str, float] = {}
    for i, band in enumerate(config.bands):
        band_spec = config.band_psf[band]
        image_obs[band], snr_effective[band] = _make_band_obs(
            source,
            truth,
            _build_band_psf(band_spec, band, z, mock=True),
            _build_band_psf(band_spec, band, z),
            image_pars,
            band,
            band_snrs[band],
            seeds[i],
            spec.render_oversample,
            flux_unit='uJy (band-averaged f_nu)' if is_catalog else None,
            noise_model=config.noise_model,
        )

    grism_psf_mock = _build_grism_psf(config.grism_psf, z, mock=True)
    grism_psf_fit = _build_grism_psf(config.grism_psf, z)
    kernel_size_mock = _grism_psf_kernel_size(config, spec, mock=True)
    kernel_size_fit = _grism_psf_kernel_size(config, spec)
    single_roll = len(config.grism_rolls_deg) == 1
    if config.noise_model == 'poisson':
        grism_sigma_bkg = grism_sigma_bkg_per_pass(
            _grism_reference_line_norm(config, spec.render_oversample)
        )
        lambda_obs_a = HALPHA_REST_A * (1.0 + z)
        electrons_per_f17 = float(grism_electrons_per_f17_per_pass(lambda_obs_a))
    else:
        grism_sigma_bkg = None
        electrons_per_f17 = None
    grism_obs = {}
    for j, roll in enumerate(config.grism_rolls_deg):
        grism_obs[f'roll{j}'], snr_effective[f'line_roll{j}'] = _make_roll_obs(
            source,
            truth,
            grism_psf_mock,
            grism_psf_fit,
            config,
            z,
            roll,
            single_roll,
            line_snr,
            seeds[len(config.bands) + j],
            spec.render_oversample,
            kernel_size_mock,
            kernel_size_fit,
            flux_unit=(
                '1e-17 erg/s/cm2 (integrated line flux)' if is_catalog else None
            ),
            noise_model=config.noise_model,
            sigma_bkg=grism_sigma_bkg,
            electrons_per_f17=electrons_per_f17,
        )
    # roll coadd: independent noise per roll, so the joint-fit line depth is
    # the quadrature sum (matches how the labeled total SNR is defined)
    snr_effective['line_total'] = float(
        np.sqrt(
            sum(
                snr_effective[f'line_roll{j}'] ** 2
                for j in range(len(config.grism_rolls_deg))
            )
        )
    )

    return FitInputs(
        source=source,
        priors=priors,
        image_obs=image_obs,
        grism_obs=grism_obs,
        truth=truth,
        snr_effective=snr_effective,
    )
