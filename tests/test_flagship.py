"""Flagship integration test: joint Roman-like broadband + grism inference.

The headline scientific claim:

    At Roman-like noise + PSF + dispersion, joint broadband-photometric +
    slitless-grism inference with NUTS recovers the full 17-dim model
    parameter set within joint Nsigma < 3, demonstrating end-to-end
    correctness of the kinematic lensing forward model and inference
    pipeline.

This is the **top-level visual regression test** for the repo: every
diagnostic plot doubles as a presentation / paper asset. Outputs sort first
alphabetically in ``tests/out/`` via the ``00_flagship/`` directory.

Configuration (Roman-like), dev config:
- F087 broadband: 32x32 image at 0.11 arcsec/pixel (native Roman), PSF FWHM 0.18"
- Grism (Roman G150-like): 32x32 dispersed image, dispersion 1.1 nm/pix,
  lambda_ref derived from Halpha at z=1.0 (lambda_obs ~ 1313 nm, within the
  Roman grism range ~1.0-1.93 um)
- Galaxy: z=1.0, vcirc=200 km/s, hlr~0.3", inc~53 deg (cosi=0.6), theta_int=pi/4
- Shear: g1=0.02, g2=-0.01
- SNR (matched-filter): 100 broadband, 150 grism

Sample space (17-dim):
- Geometry:        cosi, theta_int, g1, g2
- Velocity:        vel.v0, vel.vcirc, vel.rscale (CenteredVelocityModel -- no vel.x0/y0)
- Broadband F087:  F087.flux, F087.rscale, F087.x0, F087.y0   (h_over_r fixed)
- Emission line:   Halpha.flux, Halpha.rscale, Halpha.x0, Halpha.y0,
                   Halpha.dispersion                          (h_over_r fixed)
- Continuum under line:
                   Halpha.cont.flux_per_nm                           (others fixed to line spatial truth)
The continuum is intentionally non-zero so the flat continuum trace is
visible above the noise floor in the dispersed grism image.

Pass criterion (single, matching existing diagnostic convention):
    joint Nsigma > 3 -> pytest.fail
    2 < joint Nsigma <= 3 -> warnings.warn
    <= 2 -> pass silently

R-hat / ESS / per-param recovery are computed and saved as diagnostics but
do NOT gate the test. They surface in the plots for human review.

Observing config (``--flagship-production`` flag, orthogonal to the sampler-
depth ``--flagship-long`` flag):
- dev (default): 1 broadband (F087) + 1 grism roll -- the config documented
  above.
- production: adds a second broadband band (F158) and disperses the same
  galaxy through 4 grism rolls (0/45/90/135 deg). A strict superset of the
  dev scene (same F087 + Halpha truth/priors); each band and roll is an
  independent exposure at fixed per-exposure SNR. Diagnostics land in the
  same 00_flagship/ dir with a ``_production`` suffix; when dev stats already
  exist the production run also writes ``sensitivity_comparison.txt`` (per-
  param posterior-width ratio, production vs dev).

Marked ``slow`` -- excluded from ``make test-basic``. Run via
``make test-flagship`` / ``make test-flagship-production`` (or pytest
tests/test_flagship.py directly, adding ``--flagship-production`` and/or
``--flagship-long``).
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path
from typing import Dict

import galsim
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pytest

from kl_pipe.diagnostics.imaging import (
    compute_joint_nsigma,
    plot_data_comparison_panels,
)
from kl_pipe.dispersion import build_grism_pars_for_line
from kl_pipe.intensity import InclinedExponentialModel
from kl_pipe.lines import EmissionLine, LINE_LAMBDAS
from kl_pipe.noise import grism_line_noise
from kl_pipe.observation import (
    build_grism_obs,
    build_image_obs,
    build_velocity_obs,
)
from kl_pipe.parameters import ImagePars
from kl_pipe.priors import Gaussian, PriorDict, TruncatedNormal, make_tf_prior
from kl_pipe.sampling import (
    InferenceTask,
    NumpyroSamplerConfig,
    build_sampler,
)
from kl_pipe.sampling.diagnostics import (
    plot_corner,
    plot_recovery,
    plot_trace,
)
from kl_pipe.source import SourceModel
from kl_pipe.synthetic import SyntheticIntensity
from kl_pipe.utils import get_test_dir
from kl_pipe.velocity import CenteredVelocityModel
from kl_pipe.render import RenderConfig


pytestmark = pytest.mark.slow


# Roman-like configuration (module-scope constants -- referenced in plot titles)
Z = 1.0
PIXEL_SCALE = 0.11  # arcsec/pixel, Roman native
IMAGE_SHAPE = (32, 32)
PSF_FWHM = 0.18  # arcsec, Roman-like (F087 broadband / grism)
SNR_BROADBAND = 100.0
SNR_GRISM = 150.0
GRISM_DISPERSION_NM_PER_PIX = 1.1

# Tully-Fisher scatter (dex) for the vel.vcirc prior. Fiducial 0.08 dex from
# Xu+2022 / Pranjal, the project's archival TFR value (ensemble sweep spans
# 0.05-0.20; obs z~1 ~0.20). The vcirc prior is a LogNormal centered on the
# truth vcirc (= TF median) rather than an ad-hoc Gaussian -- the TF relation
# is genuine external information KL inference is entitled to use.
SIGMA_TF_DEX = 0.08

# Spatial oversample factor for both broadband and grism obs. Default is 5;
# 3 is acceptable here because PSF FWHM (0.18") is ~1.6 pixels at 0.11"/pix
# and the resulting profile is band-limited enough that osf=3 gives sub-1%
# bias on rendered images. Cuts FFT work ~3x vs default.
SPATIAL_OVERSAMPLE = 3

# Sampler settings. Uses the Laplace preconditioner (precondition='laplace'):
# NUTS starts at the MAP with a fixed inverse-Hessian mass matrix, so warmup is
# short (just step-size adaptation) instead of the ~300 iters dense-mass needs
# to climb from an identity metric. Validated ~5x faster (8.5 vs 42.5 min) with
# equal recovery + better convergence; see experiments/sweverett/flagship_speedup.
#
# Two configs selectable via the --flagship-long pytest flag (see conftest.py):
#   short (default): fast gating run.
#   long: production-depth run for cleaner posteriors (make test-flagship-long).
# Only sampler depth differs between them -- truth, priors, and the joint-Nsigma
# pass criterion are identical.
SHORT_CONFIG = {
    'n_warmup': 50,
    'n_samples': 300,
    'n_chains': 2,
    'max_tree_depth': 8,
    'chain_method': 'vectorized',  # 2 chains batched -- modest peak memory
}
# Long mode: 4 chains vectorized, 2000 samples, depth 10 -- production-depth for
# clean posteriors (8000 draws). Earlier this config was SIGKILLed at the end of
# sampling ("zsh: killed" at 100%); root cause was NOT the sampler but the
# wrapper computing log_prob by vmap-ing the FFT-rendering likelihood over all
# n_samples*n_chains at once (a transient that scaled with sample count and
# OOM'd). Fixed in numpyro.py via chunked evaluation (_batched_log_posterior_
# chunked); peak memory now bounded regardless of sample count.
LONG_CONFIG = {
    'n_warmup': 200,
    'n_samples': 2000,
    'n_chains': 4,
    'max_tree_depth': 10,
    'chain_method': 'vectorized',
}


# Headline parameter subset for the small corner plot.
HEADLINE_PARAMS = [
    'g1',
    'g2',
    'cosi',
    'theta_int',
    'vel.vcirc',
    'vel.rscale',
    'F087.rscale',
    'Halpha.rscale',
]


@pytest.fixture(scope='module')
def output_dir():
    """Top-of-alphabet output directory so diagnostics surface first."""
    out_dir = get_test_dir() / 'out' / '00_flagship'
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _true_pars_dotted() -> Dict[str, float]:
    return {
        # Geometry
        'cosi': 0.6,
        'theta_int': np.pi / 4,
        'g1': 0.02,
        'g2': -0.01,
        # Velocity (CenteredVelocityModel -- vel.x0, vel.y0 are not parameters)
        'vel.v0': 10.0,
        'vel.vcirc': 200.0,
        'vel.rscale': 0.3,
        # Broadband F087
        'F087.flux': 100.0,
        'F087.rscale': 0.3,
        'F087.h_over_r': 0.1,
        'F087.x0': 0.0,
        'F087.y0': 0.0,
        # Halpha emission line
        'Halpha.flux': 100.0,
        'Halpha.rscale': 0.25,
        'Halpha.h_over_r': 0.1,
        'Halpha.x0': 0.0,
        'Halpha.y0': 0.0,
        'Halpha.dispersion': 50.0,
        # Continuum under Halpha line. Spatial profile fixed equal to the line
        # spatial truth; only the continuum flux is sampled. The continuum
        # produces a flat trace across the dispersed grism image visible above
        # the noise floor at this SNR.
        'Halpha.cont.flux_per_nm': 25.0,
        'Halpha.cont.rscale': 0.25,
        'Halpha.cont.h_over_r': 0.1,
        'Halpha.cont.x0': 0.0,
        'Halpha.cont.y0': 0.0,
        # Redshift (fixed in prior)
        'z': Z,
    }


def _flagship_prior_spec(true: Dict[str, float]) -> Dict[str, object]:
    """Prior spec (dict) for the dev config; wrapped by ``_flagship_priors``.

    Split out from the PriorDict wrapper so the production config can extend
    it with the second broadband band without duplicating the shared entries.
    """
    return {
        # Geometry
        'cosi': TruncatedNormal(0.6, 0.15, 0.05, 0.99),
        'theta_int': TruncatedNormal(np.pi / 4, 0.3, 0.0, np.pi / 2),
        # Wide, uninformative shear priors so the posterior widths reflect the
        # data's shear constraint, not the prior. Sigma 0.2 matches the published
        # Roman KL prior half-width (Xu+ 2023) as an isotropic Gaussian. Unbounded:
        # the model is stable over the reach (|g|=1 is 5 sigma) and truncation
        # would re-inject a prior edge and break isotropy per-component.
        'g1': Gaussian(0.0, 0.2),
        'g2': Gaussian(0.0, 0.2),
        # Velocity. vcirc uses the Tully-Fisher LogNormal prior (median = truth,
        # scatter SIGMA_TF_DEX) instead of an ad-hoc Gaussian.
        'vel.v0': Gaussian(10.0, 10.0),
        'vel.vcirc': make_tf_prior(true['vel.vcirc'], SIGMA_TF_DEX),
        'vel.rscale': TruncatedNormal(0.3, 0.1, 0.05, 1.0),
        # Broadband F087
        'F087.flux': TruncatedNormal(100.0, 20.0, 30.0, 250.0),
        'F087.rscale': TruncatedNormal(0.3, 0.08, 0.05, 1.0),
        'F087.h_over_r': 0.1,  # fixed
        'F087.x0': TruncatedNormal(0.0, 0.1, -0.5, 0.5),
        'F087.y0': TruncatedNormal(0.0, 0.1, -0.5, 0.5),
        # Halpha emission line
        'Halpha.flux': TruncatedNormal(100.0, 20.0, 30.0, 250.0),
        'Halpha.rscale': TruncatedNormal(0.25, 0.08, 0.05, 1.0),
        'Halpha.h_over_r': 0.1,  # fixed
        'Halpha.x0': TruncatedNormal(0.0, 0.1, -0.5, 0.5),
        'Halpha.y0': TruncatedNormal(0.0, 0.1, -0.5, 0.5),
        'Halpha.dispersion': TruncatedNormal(50.0, 20.0, 5.0, 150.0),
        # Continuum under Halpha (only flux sampled; spatial profile fixed
        # to line spatial truth so the inference doesn't have to also solve
        # a continuum-line spatial degeneracy at this SNR).
        'Halpha.cont.flux_per_nm': TruncatedNormal(25.0, 15.0, 0.0, 200.0),
        'Halpha.cont.rscale': true['Halpha.cont.rscale'],
        'Halpha.cont.h_over_r': true['Halpha.cont.h_over_r'],
        'Halpha.cont.x0': true['Halpha.cont.x0'],
        'Halpha.cont.y0': true['Halpha.cont.y0'],
        # Redshift fixed
        'z': Z,
    }


def _flagship_priors(true: Dict[str, float]) -> PriorDict:
    """Moderately-narrow TruncatedNormal priors centered on truth; fix h_over_r + z.

    theta_int is restricted to [0, pi/2] (single position-angle quadrant) --
    the full [0, pi] range exposes the classic sin(2theta) two-mode
    degeneracy that destroys NUTS mixing at this dimensionality.
    """
    return PriorDict(_flagship_prior_spec(true))


# ============================================================================
# Production observing config (2 broadband F087+F158 + 4 grism rolls)
# ============================================================================
#
# The production config is a strict SUPERSET of the dev config: same F087
# broadband, same Halpha line (+continuum), same shared geometry/velocity
# truth and priors. It adds a second broadband band (F158, mild color
# gradient) and disperses the SAME sky galaxy through 4 grism roll angles.
# Roll angle is a pure WCS rotation (the sky is fixed; the detector -- and
# thus the dispersion axis -- rotates); the likelihood auto-groups the rolls
# into one shared cube. Each broadband band and each roll is an independent
# exposure at fixed per-exposure SNR (independent noise), so total information
# grows with band/roll count -- this is the sensitivity lever under study.
ROLL_ANGLES_DEG = (0.0, 45.0, 90.0, 135.0)


def _true_pars_production() -> Dict[str, float]:
    """Dev truth + a second broadband band F158 (brighter, larger scale)."""
    true = _true_pars_dotted()
    true.update(
        {
            'F158.flux': 120.0,
            'F158.rscale': 0.35,
            'F158.h_over_r': 0.1,
            'F158.x0': 0.0,
            'F158.y0': 0.0,
        }
    )
    return true


def _production_priors(true: Dict[str, float]) -> PriorDict:
    """Dev prior spec extended with the F158 broadband band."""
    spec = _flagship_prior_spec(true)
    spec.update(
        {
            'F158.flux': TruncatedNormal(120.0, 25.0, 30.0, 300.0),
            'F158.rscale': TruncatedNormal(0.35, 0.08, 0.05, 1.0),
            'F158.h_over_r': 0.1,  # fixed
            'F158.x0': TruncatedNormal(0.0, 0.1, -0.5, 0.5),
            'F158.y0': TruncatedNormal(0.0, 0.1, -0.5, 0.5),
        }
    )
    return PriorDict(spec)


def _wcs_with_pc(shape, pixel_scale: float, rotation_radians: float):
    """Astropy WCS carrying a pure roll rotation (production multi-roll grism).

    A nonzero PC-matrix rotation is read back by the forward model via
    ``image_rotation_from_wcs``; the same sky galaxy therefore disperses
    along a rotated detector axis for each roll.
    """
    from astropy.wcs import WCS

    Nrow, Ncol = shape
    c = float(np.cos(rotation_radians))
    s = float(np.sin(rotation_radians))
    wcs = WCS(naxis=2)
    wcs.wcs.pc = np.array([[c, -s], [s, c]])
    wcs.wcs.cdelt = np.array([pixel_scale, pixel_scale])
    wcs.wcs.crpix = np.array([Ncol / 2, Nrow / 2])
    wcs.wcs.crval = np.array([0.0, 0.0])
    wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    wcs.wcs.cunit = ['arcsec', 'arcsec']
    wcs.pixel_shape = (Ncol, Nrow)
    wcs.wcs.set()
    return wcs


def _band_flat_pars(true: Dict[str, float], band_key: str) -> Dict[str, float]:
    """Flat (un-dotted) intensity pars for one broadband band, for SyntheticIntensity."""
    return {
        'cosi': true['cosi'],
        'theta_int': true['theta_int'],
        'g1': true['g1'],
        'g2': true['g2'],
        'flux': true[f'{band_key}.flux'],
        'rscale': true[f'{band_key}.rscale'],
        'h_over_r': true[f'{band_key}.h_over_r'],
        'x0': true[f'{band_key}.x0'],
        'y0': true[f'{band_key}.y0'],
    }


def _make_broadband_channel(source, true, image_pars, psf, band_key, seed):
    """Broadband channel: GalSim SyntheticIntensity data + obs carrying it.

    Synthetic data is rendered by GalSim (an independent renderer), not the
    model under test -- the flagship's cross-check against the forward model.
    Returns a channel dict with obs + the true/noisy images + variance.
    """
    int_model = source.broadband_models[band_key]
    synth = SyntheticIntensity(
        _band_flat_pars(true, band_key), model_type='exponential', seed=seed, psf=psf
    )
    data_noisy = synth.generate(
        image_pars, snr=SNR_BROADBAND, seed=seed, include_poisson=False
    )
    obs = build_image_obs(
        image_pars,
        psf=psf,
        render_config=RenderConfig(oversample=SPATIAL_OVERSAMPLE),
        data=jnp.asarray(data_noisy),
        variance=synth.variance,
        int_model=int_model,
        broadband_key=band_key,
    )
    return {
        'obs': obs,
        'true': np.asarray(synth.data_true),
        'noisy': np.asarray(data_noisy),
        'variance': np.asarray(synth.variance),
        'band_key': band_key,
    }


def _make_grism_channel(source, true, image_pars, psf, angle_deg, seed, roll_key):
    """One grism roll: (optionally rotated) WCS obs + model-rendered truth +
    independent Gaussian noise at fixed per-exposure SNR.

    angle_deg == 0 reuses the dev image_pars (default WCS) so the dev config's
    single roll is unchanged; nonzero angles carry a rotated-WCS ImagePars.
    """
    if angle_deg == 0.0:
        ip = image_pars
    else:
        ip = ImagePars(
            shape=IMAGE_SHAPE,
            wcs=_wcs_with_pc(IMAGE_SHAPE, PIXEL_SCALE, np.deg2rad(angle_deg)),
            indexing='ij',
        )
    grism_pars = build_grism_pars_for_line(
        LINE_LAMBDAS['Halpha'],
        redshift=Z,
        image_pars=ip,
        dispersion=GRISM_DISPERSION_NM_PER_PIX,
    )
    grism_obs_clean = build_grism_obs(grism_pars, z=Z, psf=psf)
    data_true = np.asarray(source.render_grism(true, grism_obs_clean))
    # SNR_GRISM is the emission-LINE matched-filter SNR: normalize the noise on
    # the line only (continuum zeroed), not the continuum-inflated whole stamp
    # (see kl_pipe.noise.grism_line_noise)
    line_true_pars = {
        k: (0.0 if k.endswith('.cont.flux_per_nm') else v) for k, v in true.items()
    }
    line_true = np.asarray(source.render_grism(line_true_pars, grism_obs_clean))
    data_noisy, var = grism_line_noise(data_true, line_true, SNR_GRISM, seed)
    obs = build_grism_obs(
        grism_pars,
        z=Z,
        psf=psf,
        render_config=RenderConfig(oversample=SPATIAL_OVERSAMPLE),
        data=jnp.asarray(data_noisy),
        variance=var,
    )
    return {
        'obs': obs,
        'true': data_true,
        'noisy': data_noisy,
        'variance': var,
        'roll_key': roll_key,
        'angle_deg': angle_deg,
    }


def _build_config(production: bool):
    """Assemble the source, priors, truth, and per-channel obs for a config.

    Returns (source, priors, true, image_channels, grism_channels). Channels
    are dicts keyed by band name / roll name; each carries obs + true/noisy
    images + variance for both inference and diagnostics.
    """
    image_pars = ImagePars(shape=IMAGE_SHAPE, pixel_scale=PIXEL_SCALE, indexing='ij')
    psf = galsim.Gaussian(fwhm=PSF_FWHM)

    def _line_source(broadband_models):
        return SourceModel(
            velocity_model=CenteredVelocityModel(),
            broadband_models=broadband_models,
            emission_lines={
                'Halpha': EmissionLine(
                    intensity=InclinedExponentialModel(),
                    continuum=InclinedExponentialModel(),
                )
            },
        )

    if production:
        true = _true_pars_production()
        priors = _production_priors(true)
        source = _line_source(
            {
                'F087': InclinedExponentialModel(),
                'F158': InclinedExponentialModel(),
            }
        )
        band_keys = ('F087', 'F158')
        roll_angles = ROLL_ANGLES_DEG
    else:
        true = _true_pars_dotted()
        priors = _flagship_priors(true)
        source = _line_source({'F087': InclinedExponentialModel()})
        band_keys = ('F087',)
        roll_angles = (0.0,)

    # Distinct noise seeds per channel (independent exposures). Dev keeps its
    # historical seeds (F087=42, roll0 grism=43) so its data is unchanged.
    band_seeds = {'F087': 42, 'F158': 52}
    image_channels = {
        b: _make_broadband_channel(source, true, image_pars, psf, b, band_seeds[b])
        for b in band_keys
    }
    grism_channels = {}
    for i, a in enumerate(roll_angles):
        seed = 43 if not production else 100 + i
        grism_channels[f'roll{i}'] = _make_grism_channel(
            source, true, image_pars, psf, a, seed, f'roll{i}'
        )
    return source, priors, true, image_channels, grism_channels


def _write_sensitivity_comparison(output_path, dev_stats, prod_stats):
    """Per-parameter posterior-width comparison: dev vs production.

    ratio = prod_std / dev_std; < 1 means the production observing config
    tightened that parameter's marginal.
    """
    shared = sorted(set(dev_stats) & set(prod_stats))
    lines = ['=' * 72]
    lines.append('FLAGSHIP SENSITIVITY COMPARISON  (posterior std: dev vs production)')
    lines.append('=' * 72)
    lines.append(f"{'Parameter':<22} {'dev std':>12} {'prod std':>12} {'prod/dev':>10}")
    lines.append('-' * 72)
    for name in shared:
        d = float(dev_stats[name]['std'])
        p = float(prod_stats[name]['std'])
        ratio = p / d if d > 0 else float('nan')
        lines.append(f'{name:<22} {d:>12.5g} {p:>12.5g} {ratio:>10.3f}')
    lines.append('=' * 72)
    lines.append(
        'ratio < 1: production tightened this marginal. Params only in '
        'production (e.g. F158.*) are omitted.'
    )
    Path(output_path).write_text('\n'.join(lines))


def _plot_ground_truth_overview(
    data_F087_noisy: np.ndarray,
    data_F087_true: np.ndarray,
    data_grism_noisy: np.ndarray,
    data_grism_true: np.ndarray,
    velocity_true: np.ndarray,
    intensity_true: np.ndarray,
    output_path: Path,
    config_label: str = '',
):
    """6-panel composite: noisy data, noiseless truth, and intrinsic fields.

    For the production config this shows the F087 band + roll0 as
    representatives; every band and roll gets its own comparison panel via
    ``plot_data_comparison_panels``.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    panels = [
        (axes[0, 0], data_F087_noisy, 'F087 broadband (data, with noise)', 'viridis'),
        (axes[0, 1], data_F087_true, 'F087 broadband (truth)', 'viridis'),
        (
            axes[0, 2],
            intensity_true,
            'Intrinsic F087 intensity (no noise, no PSF)',
            'viridis',
        ),
        (axes[1, 0], data_grism_noisy, 'Halpha grism (data, with noise)', 'magma'),
        (axes[1, 1], data_grism_true, 'Halpha grism (truth)', 'magma'),
        (axes[1, 2], velocity_true, 'LOS velocity field (km/s)', 'RdBu_r'),
    ]
    for ax, img, title, cmap in panels:
        if cmap == 'RdBu_r':
            vmax = float(np.max(np.abs(img)))
            im = ax.imshow(
                np.asarray(img), cmap=cmap, vmin=-vmax, vmax=vmax, origin='lower'
            )
        else:
            im = ax.imshow(np.asarray(img), cmap=cmap, origin='lower')
        ax.set_title(title, fontsize=11)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(
        f'Flagship: Roman-like joint F087 + Halpha grism inference at z={Z:.2f} '
        f'(SNR {SNR_BROADBAND:.0f}/{SNR_GRISM:.0f}){config_label}',
        fontsize=13,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def _save_summary_txt(
    output_path: Path,
    result,
    runtime_sec: float,
    true_pars: Dict[str, float],
    joint_nsigma: float,
    sampler_config: Dict[str, int],
):
    """Plain-text dump: sampler stats, R-hat, ESS, per-param recovery."""
    summary = result.get_summary()
    rhats = result.get_rhat() if sampler_config['n_chains'] > 1 else None
    ess = result.get_ess()
    lines = []
    lines.append('=' * 80)
    lines.append('FLAGSHIP TEST SUMMARY')
    lines.append('=' * 80)
    lines.append(f'Sampler:     numpyro NUTS')
    lines.append(
        f'Chains:      {sampler_config["n_chains"]}    '
        f'Warmup: {sampler_config["n_warmup"]}    '
        f'Samples/chain: {sampler_config["n_samples"]}'
    )
    lines.append(f'Runtime:     {runtime_sec:.1f} s')
    lines.append(f'Joint Nsigma: {joint_nsigma:.3f}')
    if result.acceptance_fraction is not None:
        lines.append(f'Acceptance:  {result.acceptance_fraction:.2%}')
    lines.append('')
    lines.append(
        f"{'Parameter':<22} {'truth':>10} {'mean':>10} {'std':>10} "
        f"{'R-hat':>8} {'ESS':>8}  err"
    )
    lines.append('-' * 80)
    for name in result.param_names:
        s = summary[name]
        truth = true_pars.get(name, float('nan'))
        rhat = rhats[name] if rhats is not None else float('nan')
        ess_val = ess[name] if ess is not None else float('nan')
        if abs(truth) > 1e-9:
            err = abs(s['mean'] - truth) / abs(truth)
            err_str = f'{err:.1%}'
        else:
            err_str = 'N/A'
        lines.append(
            f'{name:<22} {truth:>10.4f} {s["mean"]:>10.4f} {s["std"]:>10.4f} '
            f'{rhat:>8.3f} {ess_val:>8.0f}  {err_str}'
        )
    lines.append('=' * 80)
    output_path.write_text('\n'.join(lines))


class TestFlagship:
    """Top-level visual regression: joint Roman-like broadband + grism inference."""

    def test_recover_joint_phot_grism(self, output_dir, request):
        """End-to-end: synth Roman-like data, run NUTS, validate via joint Nsigma."""
        long_mode = request.config.getoption('--flagship-long')
        production = request.config.getoption('--flagship-production')
        sampler_config = LONG_CONFIG if long_mode else SHORT_CONFIG
        # filename suffix + plot labels distinguish the two configs' outputs in
        # the shared 00_flagship/ dir; dev keeps its historical filenames.
        suffix = '_production' if production else ''
        plot_test_name = 'flagship_production' if production else 'flagship'
        config_label = ' -- production (2 band + 4 roll)' if production else ''
        print(
            f"\nFlagship config: {'production' if production else 'dev'} obs, "
            f"{'long' if long_mode else 'short'} sampler -> {sampler_config}"
        )

        source, priors, pars_dotted, image_channels, grism_channels = _build_config(
            production
        )
        image_obs = {b: ch['obs'] for b, ch in image_channels.items()}
        grism_obs = {k: ch['obs'] for k, ch in grism_channels.items()}
        print(
            f"Channels: {len(image_obs)} broadband ({', '.join(image_obs)}), "
            f"{len(grism_obs)} grism roll(s) ({', '.join(grism_obs)})"
        )

        image_pars = ImagePars(
            shape=IMAGE_SHAPE, pixel_scale=PIXEL_SCALE, indexing='ij'
        )

        # Truth-side intrinsic velocity + F087 intensity images (for the hero plot)
        vel_obs_clean = build_velocity_obs(image_pars)
        img_obs_clean = build_image_obs(image_pars, broadband_key='F087')
        velocity_true = np.asarray(source.render_velocity(pars_dotted, vel_obs_clean))
        intensity_true = np.asarray(
            source.render_broadband(pars_dotted, img_obs_clean, 'F087')
        )

        # ====================================================================
        # InferenceTask
        # ====================================================================
        task = InferenceTask.from_obs(
            source,
            priors,
            image_obs=image_obs,
            grism_obs=grism_obs,
        )

        # ====================================================================
        # Hero plot (before sampling, so something useful exists even on failure)
        # F087 + roll0 are shown as representatives; every channel gets its own
        # comparison panel below.
        # ====================================================================
        ch_F087 = image_channels['F087']
        ch_roll0 = grism_channels['roll0']
        _plot_ground_truth_overview(
            data_F087_noisy=ch_F087['noisy'],
            data_F087_true=ch_F087['true'],
            data_grism_noisy=ch_roll0['noisy'],
            data_grism_true=ch_roll0['true'],
            velocity_true=velocity_true,
            intensity_true=intensity_true,
            output_path=output_dir / f'ground_truth_overview{suffix}.png',
            config_label=config_label,
        )

        # ====================================================================
        # Run NUTS
        # ====================================================================

        config = NumpyroSamplerConfig(
            n_samples=sampler_config['n_samples'],
            n_warmup=sampler_config['n_warmup'],
            n_chains=sampler_config['n_chains'],
            chain_method=sampler_config['chain_method'],
            seed=42,
            progress=False,
            reparam_strategy='prior',
            dense_mass=True,  # (inert under precondition='laplace', which uses
            # the fixed inv-Hessian metric; kept for the non-preconditioned path)
            precondition='laplace',  # MAP init + fixed Laplace mass -> ~5x faster
            # warmup on this correlated joint posterior (see flagship_speedup)
            target_accept_prob=0.8,
            max_tree_depth=sampler_config['max_tree_depth'],
            init_strategy='prior',  # narrow priors are centered on truth
        )
        sampler = build_sampler('numpyro', task, config)
        start = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = sampler.run()
        runtime = time.time() - start
        print(f'\nFlagship NUTS run: {result.n_samples} samples in {runtime:.1f}s')

        # ====================================================================
        # Recovery diagnostics
        # ====================================================================

        # MAP from samples
        map_idx = int(np.argmax(result.log_prob))
        map_pars_dotted = {
            name: float(result.samples[map_idx, i])
            for i, name in enumerate(result.param_names)
        }
        # Merge truth fixed params (priors fixed values) so render has full keys
        for name, val in task.fixed_params.items():
            map_pars_dotted.setdefault(name, val)

        # Per-channel data-comparison panels (one per band + one per roll).
        # dev keeps its historical data_type labels ('grism', not 'grism_roll0').
        for band_key, ch in image_channels.items():
            model_eval = np.asarray(
                source.render_broadband(map_pars_dotted, ch['obs'], band_key)
            )
            plot_data_comparison_panels(
                data_noisy=ch['noisy'],
                data_true=ch['true'],
                model_eval=model_eval,
                test_name=plot_test_name,
                output_dir=output_dir,
                data_type=f'{band_key}_broadband',
                variance=ch['variance'],
                n_params=task.n_params,
            )
        for roll_key, ch in grism_channels.items():
            model_eval = np.asarray(source.render_grism(map_pars_dotted, ch['obs']))
            data_type = f'grism_{roll_key}' if production else 'grism'
            plot_data_comparison_panels(
                data_noisy=ch['noisy'],
                data_true=ch['true'],
                model_eval=model_eval,
                test_name=plot_test_name,
                output_dir=output_dir,
                data_type=data_type,
                variance=ch['variance'],
                n_params=task.n_params,
            )

        # Full corner (large, but readable on monitor)
        sampler_info = {
            'name': 'numpyro NUTS',
            'runtime': runtime,
            'settings': {
                'chains': sampler_config['n_chains'],
                'warmup': sampler_config['n_warmup'],
                'samples': sampler_config['n_samples'],
                'n_bands': len(image_obs),
                'n_rolls': len(grism_obs),
                'SNR_broadband': SNR_BROADBAND,
                'SNR_grism': SNR_GRISM,
                'z': Z,
            },
        }
        # smooth/smooth1d (bin units) de-jag the marginals for presentation; mild
        # (1.0) so structure is preserved -- these are slide/paper assets.
        fig_corner = plot_corner(
            result,
            true_values=pars_dotted,
            map_values=map_pars_dotted,
            sampler_info=sampler_info,
            smooth=1.0,
            smooth1d=1.0,
        )
        fig_corner.savefig(
            output_dir / f'corner_full{suffix}.png', dpi=120, bbox_inches='tight'
        )
        plt.close(fig_corner)

        # Headline subset corner (presentation-ready)
        fig_headline = plot_corner(
            result,
            params=HEADLINE_PARAMS,
            true_values=pars_dotted,
            map_values=map_pars_dotted,
            sampler_info=sampler_info,
            include_derived=True,
            smooth=1.0,
            smooth1d=1.0,
        )
        fig_headline.savefig(
            output_dir / f'corner_headline{suffix}.png', dpi=150, bbox_inches='tight'
        )
        plt.close(fig_headline)

        # Truth-vs-recovered + joint Nsigma
        fig_recov, recovery_stats = plot_recovery(
            result,
            pars_dotted,
            output_path=output_dir / f'truth_vs_recovered{suffix}.png',
            sampler_name='numpyro NUTS',
        )
        plt.close(fig_recov)

        # Trace
        fig_trace = plot_trace(result)
        fig_trace.savefig(
            output_dir / f'trace{suffix}.png', dpi=120, bbox_inches='tight'
        )
        plt.close(fig_trace)

        # Text summary (truth, mean, std, R-hat, ESS, error)
        joint_nsigma = float(recovery_stats['joint_nsigma'])
        _save_summary_txt(
            output_dir / f'summary{suffix}.txt',
            result,
            runtime_sec=runtime,
            true_pars=pars_dotted,
            joint_nsigma=joint_nsigma,
            sampler_config=sampler_config,
        )

        # Machine-readable posterior stats (per-param mean/std) for the dev-vs-
        # production sensitivity comparison. Written every run; the production
        # run additionally emits the comparison when the dev stats exist.
        posterior_summary = result.get_summary()
        posterior_stats = {
            name: {
                'truth': float(pars_dotted.get(name, float('nan'))),
                'mean': float(posterior_summary[name]['mean']),
                'std': float(posterior_summary[name]['std']),
            }
            for name in result.param_names
        }
        (output_dir / f'posterior_stats{suffix}.json').write_text(
            json.dumps(posterior_stats, indent=2)
        )
        if production:
            dev_stats_path = output_dir / 'posterior_stats.json'
            if dev_stats_path.exists():
                dev_stats = json.loads(dev_stats_path.read_text())
                _write_sensitivity_comparison(
                    output_dir / 'sensitivity_comparison.txt',
                    dev_stats,
                    posterior_stats,
                )
                print(
                    'Wrote sensitivity_comparison.txt (dev vs production '
                    'posterior widths)'
                )
            else:
                print(
                    'No dev posterior_stats.json found; run the dev config '
                    'first to get sensitivity_comparison.txt'
                )

        # ====================================================================
        # Pass criterion (single, from existing diagnostic convention)
        # ====================================================================
        config_name = 'production' if production else 'dev'
        print(f'Joint Nsigma ({config_name}) = {joint_nsigma:.3f}')
        if joint_nsigma > 3.0:
            pytest.fail(
                f'Flagship recovery failed ({config_name} config): joint Nsigma = '
                f'{joint_nsigma:.2f} > 3.0. See {output_dir} for diagnostics.'
            )
        elif joint_nsigma > 2.0:
            warnings.warn(
                f'Flagship recovery marginal ({config_name} config): joint Nsigma = '
                f'{joint_nsigma:.2f} (>2, <=3). See {output_dir} for diagnostics.'
            )
