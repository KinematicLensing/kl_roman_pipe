"""GalSim-free task builders for the Vista GH200 benchmark kit.

Reconstructs the two canonical benchmark configs used throughout the
production-speedups campaign, with the same shapes, priors, and grids:

  Q (quick-dev / flagship): 1 broadband (F087) + 1 grism roll + Halpha
      (+continuum), 32x32 @ 0.11", oversample 3, Nlam ~25.
      Mirrors experiments/sweverett/production_speedups/profile_flagship.py.
  P (production): 2 broadband (F087, F158) + 4 rolls (0/45/90/135 deg,
      rotated WCS) + Halpha + [NII] doublet blend, Nlam ~32.
      Mirrors profile_post_a3.py:build_production_setup.

Differences vs those profile scripts (both benchmark-neutral, adopted so no
galsim install is needed on Vista):
  1. PSF: GaussianPSFShim (psf_numpy.py) instead of galsim.Gaussian.
     Identical kernel grid sizes; kernel values match galsim to ~1e-5
     (galsim's own drawImage FFT accuracy -- the analytic kernel is exact).
  2. Broadband synthetic data: rendered with the model itself
     (source.render_broadband at truth) + Gaussian noise at the
     matched-filter variance (||I||^2 / SNR^2), instead of
     kl_pipe.synthetic.SyntheticIntensity (module imports galsim).
     Same variance convention as noise.add_intensity_noise; data realism
     is irrelevant for timing.

IMPORTANT import-order contract: import this module only AFTER jax
precision config is finalized (bench_matrix.py handles this). Importing
kl_pipe.source force-enables x64 at import; in fp32 mode bench_matrix
intercepts that (see _install_fp32_intercept there).
"""

from __future__ import annotations

import dataclasses

import numpy as np

from psf_numpy import (
    GaussianPSFShim,
    install_galsim_stub,
    install_psf_patch,
)

# install runtime shims BEFORE any kl_pipe import below
install_galsim_stub()

import jax.numpy as jnp  # noqa: E402

install_psf_patch()  # needs kl_pipe.psf importable; galsim stub already in

from astropy.wcs import WCS  # noqa: E402

from kl_pipe.dispersion import GrismPars, build_grism_pars_for_line  # noqa: E402
from kl_pipe.intensity import InclinedExponentialModel  # noqa: E402
from kl_pipe.lines import LINE_LAMBDAS, EmissionLine  # noqa: E402
from kl_pipe.observation import build_grism_obs, build_image_obs  # noqa: E402
from kl_pipe.parameters import ImagePars  # noqa: E402
from kl_pipe.priors import Gaussian, PriorDict, TruncatedNormal  # noqa: E402
from kl_pipe.render import RenderConfig  # noqa: E402
from kl_pipe.sampling import InferenceTask  # noqa: E402
from kl_pipe.source import SourceModel  # noqa: E402
from kl_pipe.velocity import CenteredVelocityModel  # noqa: E402

# ---------------------------------------------------------------------------
# flagship constants (tests/test_flagship.py)
# ---------------------------------------------------------------------------
Z = 1.0
PIXEL_SCALE = 0.11
IMAGE_SHAPE = (32, 32)
PSF_FWHM = 0.18
SNR_BROADBAND = 100.0
SNR_GRISM = 150.0
GRISM_DISPERSION_NM_PER_PIX = 1.1
SPATIAL_OVERSAMPLE = 3

ROLL_ANGLES_DEG = (0.0, 45.0, 90.0, 135.0)

BLEND_REST_NM = (
    LINE_LAMBDAS['NII6548'],
    LINE_LAMBDAS['Halpha'],
    LINE_LAMBDAS['NII6584'],
)

# additive perturbation scales (~0.3-0.5 prior sigma); superset of Q and P
PERTURB_SCALES = {
    'cosi': 0.05,
    'theta_int': 0.10,
    'g1': 0.012,
    'g2': 0.012,
    'vel.v0': 3.0,
    'vel.vcirc': 15.0,
    'vel.rscale': 0.03,
    'F087.flux': 7.0,
    'F087.rscale': 0.025,
    'F087.x0': 0.03,
    'F087.y0': 0.03,
    'F158.flux': 7.0,
    'F158.rscale': 0.025,
    'F158.x0': 0.03,
    'F158.y0': 0.03,
    'Halpha.flux': 7.0,
    'Halpha.rscale': 0.025,
    'Halpha.x0': 0.03,
    'Halpha.y0': 0.03,
    'Halpha.dispersion': 8.0,
    'Halpha.cont.flux_per_nm': 5.0,
    'NII6584.flux': 3.0,
}


def true_pars_flagship():
    return {
        'cosi': 0.6,
        'theta_int': np.pi / 4,
        'g1': 0.02,
        'g2': -0.01,
        'vel.v0': 10.0,
        'vel.vcirc': 200.0,
        'vel.rscale': 0.3,
        'F087.flux': 100.0,
        'F087.rscale': 0.3,
        'F087.h_over_r': 0.1,
        'F087.x0': 0.0,
        'F087.y0': 0.0,
        'Halpha.flux': 100.0,
        'Halpha.rscale': 0.25,
        'Halpha.h_over_r': 0.1,
        'Halpha.x0': 0.0,
        'Halpha.y0': 0.0,
        'Halpha.dispersion': 50.0,
        'Halpha.cont.flux_per_nm': 25.0,
        'Halpha.cont.rscale': 0.25,
        'Halpha.cont.h_over_r': 0.1,
        'Halpha.cont.x0': 0.0,
        'Halpha.cont.y0': 0.0,
        'z': Z,
    }


def priors_flagship(true):
    return PriorDict(
        {
            'cosi': TruncatedNormal(0.6, 0.15, 0.05, 0.99),
            'theta_int': TruncatedNormal(np.pi / 4, 0.3, 0.0, np.pi / 2),
            'g1': TruncatedNormal(0.0, 0.04, -0.1, 0.1),
            'g2': TruncatedNormal(0.0, 0.04, -0.1, 0.1),
            'vel.v0': Gaussian(10.0, 10.0),
            'vel.vcirc': TruncatedNormal(200.0, 50.0, 80.0, 400.0),
            'vel.rscale': TruncatedNormal(0.3, 0.1, 0.05, 1.0),
            'F087.flux': TruncatedNormal(100.0, 20.0, 30.0, 250.0),
            'F087.rscale': TruncatedNormal(0.3, 0.08, 0.05, 1.0),
            'F087.h_over_r': 0.1,
            'F087.x0': TruncatedNormal(0.0, 0.1, -0.5, 0.5),
            'F087.y0': TruncatedNormal(0.0, 0.1, -0.5, 0.5),
            'Halpha.flux': TruncatedNormal(100.0, 20.0, 30.0, 250.0),
            'Halpha.rscale': TruncatedNormal(0.25, 0.08, 0.05, 1.0),
            'Halpha.h_over_r': 0.1,
            'Halpha.x0': TruncatedNormal(0.0, 0.1, -0.5, 0.5),
            'Halpha.y0': TruncatedNormal(0.0, 0.1, -0.5, 0.5),
            'Halpha.dispersion': TruncatedNormal(50.0, 20.0, 5.0, 150.0),
            'Halpha.cont.flux_per_nm': TruncatedNormal(25.0, 15.0, 0.0, 200.0),
            'Halpha.cont.rscale': true['Halpha.cont.rscale'],
            'Halpha.cont.h_over_r': true['Halpha.cont.h_over_r'],
            'Halpha.cont.x0': true['Halpha.cont.x0'],
            'Halpha.cont.y0': true['Halpha.cont.y0'],
            'z': Z,
        }
    )


# ---------------------------------------------------------------------------
# obs builders (galsim-free)
# ---------------------------------------------------------------------------


def _make_band_obs(source, true_pars, psf, image_pars, band_key, seed):
    """Broadband obs: model-rendered truth + Gaussian noise at matched-filter
    variance (||I||^2 / SNR^2 -- same convention as noise.add_intensity_noise).
    """
    int_model = source.broadband_models[band_key]
    obs_clean = build_image_obs(
        image_pars,
        psf=psf,
        render_config=RenderConfig(oversample=SPATIAL_OVERSAMPLE),
        int_model=int_model,
        broadband_key=band_key,
    )
    data_true = np.asarray(source.render_broadband(true_pars, obs_clean, band_key))
    var = float(np.sum(data_true**2)) / SNR_BROADBAND**2
    rng = np.random.default_rng(seed)
    data_noisy = data_true + rng.normal(0.0, np.sqrt(var), size=data_true.shape)
    return build_image_obs(
        image_pars,
        psf=psf,
        render_config=RenderConfig(oversample=SPATIAL_OVERSAMPLE),
        data=jnp.asarray(data_noisy),
        variance=var,
        int_model=int_model,
        broadband_key=band_key,
    )


def build_flagship_task(data_seed_offset=0):
    """Q config. Returns (source, priors, task, obs_F087, obs_grism, true).

    data_seed_offset shifts every mock-noise seed: offset k builds the SAME
    scene with a DIFFERENT noise realization. Distinct data changes the
    constants baked into the jitted posterior, so distinct offsets model the
    per-galaxy recompile cost a production ensemble pays.
    """
    true = true_pars_flagship()
    image_pars = ImagePars(shape=IMAGE_SHAPE, pixel_scale=PIXEL_SCALE, indexing='ij')
    psf = GaussianPSFShim(fwhm=PSF_FWHM)

    grism_pars = build_grism_pars_for_line(
        LINE_LAMBDAS['Halpha'],
        redshift=Z,
        image_pars=image_pars,
        dispersion=GRISM_DISPERSION_NM_PER_PIX,
    )

    source = SourceModel(
        velocity_model=CenteredVelocityModel(),
        broadband_models={'F087': InclinedExponentialModel()},
        emission_lines={
            'Halpha': EmissionLine(
                intensity=InclinedExponentialModel(),
                continuum=InclinedExponentialModel(),
            )
        },
    )

    obs_F087 = _make_band_obs(
        source, true, psf, image_pars, 'F087', seed=42 + data_seed_offset
    )

    grism_obs_clean = build_grism_obs(grism_pars, z=Z, psf=psf)
    data_grism_true = np.asarray(source.render_grism(true, grism_obs_clean))
    var_grism = float(np.sum(data_grism_true**2)) / SNR_GRISM**2
    rng = np.random.default_rng(43 + data_seed_offset)
    data_grism_noisy = data_grism_true + rng.normal(
        0.0, np.sqrt(var_grism), size=data_grism_true.shape
    )
    obs_grism = build_grism_obs(
        grism_pars,
        z=Z,
        psf=psf,
        render_config=RenderConfig(oversample=SPATIAL_OVERSAMPLE),
        data=jnp.asarray(data_grism_noisy),
        variance=var_grism,
    )

    priors = priors_flagship(true)
    task = InferenceTask.from_obs(
        source,
        priors,
        image_obs={'F087': obs_F087},
        grism_obs={'roll0': obs_grism},
    )
    return source, priors, task, obs_F087, obs_grism, true


# ---------------------------------------------------------------------------
# production config P
# ---------------------------------------------------------------------------


def _wcs_with_pc(shape, pixel_scale, rotation_radians):
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


def true_pars_production():
    true = true_pars_flagship()
    true.update(
        {
            'F158.flux': 120.0,
            'F158.rscale': 0.35,
            'F158.h_over_r': 0.1,
            'F158.x0': 0.0,
            'F158.y0': 0.0,
            'NII6584.flux': 30.0,
            'NII6548.flux': 10.2,  # ~0.34 x NII6584 (fixed doublet ratio)
        }
    )
    return true


def priors_production(true):
    base = priors_flagship(true)
    d = dict(base._param_spec)  # private; acceptable in a benchmark kit
    d.update(
        {
            'F158.flux': TruncatedNormal(120.0, 25.0, 30.0, 300.0),
            'F158.rscale': TruncatedNormal(0.35, 0.08, 0.05, 1.0),
            'F158.h_over_r': 0.1,
            'F158.x0': TruncatedNormal(0.0, 0.1, -0.5, 0.5),
            'F158.y0': TruncatedNormal(0.0, 0.1, -0.5, 0.5),
            'NII6584.flux': TruncatedNormal(30.0, 10.0, 0.0, 150.0),
            'NII6548.flux': true['NII6548.flux'],
        }
    )
    return PriorDict(d)


def _build_production_source():
    return SourceModel(
        velocity_model=CenteredVelocityModel(),
        broadband_models={
            'F087': InclinedExponentialModel(),
            'F158': InclinedExponentialModel(),
        },
        emission_lines={
            'Halpha': EmissionLine(
                intensity=InclinedExponentialModel(),
                continuum=InclinedExponentialModel(),
            ),
            'NII6548': EmissionLine(intensity_key='Halpha', dispersion_key='Halpha'),
            'NII6584': EmissionLine(intensity_key='Halpha', dispersion_key='Halpha'),
        },
    )


def _build_roll_obs(source, true_pars, psf, rotation_rad, seed, window_kms=3000.0):
    """One production grism obs: rotated WCS + [NII]-blend window + data."""
    wcs = _wcs_with_pc(IMAGE_SHAPE, PIXEL_SCALE, rotation_rad)
    ip = ImagePars(shape=IMAGE_SHAPE, wcs=wcs, indexing='ij')
    gp = GrismPars(
        image_pars=ip,
        dispersion=GRISM_DISPERSION_NM_PER_PIX,
        lambda_ref=LINE_LAMBDAS['Halpha'] * (1.0 + Z),
        dispersion_angle_detector=0.0,
    )
    blend_cp = gp.to_cube_pars(
        Z, velocity_window_kms=window_kms, line_lambdas_rest=BLEND_REST_NM
    )
    obs_clean = build_grism_obs(gp, z=Z, psf=psf)
    # known API gap: build_grism_obs hard-codes the Halpha-only window;
    # widen to the blend window (same workaround as profile_post_a3.py)
    obs_clean = dataclasses.replace(obs_clean, cube_pars=blend_cp)
    data_true = np.asarray(source.render_grism(true_pars, obs_clean))
    var = float(np.sum(data_true**2)) / SNR_GRISM**2
    rng = np.random.default_rng(seed)
    data_noisy = data_true + rng.normal(0.0, np.sqrt(var), size=data_true.shape)
    obs = build_grism_obs(
        gp,
        z=Z,
        psf=psf,
        render_config=RenderConfig(oversample=SPATIAL_OVERSAMPLE),
        data=jnp.asarray(data_noisy),
        variance=var,
    )
    return dataclasses.replace(obs, cube_pars=blend_cp)


def build_production_setup(window_kms=3000.0, data_seed_offset=0):
    """P config. Returns (source, priors, image_obs, grism_obs, true).

    data_seed_offset: see build_flagship_task.
    """
    true = true_pars_production()
    source = _build_production_source()
    psf = GaussianPSFShim(fwhm=PSF_FWHM)
    image_pars = ImagePars(shape=IMAGE_SHAPE, pixel_scale=PIXEL_SCALE, indexing='ij')

    image_obs = {
        'F087': _make_band_obs(
            source, true, psf, image_pars, 'F087', seed=42 + data_seed_offset
        ),
        'F158': _make_band_obs(
            source, true, psf, image_pars, 'F158', seed=52 + data_seed_offset
        ),
    }
    grism_obs = {
        f'roll{i}': _build_roll_obs(
            source,
            true,
            psf,
            np.deg2rad(a),
            seed=100 + i + data_seed_offset,
            window_kms=window_kms,
        )
        for i, a in enumerate(ROLL_ANGLES_DEG)
    }
    priors = priors_production(true)
    return source, priors, image_obs, grism_obs, true


def build_production_task(window_kms=3000.0, data_seed_offset=0):
    source, priors, image_obs, grism_obs, true = build_production_setup(
        window_kms, data_seed_offset=data_seed_offset
    )
    task = InferenceTask.from_obs(
        source, priors, image_obs=image_obs, grism_obs=grism_obs
    )
    return source, priors, task, image_obs, grism_obs, true


# ---------------------------------------------------------------------------
# perturbed evaluation point (never benchmark gradients AT truth)
# ---------------------------------------------------------------------------


def perturbed_theta(priors, true_pars, seed=7):
    sampled_names = tuple(priors.sampled_names)
    theta_true = np.array([true_pars[n] for n in sampled_names], dtype=np.float64)
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=len(sampled_names))
    offsets = np.array([PERTURB_SCALES[n] for n in sampled_names])
    return jnp.asarray(theta_true + signs * offsets), sampled_names
