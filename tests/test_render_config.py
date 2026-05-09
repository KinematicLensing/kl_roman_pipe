"""Regression tests for RenderConfig + obs single source of truth (Bug B).

Pins the structural fix from PR #41 (Commit 6, ee2f171): obs.render_config
is the single source of truth for grid sizing, and InferenceTask refuses to
silently drift from it.

Each test pins a specific failure mode that would have triggered the
Issue #42 crash under the old architecture.
"""

import jax

jax.config.update('jax_enable_x64', True)

import galsim
import jax.numpy as jnp
import numpy as np
import pytest

from kl_pipe.intensity import InclinedExponentialModel
from kl_pipe.observation import build_image_obs
from kl_pipe.parameters import ImagePars
from kl_pipe.pixel import BoxPixel
from kl_pipe.priors import LogUniform, PriorDict, Uniform
from kl_pipe.render import RenderConfig
from kl_pipe.sampling import InferenceTask


@pytest.fixture
def setup():
    """Build the Issue #42 reproducer state."""
    Nbins = 64
    pixel_scale = 0.1
    image_pars = ImagePars(shape=(Nbins, Nbins), pixel_scale=pixel_scale, indexing='ij')
    psf = galsim.Gaussian(fwhm=pixel_scale * 5)
    model = InclinedExponentialModel()
    rng = np.random.default_rng(42)
    data = rng.normal(size=image_pars.shape)
    variance = np.ones_like(data)
    return image_pars, psf, model, data, variance


@pytest.fixture
def tight_priors():
    """Priors that imply oversample > the default (5)."""
    return PriorDict(
        {
            'cosi': Uniform(0.01, 0.99),
            'theta_int': Uniform(0, np.pi),
            'flux': LogUniform(0.01, 1000.0),
            'int_rscale': Uniform(1.0, 2.0),
            'int_h_over_r': 0.2,
            'g1': 0.0,
            'g2': 0.0,
            'int_x0': 0.0,
            'int_y0': 0.0,
        }
    )


@pytest.fixture
def loose_priors():
    """Priors that fit within the default (5) oversample."""
    return PriorDict(
        {
            'cosi': Uniform(0.5, 0.99),
            'theta_int': Uniform(0, np.pi),
            'flux': LogUniform(0.01, 1000.0),
            'int_rscale': Uniform(2.0, 4.0),
            'int_h_over_r': 0.2,
            'g1': 0.0,
            'g2': 0.0,
            'int_x0': 0.0,
            'int_y0': 0.0,
        }
    )


def test_obs_rc_is_single_source_of_truth(setup):
    """obs.oversample property is read from obs.render_config — no drift possible."""
    image_pars, psf, model, data, variance = setup

    # default rc
    obs = build_image_obs(
        image_pars,
        psf=psf,
        data=jnp.array(data),
        variance=jnp.array(variance),
        int_model=model,
    )
    assert obs.oversample == obs.render_config.oversample
    assert obs.render_config is not None

    # explicit rc
    rc = RenderConfig(oversample=11)
    obs2 = build_image_obs(
        image_pars,
        psf=psf,
        data=jnp.array(data),
        variance=jnp.array(variance),
        int_model=model,
        render_config=rc,
    )
    assert obs2.oversample == 11
    assert obs2.render_config.oversample == 11


def test_obs_kspace_psf_fft_shape_matches_rc(setup):
    """PSF FFT shape is consistent with rc.oversample (wrap divisibility)."""
    image_pars, psf, model, data, variance = setup
    rc = RenderConfig(oversample=7)
    obs = build_image_obs(
        image_pars,
        psf=psf,
        data=jnp.array(data),
        variance=jnp.array(variance),
        int_model=model,
        render_config=rc,
    )
    # wrap path inside _kspace_render_core requires this divisibility
    pad_row, pad_col = obs.kspace_psf_fft.shape
    assert pad_row % rc.oversample == 0
    assert pad_col % rc.oversample == 0


def test_issue_42_does_not_crash(setup, tight_priors):
    """Issue #42 reproducer — must not crash mid-JIT.

    Old code: build obs with default oversample=5; InferenceTask recomputes
    oversample=13 from priors; PSF FFT (640, 640) cannot reshape to
    (64, 13, 64, 13) (832, 832); TypeError mid-JIT.

    New code: InferenceTask reads rc from obs and validates against priors;
    raises ValueError with rebuild instructions if priors imply more
    demanding rc than obs was built for.
    """
    image_pars, psf, model, data, variance = setup
    obs = build_image_obs(
        image_pars,
        psf=psf,
        data=jnp.array(data),
        variance=jnp.array(variance),
        int_model=model,
    )

    # tight priors imply oversample > 5; expect loud raise, not crash mid-JIT
    with pytest.raises(ValueError, match='Priors imply oversample'):
        InferenceTask.from_intensity_obs(model, tight_priors, obs)


def test_priors_tighter_than_obs_rc_ok(setup, loose_priors):
    """Priors that fit within obs's pre-built rc — task constructs cleanly."""
    image_pars, psf, model, data, variance = setup
    obs = build_image_obs(
        image_pars,
        psf=psf,
        data=jnp.array(data),
        variance=jnp.array(variance),
        int_model=model,
    )
    # priors imply small oversample; obs was built with default 5 -> fine
    task = InferenceTask.from_intensity_obs(model, loose_priors, obs)
    assert task is not None
    # task uses the obs's rc (not a recomputed one)
    assert task._render_configs['intensity'].oversample == obs.render_config.oversample


def test_priors_wider_than_obs_rc_raises(setup, tight_priors):
    """Priors that demand larger rc than obs was built for — loud failure."""
    image_pars, psf, model, data, variance = setup
    # build obs with default oversample=5; tight priors will require more
    obs = build_image_obs(
        image_pars,
        psf=psf,
        data=jnp.array(data),
        variance=jnp.array(variance),
        int_model=model,
    )
    with pytest.raises(ValueError) as excinfo:
        InferenceTask.from_intensity_obs(model, tight_priors, obs)
    msg = str(excinfo.value)
    assert 'oversample' in msg
    # message must point at the fix (for_priors)
    assert 'for_priors' in msg


def test_for_priors_obs_construction_works(setup, tight_priors):
    """Recommended fix from Issue #42 raise message — verify it actually works."""
    image_pars, psf, model, data, variance = setup
    rc = RenderConfig.for_priors(
        model,
        tight_priors,
        image_pars.pixel_scale,
        pixel_response=BoxPixel(image_pars.pixel_scale),
    )
    obs = build_image_obs(
        image_pars,
        psf=psf,
        data=jnp.array(data),
        variance=jnp.array(variance),
        int_model=model,
        render_config=rc,
    )
    # task construction succeeds
    task = InferenceTask.from_intensity_obs(model, tight_priors, obs)
    # likelihood evaluates without crashing
    theta_test = jnp.array([0.3, 0.0, 0.0, 0.0, 12.0, 1.5, 0.2, 0.0, 0.0])
    log_prob = task.likelihood_fn(theta_test)
    assert jnp.isfinite(log_prob)
