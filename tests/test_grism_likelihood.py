"""
Tests for the SourceModel grism likelihood via ``InferenceTask.from_obs``.

Covers:
  - Unit tests: likelihood eval + JIT + grad + factory construction for
    grism-only and joint photometry+grism patterns.
  - One smoke optimizer-recovery test for ``Halpha.flux`` + ``vel.vcirc`` +
    ``Halpha.dispersion`` at SNR=1000.

The full joint phot+grism likelihood slice lives in
``test_likelihood_slices.py::test_recover_joint_phot_grism_base`` (B.4b),
which carries the standard slice-grid + per-channel data comparison
panels (broadband + grism) for the joint case.

The smoke recovery test fixes most parameters and optimizes over a small subset
to confirm end-to-end inference works. It is not a tight tolerance gate.

Continuum coverage uses ``EmissionLine(continuum=...)`` with
``Halpha.cont.flux_per_nm=0.05`` fixed. The continuum's spatial parameters are fixed
equal to the line's own spatial parameters (i.e. the continuum and the line
share a single spatial profile in this test).
"""

import os
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import pytest
import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

jax.config.update('jax_enable_x64', True)

import galsim

from kl_pipe.parameters import ImagePars
from kl_pipe.velocity import OffsetVelocityModel
from kl_pipe.intensity import InclinedExponentialModel
from kl_pipe.source import SourceModel
from kl_pipe.lines import EmissionLine, LINE_LAMBDAS
from kl_pipe.dispersion import build_grism_pars_for_line
from kl_pipe.observation import build_image_obs, build_grism_obs
from kl_pipe.priors import PriorDict, Uniform
from kl_pipe.sampling.task import InferenceTask
from test_utils import TestConfig

# output directory for diagnostic plots
OUT_DIR = os.path.join(os.path.dirname(__file__), 'out', 'grism_likelihood')
os.makedirs(OUT_DIR, exist_ok=True)


# =============================================================================
# Shared fixtures
# =============================================================================

_IMAGE_PARS = ImagePars(shape=(32, 32), pixel_scale=0.11, indexing='ij')
_Z = 1.0

# Truth parameters in the SourceModel dotted-key namespace.
# Continuum spatial pars are fixed equal to the line's own spatial pars,
# so the continuum and the line share a single spatial profile.
_TRUE_PARS = {
    # shared geometry (unprefixed)
    'cosi': 0.6,
    'theta_int': 0.7,
    'g1': 0.0,
    'g2': 0.0,
    'z': _Z,
    # velocity (vel. prefix)
    'vel.v0': 0.0,
    'vel.vcirc': 200.0,
    'vel.rscale': 0.4,
    'vel.x0': 0.0,
    'vel.y0': 0.0,
    # broadband F087
    'F087.flux': 100.0,
    'F087.rscale': 0.3,
    'F087.h_over_r': 0.15,
    'F087.x0': 0.0,
    'F087.y0': 0.0,
    # Halpha emission line — own spatial profile + continuum
    'Halpha.flux': 100.0,
    'Halpha.rscale': 0.3,
    'Halpha.h_over_r': 0.15,
    'Halpha.x0': 0.0,
    'Halpha.y0': 0.0,
    'Halpha.dispersion': 50.0,
    # Halpha continuum (spatial pars fixed = line spatial pars)
    'Halpha.cont.flux_per_nm': 0.05,
    'Halpha.cont.rscale': 0.3,
    'Halpha.cont.h_over_r': 0.15,
    'Halpha.cont.x0': 0.0,
    'Halpha.cont.y0': 0.0,
}


def _make_priors():
    """Production-typical priors: Halpha.flux, vel.vcirc, Halpha.dispersion
    sampled (Uniform); everything else fixed at truth. Used by the unit
    + slice + recovery tests so the sampled namespace is consistent."""
    priors_dict = {k: float(v) for k, v in _TRUE_PARS.items()}
    priors_dict['Halpha.flux'] = Uniform(20.0, 200.0)
    priors_dict['vel.vcirc'] = Uniform(100.0, 300.0)
    priors_dict['Halpha.dispersion'] = Uniform(20.0, 150.0)
    return PriorDict(priors_dict)


def _theta_sampled_truth(priors):
    """Pack the sampled-truth vector in priors.sampled_names order."""
    return jnp.array([_TRUE_PARS[n] for n in priors.sampled_names])


@pytest.fixture(scope='module')
def test_config():
    """Test configuration matching the convention in test_likelihood_slices."""
    out_dir = Path(__file__).parent / 'out' / 'grism_likelihood'
    config = TestConfig(out_dir, include_poisson_noise=False)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    return config


@pytest.fixture(scope='module')
def source():
    """SourceModel: OffsetVelocity + broadband F087 + Halpha with continuum.

    Per-line continuum exercises the ``EmissionLine(continuum=...)`` code
    path. The continuum carries its own ``Halpha.cont.*`` dotted keys; its
    spatial truth is set equal to the line's own spatial truth in
    ``_TRUE_PARS`` so the continuum and the line share a single spatial
    profile in this test.
    """
    return SourceModel(
        velocity_model=OffsetVelocityModel(),
        broadband_models={'F087': InclinedExponentialModel()},
        emission_lines={
            'Halpha': EmissionLine(
                intensity=InclinedExponentialModel(),
                continuum=InclinedExponentialModel(),
            ),
        },
    )


@pytest.fixture(scope='module')
def priors():
    return _make_priors()


@pytest.fixture(scope='module')
def theta_true_sampled(priors):
    return _theta_sampled_truth(priors)


@pytest.fixture(scope='module')
def roman_psf():
    """Roman-like Gaussian PSF (0.18" FWHM at the Roman grism central λ)."""
    return galsim.Gaussian(fwhm=0.18)


@pytest.fixture(scope='module')
def grism_pars():
    return build_grism_pars_for_line(
        LINE_LAMBDAS['Halpha'],
        redshift=_Z,
        image_pars=_IMAGE_PARS,
        dispersion=1.1,
    )


def _build_grism_synthetic(
    source, pars, grism_pars, psf, snr, seed=0, render_config=None
):
    """Render a clean grism image, add Gaussian noise calibrated to a target
    matched-filter SNR, return (data, variance, obs_with_data)."""
    obs_no_data = build_grism_obs(
        grism_pars, z=_Z, psf=psf, render_config=render_config
    )
    clean = source.render_grism(pars, obs_no_data)
    clean = np.asarray(clean)

    # matched-filter SNR for a known signal vs constant-variance Gaussian noise:
    #   SNR = sqrt(sum(signal^2) / variance)
    # → variance = sum(signal^2) / SNR^2
    signal_power = float(np.sum(clean**2))
    variance = signal_power / snr**2

    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, np.sqrt(variance), size=clean.shape)
    data = clean + noise

    obs = build_grism_obs(
        grism_pars,
        z=_Z,
        psf=psf,
        data=jnp.asarray(data),
        variance=float(variance),
        render_config=render_config,
    )
    return jnp.asarray(data), float(variance), obs


def _build_intensity_synthetic(source, pars, psf, snr, seed=1, band_key='F087'):
    """Render a clean broadband intensity image and add Gaussian noise."""
    obs_no_data = build_image_obs(_IMAGE_PARS, psf=psf, broadband_key=band_key)
    clean = source.render_broadband(pars, obs_no_data, band_key=band_key)
    clean = np.asarray(clean)

    signal_power = float(np.sum(clean**2))
    variance = signal_power / snr**2

    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, np.sqrt(variance), size=clean.shape)
    data = clean + noise

    obs = build_image_obs(
        _IMAGE_PARS,
        psf=psf,
        data=jnp.asarray(data),
        variance=float(variance),
        broadband_key=band_key,
    )
    return jnp.asarray(data), float(variance), obs


@pytest.fixture(scope='module')
def grism_obs_high_snr(source, grism_pars, roman_psf):
    """High-SNR grism synthetic obs (SNR=1000) — used for unit + recovery tests."""
    _, _, obs = _build_grism_synthetic(
        source, _TRUE_PARS, grism_pars, roman_psf, snr=1000, seed=0
    )
    return obs


@pytest.fixture(scope='module')
def image_obs_high_snr(source, roman_psf):
    """High-SNR broadband intensity synthetic obs (SNR=1000)."""
    _, _, obs = _build_intensity_synthetic(
        source, _TRUE_PARS, roman_psf, snr=1000, seed=1
    )
    return obs


# =============================================================================
# Unit tests: eval / JIT / grad / factories
# =============================================================================


class TestGrismLikelihoodUnits:
    """Smoke tests confirming the SourceModel grism likelihood evaluates,
    JIT-compiles, and produces finite gradients."""

    def test_log_likelihood_grism_evaluates(
        self, source, priors, theta_true_sampled, grism_obs_high_snr
    ):
        """log_like at truth is finite."""
        task = InferenceTask.from_obs(
            source, priors, grism_obs={'roll0': grism_obs_high_snr}
        )
        log_l = task.likelihood_fn(theta_true_sampled)
        assert jnp.isfinite(log_l), f"log_like not finite: {log_l}"

    def test_log_likelihood_grism_jit_compiles(
        self, source, priors, theta_true_sampled, grism_obs_high_snr
    ):
        """task.likelihood_fn is already JIT-compiled; re-invoke to confirm."""
        task = InferenceTask.from_obs(
            source, priors, grism_obs={'roll0': grism_obs_high_snr}
        )
        log_l = task.likelihood_fn(theta_true_sampled)
        assert jnp.isfinite(log_l)

    def test_log_likelihood_grism_grad_finite(
        self, source, priors, theta_true_sampled, grism_obs_high_snr
    ):
        """jax.grad of grism likelihood produces finite values for all sampled params."""
        task = InferenceTask.from_obs(
            source, priors, grism_obs={'roll0': grism_obs_high_snr}
        )
        grad_fn = jax.grad(task.likelihood_fn)
        grad = grad_fn(theta_true_sampled)
        assert jnp.all(jnp.isfinite(grad)), (
            f"non-finite gradient components: indices "
            f"{jnp.where(~jnp.isfinite(grad))[0].tolist()}"
        )

    def test_from_obs_grism_builds_task(self, source, priors, grism_obs_high_snr):
        """InferenceTask.from_obs builds a working grism-only task with
        finite log_posterior on a prior-sampled theta."""
        task = InferenceTask.from_obs(
            source, priors, grism_obs={'roll0': grism_obs_high_snr}
        )
        theta_sampled = task.sample_prior(jax.random.PRNGKey(0), n_samples=1)[0]
        log_post = task.log_posterior(theta_sampled)
        assert jnp.isfinite(log_post), f"log_posterior not finite: {log_post}"

    def test_from_obs_joint_photometry_grism_builds_task(
        self, source, priors, grism_obs_high_snr, image_obs_high_snr
    ):
        """from_obs builds a joint photometry+grism task with finite log_posterior."""
        task = InferenceTask.from_obs(
            source,
            priors,
            image_obs={'F087': image_obs_high_snr},
            grism_obs={'roll0': grism_obs_high_snr},
        )
        theta_sampled = task.sample_prior(jax.random.PRNGKey(0), n_samples=1)[0]
        log_post = task.log_posterior(theta_sampled)
        assert jnp.isfinite(log_post)


# =============================================================================
# Likelihood slice tests
# =============================================================================


# Note: the joint phot+grism slice test that used to live here was merged
# into test_likelihood_slices.py::test_recover_joint_phot_grism_base, which
# carries the standard slice-grid + data-comparison-panel diagnostic for
# the joint F087 broadband + Halpha grism case.


# =============================================================================
# Smoke optimizer-recovery test
# =============================================================================


def test_grism_optimizer_recovery_smoke(source, grism_pars, roman_psf):
    """Smoke test: recover Halpha.flux + vel.vcirc + Halpha.dispersion at SNR=1000.

    Fixes everything else at truth; optimizes the 3 free params from a
    perturbed initial guess. Confirms end-to-end JAX gradient + scipy
    optimizer path works for grism inference under SourceModel.

    Tolerances are loose (±15% relative). Runs on the slice path, where
    the dispersion response is sharp enough for that bound at SNR=1000
    (the exact analytic model's dispersion noise floor is ~10% here);
    the analytic default's optimizer coverage lives in
    test_optimizer_recovery.py.
    """
    from kl_pipe.render import RenderConfig

    priors = _make_priors()
    _, _, obs_slice = _build_grism_synthetic(
        source,
        _TRUE_PARS,
        grism_pars,
        roman_psf,
        snr=1000,
        seed=0,
        render_config=RenderConfig(oversample=5, dispersal_method='slice'),
    )
    task = InferenceTask.from_obs(source, priors, grism_obs={'roll0': obs_slice})
    log_like_fn = task.likelihood_fn
    grad_fn = jax.jit(jax.grad(log_like_fn))

    sampled_names = list(priors.sampled_names)
    free_names = ['Halpha.flux', 'vel.vcirc', 'Halpha.dispersion']
    free_indices = [sampled_names.index(n) for n in free_names]

    theta_true_sampled = np.array(_theta_sampled_truth(priors))
    # 15% perturbation on free params for the initial guess
    rng = np.random.default_rng(42)
    theta_init = theta_true_sampled.copy()
    for idx in free_indices:
        theta_init[idx] = theta_init[idx] * (1 + 0.15 * rng.standard_normal())

    free_indices_arr = jnp.asarray(free_indices)

    def objective(x_free):
        theta = jnp.asarray(theta_init).at[free_indices_arr].set(jnp.asarray(x_free))
        return -float(log_like_fn(theta))

    def gradient(x_free):
        theta = jnp.asarray(theta_init).at[free_indices_arr].set(jnp.asarray(x_free))
        g = grad_fn(theta)
        return -np.asarray(g)[free_indices]

    x0 = np.array([theta_init[i] for i in free_indices])
    bounds = [(20.0, 200.0), (50.0, 400.0), (10.0, 200.0)]

    result = minimize(
        objective,
        x0=x0,
        method='L-BFGS-B',
        jac=gradient,
        bounds=bounds,
        options={'maxiter': 500, 'ftol': 1e-8},
    )

    assert (
        result.success or result.status == 1
    ), f"optimizer did not converge: {result.message}"

    recovered = {n: float(result.x[i]) for i, n in enumerate(free_names)}
    truth = {n: float(_TRUE_PARS[n]) for n in free_names}

    for name in free_names:
        rel_err = abs(recovered[name] - truth[name]) / abs(truth[name])
        assert rel_err < 0.15, (
            f"{name} recovery {recovered[name]:.3f} vs truth {truth[name]:.3f} "
            f"= {rel_err:.2%} relative error (loose 15% smoke tolerance). "
            f"All free params: recovered={recovered}, truth={truth}"
        )
