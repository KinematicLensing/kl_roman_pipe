"""Chi-squared calibration of the noise bookkeeping, per observation type.

The strict likelihood-slice tests run on noise-free data (they validate
the forward model and its constraining power). This module is the one
place that checks the noise wiring directly: data drawn with a known
variance must give chi-squared per data point close to 1 at the true
parameters, evaluated through the full likelihood machinery. A wiring
bug -- a wrong variance scale, a missing square root, a pixel-count
normalization -- moves chi-squared per point by that factor and fails
loudly. The window is the standard statistical spread of chi-squared
(five sigma of sqrt(2/N)); the seed is pinned, so the check is
deterministic.

Chi-squared is extracted as -2 * (loglike(noisy data) - loglike(clean
data)), both evaluated at truth through the same likelihood, so any
data-independent normalization terms cancel exactly.
"""

import numpy as np
import pytest
import jax.numpy as jnp
import galsim as gs

from kl_pipe.dispersion import build_grism_pars_for_line
from kl_pipe.lines import EmissionLine, LINE_LAMBDAS
from kl_pipe.intensity import InclinedExponentialModel
from kl_pipe.observation import (
    build_grism_obs,
    build_image_obs,
    build_velocity_obs,
)
from kl_pipe.parameters import ImagePars
from kl_pipe.priors import PriorDict, Uniform
from kl_pipe.render import RenderConfig
from kl_pipe.sampling import InferenceTask
from kl_pipe.source import SourceModel
from kl_pipe.velocity import OffsetVelocityModel

_Z = 1.0
_PARS = {
    'cosi': 0.6,
    'theta_int': 0.785,
    'g1': 0.0,
    'g2': 0.0,
    'z': _Z,
    'vel.v0': 10.0,
    'vel.vcirc': 200.0,
    'vel.rscale': 0.4,
    'vel.x0': 0.0,
    'vel.y0': 0.0,
    'F087.flux': 100.0,
    'F087.rscale': 0.3,
    'F087.h_over_r': 0.1,
    'F087.x0': 0.0,
    'F087.y0': 0.0,
    'Halpha.flux': 100.0,
    'Halpha.rscale': 0.2,
    'Halpha.h_over_r': 0.1,
    'Halpha.x0': 0.05,
    'Halpha.y0': 0.0,
    'Halpha.dispersion': 50.0,
}
_SEED = 7


@pytest.fixture(scope='module')
def scene():
    psf = gs.Gaussian(fwhm=0.18)
    image_pars = ImagePars(shape=(32, 32), pixel_scale=0.11, indexing='ij')
    source = SourceModel(
        velocity_model=OffsetVelocityModel(),
        broadband_models={'F087': InclinedExponentialModel()},
        emission_lines={'Halpha': EmissionLine(intensity=InclinedExponentialModel())},
    )
    return image_pars, psf, source


def _priors():
    # one sampled parameter (the likelihood needs a theta); rest fixed
    priors = {k: v for k, v in _PARS.items()}
    priors['vel.vcirc'] = Uniform(100.0, 350.0)
    return PriorDict(priors)


def _chi2_per_point(task_noisy, task_clean, n_points):
    priors = _priors()
    theta = jnp.asarray([_PARS[n] for n in priors.sampled_names])
    ll_noisy = float(task_noisy.log_likelihood(theta))
    ll_clean = float(task_clean.log_likelihood(theta))
    return -2.0 * (ll_noisy - ll_clean) / n_points


def _assert_calibrated(chi2_per_point, n_points, label):
    window = 5.0 * np.sqrt(2.0 / n_points)
    assert abs(chi2_per_point - 1.0) < window, (
        f"{label}: chi2 per data point {chi2_per_point:.4f} outside "
        f"1 +/- {window:.4f}; the noise variance is mis-wired somewhere "
        f"in this channel's likelihood"
    )


def test_image_obs_chi2(scene):
    image_pars, psf, source = scene
    obs0 = build_image_obs(
        image_pars,
        psf=psf,
        int_model=source.broadband_models['F087'],
        broadband_key='F087',
    )
    clean = np.asarray(source.render_broadband(_PARS, obs0, 'F087'))
    var = float((clean**2).sum()) / 1000.0**2
    rng = np.random.default_rng(_SEED)
    noisy = clean + rng.normal(0.0, np.sqrt(var), clean.shape)

    def make_task(data):
        obs = build_image_obs(
            image_pars,
            psf=psf,
            data=jnp.asarray(data),
            variance=var,
            int_model=source.broadband_models['F087'],
            broadband_key='F087',
        )
        return InferenceTask.from_obs(source, _priors(), image_obs={'F087': obs})

    chi2 = _chi2_per_point(make_task(noisy), make_task(clean), clean.size)
    _assert_calibrated(chi2, clean.size, 'image obs')


def test_grism_obs_chi2(scene):
    image_pars, psf, source = scene
    grism_pars = build_grism_pars_for_line(
        LINE_LAMBDAS['Halpha'], redshift=_Z, image_pars=image_pars, dispersion=1.1
    )
    rc = RenderConfig(
        oversample=5, dispersal_method='analytic', line_window_halfwidth=25
    )
    obs0 = build_grism_obs(grism_pars, z=_Z, psf=psf, render_config=rc)
    clean = np.asarray(source.render_grism(_PARS, obs0))
    var = float((clean**2).sum()) / 1000.0**2
    rng = np.random.default_rng(_SEED)
    noisy = clean + rng.normal(0.0, np.sqrt(var), clean.shape)

    def make_task(data):
        obs = build_grism_obs(
            grism_pars,
            z=_Z,
            psf=psf,
            render_config=rc,
            data=jnp.asarray(data),
            variance=var,
        )
        return InferenceTask.from_obs(source, _priors(), grism_obs={'roll0': obs})

    chi2 = _chi2_per_point(make_task(noisy), make_task(clean), clean.size)
    _assert_calibrated(chi2, clean.size, 'grism obs')


def test_velocity_obs_chi2(scene):
    image_pars, psf, source = scene
    from kl_pipe.utils import build_map_grid_from_image_pars

    X, Y = build_map_grid_from_image_pars(image_pars)
    theta_vel = jnp.asarray(
        [
            _PARS['cosi'],
            _PARS['theta_int'],
            _PARS['g1'],
            _PARS['g2'],
            _PARS['vel.v0'],
            _PARS['vel.vcirc'],
            _PARS['vel.rscale'],
            _PARS['vel.x0'],
            _PARS['vel.y0'],
        ]
    )
    clean = np.asarray(source.velocity_model(theta_vel, 'obs', X, Y))
    var = 25.0  # (km/s)^2, arbitrary known value
    rng = np.random.default_rng(_SEED)
    noisy = clean + rng.normal(0.0, np.sqrt(var), clean.shape)

    def make_task(data):
        obs = build_velocity_obs(image_pars, data=jnp.asarray(data), variance=var)
        return InferenceTask.from_obs(source, _priors(), velocity_obs=obs)

    chi2 = _chi2_per_point(make_task(noisy), make_task(clean), clean.size)
    _assert_calibrated(chi2, clean.size, 'velocity obs')


def _shot_noise_variance_map(clean):
    """Physical-style variance map with both terms mattering: background at
    the bg-only matched-filter SNR ~50 level, shot noise scaled so the peak
    pixel's Poisson variance roughly matches the background variance."""
    from kl_pipe.noise import physical_variance_map

    sigma_bg = float(np.sqrt((clean**2).sum())) / 50.0
    electrons_per_flux = float(clean.max()) / sigma_bg**2
    return physical_variance_map(clean, sigma_bg, electrons_per_flux)


def test_image_obs_chi2_variance_map(scene):
    # per-pixel (background + shot) variance through the full likelihood:
    # data drawn from the same map must give chi2/point ~ 1, exactly as the
    # uniform-variance cases above
    from kl_pipe.noise import add_map_noise

    image_pars, psf, source = scene
    obs0 = build_image_obs(
        image_pars,
        psf=psf,
        int_model=source.broadband_models['F087'],
        broadband_key='F087',
    )
    clean = np.asarray(source.render_broadband(_PARS, obs0, 'F087'))
    var_map = _shot_noise_variance_map(clean)
    noisy = add_map_noise(clean, var_map, seed=_SEED)

    def make_task(data):
        obs = build_image_obs(
            image_pars,
            psf=psf,
            data=jnp.asarray(data),
            variance=jnp.asarray(var_map),
            int_model=source.broadband_models['F087'],
            broadband_key='F087',
        )
        return InferenceTask.from_obs(source, _priors(), image_obs={'F087': obs})

    chi2 = _chi2_per_point(make_task(noisy), make_task(clean), clean.size)
    _assert_calibrated(chi2, clean.size, 'image obs, variance map')


def test_grism_obs_chi2_variance_map(scene):
    from kl_pipe.noise import add_map_noise

    image_pars, psf, source = scene
    grism_pars = build_grism_pars_for_line(
        LINE_LAMBDAS['Halpha'], redshift=_Z, image_pars=image_pars, dispersion=1.1
    )
    rc = RenderConfig(
        oversample=5, dispersal_method='analytic', line_window_halfwidth=25
    )
    obs0 = build_grism_obs(grism_pars, z=_Z, psf=psf, render_config=rc)
    clean = np.asarray(source.render_grism(_PARS, obs0))
    var_map = _shot_noise_variance_map(clean)
    noisy = add_map_noise(clean, var_map, seed=_SEED)

    def make_task(data):
        obs = build_grism_obs(
            grism_pars,
            z=_Z,
            psf=psf,
            render_config=rc,
            data=jnp.asarray(data),
            variance=jnp.asarray(var_map),
        )
        return InferenceTask.from_obs(source, _priors(), grism_obs={'roll0': obs})

    chi2 = _chi2_per_point(make_task(noisy), make_task(clean), clean.size)
    _assert_calibrated(chi2, clean.size, 'grism obs, variance map')


def test_optimizer_recovers_truth_under_variance_map(scene):
    # gradient-descent wiring under a per-pixel variance map: on NOISE-FREE
    # data the likelihood optimum is the truth by construction (the map only
    # reweights pixels), so the optimizer must return it to within optimizer
    # tolerance; any gradient or weighting bug under map variance moves the
    # recovered point far outside these bounds
    from scipy.optimize import minimize

    from kl_pipe.priors import Uniform as U

    image_pars, psf, source = scene
    obs0 = build_image_obs(
        image_pars,
        psf=psf,
        int_model=source.broadband_models['F087'],
        broadband_key='F087',
    )
    clean = np.asarray(source.render_broadband(_PARS, obs0, 'F087'))
    var_map = _shot_noise_variance_map(clean)

    priors = {k: v for k, v in _PARS.items()}
    priors['F087.flux'] = U(50.0, 200.0)
    priors['F087.rscale'] = U(0.1, 0.6)
    priors['cosi'] = U(0.2, 0.95)
    prior_dict = PriorDict(priors)

    obs = build_image_obs(
        image_pars,
        psf=psf,
        data=jnp.asarray(clean),
        variance=jnp.asarray(var_map),
        int_model=source.broadband_models['F087'],
        broadband_key='F087',
    )
    task = InferenceTask.from_obs(source, prior_dict, image_obs={'F087': obs})
    grad_fn = task.get_log_posterior_and_grad_fn()

    def neg_logpost(th):
        val, grad = grad_fn(jnp.asarray(th))
        return -float(val), -np.asarray(grad, dtype=np.float64)

    truth_theta = np.asarray([_PARS[n] for n in prior_dict.sampled_names])
    start = truth_theta * np.array([1.1, 0.9, 1.05])
    # bounds keep the line search inside the uniform prior support; an
    # unbounded first step lands at log-prior -inf and stalls the optimizer
    result = minimize(
        neg_logpost,
        start,
        jac=True,
        method='L-BFGS-B',
        bounds=task.get_bounds(),
        options={'maxiter': 200},
    )
    assert result.success, result.message
    for name, got, want in zip(prior_dict.sampled_names, result.x, truth_theta):
        assert got == pytest.approx(want, rel=1e-4), (
            f'{name}: optimizer returned {got}, truth {want} (noise-free '
            f'optimum must be the truth)'
        )
