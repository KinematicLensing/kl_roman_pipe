"""
Tests for the grism likelihood + ``InferenceTask`` factory hookup.

Covers:
  - Unit tests: eval / JIT compile / grad / factory construction (grism-only
    and joint photometry+grism)
  - Likelihood slice tests: ``Ha_flux`` at SNR in {100, 1000}; ``vcirc``
    and ``vel_dispersion`` at SNR=1000.
  - One smoke optimizer-recovery test for ``Ha_flux`` + ``vcirc`` +
    ``vel_dispersion`` at SNR=1000.

The smoke recovery test fixes most parameters and optimizes over a small subset
to confirm end-to-end inference works. It is not a tight tolerance gate;
tighter parameter recovery awaits the centroid-decoupling refactor described
in ``docs/plans/phase3_sourcemodel_refactor.md``.

Known limitations of the current grism likelihood (see
``docs/plans/grism_inference_plan.md``):
  - The photometric image and the emission cube inside ``render_grism``
    share ``kl_model.intensity_model``'s single ``int_x0``/``int_y0``
    centroid pair. Independent astrometric solutions across channels are
    not yet supported. Tracked in ``docs/plans/phase3_sourcemodel_refactor.md``.
"""

import os
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
from kl_pipe.model import KLModel
from kl_pipe.spectral import (
    SpectralConfig,
    SpectralModel,
    halpha_line,
    HALPHA,
)
from kl_pipe.dispersion import build_grism_pars_for_line
from kl_pipe.observation import build_image_obs, build_grism_obs
from kl_pipe.priors import PriorDict, Uniform, Gaussian
from kl_pipe.sampling.task import InferenceTask
from kl_pipe.likelihood import (
    _log_likelihood_grism,
    _log_likelihood_joint_photometry_grism,
    create_jitted_likelihood_grism,
    create_jitted_likelihood_joint_photometry_grism,
)

# output directory for diagnostic plots
OUT_DIR = os.path.join(os.path.dirname(__file__), 'out', 'grism_likelihood')
os.makedirs(OUT_DIR, exist_ok=True)


# =============================================================================
# Shared fixtures
# =============================================================================

_IMAGE_PARS = ImagePars(shape=(32, 32), pixel_scale=0.11, indexing='ij')
_SHARED_PARS = {'cosi', 'theta_int', 'g1', 'g2'}
_Z = 1.0

# Truth parameters spanning velocity, intensity, and spectral sub-models. The
# spectral sub-model adds ``z`` plus per-line ``Ha_flux``, ``Ha_cont``,
# and shared ``vel_dispersion``.
_TRUE_PARS = {
    # velocity
    'v0': 0.0,
    'vcirc': 200.0,
    'vel_rscale': 0.4,
    'vel_x0': 0.0,
    'vel_y0': 0.0,
    # intensity
    'flux': 100.0,
    'int_rscale': 0.3,
    'int_h_over_r': 0.15,
    'int_x0': 0.0,
    'int_y0': 0.0,
    # shared geometry
    'cosi': 0.6,
    'theta_int': 0.7,
    'g1': 0.0,
    'g2': 0.0,
    # spectral
    'z': _Z,
    'vel_dispersion': 50.0,
    'Ha_flux': 100.0,
    'Ha_cont': 0.05,
}


@pytest.fixture(scope='module')
def kl_model():
    """KLModel with Hα emission line, OffsetVelocityModel, InclinedExponential."""
    vel_model = OffsetVelocityModel()
    int_model = InclinedExponentialModel()
    spec_config = SpectralConfig(lines=(halpha_line(),), spectral_oversample=5)
    spec_model = SpectralModel(spec_config, int_model, vel_model)
    return KLModel(
        vel_model,
        int_model,
        shared_pars=_SHARED_PARS,
        spectral_model=spec_model,
    )


@pytest.fixture(scope='module')
def theta_true(kl_model):
    """Composite theta array at truth."""
    return kl_model.pars2theta(_TRUE_PARS)


@pytest.fixture(scope='module')
def roman_psf():
    """Roman-like Gaussian PSF (0.18" FWHM at the Roman grism central λ)."""
    return galsim.Gaussian(fwhm=0.18)


@pytest.fixture(scope='module')
def grism_pars():
    return build_grism_pars_for_line(
        HALPHA.lambda_rest,
        redshift=_Z,
        image_pars=_IMAGE_PARS,
        dispersion=1.1,
    )


def _build_grism_synthetic(kl_model, theta_true, grism_pars, psf, snr, seed=0):
    """Render a clean grism image, add Gaussian noise calibrated to a target
    matched-filter SNR, return (data, variance, obs_with_data)."""
    obs_no_data = build_grism_obs(grism_pars, z=_Z, psf=psf)
    clean = kl_model.render_grism(theta_true, obs_no_data)
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
    )
    return jnp.asarray(data), float(variance), obs


def _build_intensity_synthetic(kl_model, theta_true, psf, snr, seed=1):
    """Render a clean broadband intensity image and add Gaussian noise."""
    obs_no_data = build_image_obs(
        _IMAGE_PARS, int_model=kl_model.intensity_model, psf=psf
    )
    theta_int = kl_model.get_intensity_pars(theta_true)
    clean = kl_model.intensity_model.render_image(
        theta_int, obs=obs_no_data, render_config=obs_no_data.render_config
    )
    clean = np.asarray(clean)

    signal_power = float(np.sum(clean**2))
    variance = signal_power / snr**2

    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, np.sqrt(variance), size=clean.shape)
    data = clean + noise

    obs = build_image_obs(
        _IMAGE_PARS,
        int_model=kl_model.intensity_model,
        psf=psf,
        data=jnp.asarray(data),
        variance=float(variance),
    )
    return jnp.asarray(data), float(variance), obs


@pytest.fixture(scope='module')
def grism_obs_high_snr(kl_model, theta_true, grism_pars, roman_psf):
    """High-SNR grism synthetic obs (SNR=1000) — used for unit + recovery tests."""
    _, _, obs = _build_grism_synthetic(
        kl_model, theta_true, grism_pars, roman_psf, snr=1000, seed=0
    )
    return obs


@pytest.fixture(scope='module')
def image_obs_high_snr(kl_model, theta_true, roman_psf):
    """High-SNR broadband intensity synthetic obs (SNR=1000)."""
    _, _, obs = _build_intensity_synthetic(
        kl_model, theta_true, roman_psf, snr=1000, seed=1
    )
    return obs


# =============================================================================
# Unit tests: eval / JIT / grad / factories
# =============================================================================


class TestGrismLikelihoodUnits:
    """Smoke tests confirming the grism likelihood evaluates, JITs, and grads."""

    def test_log_likelihood_grism_evaluates(
        self, kl_model, theta_true, grism_obs_high_snr
    ):
        """log_like at theta_true is finite."""
        log_l = _log_likelihood_grism(theta_true, grism_obs_high_snr, kl_model)
        assert jnp.isfinite(log_l), f"log_like not finite: {log_l}"

    def test_log_likelihood_grism_jit_compiles(
        self, kl_model, theta_true, grism_obs_high_snr
    ):
        """JIT-compiled grism likelihood compiles and runs."""
        log_like_fn = create_jitted_likelihood_grism(kl_model, grism_obs_high_snr)
        log_l = log_like_fn(theta_true)
        assert jnp.isfinite(log_l)

    def test_log_likelihood_grism_grad_finite(
        self, kl_model, theta_true, grism_obs_high_snr
    ):
        """jax.grad of grism likelihood produces finite values for all params."""
        log_like_fn = create_jitted_likelihood_grism(kl_model, grism_obs_high_snr)
        grad_fn = jax.grad(log_like_fn)
        grad = grad_fn(theta_true)
        assert jnp.all(jnp.isfinite(grad)), (
            f"non-finite gradient components: indices "
            f"{jnp.where(~jnp.isfinite(grad))[0].tolist()}"
        )

    def test_from_grism_obs_factory_builds_task(
        self, kl_model, grism_obs_high_snr, theta_true
    ):
        """InferenceTask.from_grism_obs builds a working task with finite log_posterior."""
        # at least one sampled prior so theta_sampled is non-empty.
        priors_dict = {k: float(v) for k, v in _TRUE_PARS.items()}
        priors_dict['Ha_flux'] = Uniform(20.0, 200.0)
        priors = PriorDict(priors_dict)

        task = InferenceTask.from_grism_obs(kl_model, priors, grism_obs_high_snr)

        theta_sampled = task.sample_prior(jax.random.PRNGKey(0), n_samples=1)[0]
        log_post = task.log_posterior(theta_sampled)
        assert jnp.isfinite(log_post), f"log_posterior not finite: {log_post}"

    def test_from_joint_photometry_grism_obs_factory_builds_task(
        self, kl_model, grism_obs_high_snr, image_obs_high_snr, theta_true
    ):
        """from_joint_photometry_grism_obs builds a working task."""
        priors_dict = {k: float(v) for k, v in _TRUE_PARS.items()}
        priors_dict['Ha_flux'] = Uniform(20.0, 200.0)
        priors = PriorDict(priors_dict)

        task = InferenceTask.from_joint_photometry_grism_obs(
            kl_model, priors, image_obs_high_snr, grism_obs_high_snr
        )

        theta_sampled = task.sample_prior(jax.random.PRNGKey(0), n_samples=1)[0]
        log_post = task.log_posterior(theta_sampled)
        assert jnp.isfinite(log_post)


# =============================================================================
# Likelihood slice tests
# =============================================================================


def _slice_log_likelihood(log_like_fn, theta_true, param_idx, values):
    """Evaluate log_like along a single-parameter slice (others fixed at truth)."""
    log_ls = []
    for v in values:
        theta = theta_true.at[param_idx].set(v)
        log_ls.append(float(log_like_fn(theta)))
    return np.asarray(log_ls)


def _save_slice_plot(values, log_ls, true_val, peak_val, param_name, snr, out_dir):
    """Save a slice diagnostic plot."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(values, log_ls, 'k-', linewidth=1.0)
    ax.axvline(true_val, color='green', linestyle='--', label=f'truth = {true_val}')
    ax.axvline(peak_val, color='red', linestyle=':', label=f'peak = {peak_val:.3f}')
    ax.set_xlabel(param_name)
    ax.set_ylabel('log likelihood')
    ax.set_title(f'{param_name} slice @ SNR={snr}')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f'slice_{param_name}_snr{snr}.png'), dpi=120)
    plt.close(fig)


@pytest.mark.parametrize(
    "param_name,prior_range,snr,tol_frac",
    [
        # Ha_flux is well-constrained by the integrated line signal at both SNRs.
        ('Ha_flux', (20.0, 200.0), 100, 0.15),
        ('Ha_flux', (20.0, 200.0), 1000, 0.15),
        # vcirc shifts the line wavelength across the disk; needs high SNR
        # because the projected velocity gradient onto the dispersion axis is
        # reduced by sin(i) * cos(theta_int_vs_dispersion).
        ('vcirc', (100.0, 300.0), 1000, 0.15),
        # vel_dispersion is identifiable post-LSF-refactor (sigma_eff = vel_disp);
        # the line width on the detector reflects the kinematic dispersion plus
        # PSF+dispersion broadening (constant across the grid).
        ('vel_dispersion', (20.0, 150.0), 1000, 0.15),
    ],
)
def test_grism_likelihood_slice_peaks_near_truth(
    snr,
    param_name,
    prior_range,
    tol_frac,
    kl_model,
    theta_true,
    grism_pars,
    roman_psf,
):
    """Likelihood slice along one parameter peaks near the truth.

    Tolerance is fractional (peak within ±tol_frac * |truth|). Loose by design
    — the slice is a sanity check that the likelihood is unimodal and peaked,
    not a tight gating tolerance.
    """
    # build fresh synthetic data at this SNR
    _, _, obs = _build_grism_synthetic(
        kl_model, theta_true, grism_pars, roman_psf, snr=snr, seed=snr * 10
    )
    log_like_fn = create_jitted_likelihood_grism(kl_model, obs)

    # locate the param index in the composite theta
    param_names = kl_model.PARAMETER_NAMES
    assert (
        param_name in param_names
    ), f"{param_name} not in kl_model.PARAMETER_NAMES = {param_names}"
    param_idx = param_names.index(param_name)
    true_val = float(_TRUE_PARS[param_name])

    # 25-point slice over the prior range
    values = np.linspace(prior_range[0], prior_range[1], 25)
    log_ls = _slice_log_likelihood(log_like_fn, theta_true, param_idx, values)

    # peak should be near truth
    peak_idx = int(np.argmax(log_ls))
    peak_val = float(values[peak_idx])

    _save_slice_plot(values, log_ls, true_val, peak_val, param_name, snr, OUT_DIR)

    abs_err = abs(peak_val - true_val)
    rel_err = abs_err / abs(true_val) if abs(true_val) > 0 else abs_err
    assert rel_err < tol_frac, (
        f"{param_name} slice peak {peak_val:.3f} differs from truth "
        f"{true_val:.3f} by {rel_err:.2%}; tolerance {tol_frac:.0%} "
        f"(SNR={snr}). See {OUT_DIR}/slice_{param_name}_snr{snr}.png"
    )


# =============================================================================
# Smoke optimizer-recovery test
# =============================================================================


def test_grism_optimizer_recovery_smoke(kl_model, theta_true, grism_obs_high_snr):
    """Smoke test: recover Ha_flux + vcirc + vel_dispersion at SNR=1000.

    Fixes everything else at truth; optimizes the 3 free params from a
    perturbed initial guess. Confirms end-to-end JAX gradient + scipy
    optimizer path works for grism inference.

    Tolerances are loose (±15% relative). vel_dispersion is recoverable
    post-LSF-refactor (sigma_eff = vel_disp).
    """
    log_like_fn = create_jitted_likelihood_grism(kl_model, grism_obs_high_snr)
    grad_fn = jax.jit(jax.grad(log_like_fn))

    param_names = kl_model.PARAMETER_NAMES
    free_names = ['Ha_flux', 'vcirc', 'vel_dispersion']
    free_indices = [param_names.index(n) for n in free_names]

    theta_init = np.array(theta_true)
    # 15% perturbation on free params for the initial guess
    rng = np.random.default_rng(42)
    for idx in free_indices:
        theta_init[idx] = theta_init[idx] * (1 + 0.15 * rng.standard_normal())

    def objective(x_free):
        theta = (
            jnp.asarray(theta_init)
            .at[jnp.asarray(free_indices)]
            .set(jnp.asarray(x_free))
        )
        return -float(log_like_fn(theta))

    def gradient(x_free):
        theta = (
            jnp.asarray(theta_init)
            .at[jnp.asarray(free_indices)]
            .set(jnp.asarray(x_free))
        )
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
