"""
Tests for the SourceModel grism likelihood via ``InferenceTask.from_obs``.

Covers:
  - Unit tests: likelihood eval + JIT + grad + factory construction for
    grism-only and joint photometry+grism patterns.
  - Likelihood slice tests: ``Halpha.flux`` at SNR in {100, 1000}; ``vel.vcirc``
    and ``Halpha.dispersion`` at SNR=1000.
  - One smoke optimizer-recovery test for ``Halpha.flux`` + ``vel.vcirc`` +
    ``Halpha.dispersion`` at SNR=1000.

The smoke recovery test fixes most parameters and optimizes over a small subset
to confirm end-to-end inference works. It is not a tight tolerance gate.

Continuum coverage uses ``EmissionLine(continuum=...)`` with
``Halpha.cont.flux=0.05`` fixed. The continuum's spatial parameters are fixed
equal to the line's own spatial parameters (i.e. the continuum and the line
share a single spatial profile in this test).
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
from kl_pipe.source import SourceModel
from kl_pipe.lines import EmissionLine, LINE_LAMBDAS
from kl_pipe.dispersion import build_grism_pars_for_line
from kl_pipe.observation import build_image_obs, build_grism_obs
from kl_pipe.priors import PriorDict, Uniform
from kl_pipe.sampling.task import InferenceTask

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
    'Halpha.cont.flux': 0.05,
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


def _build_grism_synthetic(source, pars, grism_pars, psf, snr, seed=0):
    """Render a clean grism image, add Gaussian noise calibrated to a target
    matched-filter SNR, return (data, variance, obs_with_data)."""
    obs_no_data = build_grism_obs(grism_pars, z=_Z, psf=psf)
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


def _slice_log_likelihood(log_like_fn, theta_sampled, param_idx, values):
    """Evaluate log_like along a single-parameter slice (others fixed at truth)."""
    log_ls = []
    for v in values:
        theta = theta_sampled.at[param_idx].set(v)
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
    fname = f'slice_{param_name.replace(".", "_")}_snr{snr}.png'
    fig.savefig(os.path.join(out_dir, fname), dpi=120)
    plt.close(fig)


@pytest.mark.parametrize(
    "param_name,prior_range,snr,tol_frac",
    [
        # Halpha.flux is well-constrained by the integrated line signal at both SNRs.
        ('Halpha.flux', (20.0, 200.0), 100, 0.15),
        ('Halpha.flux', (20.0, 200.0), 1000, 0.15),
        # vel.vcirc shifts the line wavelength across the disk; needs high SNR
        # because the projected velocity gradient onto the dispersion axis is
        # reduced by sin(i) * cos(theta_int_vs_dispersion).
        ('vel.vcirc', (100.0, 300.0), 1000, 0.15),
        # Halpha.dispersion is identifiable post-LSF-refactor (sigma_eff = vel_disp);
        # the line width on the detector reflects the kinematic dispersion plus
        # PSF+dispersion broadening (constant across the grid).
        ('Halpha.dispersion', (20.0, 150.0), 1000, 0.15),
    ],
)
def test_grism_likelihood_slice_peaks_near_truth(
    snr,
    param_name,
    prior_range,
    tol_frac,
    source,
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
        source, _TRUE_PARS, grism_pars, roman_psf, snr=snr, seed=snr * 10
    )

    priors = _make_priors()
    task = InferenceTask.from_obs(source, priors, grism_obs={'roll0': obs})
    log_like_fn = task.likelihood_fn

    # locate the param index in the sampled-name vector
    sampled_names = list(priors.sampled_names)
    assert (
        param_name in sampled_names
    ), f"{param_name} not in priors.sampled_names = {sampled_names}"
    param_idx = sampled_names.index(param_name)
    true_val = float(_TRUE_PARS[param_name])

    theta_sampled = _theta_sampled_truth(priors)

    # 25-point slice over the prior range
    values = np.linspace(prior_range[0], prior_range[1], 25)
    log_ls = _slice_log_likelihood(log_like_fn, theta_sampled, param_idx, values)

    # peak should be near truth
    peak_idx = int(np.argmax(log_ls))
    peak_val = float(values[peak_idx])

    _save_slice_plot(values, log_ls, true_val, peak_val, param_name, snr, OUT_DIR)

    abs_err = abs(peak_val - true_val)
    rel_err = abs_err / abs(true_val) if abs(true_val) > 0 else abs_err
    assert rel_err < tol_frac, (
        f"{param_name} slice peak {peak_val:.3f} differs from truth "
        f"{true_val:.3f} by {rel_err:.2%}; tolerance {tol_frac:.0%} "
        f"(SNR={snr}). See {OUT_DIR}/slice_{param_name.replace('.', '_')}_snr{snr}.png"
    )


# =============================================================================
# Smoke optimizer-recovery test
# =============================================================================


def test_grism_optimizer_recovery_smoke(source, grism_obs_high_snr):
    """Smoke test: recover Halpha.flux + vel.vcirc + Halpha.dispersion at SNR=1000.

    Fixes everything else at truth; optimizes the 3 free params from a
    perturbed initial guess. Confirms end-to-end JAX gradient + scipy
    optimizer path works for grism inference under SourceModel.

    Tolerances are loose (±15% relative).
    """
    priors = _make_priors()
    task = InferenceTask.from_obs(
        source, priors, grism_obs={'roll0': grism_obs_high_snr}
    )
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
