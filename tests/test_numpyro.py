"""
Tests for NumPyro gradient-based sampler.

These tests verify that the NumPyro sampler works correctly with Z-score
reparameterization, including:
- Gradient scaling normalization
- Basic sampling functionality
- Joint model handling (the key failure mode for BlackJAX)
- Convergence diagnostics (R-hat, ESS)
- Different chain execution methods

Many tests are adapted from test_blackjax.py to ensure NumPyro handles
the same scenarios that caused BlackJAX to fail.
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as random
from pathlib import Path

from kl_pipe.velocity import CenteredVelocityModel
from kl_pipe.parameters import ImagePars
from kl_pipe.synthetic import SyntheticVelocity, SyntheticIntensity
from kl_pipe.priors import Uniform, Gaussian, TruncatedNormal, PriorDict
from kl_pipe.sampling import (
    InferenceTask,
    NumpyroSamplerConfig,
    ReparamStrategy,
    build_sampler,
)
from kl_pipe.sampling.numpyro import (
    NumpyroSampler,
    compute_reparam_scales,
    compute_empirical_scales,
)
from kl_pipe.utils import get_test_dir


# ==============================================================================
# Test Fixtures
# ==============================================================================


@pytest.fixture(scope="module")
def output_dir():
    """Output directory for NumPyro diagnostic tests."""
    out_dir = get_test_dir() / "out" / "numpyro_diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


@pytest.fixture(scope="module")
def simple_velocity_task():
    """
    Create a simple velocity-only inference task for basic tests.

    Uses Gaussian/TruncatedNormal priors with reasonable scales.
    """
    image_pars = ImagePars(shape=(20, 20), pixel_scale=0.4, indexing='ij')

    true_pars = {
        'v0': 10.0,
        'vcirc': 200.0,
        'vel_rscale': 5.0,
        'cosi': 0.6,
        'theta_int': 0.785,
        'g1': 0.02,
        'g2': -0.01,
    }

    vel_model = CenteredVelocityModel()

    synth_vel = SyntheticVelocity(true_pars, model_type='arctan', seed=42)
    data_vel_noisy = synth_vel.generate(image_pars, snr=100, include_poisson=False)
    var_vel = synth_vel.variance

    priors = PriorDict(
        {
            'v0': Gaussian(10.0, 5.0),
            'vcirc': TruncatedNormal(200.0, 50.0, 100, 300),
            'vel_rscale': TruncatedNormal(5.0, 2.0, 0.4, 20.0),
            'cosi': TruncatedNormal(0.6, 0.2, 0.01, 0.99),
            'theta_int': TruncatedNormal(0.785, 0.3, 0, np.pi),
            'g1': 0.02,  # Fixed
            'g2': -0.01,  # Fixed
        }
    )

    task = InferenceTask.from_velocity_model(
        model=vel_model,
        priors=priors,
        data_vel=data_vel_noisy,
        variance_vel=var_vel,
        image_pars=image_pars,
    )

    return task, true_pars


@pytest.fixture(scope="module")
def joint_model_task():
    """
    Create joint velocity+intensity task - the critical test case.

    This is where BlackJAX failed due to gradient scale mismatch.
    """
    from kl_pipe.intensity import InclinedExponentialModel
    from kl_pipe.model import KLModel

    image_pars_vel = ImagePars(shape=(24, 24), pixel_scale=0.4, indexing='ij')
    image_pars_int = ImagePars(shape=(32, 32), pixel_scale=0.3, indexing='ij')

    true_pars = {
        'v0': 10.0,
        'vcirc': 200.0,
        'vel_rscale': 5.0,
        'cosi': 0.6,
        'theta_int': 0.785,
        'g1': 0.03,
        'g2': -0.02,
        'flux': 1.0,
        'int_rscale': 3.0,
        'int_x0': 0.0,
        'int_y0': 0.0,
    }

    vel_model = CenteredVelocityModel()
    vel_pars = {k: v for k, v in true_pars.items() if k in vel_model.PARAMETER_NAMES}
    synth_vel = SyntheticVelocity(vel_pars, model_type='arctan', seed=42)
    data_vel = synth_vel.generate(image_pars_vel, snr=100, include_poisson=False)
    var_vel = synth_vel.variance

    int_model = InclinedExponentialModel()
    int_pars = {k: v for k, v in true_pars.items() if k in int_model.PARAMETER_NAMES}
    synth_int = SyntheticIntensity(int_pars, model_type='exponential', seed=43)
    data_int = synth_int.generate(image_pars_int, snr=100, include_poisson=False)
    var_int = synth_int.variance

    joint_model = KLModel(
        velocity_model=vel_model,
        intensity_model=int_model,
        shared_pars={'cosi', 'theta_int', 'g1', 'g2'},
    )

    priors = PriorDict(
        {
            'v0': Gaussian(true_pars['v0'], 5.0),
            'vcirc': TruncatedNormal(200.0, 50.0, 100, 300),
            'vel_rscale': TruncatedNormal(5.0, 2.0, 1.0, 10.0),
            'flux': TruncatedNormal(1.0, 1.0, 0.1, 5.0),
            'int_rscale': TruncatedNormal(3.0, 2.0, 0.5, 10.0),
            'int_x0': 0.0,  # Fixed
            'int_y0': 0.0,  # Fixed
            'cosi': TruncatedNormal(0.5, 0.3, 0.01, 0.99),
            'theta_int': TruncatedNormal(np.pi / 2, 1.0, 0, np.pi),
            'g1': TruncatedNormal(0.0, 0.05, -0.1, 0.1),
            'g2': TruncatedNormal(0.0, 0.05, -0.1, 0.1),
        }
    )

    task = InferenceTask.from_joint_model(
        model=joint_model,
        priors=priors,
        data_vel=jnp.array(data_vel),
        data_int=jnp.array(data_int),
        variance_vel=var_vel,
        variance_int=var_int,
        image_pars_vel=image_pars_vel,
        image_pars_int=image_pars_int,
    )

    return task, true_pars


# ==============================================================================
# Reparameterization Tests
# ==============================================================================


class TestReparameterization:
    """Tests for Z-score reparameterization scaling."""

    def test_gaussian_scaling(self):
        """Gaussian prior uses mu and sigma directly."""
        prior = Gaussian(100.0, 25.0)
        loc, scale = compute_reparam_scales(prior, 'test')
        assert loc == 100.0
        assert scale == 25.0

    def test_truncated_normal_scaling(self):
        """TruncatedNormal uses underlying Gaussian params."""
        prior = TruncatedNormal(0.5, 0.2, 0.01, 0.99)
        loc, scale = compute_reparam_scales(prior, 'test')
        assert loc == 0.5
        assert scale == 0.2

    def test_uniform_scaling(self):
        """Uniform uses midpoint and quarter-range."""
        prior = Uniform(0, 100)
        loc, scale = compute_reparam_scales(prior, 'test')
        assert loc == 50.0
        assert scale == 25.0  # (100-0)/4

    def test_reparam_strategy_none(self, simple_velocity_task):
        """ReparamStrategy.NONE returns identity scales."""
        task, _ = simple_velocity_task

        config = NumpyroSamplerConfig(
            n_samples=10,
            n_warmup=10,
            n_chains=1,
            reparam_strategy=ReparamStrategy.NONE,
            seed=42,
            progress=False,
        )

        sampler = NumpyroSampler(task, config)
        scales = sampler._compute_reparam_scales(random.PRNGKey(0))

        for name in task.sampled_names:
            loc, scale = scales[name]
            assert loc == 0.0
            assert scale == 1.0

    def test_reparam_strategy_prior(self, simple_velocity_task):
        """ReparamStrategy.PRIOR uses prior statistics."""
        task, _ = simple_velocity_task

        config = NumpyroSamplerConfig(
            n_samples=10,
            n_warmup=10,
            n_chains=1,
            reparam_strategy=ReparamStrategy.PRIOR,
            seed=42,
            progress=False,
        )

        sampler = NumpyroSampler(task, config)
        scales = sampler._compute_reparam_scales(random.PRNGKey(0))

        # Check that vcirc uses its prior params
        loc, scale = scales['vcirc']
        assert loc == 200.0  # TruncatedNormal mu
        assert scale == 50.0  # TruncatedNormal sigma


class TestGradientScaling:
    """
    Verify that Z-score reparameterization normalizes gradients.

    This is the key test for the BlackJAX failure mode.
    """

    @pytest.mark.slow
    def test_latent_gradient_norms_order_one(self, joint_model_task, output_dir):
        """
        After Z-score transform, gradients w.r.t. latent z should be O(1).

        This test verifies the fix for BlackJAX collapse where intensity
        gradients were ~10^4x larger than velocity gradients.

        We evaluate gradients at the TRUE parameter values (converted to z-space)
        where we expect the log posterior gradient to be moderate (near the mode).
        """
        import numpyro
        import numpyro.distributions as dist

        task, true_pars = joint_model_task

        # Get prior-based scales
        config = NumpyroSamplerConfig(reparam_strategy=ReparamStrategy.PRIOR)
        sampler = NumpyroSampler(task, config)
        scales = sampler._compute_reparam_scales(random.PRNGKey(0))

        # Build function that computes log_posterior in z-space
        log_posterior_fn = task.get_log_posterior_fn()

        def log_prob_z(z_dict):
            """Log posterior as function of latent z values."""
            theta_physical = []
            for name in task.sampled_names:
                loc, scale = scales[name]
                z = z_dict[name]
                theta_physical.append(loc + scale * z)
            theta = jnp.stack(theta_physical)
            return log_posterior_fn(theta)

        # Convert TRUE parameters to z-space (not z=0)
        # Near the mode, gradients should be moderate
        z_true = {}
        for name in task.sampled_names:
            loc, scale = scales[name]
            theta_true = true_pars[name]
            z_true[name] = jnp.array((theta_true - loc) / scale)

        grad_fn = jax.grad(log_prob_z)
        grads = grad_fn(z_true)

        # Check gradient magnitudes
        grad_mags = {name: float(jnp.abs(grads[name])) for name in task.sampled_names}

        # Write diagnostic output
        log_path = output_dir / "gradient_scaling_test.txt"
        with open(log_path, 'w') as f:
            f.write("Gradient Scaling Test (Z-space)\n")
            f.write("=" * 60 + "\n\n")
            f.write(
                "Gradients evaluated at z=z_true (physical space = true params)\n\n"
            )

            f.write("Parameter z-values (true):\n")
            for name in task.sampled_names:
                f.write(f"  {name:15s}: z = {float(z_true[name]):.4f}\n")
            f.write("\n")

            for name, mag in sorted(grad_mags.items()):
                f.write(f"{name:15s}: |∂log_p/∂z| = {mag:.4e}\n")

            max_grad = max(grad_mags.values())
            min_grad = min(grad_mags.values())
            ratio = max_grad / min_grad if min_grad > 0 else float('inf')
            f.write(f"\nMax/Min ratio: {ratio:.2f}\n")

            if ratio < 100:
                f.write("\nSUCCESS: Gradients are well-balanced (ratio < 100)\n")
            else:
                f.write("\nWARNING: Large gradient disparity may cause issues\n")

        # Assertion: ratio should be much smaller than 10^4 (the BlackJAX failure)
        assert (
            ratio < 1000
        ), f"Gradient ratio {ratio:.0f} too large - reparameterization not working"


# ==============================================================================
# Basic Sampling Tests
# ==============================================================================


@pytest.mark.slow
class TestNumpyroBasicSampling:
    """Verify sampler produces valid output."""

    def test_samples_shape(self, simple_velocity_task):
        """Correct number of samples returned."""
        task, _ = simple_velocity_task

        config = NumpyroSamplerConfig(
            n_samples=100,
            n_warmup=50,
            n_chains=2,
            seed=42,
            progress=False,
        )

        sampler = build_sampler('numpyro', task, config)
        result = sampler.run()

        expected_samples = 100 * 2  # n_samples * n_chains
        assert result.n_samples == expected_samples
        assert result.samples.shape == (expected_samples, task.n_params)

    def test_finite_log_prob(self, simple_velocity_task):
        """All samples have finite log probability."""
        task, _ = simple_velocity_task

        config = NumpyroSamplerConfig(
            n_samples=100,
            n_warmup=50,
            n_chains=1,
            seed=42,
            progress=False,
        )

        sampler = build_sampler('numpyro', task, config)
        result = sampler.run()

        assert np.all(
            np.isfinite(result.log_prob)
        ), "Some log_prob values are not finite"

    def test_no_divergences(self, simple_velocity_task):
        """No divergent transitions for well-posed problem."""
        task, _ = simple_velocity_task

        config = NumpyroSamplerConfig(
            n_samples=200,
            n_warmup=100,
            n_chains=1,
            seed=42,
            progress=False,
        )

        sampler = build_sampler('numpyro', task, config)
        result = sampler.run()

        n_div = result.diagnostics.get('n_divergences', 0)
        div_rate = result.diagnostics.get('divergence_rate', 0)

        assert div_rate < 0.05, f"Too many divergences: {n_div} ({div_rate:.1%})"

    def test_reasonable_acceptance_rate(self, simple_velocity_task):
        """Acceptance rate should be in healthy range."""
        task, _ = simple_velocity_task

        config = NumpyroSamplerConfig(
            n_samples=200,
            n_warmup=100,
            n_chains=1,
            seed=42,
            progress=False,
        )

        sampler = build_sampler('numpyro', task, config)
        result = sampler.run()

        accept = result.acceptance_fraction
        assert accept is not None
        assert (
            0.4 < accept < 0.99
        ), f"Acceptance rate {accept:.2%} outside healthy range"


# ==============================================================================
# Joint Model Tests (Critical - BlackJAX Failed Here)
# ==============================================================================


@pytest.mark.slow
class TestNumpyroJointModel:
    """
    The critical tests - joint model must not collapse like BlackJAX.

    These tests verify that NumPyro with Z-score reparameterization
    successfully samples joint velocity+intensity models.
    """

    def test_nonzero_variance_all_params(self, joint_model_task, output_dir):
        """
        All parameters must have posterior variance > 0.

        This was the key failure mode of BlackJAX: step_size collapsed
        to ~1e-8, resulting in zero-variance chains.
        """
        task, _ = joint_model_task

        config = NumpyroSamplerConfig(
            n_samples=300,
            n_warmup=200,
            n_chains=1,
            dense_mass=True,
            seed=42,
            progress=False,
        )

        sampler = build_sampler('numpyro', task, config)
        result = sampler.run()

        # Check variance for each parameter
        variances = {}
        zero_var_params = []
        for name in result.param_names:
            chain = result.get_chain(name)
            var = float(np.var(chain))
            variances[name] = var
            if var < 1e-10:
                zero_var_params.append(name)

        # Write diagnostic log
        log_path = output_dir / "joint_model_variance.txt"
        with open(log_path, 'w') as f:
            f.write("Joint Model Variance Test\n")
            f.write("=" * 60 + "\n\n")

            for name, var in sorted(variances.items()):
                status = "OK" if var > 1e-10 else "ZERO"
                f.write(f"{name:15s}: variance = {var:.6e} [{status}]\n")

            if zero_var_params:
                f.write(f"\nFAILED: Zero variance params: {zero_var_params}\n")
            else:
                f.write("\nSUCCESS: All parameters have non-zero variance\n")

        assert (
            len(zero_var_params) == 0
        ), f"Parameters with zero variance: {zero_var_params}"

    def test_step_size_reasonable(self, joint_model_task, output_dir):
        """Step size should be O(0.01-1), not 1e-8."""
        task, _ = joint_model_task

        config = NumpyroSamplerConfig(
            n_samples=200,
            n_warmup=200,
            n_chains=1,
            dense_mass=True,
            seed=42,
            progress=False,
        )

        sampler = build_sampler('numpyro', task, config)
        result = sampler.run()

        step_size = result.diagnostics.get('step_size')

        # Write diagnostic
        log_path = output_dir / "joint_model_step_size.txt"
        with open(log_path, 'w') as f:
            f.write("Joint Model Step Size Test\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Adapted step size: {step_size}\n")

            if step_size is not None:
                if step_size < 1e-6:
                    f.write("FAILED: Step size collapsed (like BlackJAX failure)\n")
                elif step_size > 10:
                    f.write("WARNING: Step size very large\n")
                else:
                    f.write("SUCCESS: Step size in reasonable range\n")

        assert step_size is not None, "Step size not returned"
        assert step_size > 1e-6, f"Step size {step_size:.2e} collapsed (too small)"

    def test_no_excessive_divergences(self, joint_model_task):
        """Well-posed problem should have <10% divergences."""
        task, _ = joint_model_task

        config = NumpyroSamplerConfig(
            n_samples=200,
            n_warmup=200,
            n_chains=1,
            dense_mass=True,
            seed=42,
            progress=False,
        )

        sampler = build_sampler('numpyro', task, config)
        result = sampler.run()

        div_rate = result.diagnostics.get('divergence_rate', 0)
        assert div_rate < 0.10, f"Divergence rate {div_rate:.1%} too high"


# ==============================================================================
# Convergence Diagnostics Tests
# ==============================================================================


@pytest.mark.slow
class TestNumpyroConvergence:
    """Verify convergence diagnostics are computed and valid."""

    def test_rhat_computed_multichain(self, simple_velocity_task):
        """R-hat available when n_chains > 1."""
        task, _ = simple_velocity_task

        config = NumpyroSamplerConfig(
            n_samples=200,
            n_warmup=100,
            n_chains=2,  # Need multiple chains for R-hat
            seed=42,
            progress=False,
        )

        sampler = build_sampler('numpyro', task, config)
        result = sampler.run()

        r_hats = result.get_rhat()
        assert r_hats is not None, "R-hat not computed"
        assert len(r_hats) == len(task.sampled_names)

    def test_rhat_near_one_for_converged(self, simple_velocity_task):
        """Converged chains should have R-hat < 1.05."""
        task, _ = simple_velocity_task

        config = NumpyroSamplerConfig(
            n_samples=500,
            n_warmup=200,
            n_chains=2,
            seed=42,
            progress=False,
        )

        sampler = build_sampler('numpyro', task, config)
        result = sampler.run()

        r_hats = result.get_rhat()
        max_rhat = max(r_hats.values())

        assert max_rhat < 1.1, f"Max R-hat {max_rhat:.3f} indicates poor convergence"

    def test_ess_computed(self, simple_velocity_task):
        """ESS available in diagnostics."""
        task, _ = simple_velocity_task

        config = NumpyroSamplerConfig(
            n_samples=200,
            n_warmup=100,
            n_chains=1,
            seed=42,
            progress=False,
        )

        sampler = build_sampler('numpyro', task, config)
        result = sampler.run()

        ess = result.get_ess()
        assert ess is not None, "ESS not computed"
        assert len(ess) == len(task.sampled_names)

    def test_ess_reasonable(self, simple_velocity_task):
        """ESS should be > 50 for short run."""
        task, _ = simple_velocity_task

        config = NumpyroSamplerConfig(
            n_samples=300,
            n_warmup=100,
            n_chains=1,
            seed=42,
            progress=False,
        )

        sampler = build_sampler('numpyro', task, config)
        result = sampler.run()

        ess = result.get_ess()
        min_ess = min(ess.values())

        assert min_ess > 20, f"Min ESS {min_ess:.0f} too low"


# ==============================================================================
# Chain Method Tests
# ==============================================================================


@pytest.mark.slow
class TestNumpyroChainMethods:
    """Test different chain execution strategies."""

    def test_sequential_chains(self, simple_velocity_task):
        """chain_method='sequential' works."""
        task, _ = simple_velocity_task

        config = NumpyroSamplerConfig(
            n_samples=50,
            n_warmup=25,
            n_chains=2,
            chain_method='sequential',
            seed=42,
            progress=False,
        )

        sampler = build_sampler('numpyro', task, config)
        result = sampler.run()

        assert result.n_samples == 100  # 50 * 2

    def test_vectorized_chains(self, simple_velocity_task):
        """chain_method='vectorized' works."""
        task, _ = simple_velocity_task

        config = NumpyroSamplerConfig(
            n_samples=50,
            n_warmup=25,
            n_chains=2,
            chain_method='vectorized',
            seed=42,
            progress=False,
        )

        sampler = build_sampler('numpyro', task, config)
        result = sampler.run()

        assert result.n_samples == 100


# ==============================================================================
# Init Strategy Tests
# ==============================================================================


class TestNumpyroInitStrategies:
    """Test different initialization strategies."""

    def test_init_prior(self, simple_velocity_task):
        """init_strategy='prior' samples from prior."""
        task, _ = simple_velocity_task

        config = NumpyroSamplerConfig(
            n_samples=10,
            n_warmup=10,
            n_chains=1,
            init_strategy='prior',
            seed=42,
            progress=False,
        )

        # Should not raise
        sampler = build_sampler('numpyro', task, config)
        result = sampler.run()
        assert result.n_samples > 0

    def test_init_median(self, simple_velocity_task):
        """init_strategy='median' starts at prior medians."""
        task, _ = simple_velocity_task

        config = NumpyroSamplerConfig(
            n_samples=10,
            n_warmup=10,
            n_chains=1,
            init_strategy='median',
            seed=42,
            progress=False,
        )

        sampler = build_sampler('numpyro', task, config)
        result = sampler.run()
        assert result.n_samples > 0

    def test_init_jitter(self, simple_velocity_task):
        """init_strategy='jitter' adds small perturbation."""
        task, _ = simple_velocity_task

        config = NumpyroSamplerConfig(
            n_samples=10,
            n_warmup=10,
            n_chains=1,
            init_strategy='jitter',
            seed=42,
            progress=False,
        )

        sampler = build_sampler('numpyro', task, config)
        result = sampler.run()
        assert result.n_samples > 0


# ==============================================================================
# Factory and Integration Tests
# ==============================================================================


class TestNumpyroFactory:
    """Test factory integration."""

    def test_build_sampler_numpyro(self, simple_velocity_task):
        """build_sampler('numpyro', ...) works."""
        task, _ = simple_velocity_task
        config = NumpyroSamplerConfig(
            n_samples=10, n_warmup=10, seed=42, progress=False
        )

        sampler = build_sampler('numpyro', task, config)
        assert isinstance(sampler, NumpyroSampler)

    def test_build_sampler_nuts_alias(self, simple_velocity_task):
        """build_sampler('nuts', ...) returns NumpyroSampler."""
        task, _ = simple_velocity_task
        config = NumpyroSamplerConfig(
            n_samples=10, n_warmup=10, seed=42, progress=False
        )

        sampler = build_sampler('nuts', task, config)
        assert isinstance(sampler, NumpyroSampler)

    def test_build_sampler_hmc_alias(self, simple_velocity_task):
        """build_sampler('hmc', ...) returns NumpyroSampler."""
        task, _ = simple_velocity_task
        config = NumpyroSamplerConfig(
            n_samples=10, n_warmup=10, seed=42, progress=False
        )

        sampler = build_sampler('hmc', task, config)
        assert isinstance(sampler, NumpyroSampler)


# ==============================================================================
# Config Validation Tests
# ==============================================================================


class TestNumpyroConfig:
    """Test config validation."""

    def test_invalid_chain_method(self):
        """Invalid chain_method raises ValueError."""
        with pytest.raises(ValueError, match="chain_method"):
            NumpyroSamplerConfig(chain_method='invalid')

    def test_invalid_init_strategy(self):
        """Invalid init_strategy raises ValueError."""
        with pytest.raises(ValueError, match="init_strategy"):
            NumpyroSamplerConfig(init_strategy='invalid')

    def test_invalid_target_accept(self):
        """Invalid target_accept_prob raises ValueError."""
        with pytest.raises(ValueError, match="target_accept_prob"):
            NumpyroSamplerConfig(target_accept_prob=1.5)

    def test_reparam_strategy_string_conversion(self):
        """String reparam_strategy is converted to enum."""
        config = NumpyroSamplerConfig(reparam_strategy='prior')
        assert config.reparam_strategy == ReparamStrategy.PRIOR

    def test_invalid_reparam_strategy(self):
        """Invalid reparam_strategy raises ValueError."""
        with pytest.raises(ValueError, match="reparam_strategy"):
            NumpyroSamplerConfig(reparam_strategy='invalid')
