"""
Integration tests for MCMC sampling infrastructure.

Tests the sampling module:
- InferenceTask creation
- Sampler factory
- Basic sampler functionality
- Parameter recovery (marked slow)
"""

import pytest
import numpy as np
import jax.numpy as jnp
import jax.random as random
from pathlib import Path

from kl_pipe.velocity import CenteredVelocityModel
from kl_pipe.parameters import ImagePars
from kl_pipe.synthetic import SyntheticVelocity
from kl_pipe.priors import Uniform, Gaussian, TruncatedNormal, PriorDict
from kl_pipe.sampling import (
    InferenceTask,
    SamplerResult,
    BaseSamplerConfig,
    EnsembleSamplerConfig,
    NestedSamplerConfig,
    GradientSamplerConfig,
    build_sampler,
    get_available_samplers,
)
from kl_pipe.sampling.base import Sampler


# ==============================================================================
# Test Configuration
# ==============================================================================


def get_test_dir() -> Path:
    """Get test directory path."""
    return Path(__file__).parent


@pytest.fixture(scope="module")
def test_output_dir():
    """Create output directory for test artifacts."""
    out_dir = get_test_dir() / "out" / "sampling"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


# ==============================================================================
# Simple Test Problem Fixtures
# ==============================================================================


@pytest.fixture(scope="module")
def simple_velocity_problem():
    """
    Create a simple velocity inference task for testing.

    Uses CenteredVelocityModel with only 2 sampled parameters for speed.
    """
    # True parameters
    true_pars = {
        'cosi': 0.6,
        'theta_int': 0.785,
        'g1': 0.0,
        'g2': 0.0,
        'v0': 10.0,
        'vcirc': 200.0,
        'vel_rscale': 5.0,
    }

    # Generate synthetic data
    image_pars = ImagePars(shape=(24, 24), pixel_scale=0.4, indexing='ij')
    synth = SyntheticVelocity(true_pars, model_type='arctan', seed=42)
    data_noisy = synth.generate(image_pars, snr=100, include_poisson=False)
    variance = synth.variance

    # Define priors (sample only vcirc and cosi for speed)
    priors = PriorDict(
        {
            'vcirc': Uniform(100, 300),
            'cosi': TruncatedNormal(0.5, 0.3, 0.1, 0.99),
            'theta_int': 0.785,  # Fixed
            'g1': 0.0,  # Fixed
            'g2': 0.0,  # Fixed
            'v0': 10.0,  # Fixed
            'vel_rscale': 5.0,  # Fixed
        }
    )

    # Create model and task
    model = CenteredVelocityModel()
    task = InferenceTask.from_velocity_model(
        model, priors, jnp.array(data_noisy), variance, image_pars
    )

    return task, true_pars


# ==============================================================================
# InferenceTask Tests
# ==============================================================================


class TestInferenceTask:
    """Tests for InferenceTask class."""

    def test_from_velocity_model(self, simple_velocity_problem):
        """Can create task from velocity model."""
        task, true_pars = simple_velocity_problem

        assert task.n_params == 2  # cosi and vcirc
        assert 'cosi' in task.sampled_names
        assert 'vcirc' in task.sampled_names
        assert task.fixed_params['v0'] == 10.0

    def test_log_posterior_finite(self, simple_velocity_problem):
        """Log posterior is finite for valid parameters."""
        task, true_pars = simple_velocity_problem

        # Sample from prior
        key = random.PRNGKey(42)
        theta = task.sample_prior(key, 1)[0]

        log_prob_fn = task.get_log_posterior_fn()
        lp = log_prob_fn(theta)

        assert np.isfinite(lp)

    def test_log_posterior_gradient(self, simple_velocity_problem):
        """Can compute gradient of log posterior."""
        task, true_pars = simple_velocity_problem

        key = random.PRNGKey(42)
        theta = task.sample_prior(key, 1)[0]

        grad_fn = task.get_log_posterior_and_grad_fn()
        lp, grad = grad_fn(theta)

        assert np.isfinite(lp)
        assert all(np.isfinite(grad))

    def test_bounds(self, simple_velocity_problem):
        """Can get parameter bounds."""
        task, _ = simple_velocity_problem

        bounds = task.get_bounds()
        assert len(bounds) == task.n_params

        # Check bounds are reasonable
        for low, high in bounds:
            if low is not None and high is not None:
                assert low < high


# ==============================================================================
# InferenceTask Mask Tests
# ==============================================================================


class TestInferenceTaskMask:
    """Tests for InferenceTask with pixel masks."""

    def test_velocity_with_mask(self):
        """InferenceTask.from_velocity_model with mask produces finite posterior."""
        true_pars = {
            'cosi': 0.6,
            'theta_int': 0.785,
            'g1': 0.0,
            'g2': 0.0,
            'v0': 10.0,
            'vcirc': 200.0,
            'vel_rscale': 5.0,
        }

        image_pars = ImagePars(shape=(24, 24), pixel_scale=0.4, indexing='ij')
        synth = SyntheticVelocity(true_pars, model_type='arctan', seed=42)
        data_noisy = synth.generate(image_pars, snr=100, include_poisson=False)
        variance = synth.variance

        mask = np.ones(image_pars.shape, dtype=bool)
        mask[12:, 12:] = False  # mask one quadrant

        priors = PriorDict(
            {
                'vcirc': Uniform(100, 300),
                'cosi': TruncatedNormal(0.5, 0.3, 0.1, 0.99),
                'theta_int': 0.785,
                'g1': 0.0,
                'g2': 0.0,
                'v0': 10.0,
                'vel_rscale': 5.0,
            }
        )

        model = CenteredVelocityModel()
        task = InferenceTask.from_velocity_model(
            model,
            priors,
            jnp.array(data_noisy),
            variance,
            image_pars,
            mask_vel=jnp.array(mask),
        )

        assert task.mask['velocity'] is not None
        key = random.PRNGKey(42)
        theta = task.sample_prior(key, 1)[0]
        lp = task.log_posterior(theta)
        assert np.isfinite(lp)

    def test_intensity_with_mask(self):
        """InferenceTask.from_intensity_model with mask produces finite posterior."""
        from kl_pipe.intensity import InclinedExponentialModel
        from kl_pipe.synthetic import SyntheticIntensity

        true_pars = {
            'cosi': 0.7,
            'theta_int': 0.785,
            'g1': 0.0,
            'g2': 0.0,
            'flux': 1.0,
            'int_rscale': 3.0,
            'int_h_over_r': 0.1,
            'int_x0': 0.0,
            'int_y0': 0.0,
        }

        image_pars = ImagePars(shape=(24, 24), pixel_scale=0.4, indexing='ij')
        synth = SyntheticIntensity(true_pars, model_type='exponential', seed=42)
        data_noisy = synth.generate(image_pars, snr=100, include_poisson=False)
        variance = synth.variance

        mask = np.ones(image_pars.shape, dtype=bool)
        mask[12:, 12:] = False

        priors = PriorDict(
            {
                'flux': Uniform(0.1, 10.0),
                'int_rscale': Uniform(0.5, 10.0),
                'cosi': 0.7,
                'theta_int': 0.785,
                'g1': 0.0,
                'g2': 0.0,
                'int_h_over_r': 0.1,
                'int_x0': 0.0,
                'int_y0': 0.0,
            }
        )

        model = InclinedExponentialModel()
        task = InferenceTask.from_intensity_model(
            model,
            priors,
            jnp.array(data_noisy),
            variance,
            image_pars,
            mask_int=jnp.array(mask),
        )

        assert task.mask['intensity'] is not None
        key = random.PRNGKey(42)
        theta = task.sample_prior(key, 1)[0]
        lp = task.log_posterior(theta)
        assert np.isfinite(lp)

    def test_joint_with_masks(self):
        """InferenceTask.from_joint_model with both masks produces finite posterior."""
        from kl_pipe.intensity import InclinedExponentialModel
        from kl_pipe.velocity import OffsetVelocityModel
        from kl_pipe.model import KLModel
        from kl_pipe.synthetic import SyntheticIntensity

        true_pars = {
            'cosi': 0.6,
            'theta_int': 0.785,
            'g1': 0.0,
            'g2': 0.0,
            'v0': 10.0,
            'vcirc': 200.0,
            'vel_rscale': 5.0,
            'vel_x0': 0.0,
            'vel_y0': 0.0,
            'flux': 1.0,
            'int_rscale': 3.0,
            'int_h_over_r': 0.1,
            'int_x0': 0.0,
            'int_y0': 0.0,
        }

        ip_vel = ImagePars(shape=(24, 24), pixel_scale=0.4, indexing='ij')
        ip_int = ImagePars(shape=(24, 24), pixel_scale=0.4, indexing='ij')

        vel_pars = {
            k: v
            for k, v in true_pars.items()
            if k in OffsetVelocityModel.PARAMETER_NAMES
        }
        synth_vel = SyntheticVelocity(vel_pars, model_type='arctan', seed=42)
        data_vel = synth_vel.generate(ip_vel, snr=100, include_poisson=False)
        var_vel = synth_vel.variance

        int_pars = {
            k: v
            for k, v in true_pars.items()
            if k in InclinedExponentialModel.PARAMETER_NAMES
        }
        synth_int = SyntheticIntensity(int_pars, model_type='exponential', seed=43)
        data_int = synth_int.generate(ip_int, snr=100, include_poisson=False)
        var_int = synth_int.variance

        mask_vel = np.ones(ip_vel.shape, dtype=bool)
        mask_vel[12:, 12:] = False
        mask_int = np.ones(ip_int.shape, dtype=bool)
        mask_int[:12, :12] = False

        vel_model = OffsetVelocityModel()
        int_model = InclinedExponentialModel()
        joint_model = KLModel(
            vel_model, int_model, shared_pars={'cosi', 'theta_int', 'g1', 'g2'}
        )

        priors = PriorDict(
            {
                'vcirc': Uniform(100, 300),
                'cosi': 0.6,
                'theta_int': 0.785,
                'g1': 0.0,
                'g2': 0.0,
                'v0': 10.0,
                'vel_rscale': 5.0,
                'vel_x0': 0.0,
                'vel_y0': 0.0,
                'flux': 1.0,
                'int_rscale': 3.0,
                'int_h_over_r': 0.1,
                'int_x0': 0.0,
                'int_y0': 0.0,
            }
        )

        task = InferenceTask.from_joint_model(
            joint_model,
            priors,
            jnp.array(data_vel),
            jnp.array(data_int),
            var_vel,
            var_int,
            ip_vel,
            ip_int,
            mask_vel=jnp.array(mask_vel),
            mask_int=jnp.array(mask_int),
        )

        assert task.mask['velocity'] is not None
        assert task.mask['intensity'] is not None
        key = random.PRNGKey(42)
        theta = task.sample_prior(key, 1)[0]
        lp = task.log_posterior(theta)
        assert np.isfinite(lp)

    def test_mask_none_backward_compat(self):
        """from_velocity_model without mask arg works identically to before."""
        true_pars = {
            'cosi': 0.6,
            'theta_int': 0.785,
            'g1': 0.0,
            'g2': 0.0,
            'v0': 10.0,
            'vcirc': 200.0,
            'vel_rscale': 5.0,
        }

        image_pars = ImagePars(shape=(24, 24), pixel_scale=0.4, indexing='ij')
        synth = SyntheticVelocity(true_pars, model_type='arctan', seed=42)
        data_noisy = synth.generate(image_pars, snr=100, include_poisson=False)
        variance = synth.variance

        priors = PriorDict(
            {
                'vcirc': Uniform(100, 300),
                'cosi': TruncatedNormal(0.5, 0.3, 0.1, 0.99),
                'theta_int': 0.785,
                'g1': 0.0,
                'g2': 0.0,
                'v0': 10.0,
                'vel_rscale': 5.0,
            }
        )

        model = CenteredVelocityModel()
        task = InferenceTask.from_velocity_model(
            model,
            priors,
            jnp.array(data_noisy),
            variance,
            image_pars,
        )

        # mask field should be empty dict or contain None
        assert task.mask.get('velocity') is None

        key = random.PRNGKey(42)
        theta = task.sample_prior(key, 1)[0]
        lp = task.log_posterior(theta)
        assert np.isfinite(lp)


# ==============================================================================
# Config Tests
# ==============================================================================


class TestConfigs:
    """Tests for sampler configuration classes."""

    def test_ensemble_config_defaults(self):
        """EnsembleSamplerConfig has sensible defaults."""
        config = EnsembleSamplerConfig()

        assert config.n_walkers >= 2
        assert config.n_iterations > config.burn_in
        assert config.thin >= 1

    def test_ensemble_config_validation(self):
        """EnsembleSamplerConfig validates inputs."""
        with pytest.raises(ValueError):
            EnsembleSamplerConfig(n_walkers=1)

        with pytest.raises(ValueError):
            EnsembleSamplerConfig(n_iterations=100, burn_in=200)

    def test_nested_config(self):
        """NestedSamplerConfig works correctly."""
        config = NestedSamplerConfig(n_live=500, seed=42)

        assert config.n_live == 500
        assert config.seed == 42

    def test_gradient_config(self):
        """GradientSamplerConfig works correctly."""
        config = GradientSamplerConfig(n_samples=1000, algorithm='nuts')

        assert config.n_samples == 1000
        assert config.algorithm == 'nuts'

        with pytest.raises(ValueError):
            GradientSamplerConfig(algorithm='invalid')


# ==============================================================================
# Factory Tests
# ==============================================================================


class TestFactory:
    """Tests for sampler factory and registry."""

    def test_available_samplers(self):
        """Can get list of available samplers."""
        samplers = get_available_samplers()

        assert 'emcee' in samplers
        assert 'nautilus' in samplers
        assert 'blackjax' in samplers

    def test_build_emcee(self, simple_velocity_problem):
        """Can build emcee sampler."""
        task, _ = simple_velocity_problem

        config = EnsembleSamplerConfig(n_walkers=8, n_iterations=10, burn_in=2)
        sampler = build_sampler('emcee', task, config)

        assert isinstance(sampler, Sampler)
        assert sampler.name == 'EmceeSampler'

    def test_build_case_insensitive(self, simple_velocity_problem):
        """Factory handles case-insensitive names."""
        task, _ = simple_velocity_problem
        config = EnsembleSamplerConfig(n_walkers=8, n_iterations=10, burn_in=2)

        sampler1 = build_sampler('EMCEE', task, config)
        sampler2 = build_sampler('EmCeE', task, config)

        assert type(sampler1) == type(sampler2)

    def test_build_unknown_raises(self, simple_velocity_problem):
        """Factory raises error for unknown sampler."""
        task, _ = simple_velocity_problem

        with pytest.raises(ValueError, match="Unknown sampler"):
            build_sampler('nonexistent', task)

    def test_build_with_default_config(self, simple_velocity_problem):
        """Factory creates default config if none provided."""
        task, _ = simple_velocity_problem

        sampler = build_sampler('emcee', task)
        assert sampler.config is not None

    def test_ultranest_not_implemented(self, simple_velocity_problem):
        """UltraNest raises NotImplementedError."""
        task, _ = simple_velocity_problem
        config = NestedSamplerConfig()

        with pytest.raises(NotImplementedError):
            build_sampler('ultranest', task, config)


# ==============================================================================
# SamplerResult Tests
# ==============================================================================


class TestSamplerResult:
    """Tests for SamplerResult class."""

    def test_result_properties(self):
        """SamplerResult has correct properties."""
        samples = np.random.randn(100, 2)
        log_prob = np.random.randn(100)

        result = SamplerResult(
            samples=samples,
            log_prob=log_prob,
            param_names=['a', 'b'],
            fixed_params={'c': 5.0},
        )

        assert result.n_samples == 100
        assert result.n_params == 2

    def test_get_chain(self):
        """Can retrieve chains for individual parameters."""
        samples = np.array([[1, 2], [3, 4], [5, 6]])
        log_prob = np.zeros(3)

        result = SamplerResult(
            samples=samples,
            log_prob=log_prob,
            param_names=['a', 'b'],
            fixed_params={'c': 5.0},
        )

        chain_a = result.get_chain('a')
        assert np.allclose(chain_a, [1, 3, 5])

        chain_b = result.get_chain('b')
        assert np.allclose(chain_b, [2, 4, 6])

        # Fixed parameter
        chain_c = result.get_chain('c')
        assert np.allclose(chain_c, [5.0, 5.0, 5.0])

    def test_get_summary(self):
        """Summary statistics are computed correctly."""
        # Use known distribution
        np.random.seed(42)
        samples = np.random.randn(10000, 1)
        log_prob = np.zeros(10000)

        result = SamplerResult(
            samples=samples,
            log_prob=log_prob,
            param_names=['a'],
            fixed_params={},
        )

        summary = result.get_summary()

        assert np.isclose(summary['a']['mean'], 0.0, atol=0.05)
        assert np.isclose(summary['a']['std'], 1.0, atol=0.05)

    def test_to_dict(self):
        """Can convert result to dictionary."""
        samples = np.random.randn(10, 2)
        log_prob = np.random.randn(10)

        result = SamplerResult(
            samples=samples,
            log_prob=log_prob,
            param_names=['a', 'b'],
            fixed_params={'c': 5.0},
        )

        d = result.to_dict(include_samples=False)
        assert 'summary' in d
        assert 'samples' not in d

        d = result.to_dict(include_samples=True)
        assert 'samples' in d


# ==============================================================================
# Emcee Sampler Tests
# ==============================================================================


class TestEmceeSampler:
    """Tests for emcee backend."""

    @pytest.mark.slow
    def test_emcee_runs(self, simple_velocity_problem):
        """emcee sampler runs without errors."""
        task, _ = simple_velocity_problem

        config = EnsembleSamplerConfig(
            n_walkers=16,
            n_iterations=50,
            burn_in=10,
            seed=42,
            progress=False,
        )

        sampler = build_sampler('emcee', task, config)
        result = sampler.run()

        assert isinstance(result, SamplerResult)
        assert result.n_samples > 0
        assert result.samples.shape[1] == task.n_params
        assert result.acceptance_fraction is not None

    @pytest.mark.slow
    def test_emcee_chains_stored(self, simple_velocity_problem):
        """emcee stores full chains."""
        task, _ = simple_velocity_problem

        config = EnsembleSamplerConfig(
            n_walkers=16,
            n_iterations=50,
            burn_in=10,
            seed=42,
            progress=False,
        )

        sampler = build_sampler('emcee', task, config)
        result = sampler.run()

        assert result.chains is not None
        assert result.chains.shape[0] == 50  # n_iterations
        assert result.chains.shape[1] == 16  # n_walkers


# ==============================================================================
# Parameter Recovery Tests (Slow)
# ==============================================================================


@pytest.mark.slow
class TestParameterRecovery:
    """Tests for parameter recovery with samplers."""

    def test_emcee_recovers_vcirc(self, simple_velocity_problem):
        """emcee recovers vcirc within tolerance."""
        task, true_pars = simple_velocity_problem

        config = EnsembleSamplerConfig(
            n_walkers=32,
            n_iterations=500,
            burn_in=100,
            seed=42,
            progress=False,
        )

        sampler = build_sampler('emcee', task, config)
        result = sampler.run()

        summary = result.get_summary()

        # Check vcirc recovery (true = 200)
        vcirc_median = summary['vcirc']['quantiles'][0.5]
        true_vcirc = true_pars['vcirc']
        rel_error = abs(vcirc_median - true_vcirc) / true_vcirc

        assert rel_error < 0.10, f"vcirc error {rel_error:.1%} exceeds 10%"

    def test_emcee_recovers_cosi(self, simple_velocity_problem):
        """emcee recovers cosi within tolerance."""
        task, true_pars = simple_velocity_problem

        config = EnsembleSamplerConfig(
            n_walkers=32,
            n_iterations=500,
            burn_in=100,
            seed=42,
            progress=False,
        )

        sampler = build_sampler('emcee', task, config)
        result = sampler.run()

        summary = result.get_summary()

        # Check cosi recovery (true = 0.6)
        cosi_median = summary['cosi']['quantiles'][0.5]
        true_cosi = true_pars['cosi']
        rel_error = abs(cosi_median - true_cosi) / true_cosi

        assert rel_error < 0.15, f"cosi error {rel_error:.1%} exceeds 15%"


# ==============================================================================
# Diagnostics Tests
# ==============================================================================


class TestDiagnostics:
    """Tests for diagnostic utilities."""

    def test_print_summary(self, simple_velocity_problem, capsys):
        """print_summary works without errors."""
        from kl_pipe.sampling.diagnostics import print_summary

        # Create a mock result
        samples = np.random.randn(100, 2)
        log_prob = np.zeros(100)

        result = SamplerResult(
            samples=samples,
            log_prob=log_prob,
            param_names=['cosi', 'vcirc'],
            fixed_params={'v0': 10.0},
            metadata={'backend': 'test'},
        )

        print_summary(result)

        captured = capsys.readouterr()
        assert 'SAMPLING SUMMARY' in captured.out
        assert 'cosi' in captured.out
        assert 'vcirc' in captured.out

    @pytest.mark.slow
    def test_plot_functions(self, simple_velocity_problem, test_output_dir):
        """Diagnostic plot functions run without errors."""
        from kl_pipe.sampling.diagnostics import plot_trace, plot_corner

        # Create a mock result
        samples = np.random.randn(100, 2)
        log_prob = np.zeros(100)

        result = SamplerResult(
            samples=samples,
            log_prob=log_prob,
            param_names=['cosi', 'vcirc'],
            fixed_params={},
        )

        # Test trace plot
        fig = plot_trace(result)
        assert fig is not None

        # Test corner plot
        fig = plot_corner(result)
        assert fig is not None

        import matplotlib.pyplot as plt

        plt.close('all')
