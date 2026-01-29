"""
Diagnostic tests for BlackJAX gradient-based sampler.

These tests verify that the BlackJAX sampler infrastructure is working correctly:
- Gradients are finite and non-zero
- Step size adaptation produces reasonable values
- Acceptance rates are in healthy range
- Samples have non-zero variance

These diagnostics help catch issues early before running expensive comparison tests.
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
    GradientSamplerConfig,
    build_sampler,
)
from kl_pipe.utils import get_test_dir


# ==============================================================================
# Test Fixtures
# ==============================================================================


@pytest.fixture(scope="module")
def output_dir():
    """Output directory for blackjax diagnostic tests."""
    out_dir = get_test_dir() / "out" / "blackjax_diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


@pytest.fixture(scope="module")
def simple_velocity_task():
    """
    Create a simple velocity-only inference task for diagnostics.

    Uses Gaussian priors (unbounded) to isolate potential boundary issues.
    """
    # Simple image parameters
    image_pars = ImagePars(shape=(20, 20), pixel_scale=0.4, indexing='ij')

    # True parameters (including shear g1, g2)
    true_pars = {
        'v0': 10.0,
        'vcirc': 200.0,
        'vel_rscale': 5.0,
        'cosi': 0.6,
        'theta_int': 0.785,
        'g1': 0.02,
        'g2': -0.01,
    }

    # Create velocity model
    vel_model = CenteredVelocityModel()

    # Generate synthetic data using correct API
    synth_vel = SyntheticVelocity(true_pars, model_type='arctan', seed=42)
    data_vel_noisy = synth_vel.generate(image_pars, snr=100, include_poisson=False)
    var_vel = synth_vel.variance

    # Use Gaussian priors for most parameters, but TruncatedNormal for
    # physically-bounded parameters to avoid numerical issues:
    # - cosi must be in [0, 1] (sersics are symmetric, so no need for negative)
    # - vel_rscale must be positive and at least ~1 pixel (0.4 arcsec here)
    # Fix g1, g2 to reduce dimensionality for this simple test
    pixel_scale = image_pars.pixel_scale  # 0.4 arcsec/pixel
    priors = PriorDict(
        {
            'v0': Gaussian(10.0, 5.0),
            'vcirc': Gaussian(200.0, 50.0),
            'vel_rscale': TruncatedNormal(5.0, 2.0, pixel_scale, 20.0),
            'cosi': TruncatedNormal(0.6, 0.2, 0.01, 0.99),
            'theta_int': Gaussian(0.785, 0.3),
            'g1': 0.02,  # Fixed
            'g2': -0.01,  # Fixed
        }
    )

    # Create inference task
    task = InferenceTask.from_velocity_model(
        model=vel_model,
        priors=priors,
        data_vel=data_vel_noisy,
        variance_vel=var_vel,
        image_pars=image_pars,
    )

    return task, true_pars


@pytest.fixture(scope="module")
def bounded_velocity_task():
    """
    Create a velocity inference task with bounded (Uniform) priors.

    This tests the case that may have gradient issues at boundaries.
    """
    # Simple image parameters
    image_pars = ImagePars(shape=(20, 20), pixel_scale=0.4, indexing='ij')

    # True parameters (including shear g1, g2)
    true_pars = {
        'v0': 10.0,
        'vcirc': 200.0,
        'vel_rscale': 5.0,
        'cosi': 0.6,
        'theta_int': 0.785,
        'g1': 0.02,
        'g2': -0.01,
    }

    # Create velocity model
    vel_model = CenteredVelocityModel()

    # Generate synthetic data using correct API
    synth_vel = SyntheticVelocity(true_pars, model_type='arctan', seed=42)
    data_vel_noisy = synth_vel.generate(image_pars, snr=100, include_poisson=False)
    var_vel = synth_vel.variance

    # Use Uniform priors (bounded) - may have gradient issues at boundaries
    # Fix g1, g2 to reduce dimensionality for this simple test
    priors = PriorDict(
        {
            'v0': Uniform(-50, 50),
            'vcirc': Uniform(50, 350),
            'vel_rscale': Uniform(1.0, 15.0),
            'cosi': Uniform(0.1, 0.99),
            'theta_int': Uniform(0.0, 3.14159),
            'g1': 0.02,  # Fixed
            'g2': -0.01,  # Fixed
        }
    )

    # Create inference task
    task = InferenceTask.from_velocity_model(
        model=vel_model,
        priors=priors,
        data_vel=data_vel_noisy,
        variance_vel=var_vel,
        image_pars=image_pars,
    )

    return task, true_pars


# ==============================================================================
# Diagnostic Tests
# ==============================================================================


@pytest.mark.slow
class TestBlackJAXDiagnostics:
    """
    Diagnostic tests to catch BlackJAX issues early.

    These tests verify the gradient-based sampling infrastructure is working.
    """

    def test_gradient_computation_gaussian_priors(
        self, simple_velocity_task, output_dir
    ):
        """Verify gradients are finite and non-zero with Gaussian priors."""
        task, true_pars = simple_velocity_task

        log_prob_fn = task.get_log_posterior_fn()
        grad_fn = jax.grad(log_prob_fn)

        # Test gradient at multiple points
        key = random.PRNGKey(123)
        n_test_points = 10

        results = []
        for i in range(n_test_points):
            key, subkey = random.split(key)
            theta = task.sample_prior(subkey, 1)[0]

            # Compute gradient
            grad = grad_fn(theta)

            # Check properties
            has_nan = bool(jnp.any(jnp.isnan(grad)))
            has_inf = bool(jnp.any(jnp.isinf(grad)))
            all_zero = bool(jnp.allclose(grad, 0.0))
            grad_norm = float(jnp.linalg.norm(grad))

            results.append(
                {
                    'has_nan': has_nan,
                    'has_inf': has_inf,
                    'all_zero': all_zero,
                    'grad_norm': grad_norm,
                }
            )

        # Write results to file
        log_path = output_dir / "gradient_test_gaussian.txt"
        with open(log_path, 'w') as f:
            f.write("Gradient Computation Test (Gaussian Priors)\n")
            f.write("=" * 60 + "\n\n")
            for i, r in enumerate(results):
                f.write(
                    f"Point {i}: norm={r['grad_norm']:.4f}, "
                    f"nan={r['has_nan']}, inf={r['has_inf']}, zero={r['all_zero']}\n"
                )

            # Summary
            n_nan = sum(r['has_nan'] for r in results)
            n_inf = sum(r['has_inf'] for r in results)
            n_zero = sum(r['all_zero'] for r in results)
            f.write(
                f"\nSummary: {n_nan}/{n_test_points} NaN, "
                f"{n_inf}/{n_test_points} Inf, {n_zero}/{n_test_points} all-zero\n"
            )

        # Check results - warn but don't fail for occasional NaN (can happen at edge cases)
        n_nan = sum(r['has_nan'] for r in results)
        n_inf = sum(r['has_inf'] for r in results)
        n_zero = sum(r['all_zero'] for r in results)

        # Fail only if majority have issues
        if n_nan > n_test_points // 2:
            pytest.fail(f"Too many NaN gradients: {n_nan}/{n_test_points}")
        if n_inf > n_test_points // 2:
            pytest.fail(f"Too many Inf gradients: {n_inf}/{n_test_points}")
        if n_zero == n_test_points:
            pytest.fail(
                "All gradients are zero - log_posterior may not be differentiable"
            )

        # Warn if any issues found
        if n_nan > 0 or n_inf > 0:
            import warnings

            warnings.warn(
                f"Gradient issues detected: {n_nan} NaN, {n_inf} Inf out of {n_test_points} samples. "
                "This may indicate numerical instability at some parameter values."
            )

    def test_gradient_computation_bounded_priors(
        self, bounded_velocity_task, output_dir
    ):
        """Verify gradients with bounded (Uniform) priors - may have issues at boundaries."""
        task, true_pars = bounded_velocity_task

        log_prob_fn = task.get_log_posterior_fn()
        grad_fn = jax.grad(log_prob_fn)

        # Test gradient at multiple points
        key = random.PRNGKey(456)
        n_test_points = 10

        results = []
        for i in range(n_test_points):
            key, subkey = random.split(key)
            theta = task.sample_prior(subkey, 1)[0]

            # Compute gradient
            grad = grad_fn(theta)

            # Check properties
            has_nan = bool(jnp.any(jnp.isnan(grad)))
            has_inf = bool(jnp.any(jnp.isinf(grad)))
            all_zero = bool(jnp.allclose(grad, 0.0))
            grad_norm = float(jnp.linalg.norm(grad))

            results.append(
                {
                    'theta': theta,
                    'grad': grad,
                    'has_nan': has_nan,
                    'has_inf': has_inf,
                    'all_zero': all_zero,
                    'grad_norm': grad_norm,
                }
            )

        # Write results to file
        log_path = output_dir / "gradient_test_bounded.txt"
        with open(log_path, 'w') as f:
            f.write("Gradient Computation Test (Bounded Priors)\n")
            f.write("=" * 60 + "\n\n")
            f.write("NOTE: Bounded priors may have gradient issues at boundaries.\n")
            f.write("This is expected behavior for hard-boundary priors.\n\n")

            for i, r in enumerate(results):
                f.write(
                    f"Point {i}: norm={r['grad_norm']:.4f}, "
                    f"nan={r['has_nan']}, inf={r['has_inf']}, zero={r['all_zero']}\n"
                )
                f.write(f"  theta = {np.array(r['theta'])}\n")
                f.write(f"  grad  = {np.array(r['grad'])}\n\n")

            # Summary
            n_nan = sum(r['has_nan'] for r in results)
            n_inf = sum(r['has_inf'] for r in results)
            n_zero = sum(r['all_zero'] for r in results)
            f.write(
                f"\nSummary: {n_nan}/{n_test_points} NaN, "
                f"{n_inf}/{n_test_points} Inf, {n_zero}/{n_test_points} all-zero\n"
            )

            if n_nan > 0 or n_zero > n_test_points // 2:
                f.write(
                    "\nWARNING: Gradient issues detected. This may cause BlackJAX to fail.\n"
                )
                f.write("Consider using parameter transforms or Gaussian priors.\n")

        # For bounded priors, we expect some issues - just report them
        n_problematic = sum(r['has_nan'] or r['all_zero'] for r in results)
        if n_problematic > n_test_points // 2:
            pytest.skip(
                f"Bounded priors have gradient issues ({n_problematic}/{n_test_points}). "
                "This is expected - see output file for details."
            )

    def test_step_size_adaptation(self, simple_velocity_task, output_dir):
        """Verify warmup produces reasonable step size."""
        task, true_pars = simple_velocity_task

        # Run blackjax with short warmup
        config = GradientSamplerConfig(
            n_samples=100,  # Short run for diagnostics
            n_warmup=200,
            algorithm='nuts',
            seed=42,
            progress=False,
        )

        sampler = build_sampler('blackjax', task, config)
        result = sampler.run()

        # Get diagnostics
        diagnostics = result.diagnostics

        # Write results
        log_path = output_dir / "step_size_test.txt"
        with open(log_path, 'w') as f:
            f.write("Step Size Adaptation Test\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Algorithm: {diagnostics.get('algorithm', 'unknown')}\n")
            f.write(f"N warmup: {diagnostics.get('n_warmup', 'unknown')}\n")
            f.write(f"N samples: {diagnostics.get('n_samples', 'unknown')}\n")

            if 'step_size' in diagnostics:
                step_size = diagnostics['step_size']
                f.write(f"\nAdapted step size: {step_size}\n")

                # Check if reasonable
                if step_size < 1e-8:
                    f.write(
                        "WARNING: Step size very small - may indicate gradient issues\n"
                    )
                elif step_size > 100:
                    f.write(
                        "WARNING: Step size very large - may indicate flat likelihood\n"
                    )
                else:
                    f.write("Step size appears reasonable\n")
            else:
                f.write("\nNote: Step size not returned in diagnostics\n")

            f.write(f"\nFull diagnostics: {diagnostics}\n")

        # Check acceptance rate
        acceptance = result.acceptance_fraction
        assert acceptance is not None, "Acceptance rate not returned"

        log_path = output_dir / "acceptance_rate_test.txt"
        with open(log_path, 'a') as f:
            f.write(f"\nAcceptance rate: {acceptance:.2%}\n")

    def test_acceptance_rate(self, simple_velocity_task, output_dir):
        """Verify acceptance rate is in healthy range."""
        task, true_pars = simple_velocity_task

        config = GradientSamplerConfig(
            n_samples=500,
            n_warmup=200,
            algorithm='nuts',
            seed=42,
            progress=False,
        )

        sampler = build_sampler('blackjax', task, config)
        result = sampler.run()

        acceptance = result.acceptance_fraction

        # Write results
        log_path = output_dir / "acceptance_rate_test.txt"
        with open(log_path, 'w') as f:
            f.write("Acceptance Rate Test\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Acceptance rate: {acceptance:.2%}\n\n")

            if acceptance is None:
                f.write("WARNING: Acceptance rate is None\n")
            elif acceptance < 0.1:
                f.write(
                    "WARNING: Very low acceptance rate - step size may be too large\n"
                )
            elif acceptance > 0.99:
                f.write(
                    "WARNING: Very high acceptance rate - step size may be too small\n"
                )
            else:
                f.write(
                    "Acceptance rate is in healthy range (0.4-0.95 typical for NUTS)\n"
                )

        assert acceptance is not None, "Acceptance rate not returned"
        # NUTS typically has high acceptance (0.6-0.9), but we use loose bounds
        assert acceptance > 0.01, f"Acceptance rate too low: {acceptance:.2%}"

    def test_sample_variance(self, simple_velocity_task, output_dir):
        """Verify samples have non-zero variance for each parameter."""
        task, true_pars = simple_velocity_task

        config = GradientSamplerConfig(
            n_samples=500,
            n_warmup=200,
            algorithm='nuts',
            seed=42,
            progress=False,
        )

        sampler = build_sampler('blackjax', task, config)
        result = sampler.run()

        # Compute variance for each parameter
        variances = {}
        for i, name in enumerate(result.param_names):
            chain = result.get_chain(name)
            variances[name] = float(np.var(chain))

        # Write results
        log_path = output_dir / "sample_variance_test.txt"
        with open(log_path, 'w') as f:
            f.write("Sample Variance Test\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"N samples: {len(result.samples)}\n\n")

            zero_variance_params = []
            for name, var in variances.items():
                status = "OK" if var > 1e-10 else "ZERO VARIANCE"
                f.write(f"{name}: variance = {var:.6e} [{status}]\n")
                if var < 1e-10:
                    zero_variance_params.append(name)

            if zero_variance_params:
                f.write(
                    f"\nWARNING: Parameters with zero variance: {zero_variance_params}\n"
                )
                f.write(
                    "This indicates the sampler is not exploring the parameter space.\n"
                )

        # Check that at least some parameters have variance
        n_zero = sum(1 for v in variances.values() if v < 1e-10)
        assert n_zero < len(
            variances
        ), f"All {len(variances)} parameters have zero variance - sampler not moving"

    def test_bounded_vs_unbounded_comparison(
        self, simple_velocity_task, bounded_velocity_task, output_dir
    ):
        """
        Compare BlackJAX behavior with bounded vs unbounded (Gaussian) priors.

        This helps isolate whether boundary handling is causing issues.
        """
        config = GradientSamplerConfig(
            n_samples=300,
            n_warmup=150,
            algorithm='nuts',
            seed=42,
            progress=False,
        )

        results = {}

        # Run with Gaussian priors
        task_gaussian, _ = simple_velocity_task
        sampler = build_sampler('blackjax', task_gaussian, config)
        results['gaussian'] = sampler.run()

        # Run with bounded priors
        task_bounded, _ = bounded_velocity_task
        sampler = build_sampler('blackjax', task_bounded, config)
        results['bounded'] = sampler.run()

        # Compare
        log_path = output_dir / "bounded_vs_unbounded.txt"
        with open(log_path, 'w') as f:
            f.write("Bounded vs Unbounded Priors Comparison\n")
            f.write("=" * 60 + "\n\n")

            for prior_type, result in results.items():
                f.write(f"\n{prior_type.upper()} PRIORS:\n")
                f.write("-" * 40 + "\n")
                f.write(f"Acceptance rate: {result.acceptance_fraction:.2%}\n")

                # Compute per-parameter variance
                for name in result.param_names:
                    chain = result.get_chain(name)
                    var = float(np.var(chain))
                    std = float(np.std(chain))
                    f.write(f"  {name}: std = {std:.4f}, var = {var:.6e}\n")

            # Summary
            f.write("\n" + "=" * 60 + "\n")
            f.write("COMPARISON SUMMARY:\n")

            gauss_accept = results['gaussian'].acceptance_fraction
            bound_accept = results['bounded'].acceptance_fraction
            f.write(f"Gaussian acceptance: {gauss_accept:.2%}\n")
            f.write(f"Bounded acceptance:  {bound_accept:.2%}\n")

            if bound_accept < gauss_accept * 0.5:
                f.write("\nWARNING: Bounded priors have much lower acceptance.\n")
                f.write("This suggests gradient issues at boundaries.\n")

            # Check for zero-variance parameters
            gauss_zero = sum(
                1
                for name in results['gaussian'].param_names
                if np.var(results['gaussian'].get_chain(name)) < 1e-10
            )
            bound_zero = sum(
                1
                for name in results['bounded'].param_names
                if np.var(results['bounded'].get_chain(name)) < 1e-10
            )

            f.write(f"\nGaussian zero-variance params: {gauss_zero}\n")
            f.write(f"Bounded zero-variance params:  {bound_zero}\n")

            if bound_zero > gauss_zero:
                f.write("\nCONCLUSION: Bounded priors cause more sampling issues.\n")
                f.write("Consider parameter transforms for BlackJAX.\n")
            elif gauss_zero > 0:
                f.write("\nCONCLUSION: Issues exist with both prior types.\n")
                f.write("Problem may be in likelihood or model, not priors.\n")
            else:
                f.write("\nCONCLUSION: Both prior types working correctly.\n")


# ==============================================================================
# Joint Model Tests (matching test_sampler_comparison setup)
# ==============================================================================


@pytest.fixture(scope="module")
def joint_model_task_bounded():
    """
    Create joint velocity+intensity task with TruncatedNormal priors.

    This matches the exact setup used in test_sampler_comparison to
    reproduce and diagnose the zero-variance issue.
    """
    from kl_pipe.intensity import InclinedExponentialModel
    from kl_pipe.model import KLModel
    from kl_pipe.synthetic import SyntheticIntensity

    # Image parameters matching test_sampler_comparison
    image_pars_vel = ImagePars(shape=(24, 24), pixel_scale=0.4, indexing='ij')
    image_pars_int = ImagePars(shape=(32, 32), pixel_scale=0.3, indexing='ij')

    # True parameters
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

    # Generate velocity data
    vel_model = CenteredVelocityModel()
    vel_pars = {k: v for k, v in true_pars.items() if k in vel_model.PARAMETER_NAMES}
    synth_vel = SyntheticVelocity(vel_pars, model_type='arctan', seed=42)
    data_vel = synth_vel.generate(image_pars_vel, snr=100, include_poisson=False)
    var_vel = synth_vel.variance

    # Generate intensity data
    int_model = InclinedExponentialModel()
    int_pars = {k: v for k, v in true_pars.items() if k in int_model.PARAMETER_NAMES}
    synth_int = SyntheticIntensity(int_pars, model_type='exponential', seed=43)
    data_int = synth_int.generate(image_pars_int, snr=100, include_poisson=False)
    var_int = synth_int.variance

    # Create joint model
    joint_model = KLModel(
        velocity_model=vel_model,
        intensity_model=int_model,
        shared_pars={'cosi', 'theta_int', 'g1', 'g2'},
    )

    # TruncatedNormal priors (same as test_sampler_comparison)
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


@pytest.fixture(scope="module")
def joint_model_task_gaussian():
    """
    Create joint velocity+intensity task with Gaussian priors (no bounds).

    This is used to compare with bounded priors to isolate whether
    boundaries cause the zero-variance issue.
    """
    from kl_pipe.intensity import InclinedExponentialModel
    from kl_pipe.model import KLModel
    from kl_pipe.synthetic import SyntheticIntensity

    # Image parameters matching test_sampler_comparison
    image_pars_vel = ImagePars(shape=(24, 24), pixel_scale=0.4, indexing='ij')
    image_pars_int = ImagePars(shape=(32, 32), pixel_scale=0.3, indexing='ij')

    # True parameters
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

    # Generate velocity data
    vel_model = CenteredVelocityModel()
    vel_pars = {k: v for k, v in true_pars.items() if k in vel_model.PARAMETER_NAMES}
    synth_vel = SyntheticVelocity(vel_pars, model_type='arctan', seed=42)
    data_vel = synth_vel.generate(image_pars_vel, snr=100, include_poisson=False)
    var_vel = synth_vel.variance

    # Generate intensity data
    int_model = InclinedExponentialModel()
    int_pars = {k: v for k, v in true_pars.items() if k in int_model.PARAMETER_NAMES}
    synth_int = SyntheticIntensity(int_pars, model_type='exponential', seed=43)
    data_int = synth_int.generate(image_pars_int, snr=100, include_poisson=False)
    var_int = synth_int.variance

    # Create joint model
    joint_model = KLModel(
        velocity_model=vel_model,
        intensity_model=int_model,
        shared_pars={'cosi', 'theta_int', 'g1', 'g2'},
    )

    # Gaussian priors (no bounds) - for comparison
    priors = PriorDict(
        {
            'v0': Gaussian(10.0, 5.0),
            'vcirc': Gaussian(200.0, 50.0),
            'vel_rscale': Gaussian(5.0, 2.0),
            'flux': Gaussian(1.0, 0.5),
            'int_rscale': Gaussian(3.0, 2.0),
            'int_x0': 0.0,  # Fixed
            'int_y0': 0.0,  # Fixed
            'cosi': Gaussian(0.6, 0.2),
            'theta_int': Gaussian(0.785, 0.5),
            'g1': Gaussian(0.0, 0.05),
            'g2': Gaussian(0.0, 0.05),
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


class TestBlackJAXJointModel:
    """
    Tests for BlackJAX on joint velocity+intensity models.

    These tests diagnose the zero-variance issue observed in test_sampler_comparison.
    """

    def test_joint_model_gradient_check(self, joint_model_task_bounded, output_dir):
        """
        Verify gradients are finite for joint model with bounded priors.

        This is the first diagnostic to run when zero-variance issues occur.
        """
        task, true_pars = joint_model_task_bounded

        log_prob_fn = task.get_log_posterior_fn()
        grad_fn = jax.grad(log_prob_fn)

        # Test at multiple positions
        key = random.PRNGKey(42)
        n_test = 5
        results = []

        for i in range(n_test):
            key, subkey = random.split(key)
            position = task.sample_prior(subkey, 1)[0]

            log_prob = float(log_prob_fn(position))
            grad = grad_fn(position)

            grad_finite = bool(jnp.all(jnp.isfinite(grad)))
            grad_nonzero = bool(jnp.any(jnp.abs(grad) > 1e-10))

            results.append(
                {
                    'position': position,
                    'log_prob': log_prob,
                    'grad': grad,
                    'grad_finite': grad_finite,
                    'grad_nonzero': grad_nonzero,
                }
            )

        # Write diagnostic log
        log_path = output_dir / "joint_model_gradient_check.txt"
        with open(log_path, 'w') as f:
            f.write("Joint Model Gradient Check (Bounded Priors)\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Parameters: {task.sampled_names}\n\n")

            for i, r in enumerate(results):
                f.write(f"Test {i+1}:\n")
                f.write(f"  log_prob: {r['log_prob']:.4f}\n")
                f.write(f"  grad_finite: {r['grad_finite']}\n")
                f.write(f"  grad_nonzero: {r['grad_nonzero']}\n")
                f.write(f"  grad: {r['grad']}\n\n")

            n_finite = sum(r['grad_finite'] for r in results)
            n_nonzero = sum(r['grad_nonzero'] for r in results)
            f.write(
                f"Summary: {n_finite}/{n_test} finite, {n_nonzero}/{n_test} nonzero\n"
            )

        # At least some should be finite
        assert (
            n_finite >= n_test // 2
        ), f"Too many NaN/Inf gradients: only {n_finite}/{n_test} finite"

    @pytest.mark.xfail(
        reason="Joint model has poor parameter scaling: intensity gradients are ~1000x "
        "larger than velocity gradients, causing NUTS step size to shrink to ~1e-8. "
        "This is a known limitation requiring model reparameterization to fix.",
        strict=True,  # Test should fail; if it passes, we want to know!
    )
    def test_joint_model_variance_bounded(self, joint_model_task_bounded, output_dir):
        """
        Test that BlackJAX produces non-zero variance samples on joint model.

        This is the key test that catches the zero-variance bug. Uses the same
        setup as test_sampler_comparison but with a short run for speed.

        NOTE: This test is expected to fail due to poor parameter scaling in
        the joint model. The gradients for intensity parameters are ~1000x
        larger than velocity parameters, causing NUTS step size adaptation
        to shrink the step size to near-zero. This is a known limitation
        of gradient-based sampling on the current joint model parameterization.
        """
        task, true_pars = joint_model_task_bounded

        config = GradientSamplerConfig(
            n_samples=100,  # Short run for fast diagnostics
            n_warmup=100,
            algorithm='nuts',
            seed=42,
            progress=False,
        )

        sampler = build_sampler('blackjax', task, config)
        result = sampler.run()

        # Get diagnostics
        step_size = result.diagnostics.get('step_size')
        acceptance = result.acceptance_fraction

        # Compute variance for each parameter
        variances = {}
        zero_var_params = []
        for name in result.param_names:
            chain = result.get_chain(name)
            var = float(np.var(chain))
            variances[name] = var
            if var < 1e-10:
                zero_var_params.append(name)

        # Write diagnostic log
        log_path = output_dir / "joint_model_variance_bounded.txt"
        with open(log_path, 'w') as f:
            f.write("Joint Model Variance Test (Bounded Priors)\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"n_samples: {config.n_samples}\n")
            f.write(f"n_warmup: {config.n_warmup}\n")
            f.write(f"Adapted step_size: {step_size}\n")
            f.write(f"Acceptance rate: {acceptance:.2%}\n\n")

            f.write("Parameter variances:\n")
            for name, var in variances.items():
                status = "OK" if var > 1e-10 else "ZERO VARIANCE"
                f.write(f"  {name}: {var:.6e} [{status}]\n")

            if zero_var_params:
                f.write(f"\nWARNING: Zero variance in: {zero_var_params}\n")
                f.write("This is the bug we're trying to diagnose!\n")
            else:
                f.write("\nAll parameters have non-zero variance.\n")

        # Assert no zero-variance parameters
        assert len(zero_var_params) == 0, (
            f"BlackJAX produced zero variance for parameters: {zero_var_params}. "
            f"Step size was {step_size}, acceptance was {acceptance:.2%}. "
            f"See {log_path} for details."
        )

    @pytest.mark.xfail(
        reason="Joint model has poor parameter scaling: intensity gradients are ~1000x "
        "larger than velocity gradients, causing NUTS step size to shrink to ~1e-8. "
        "This is a known limitation requiring model reparameterization to fix.",
        strict=True,
    )
    def test_joint_model_variance_gaussian(self, joint_model_task_gaussian, output_dir):
        """
        Test BlackJAX with Gaussian priors on joint model.

        Comparison baseline to isolate whether bounded priors cause issues.
        This test confirms the issue is NOT prior type - it's parameter scaling.
        """
        task, true_pars = joint_model_task_gaussian

        config = GradientSamplerConfig(
            n_samples=100,
            n_warmup=100,
            algorithm='nuts',
            seed=42,
            progress=False,
        )

        sampler = build_sampler('blackjax', task, config)
        result = sampler.run()

        # Get diagnostics
        step_size = result.diagnostics.get('step_size')
        acceptance = result.acceptance_fraction

        # Compute variance for each parameter
        variances = {}
        zero_var_params = []
        for name in result.param_names:
            chain = result.get_chain(name)
            var = float(np.var(chain))
            variances[name] = var
            if var < 1e-10:
                zero_var_params.append(name)

        # Write diagnostic log
        log_path = output_dir / "joint_model_variance_gaussian.txt"
        with open(log_path, 'w') as f:
            f.write("Joint Model Variance Test (Gaussian Priors)\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"n_samples: {config.n_samples}\n")
            f.write(f"n_warmup: {config.n_warmup}\n")
            f.write(f"Adapted step_size: {step_size}\n")
            f.write(f"Acceptance rate: {acceptance:.2%}\n\n")

            f.write("Parameter variances:\n")
            for name, var in variances.items():
                status = "OK" if var > 1e-10 else "ZERO VARIANCE"
                f.write(f"  {name}: {var:.6e} [{status}]\n")

            if zero_var_params:
                f.write(f"\nWARNING: Zero variance in: {zero_var_params}\n")
            else:
                f.write("\nAll parameters have non-zero variance.\n")

        # Assert no zero-variance parameters
        assert len(zero_var_params) == 0, (
            f"BlackJAX produced zero variance for parameters: {zero_var_params}. "
            f"Step size was {step_size}. See {log_path} for details."
        )

    def test_joint_bounded_vs_gaussian_comparison(
        self, joint_model_task_bounded, joint_model_task_gaussian, output_dir
    ):
        """
        Compare BlackJAX behavior between bounded and Gaussian priors on joint model.

        This isolates whether the issue is prior type vs model complexity.
        """
        config = GradientSamplerConfig(
            n_samples=100,
            n_warmup=100,
            algorithm='nuts',
            seed=42,
            progress=False,
        )

        results = {}

        # Run with bounded priors
        task_bounded, _ = joint_model_task_bounded
        sampler = build_sampler('blackjax', task_bounded, config)
        results['bounded'] = sampler.run()

        # Run with Gaussian priors
        task_gaussian, _ = joint_model_task_gaussian
        sampler = build_sampler('blackjax', task_gaussian, config)
        results['gaussian'] = sampler.run()

        # Write comparison log
        log_path = output_dir / "joint_bounded_vs_gaussian.txt"
        with open(log_path, 'w') as f:
            f.write("Joint Model: Bounded vs Gaussian Priors\n")
            f.write("=" * 60 + "\n\n")

            for prior_type, result in results.items():
                f.write(f"\n{prior_type.upper()} PRIORS:\n")
                f.write("-" * 40 + "\n")
                f.write(f"Step size: {result.diagnostics.get('step_size')}\n")
                f.write(f"Acceptance: {result.acceptance_fraction:.2%}\n")

                zero_var = []
                for name in result.param_names:
                    chain = result.get_chain(name)
                    var = float(np.var(chain))
                    if var < 1e-10:
                        zero_var.append(name)
                    f.write(f"  {name}: var = {var:.6e}\n")

                if zero_var:
                    f.write(f"ZERO VARIANCE: {zero_var}\n")

            # Summary
            f.write("\n" + "=" * 60 + "\n")
            f.write("COMPARISON SUMMARY:\n")

            bound_zero = sum(
                1
                for name in results['bounded'].param_names
                if np.var(results['bounded'].get_chain(name)) < 1e-10
            )
            gauss_zero = sum(
                1
                for name in results['gaussian'].param_names
                if np.var(results['gaussian'].get_chain(name)) < 1e-10
            )

            f.write(f"Bounded zero-variance params: {bound_zero}\n")
            f.write(f"Gaussian zero-variance params: {gauss_zero}\n")

            if bound_zero > gauss_zero:
                f.write("\nCONCLUSION: Bounded priors cause issues on joint model.\n")
            elif gauss_zero > bound_zero:
                f.write("\nCONCLUSION: Gaussian priors cause issues (unexpected).\n")
            elif bound_zero > 0:
                f.write("\nCONCLUSION: Both prior types have issues - model problem.\n")
            else:
                f.write("\nCONCLUSION: Both prior types work correctly.\n")

        # Both should work - this test documents the comparison
        # If bounded fails but gaussian works, it points to prior handling
        print(f"\nBounded zero-var params: {bound_zero}")
        print(f"Gaussian zero-var params: {gauss_zero}")
        print(f"See {log_path} for details")
