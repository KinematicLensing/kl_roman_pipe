"""
Parameter recovery tests using gradient-based optimization.

Similar to test_likelihood_slices.py, but uses scipy.optimize with
analytical gradients from JAX instead of brute-force likelihood slicing.

This tests:
1. That JAX gradients are computed correctly
2. That optimization can recover parameters efficiently
3. Comparison of optimizer performance vs likelihood slicing

The optimizer tests complement likelihood slicing by:
- Being much faster (gradients vs brute force)
- Testing in realistic inference scenarios
- Validating gradient implementations
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize
from typing import Dict, Tuple

from kl_pipe.velocity import CenteredVelocityModel, OffsetVelocityModel
from kl_pipe.intensity import InclinedExponentialModel
from kl_pipe.model import KLModel
from kl_pipe.parameters import ImagePars
from kl_pipe.synthetic import SyntheticVelocity, SyntheticIntensity
from kl_pipe.likelihood import (
    create_jitted_likelihood_velocity,
    create_jitted_likelihood_intensity,
    create_jitted_likelihood_joint,
)
from kl_pipe.utils import build_map_grid_from_image_pars, get_test_dir
from kl_pipe.diagnostics import plot_data_comparison_panels

from test_utils import (
    TestConfig,
    check_parameter_recovery,
    assert_parameter_recovery,
    plot_parameter_comparison,
    check_degenerate_product_recovery,
)


# ==============================================================================
# Pytest Fixtures
# ==============================================================================


@pytest.fixture(scope="module")
def test_config():
    """Test configuration fixture."""
    out_dir = get_test_dir() / "out" / "optimizer_recovery"
    config = TestConfig(out_dir, include_poisson_noise=False)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    return config


@pytest.fixture
def velocity_grids(test_config):
    """Pre-computed coordinate grids for velocity tests."""
    X, Y = build_map_grid_from_image_pars(
        test_config.image_pars_velocity, unit='arcsec', centered=True
    )
    return X, Y


@pytest.fixture
def intensity_grids(test_config):
    """Pre-computed coordinate grids for intensity tests."""
    X, Y = build_map_grid_from_image_pars(
        test_config.image_pars_intensity, unit='arcsec', centered=True
    )
    return X, Y


# ==============================================================================
# Helper Functions
# ==============================================================================


def generate_synthetic_velocity_data(
    model_class,
    true_pars: Dict[str, float],
    image_pars: ImagePars,
    snr: float,
    config: TestConfig,
) -> Tuple[jnp.ndarray, jnp.ndarray, float]:
    """Generate synthetic velocity data with noise."""
    model = model_class()
    vel_pars = {k: v for k, v in true_pars.items() if k in model.PARAMETER_NAMES}

    synth = SyntheticVelocity(vel_pars, model_type='arctan', seed=config.seed)
    data_noisy = synth.generate(
        image_pars,
        snr=snr,
        seed=config.seed,
        include_poisson=config.include_poisson_noise,
    )
    variance = synth.variance
    data_true = synth.data_true

    return data_true, data_noisy, variance


def generate_synthetic_intensity_data(
    model_class,
    true_pars: Dict[str, float],
    image_pars: ImagePars,
    snr: float,
    config: TestConfig,
) -> Tuple[jnp.ndarray, jnp.ndarray, float]:
    """Generate synthetic intensity data with noise."""
    model = model_class()
    int_pars = {k: v for k, v in true_pars.items() if k in model.PARAMETER_NAMES}

    synth = SyntheticIntensity(int_pars, model_type='exponential', seed=config.seed)
    data_noisy = synth.generate(
        image_pars,
        snr=snr,
        seed=config.seed,
        include_poisson=config.include_poisson_noise,
        sersic_backend=config.sersic_backend,
    )
    variance = synth.variance
    data_true = synth.data_true

    return data_true, data_noisy, variance


def optimize_with_gradients(
    log_like_fn: callable,
    theta_init: jnp.ndarray,
    bounds: list = None,
    method: str = 'L-BFGS-B',
) -> Tuple[jnp.ndarray, dict]:
    """
    Optimize using scipy with JAX gradients.

    Parameters
    ----------
    log_like_fn : callable
        JIT-compiled log-likelihood function.
    theta_init : jnp.ndarray
        Initial parameter guess.
    bounds : list, optional
        Parameter bounds as [(low, high), ...].
    method : str, optional
        Optimization method. Default is 'L-BFGS-B'.

    Returns
    -------
    theta_opt : jnp.ndarray
        Optimized parameters.
    result : dict
        Optimization result from scipy.
    """

    # Create gradient function using JAX
    grad_fn = jax.jit(jax.grad(log_like_fn))

    # Define objective (negative log-likelihood)
    def objective(theta):
        return -float(log_like_fn(jnp.array(theta)))

    def gradient(theta):
        return -np.array(grad_fn(jnp.array(theta)))

    # Run optimization
    # ftol is the relative change tolerance: (f_k - f_{k+1})/max{|f_k|,|f_{k+1}|,1} <= ftol
    result = minimize(
        objective,
        x0=np.array(theta_init),
        method=method,
        jac=gradient,
        bounds=bounds,
        options={'maxiter': 2000, 'ftol': 1e-8},
    )

    return jnp.array(result.x), result


# ==============================================================================
# Tests: Velocity Models
# ==============================================================================


@pytest.mark.parametrize("snr", [1000, 50, 10])
def test_optimize_centered_velocity_base(snr, test_config, velocity_grids):
    """Test optimizer recovery for CenteredVelocityModel (no shear)."""

    X, Y = velocity_grids

    true_pars = {
        'cosi': 0.6,
        'theta_int': 0.785,
        'g1': 0.0,
        'g2': 0.0,
        'v0': 10.0,
        'vcirc': 200.0,
        'vel_rscale': 5.0,
    }

    model = CenteredVelocityModel()
    theta_true = model.pars2theta(true_pars)

    # Generate data
    data_true, data_noisy, variance = generate_synthetic_velocity_data(
        CenteredVelocityModel,
        true_pars,
        test_config.image_pars_velocity,
        snr,
        test_config,
    )  # Diagnostic plots
    model_eval = model(theta_true, 'obs', X, Y)
    test_name = f"opt_centered_vel_base_snr{snr}"
    plot_data_comparison_panels(
        data_noisy=np.asarray(data_noisy),
        data_true=np.asarray(data_true),
        model_eval=np.asarray(model_eval),
        test_name=test_name,
        output_dir=test_config.output_dir / test_name,
        data_type='velocity',
        variance=variance,
        n_params=len(model.PARAMETER_NAMES),
        enable_plots=test_config.enable_plots,
    )

    # Create likelihood with gradients
    log_like = create_jitted_likelihood_velocity(
        model, test_config.image_pars_velocity, variance, data_noisy
    )

    # Add small perturbation to initial guess (5% random noise)
    rng = np.random.RandomState(test_config.seed)
    theta_init = theta_true + 0.05 * theta_true * rng.randn(len(theta_true))

    # Define bounds with Tully-Fisher motivated vcirc constraints
    # Tully-Fisher: M_* ∝ v_circ^4 with ~0.3 dex scatter
    # For scale length ~5 arcsec, expect vcirc ~ 100-300 km/s for typical galaxies
    # Tighter bounds help break cosi-shear degeneracy via v_los = sqrt(1-cos²i) * v_circ
    bounds = [
        (0.1, 0.99),  # cosi (avoid edge-on/face-on singularities)
        (0.0, np.pi),  # theta_int
        (-0.1, 0.1),  # g1 (weak lensing regime)
        (-0.1, 0.1),  # g2 (weak lensing regime)
        (0.0, 50.0),  # v0 (systemic velocity)
        (100.0, 350.0),  # vcirc (Tully-Fisher motivated, was 50-500)
        (1.0, 20.0),  # vel_rscale
    ]

    # Optimize
    theta_opt, result = optimize_with_gradients(log_like, theta_init, bounds)

    # Check convergence
    assert result.success, f"Optimization failed: {result.message}"

    # Evaluate model at optimized parameters for diagnostic plots
    model_eval_opt = model(theta_opt, 'obs', X, Y)
    plot_data_comparison_panels(
        data_noisy=np.asarray(data_noisy),
        data_true=np.asarray(data_true),
        model_eval=np.asarray(model_eval_opt),
        test_name=f"{test_name}_optimized",
        output_dir=test_config.output_dir / test_name,
        data_type='velocity',
        variance=variance,
        n_params=len(model.PARAMETER_NAMES),
        model_label='Optimized Model',
        enable_plots=test_config.enable_plots,
    )

    # Check parameter recovery
    recovery_stats = {}
    pars_opt = model.theta2pars(theta_opt)

    for param_name, true_val in true_pars.items():
        recovered_val = pars_opt[param_name]
        tolerance = test_config.get_tolerance(
            snr, param_name, true_val, 'velocity', test_type='optimizer'
        )
        passed, stats = check_parameter_recovery(
            recovered_val, true_val, tolerance, param_name
        )
        recovery_stats[param_name] = stats

    # Check degenerate product vcirc*sini
    product_passed, product_stats = check_degenerate_product_recovery(
        true_pars, pars_opt, snr=snr
    )

    # Create parameter comparison plot
    exclude_params = ['cosi', 'g1', 'g2']
    plot_parameter_comparison(
        true_pars,
        pars_opt,
        recovery_stats,
        test_name,
        test_config,
        snr,
        product_stats=product_stats,
        exclude_params=exclude_params,
    )

    # Check if product passed
    if not product_passed:
        print(f"\n⚠️  Degenerate product vcirc*sini: {product_stats['formula']}")
        print(
            f"    True: {product_stats['true']:.2f}, Recovered: {product_stats['recovered']:.2f}"
        )
        print(
            f"    Rel error: {product_stats['rel_error']:.1%} (tolerance: {product_stats['tolerance']:.1%})"
        )

    assert_parameter_recovery(
        recovery_stats,
        snr,
        'Optimizer: Centered velocity (base)',
        exclude_params=['cosi', 'g1', 'g2'],
    )
    assert (
        product_passed
    ), f"Degenerate product vcirc*sini not recovered: {product_stats['rel_error']:.1%} error"


@pytest.mark.parametrize("snr", [1000, 50, 10])
def test_optimize_offset_velocity(snr, test_config, velocity_grids):
    """Test optimizer recovery for OffsetVelocityModel with shear."""

    X, Y = velocity_grids

    true_pars = {
        'cosi': 0.6,
        'theta_int': 0.785,
        'g1': 0.02,
        'g2': -0.01,
        'v0': 10.0,
        'vcirc': 200.0,
        'vel_rscale': 5.0,
        'vel_x0': 1.5,
        'vel_y0': -1.0,
    }

    model = OffsetVelocityModel()
    theta_true = model.pars2theta(true_pars)

    # Generate data
    data_true, data_noisy, variance = generate_synthetic_velocity_data(
        OffsetVelocityModel,
        true_pars,
        test_config.image_pars_velocity,
        snr,
        test_config,
    )  # Diagnostic plots
    model_eval = model(theta_true, 'obs', X, Y)
    test_name = f"opt_offset_vel_snr{snr}"
    plot_data_comparison_panels(
        data_noisy=np.asarray(data_noisy),
        data_true=np.asarray(data_true),
        model_eval=np.asarray(model_eval),
        test_name=test_name,
        output_dir=test_config.output_dir / test_name,
        data_type='velocity',
        variance=variance,
        n_params=len(model.PARAMETER_NAMES),
        enable_plots=test_config.enable_plots,
    )

    # Create likelihood
    log_like = create_jitted_likelihood_velocity(
        model, test_config.image_pars_velocity, variance, data_noisy
    )

    # Initial guess with perturbation
    rng = np.random.RandomState(test_config.seed)
    theta_init = theta_true + 0.05 * theta_true * rng.randn(len(theta_true))

    # Bounds with Tully-Fisher motivated vcirc
    extent = (
        test_config.image_pars_velocity.shape[0]
        * test_config.image_pars_velocity.pixel_scale
        / 2
    )
    bounds = [
        (0.1, 0.99),  # cosi
        (0.0, np.pi),  # theta_int
        (-0.1, 0.1),  # g1
        (-0.1, 0.1),  # g2
        (0.0, 50.0),  # v0
        (100.0, 350.0),  # vcirc (~roughly Tully-Fisher motivated)
        (1.0, 20.0),  # vel_rscale
        (-extent, extent),  # vel_x0
        (-extent, extent),  # vel_y0
    ]

    # Optimize
    theta_opt, result = optimize_with_gradients(log_like, theta_init, bounds)
    assert result.success, f"Optimization failed: {result.message}"

    # Evaluate model at optimized parameters
    model_eval_opt = model(theta_opt, 'obs', X, Y)
    plot_data_comparison_panels(
        data_noisy=np.asarray(data_noisy),
        data_true=np.asarray(data_true),
        model_eval=np.asarray(model_eval_opt),
        test_name=f"{test_name}_optimized",
        output_dir=test_config.output_dir / test_name,
        data_type='velocity',
        variance=variance,
        n_params=len(model.PARAMETER_NAMES),
        model_label='Optimized Model',
        enable_plots=test_config.enable_plots,
    )

    # Check recovery
    recovery_stats = {}
    pars_opt = model.theta2pars(theta_opt)

    for param_name, true_val in true_pars.items():
        recovered_val = pars_opt[param_name]
        tolerance = test_config.get_tolerance(
            snr, param_name, true_val, 'velocity', test_type='optimizer'
        )
        passed, stats = check_parameter_recovery(
            recovered_val, true_val, tolerance, param_name
        )
        recovery_stats[param_name] = stats

    # Check degenerate product vcirc*sini
    product_passed, product_stats = check_degenerate_product_recovery(
        true_pars, pars_opt, snr=snr
    )

    # Create parameter comparison plot
    exclude_params = ['cosi', 'g1', 'g2']
    plot_parameter_comparison(
        true_pars,
        pars_opt,
        recovery_stats,
        test_name,
        test_config,
        snr,
        product_stats=product_stats,
        exclude_params=exclude_params,
    )

    if not product_passed:
        print(f"\n⚠️  Degenerate product vcirc*sini: {product_stats['formula']}")
        print(
            f"    True: {product_stats['true']:.2f}, Recovered: {product_stats['recovered']:.2f}"
        )
        print(
            f"    Rel error: {product_stats['rel_error']:.1%} (tolerance: {product_stats['tolerance']:.1%})"
        )

    assert_parameter_recovery(
        recovery_stats,
        snr,
        'Optimizer: Offset velocity with shear',
        exclude_params=['cosi', 'g1', 'g2'],
    )
    assert (
        product_passed
    ), f"Degenerate product vcirc*sini not recovered: {product_stats['rel_error']:.1%} error"


# ==============================================================================
# Tests: Intensity Models
# ==============================================================================


@pytest.mark.parametrize("snr", [1000, 50, 10])
def test_optimize_inclined_exponential(snr, test_config, intensity_grids):
    """Test optimizer recovery for InclinedExponentialModel."""

    X, Y = intensity_grids

    true_pars = {
        'cosi': 0.7,
        'theta_int': 0.785,
        'g1': 0.03,
        'g2': -0.02,
        'flux': 1.0,
        'int_rscale': 3.0,
        'int_x0': 0.0,
        'int_y0': 0.0,
    }

    model = InclinedExponentialModel()
    theta_true = model.pars2theta(true_pars)

    # Generate data
    data_true, data_noisy, variance = generate_synthetic_intensity_data(
        InclinedExponentialModel,
        true_pars,
        test_config.image_pars_intensity,
        snr,
        test_config,
    )  # Diagnostic plots
    model_eval = model(theta_true, 'obs', X, Y)
    test_name = f"opt_inclined_exp_snr{snr}"
    plot_data_comparison_panels(
        data_noisy=np.asarray(data_noisy),
        data_true=np.asarray(data_true),
        model_eval=np.asarray(model_eval),
        test_name=test_name,
        output_dir=test_config.output_dir / test_name,
        data_type='intensity',
        variance=variance,
        n_params=len(model.PARAMETER_NAMES),
        enable_plots=test_config.enable_plots,
    )

    # Create likelihood
    log_like = create_jitted_likelihood_intensity(
        model, test_config.image_pars_intensity, variance, data_noisy
    )

    # Initial guess
    rng = np.random.RandomState(test_config.seed)
    theta_init = theta_true + 0.05 * theta_true * rng.randn(len(theta_true))

    # Bounds
    extent = (
        test_config.image_pars_intensity.shape[0]
        * test_config.image_pars_intensity.pixel_scale
        / 2
    )
    bounds = [
        (0.1, 0.99),  # cosi
        (0.0, np.pi),  # theta_int
        (-0.1, 0.1),  # g1
        (-0.1, 0.1),  # g2
        (0.1, 10.0),  # flux
        (0.5, 10.0),  # int_rscale
        (-extent, extent),  # int_x0
        (-extent, extent),  # int_y0
    ]

    # Optimize
    theta_opt, result = optimize_with_gradients(log_like, theta_init, bounds)
    assert result.success, f"Optimization failed: {result.message}"

    # Evaluate model at optimized parameters
    model_eval_opt = model(theta_opt, 'obs', X, Y)
    plot_data_comparison_panels(
        data_noisy=np.asarray(data_noisy),
        data_true=np.asarray(data_true),
        model_eval=np.asarray(model_eval_opt),
        test_name=f"{test_name}_optimized",
        output_dir=test_config.output_dir / test_name,
        data_type='intensity',
        variance=variance,
        n_params=len(model.PARAMETER_NAMES),
        model_label='Optimized Model',
        enable_plots=test_config.enable_plots,
    )

    # Check recovery
    recovery_stats = {}
    pars_opt = model.theta2pars(theta_opt)

    for param_name, true_val in true_pars.items():
        recovered_val = pars_opt[param_name]
        tolerance = test_config.get_tolerance(
            snr, param_name, true_val, 'intensity', test_type='optimizer'
        )
        passed, stats = check_parameter_recovery(
            recovered_val, true_val, tolerance, param_name
        )
        recovery_stats[param_name] = stats

    # Create parameter comparison plot (no vcirc*sini for intensity-only models)
    exclude_params = ['cosi', 'g1', 'g2']
    plot_parameter_comparison(
        true_pars,
        pars_opt,
        recovery_stats,
        test_name,
        test_config,
        snr,
        product_stats=None,
        exclude_params=exclude_params,
    )

    # No vcirc*sini check for intensity-only models (no velocity field)
    assert_parameter_recovery(
        recovery_stats,
        snr,
        'Optimizer: Inclined exponential intensity',
        exclude_params=['cosi', 'g1', 'g2'],
    )


# ==============================================================================
# Tests: PSF Recovery
# ==============================================================================


def test_optimize_inclined_exponential_with_psf(test_config, intensity_grids):
    """Test optimizer recovery for InclinedExponentialModel with PSF at SNR=1000."""
    import galsim as gs

    snr = 1000
    X, Y = intensity_grids

    true_pars = {
        'cosi': 0.7,
        'theta_int': 0.785,
        'g1': 0.03,
        'g2': -0.02,
        'flux': 1.0,
        'int_rscale': 3.0,
        'int_x0': 0.0,
        'int_y0': 0.0,
    }

    psf = gs.Gaussian(fwhm=0.625)

    # generate PSF-convolved data
    from kl_pipe.synthetic import SyntheticIntensity

    synth = SyntheticIntensity(true_pars, model_type='exponential', seed=test_config.seed, psf=psf)
    data_noisy = synth.generate(
        test_config.image_pars_intensity, snr=snr, seed=test_config.seed,
        include_poisson=test_config.include_poisson_noise, sersic_backend='galsim',
    )
    variance = synth.variance

    # configure model with same PSF
    model = InclinedExponentialModel()
    model.configure_psf(psf, test_config.image_pars_intensity.shape, test_config.image_pars_intensity.pixel_scale)
    theta_true = model.pars2theta(true_pars)

    log_like = create_jitted_likelihood_intensity(
        model, test_config.image_pars_intensity, variance, data_noisy
    )

    # optimize
    rng = np.random.RandomState(test_config.seed)
    theta_init = theta_true + 0.05 * theta_true * rng.randn(len(theta_true))

    extent = (
        test_config.image_pars_intensity.shape[0]
        * test_config.image_pars_intensity.pixel_scale
        / 2
    )
    bounds = [
        (0.1, 0.99),
        (0.0, np.pi),
        (-0.1, 0.1),
        (-0.1, 0.1),
        (0.1, 10.0),
        (0.5, 10.0),
        (-extent, extent),
        (-extent, extent),
    ]

    theta_opt, result = optimize_with_gradients(log_like, theta_init, bounds)
    assert result.success, f"Optimization failed: {result.message}"

    # check recovery
    recovery_stats = {}
    pars_opt = model.theta2pars(theta_opt)

    for param_name, true_val in true_pars.items():
        recovered_val = pars_opt[param_name]
        tolerance = test_config.get_tolerance(
            snr, param_name, true_val, 'intensity', test_type='optimizer', has_psf=True
        )
        passed, stats = check_parameter_recovery(
            recovered_val, true_val, tolerance, param_name
        )
        recovery_stats[param_name] = stats

    test_name = f"opt_inclined_exp_psf_snr{snr}"
    plot_parameter_comparison(
        true_pars, pars_opt, recovery_stats, test_name, test_config, snr,
        exclude_params=['cosi', 'g1', 'g2'],
    )

    model.clear_psf()
    assert_parameter_recovery(
        recovery_stats, snr, 'Optimizer: Inclined exponential (PSF)',
        exclude_params=['cosi', 'g1', 'g2'],
    )


def test_optimize_joint_with_psf(test_config, velocity_grids, intensity_grids):
    """Test optimizer recovery for joint model with PSF at SNR=1000."""
    import galsim as gs
    from kl_pipe.model import KLModel

    snr = 1000
    X_vel, Y_vel = velocity_grids
    X_int, Y_int = intensity_grids

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
        'int_x0': 0.0,
        'int_y0': 0.0,
    }

    psf = gs.Gaussian(fwhm=0.625)

    # generate data
    from kl_pipe.synthetic import (
        SyntheticIntensity, SyntheticVelocity, generate_sersic_intensity_2d,
    )
    from kl_pipe.velocity import OffsetVelocityModel

    synth_int = SyntheticIntensity(
        {k: v for k, v in true_pars.items() if k in InclinedExponentialModel.PARAMETER_NAMES},
        model_type='exponential', seed=test_config.seed, psf=psf,
    )
    data_int_noisy = synth_int.generate(
        test_config.image_pars_intensity, snr=snr, seed=test_config.seed,
        include_poisson=test_config.include_poisson_noise, sersic_backend='galsim',
    )
    variance_int = synth_int.variance

    int_pars_for_vel = {k: v for k, v in true_pars.items() if k in InclinedExponentialModel.PARAMETER_NAMES}
    flux_image = generate_sersic_intensity_2d(
        test_config.image_pars_velocity, backend='scipy', n_sersic=1.0,
        **{k: v for k, v in int_pars_for_vel.items() if k != 'n_sersic'},
    )

    synth_vel = SyntheticVelocity(
        {k: v for k, v in true_pars.items() if k in OffsetVelocityModel.PARAMETER_NAMES},
        model_type='arctan', seed=test_config.seed + 1,
        psf=psf, intensity_for_psf=flux_image,
    )
    data_vel_noisy = synth_vel.generate(
        test_config.image_pars_velocity, snr=snr, seed=test_config.seed + 1,
        include_poisson=test_config.include_poisson_noise,
    )
    variance_vel = synth_vel.variance

    # create joint model with PSF
    vel_model = OffsetVelocityModel()
    int_model = InclinedExponentialModel()
    joint_model = KLModel(
        vel_model, int_model, shared_pars={'cosi', 'theta_int', 'g1', 'g2'}
    )
    joint_model.configure_joint_psf(
        psf_vel=psf, psf_int=psf,
        image_shape_vel=test_config.image_pars_velocity.shape,
        pixel_scale_vel=test_config.image_pars_velocity.pixel_scale,
        image_shape_int=test_config.image_pars_intensity.shape,
        pixel_scale_int=test_config.image_pars_intensity.pixel_scale,
    )
    theta_true = joint_model.pars2theta(true_pars)

    log_like = create_jitted_likelihood_joint(
        joint_model,
        test_config.image_pars_velocity,
        test_config.image_pars_intensity,
        variance_vel,
        variance_int,
        data_vel_noisy,
        data_int_noisy,
    )

    # optimize
    rng = np.random.RandomState(test_config.seed)
    theta_init = theta_true + 0.05 * theta_true * rng.randn(len(theta_true))

    extent_vel = (
        test_config.image_pars_velocity.shape[0]
        * test_config.image_pars_velocity.pixel_scale / 2
    )
    extent_int = (
        test_config.image_pars_intensity.shape[0]
        * test_config.image_pars_intensity.pixel_scale / 2
    )

    # build bounds following joint model parameter ordering
    bounds = []
    for name in joint_model.PARAMETER_NAMES:
        if name == 'cosi':
            bounds.append((0.1, 0.99))
        elif name == 'theta_int':
            bounds.append((0.0, np.pi))
        elif name in ('g1', 'g2'):
            bounds.append((-0.1, 0.1))
        elif name == 'v0':
            bounds.append((0.0, 50.0))
        elif name == 'vcirc':
            bounds.append((100.0, 350.0))
        elif name == 'vel_rscale':
            bounds.append((1.0, 20.0))
        elif name in ('vel_x0', 'vel_y0'):
            bounds.append((-extent_vel, extent_vel))
        elif name == 'flux':
            bounds.append((0.1, 10.0))
        elif name == 'int_rscale':
            bounds.append((0.5, 10.0))
        elif name in ('int_x0', 'int_y0'):
            bounds.append((-extent_int, extent_int))
        else:
            bounds.append((None, None))

    theta_opt, result = optimize_with_gradients(log_like, theta_init, bounds)
    assert result.success, f"Optimization failed: {result.message}"

    # check recovery
    recovery_stats = {}
    pars_opt = joint_model.theta2pars(theta_opt)

    for param_name, true_val in true_pars.items():
        recovered_val = pars_opt[param_name]
        tolerance = test_config.get_tolerance(
            snr, param_name, true_val, 'velocity', test_type='optimizer', has_psf=True
        )
        passed, stats = check_parameter_recovery(
            recovered_val, true_val, tolerance, param_name
        )
        recovery_stats[param_name] = stats

    product_passed, product_stats = check_degenerate_product_recovery(
        true_pars, pars_opt, snr=snr
    )

    test_name = f"opt_joint_psf_snr{snr}"
    exclude_params = ['cosi', 'g1', 'g2']
    plot_parameter_comparison(
        true_pars, pars_opt, recovery_stats, test_name, test_config, snr,
        product_stats=product_stats, exclude_params=exclude_params,
    )

    assert_parameter_recovery(
        recovery_stats, snr, 'Optimizer: Joint model (PSF)',
        exclude_params=exclude_params,
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
