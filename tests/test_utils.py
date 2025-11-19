"""
Shared test utilities for parameter recovery tests.

This module contains common functions used by both likelihood slicing tests
and gradient-based optimizer tests. May expand further in the future.
"""

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Tuple, Optional, Callable

from kl_pipe.parameters import ImagePars
from kl_pipe.utils import get_test_dir


# ==============================================================================
# Test Configuration Data Structures
# ==============================================================================


class TestConfig:
    """
    Configuration container for parameter recovery tests.

    This makes it easy to pass configuration through pytest fixtures
    rather than using global variables.
    """

    __test__ = False  # tell pytest this is not a test class

    def __init__(
        self,
        output_dir: Path,
        enable_plots: bool = True,
        include_poisson_noise: bool = True,
        seed: int = 42,
    ):
        self.output_dir = output_dir
        # Plotting control
        self.enable_plots = enable_plots
        # Synthetic data generation
        self.include_poisson_noise = include_poisson_noise  # Whether to add shot noise
        self.seed = seed

        # SNR tolerance maps (can differ for velocity vs intensity)
        self.snr_tolerance_velocity = {
            100: 0.01,  # 1% tolerance
            50: 0.02,  # 2% tolerance
            30: 0.05,  # 5% tolerance
            10: 0.10,  # 10% tolerance
        }

        self.snr_tolerance_intensity = {
            100: 0.01,  # 1% tolerance
            50: 0.02,  # 2% tolerance
            30: 0.05,  # 5% tolerance
            10: 0.10,  # 10% tolerance
        }

        # Physical parameter boundaries
        self.param_bounds = {
            'cosi': (0.0, 0.99),
            'theta_int': (0.0, np.pi),
            'g1': (-0.1, 0.1),
            'g2': (-0.1, 0.1),
            'I0': (1e-6, None),  # Strictly positive
        }

        # Image parameters
        self.image_pars_velocity = ImagePars(
            shape=(32, 32), pixel_scale=0.3, indexing='ij'  # arcsec/pixel
        )

        self.image_pars_intensity = ImagePars(
            shape=(64, 64), pixel_scale=0.3, indexing='ij'  # arcsec/pixel
        )

    def get_tolerance(self, snr: float, data_type: str = 'velocity') -> float:
        """Get tolerance threshold for given SNR and data type."""
        if data_type == 'velocity':
            return self.snr_tolerance_velocity.get(snr, 0.05)
        elif data_type == 'intensity':
            return self.snr_tolerance_intensity.get(snr, 0.05)
        else:
            raise ValueError(f"Unknown data_type: {data_type}")


# ==============================================================================
# Parameter Bounds and Scanning
# ==============================================================================


def compute_parameter_bounds(
    param_name: str,
    true_value: float,
    config: TestConfig,
    image_pars: Optional[ImagePars] = None,
    fraction: float = 0.25,
) -> Tuple[float, float]:
    """
    Compute scan bounds for a parameter.

    Uses ±fraction around true value by default, respecting physical boundaries.

    Parameters
    ----------
    param_name : str
        Parameter name.
    true_value : float
        True parameter value.
    config : TestConfig
        Test configuration containing parameter bounds.
    image_pars : ImagePars, optional
        Image parameters (needed for x0, y0 bounds).
    fraction : float, optional
        Fractional range around true value. Default is 0.25 (±25%).

    Returns
    -------
    lower, upper : float
        Bounds for parameter scan.
    """

    # Check if parameter has physical bounds
    if param_name in config.param_bounds:
        lower_phys, upper_phys = config.param_bounds[param_name]

        # Compute ±fraction range
        delta = fraction * abs(true_value)
        lower_pct = true_value - delta
        upper_pct = true_value + delta

        # Respect physical boundaries
        if lower_phys is not None:
            lower = max(lower_pct, lower_phys)
        else:
            lower = lower_pct

        if upper_phys is not None:
            upper = min(upper_pct, upper_phys)
        else:
            upper = upper_pct

    # Special case: centroid offsets (bounded by image)
    elif param_name in ['x0', 'y0', 'vel_x0', 'vel_y0', 'int_x0', 'int_y0']:
        if image_pars is None:
            raise ValueError(f"image_pars required for bounds on {param_name}")

        # Image spans from -extent/2 to +extent/2 in arcsec
        extent = image_pars.shape[0] * image_pars.pixel_scale / 2

        delta = fraction * abs(true_value)
        lower = max(true_value - delta, -extent)
        upper = min(true_value + delta, extent)

    # Default: ±fraction
    else:
        delta = fraction * abs(true_value)
        lower = true_value - delta
        upper = true_value + delta

    return lower, upper


def slice_likelihood_1d(
    log_like_fn: Callable,
    theta_true: jnp.ndarray,
    param_idx: int,
    param_name: str,
    config: TestConfig,
    n_points: int = 50,
    image_pars: Optional[ImagePars] = None,
    scan_fraction: float = 0.25,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute 1D likelihood slice for a single parameter.

    Parameters
    ----------
    log_like_fn : callable
        JIT-compiled log-likelihood function.
    theta_true : jnp.ndarray
        True parameter array.
    param_idx : int
        Index of parameter to slice.
    param_name : str
        Name of parameter (for bounds computation).
    config : TestConfig
        Test configuration.
    n_points : int, optional
        Number of points in slice. Default is 50.
    image_pars : ImagePars, optional
        Image parameters (for x0, y0 bounds).
    scan_fraction : float, optional
        Fractional scan range. Default is 0.25.

    Returns
    -------
    param_values : jnp.ndarray
        Parameter values scanned.
    log_probs : jnp.ndarray
        Log-likelihood at each parameter value.
    """

    true_value = float(theta_true[param_idx])

    # Compute scan range
    lower, upper = compute_parameter_bounds(
        param_name, true_value, config, image_pars, fraction=scan_fraction
    )
    param_values = jnp.linspace(lower, upper, n_points)

    # Evaluate likelihood at each point
    log_probs = []
    for val in param_values:
        theta_test = theta_true.at[param_idx].set(val)
        log_prob = log_like_fn(theta_test)
        log_probs.append(float(log_prob))

    return param_values, jnp.array(log_probs)


def slice_all_parameters(
    log_like_fn: Callable,
    model,
    theta_true: jnp.ndarray,
    config: TestConfig,
    n_points: int = 50,
    image_pars: Optional[ImagePars] = None,
    scan_fraction: float = 0.25,
) -> Dict[str, Tuple[jnp.ndarray, jnp.ndarray]]:
    """
    Compute likelihood slices for all parameters.

    Parameters
    ----------
    log_like_fn : callable
        JIT-compiled log-likelihood function.
    model : Model
        Model instance (for parameter names).
    theta_true : jnp.ndarray
        True parameter array.
    config : TestConfig
        Test configuration.
    n_points : int, optional
        Number of points per slice. Default is 50.
    image_pars : ImagePars, optional
        Image parameters.
    scan_fraction : float, optional
        Fractional scan range. Default is 0.25.

    Returns
    -------
    slices : dict
        Dictionary mapping parameter names to (values, log_probs) tuples.
    """

    slices = {}

    for idx, param_name in enumerate(model.PARAMETER_NAMES):
        param_values, log_probs = slice_likelihood_1d(
            log_like_fn,
            theta_true,
            idx,
            param_name,
            config,
            n_points,
            image_pars,
            scan_fraction,
        )
        slices[param_name] = (param_values, log_probs)

    return slices


# ==============================================================================
# Diagnostic Plotting
# ==============================================================================


def plot_data_comparison_panels(
    data_noisy: jnp.ndarray,
    data_true: jnp.ndarray,
    model_eval: jnp.ndarray,
    test_name: str,
    config: TestConfig,
    data_type: str = 'velocity',
) -> None:
    """
    Create 2x3 panel diagnostic plot.

    Row 1: noisy | true | noisy - true
    Row 2: noisy | model | noisy - model

    Parameters
    ----------
    data_noisy : jnp.ndarray
        Noisy synthetic data.
    data_true : jnp.ndarray
        True noiseless data.
    model_eval : jnp.ndarray
        Model evaluation at true parameters.
    test_name : str
        Name of test (for title and filename).
    config : TestConfig
        Test configuration (for output dir and plot enable flag).
    data_type : str, optional
        Type of data ('velocity' or 'intensity'). Default is 'velocity'.
    """

    if not config.enable_plots:
        return

    # Create output directory for this test
    test_dir = config.output_dir / test_name
    test_dir.mkdir(parents=True, exist_ok=True)

    # Compute residuals
    residual_true = np.array(data_noisy - data_true)
    residual_model = np.array(data_noisy - model_eval)

    # Set up figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Common colorbar limits for data
    data_arrays = [data_noisy, data_true, model_eval]
    vmin_data = min(np.percentile(arr, 1) for arr in data_arrays)
    vmax_data = max(np.percentile(arr, 99) for arr in data_arrays)

    # Common colorbar limits for residuals
    residual_arrays = [residual_true, residual_model]
    abs_max = max(np.abs(np.percentile(arr, [1, 99])).max() for arr in residual_arrays)
    vmin_res = -abs_max
    vmax_res = abs_max

    # Row 1: noisy | true | noisy - true
    im00 = axes[0, 0].imshow(
        np.array(data_noisy).T,
        origin='lower',
        cmap='RdBu_r',
        vmin=vmin_data,
        vmax=vmax_data,
    )
    axes[0, 0].set_title('Noisy Data')
    plt.colorbar(im00, ax=axes[0, 0])

    im01 = axes[0, 1].imshow(
        np.array(data_true).T,
        origin='lower',
        cmap='RdBu_r',
        vmin=vmin_data,
        vmax=vmax_data,
    )
    axes[0, 1].set_title('True (Noiseless)')
    plt.colorbar(im01, ax=axes[0, 1])

    im02 = axes[0, 2].imshow(
        residual_true.T, origin='lower', cmap='RdBu_r', vmin=vmin_res, vmax=vmax_res
    )
    axes[0, 2].set_title('Noisy - True')
    plt.colorbar(im02, ax=axes[0, 2])

    # Row 2: noisy | model | noisy - model
    im10 = axes[1, 0].imshow(
        np.array(data_noisy).T,
        origin='lower',
        cmap='RdBu_r',
        vmin=vmin_data,
        vmax=vmax_data,
    )
    axes[1, 0].set_title('Noisy Data')
    plt.colorbar(im10, ax=axes[1, 0])

    im11 = axes[1, 1].imshow(
        np.array(model_eval).T,
        origin='lower',
        cmap='RdBu_r',
        vmin=vmin_data,
        vmax=vmax_data,
    )
    axes[1, 1].set_title('Model at True Params')
    plt.colorbar(im11, ax=axes[1, 1])

    im12 = axes[1, 2].imshow(
        residual_model.T, origin='lower', cmap='RdBu_r', vmin=vmin_res, vmax=vmax_res
    )
    axes[1, 2].set_title('Noisy - Model')
    plt.colorbar(im12, ax=axes[1, 2])

    # Labels
    for ax in axes.flat:
        ax.set_xlabel('x (pixels)')
        ax.set_ylabel('y (pixels)')

    # Overall title
    fig.suptitle(f'{test_name} - {data_type.capitalize()} Comparison', fontsize=14)
    plt.tight_layout()

    # Save
    outfile = test_dir / f"{test_name}_{data_type}_panels.png"
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close()


def plot_likelihood_slices(
    slices: Dict[str, Tuple[jnp.ndarray, jnp.ndarray]],
    true_pars: Dict[str, float],
    test_name: str,
    config: TestConfig,
    snr: float,
    tolerance: float,
) -> Dict[str, Dict[str, float]]:
    """
    Plot likelihood slices for all parameters.

    Parameters
    ----------
    slices : dict
        Dictionary mapping parameter names to (values, log_probs).
    true_pars : dict
        True parameter values.
    test_name : str
        Name of test.
    config : TestConfig
        Test configuration.
    snr : float
        Signal-to-noise ratio used.
    tolerance : float
        Tolerance threshold for recovery.

    Returns
    -------
    recovery_stats : dict
        Statistics for each parameter: true, recovered, error, pct_error, passed.
    """

    # Always compute recovery stats
    recovery_stats = {}
    for param_name, (param_values, log_probs) in slices.items():
        true_val = true_pars[param_name]
        best_idx = jnp.argmax(log_probs)
        recovered_val = float(param_values[best_idx])
        error = recovered_val - true_val
        pct_error = abs(error / true_val) if true_val != 0 else abs(error)
        passed = pct_error < tolerance

        recovery_stats[param_name] = {
            'true': true_val,
            'recovered': recovered_val,
            'error': error,
            'pct_error': pct_error,
            'passed': passed,
        }

    if not config.enable_plots:
        return recovery_stats

    # Create output directory
    test_dir = config.output_dir / test_name
    test_dir.mkdir(parents=True, exist_ok=True)

    # Determine grid layout
    n_params = len(slices)
    ncols = 3
    nrows = int(np.ceil(n_params / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows))
    axes = np.atleast_2d(axes).flatten()

    for idx, (param_name, (param_values, log_probs)) in enumerate(slices.items()):
        ax = axes[idx]
        stats = recovery_stats[param_name]

        # Plot likelihood slice
        ax.plot(param_values, log_probs, 'b-', linewidth=2)
        ax.axvline(stats['true'], color='k', linestyle='--', linewidth=2, label='True')
        ax.axvline(
            stats['recovered'],
            color='r',
            linestyle=':',
            linewidth=2,
            label=f'Peak: {stats["recovered"]:.4f} ({stats["pct_error"]*100:.1f}%)',
        )

        # Styling
        ax.set_xlabel(param_name)
        ax.set_ylabel('Log-Likelihood')
        ax.set_title(f'{param_name} - {"PASS" if stats["passed"] else "FAIL"}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Color title based on pass/fail
        title_color = 'green' if stats['passed'] else 'red'
        ax.title.set_color(title_color)

    # Hide unused subplots
    for idx in range(n_params, len(axes)):
        axes[idx].axis('off')

    # Overall title
    fig.suptitle(
        f'{test_name} - Likelihood Slices (SNR={snr}, tolerance={tolerance*100:.1f}%)',
        fontsize=14,
    )
    plt.tight_layout()

    # Save
    outfile = test_dir / f"{test_name}_likelihood_slices.png"
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close()

    return recovery_stats
