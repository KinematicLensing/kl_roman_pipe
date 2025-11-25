"""
Shared test utilities for parameter recovery tests.

This module contains common functions used by both likelihood slicing tests
and gradient-based optimizer tests. May expand further in the future.
"""

import pytest
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Tuple, Optional, Callable
from mpl_toolkits.axes_grid1 import make_axes_locatable

from kl_pipe.parameters import ImagePars
from kl_pipe.plotting import MidpointNormalize


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
        include_poisson_noise: bool = False,
        seed: int = 42,
    ):
        self.output_dir = output_dir
        # plotting control
        self.enable_plots = enable_plots
        # synthetic data generation
        self.include_poisson_noise = include_poisson_noise
        self.seed = seed

        # base tolerances (for well-constrained parameters)
        self.base_tolerance_velocity = {
            1000: 0.001,  # 0.1%
            500: 0.0025,  # 0.25%
            100: 0.005,  # 0.5%
            50: 0.01,  # 1%
            10: 0.05,  # 5%
        }
        # NOTE: same for now, but could differ in future
        self.base_tolerance_intensity = {
            1000: 0.001,  # 0.1%
            500: 0.0025,  # 0.25%
            100: 0.005,  # 0.5%
            50: 0.01,  # 1%
            10: 0.05,  # 5%
        }

        # parameter-specific scaling factors, to account for inherently weaker signals
        self.param_tolerance_scaling = {
            # shear params are ~4% of main velocity signal at our parameters
            'g1': {
                1000: 1.0,  # Well-constrained at high SNR
                500: 1.0,
                100: 1.5,  # 50% more lenient
                50: 2.0,  # 2x more lenient
                10: 5.0,  # 5x more lenient (shear SNR << 1)
            },
            'g2': {
                1000: 1.0,
                500: 1.0,
                100: 1.5,
                50: 2.0,
                10: 5.0,
            },
            # v0 is well constrainted but also small
            'v0': {
                1000: 1.0,
                500: 1.0,
                100: 1.5,
                50: 1.5,
                10: 2.5,
            },
            # could add other weak parameters here
            # ...
        }

        # absolute tolerance floor (for parameters near zero)
        # if true value is very small, relative error is misleading
        self.absolute_tolerance_floor = {
            'g1': 0.002,
            'g2': 0.002,
            'vel_x0': 0.1,
            'vel_y0': 0.1,
            'int_x0': 0.1,
            'int_y0': 0.1,
        }

        # physical parameter boundaries
        self.param_bounds = {
            'cosi': (0.0, 0.99),
            'theta_int': (0.0, np.pi),
            'g1': (-0.1, 0.1),
            'g2': (-0.1, 0.1),
            'flux': (1e-8, None),  # Strictly positive
        }

        # image parameters - specified in (Nx, Ny) for easy verification
        Nx_vel, Ny_vel = 40, 30
        self.image_pars_velocity = ImagePars(
            shape=(Nx_vel, Ny_vel), pixel_scale=0.3, indexing='xy'  # arcsec/pixel
        )

        # intensity: taller than wide (Ny > Nx) - opposite orientation
        Nx_int, Ny_int = 60, 80
        self.image_pars_intensity = ImagePars(
            shape=(Nx_int, Ny_int), pixel_scale=0.3, indexing='xy'  # arcsec/pixel
        )

        return

    def get_tolerance(
        self,
        snr: float,
        param_name: str,
        param_value: float,
        data_type: str = 'velocity',
    ) -> Dict[str, float]:
        """
        Get tolerance for parameter at given SNR.

        Returns both relative and absolute tolerances.
        Parameter passes if EITHER criterion is met.

        Parameters
        ----------
        snr : float
            Signal-to-noise ratio of data.
        param_name : str
            Name of parameter being tested.
        param_value : float
            True value of parameter (needed for absolute tolerance).
        data_type : str
            'velocity' or 'intensity'.

        Returns
        -------
        dict
            Contains 'relative' and 'absolute' tolerance values.
        """

        # get base tolerance for this SNR
        if data_type == 'velocity':
            base_tol = self.base_tolerance_velocity.get(snr, 0.05)
        else:
            base_tol = self.base_tolerance_intensity.get(snr, 0.05)

        # apply parameter-specific scaling
        if param_name in self.param_tolerance_scaling:
            scaling = self.param_tolerance_scaling[param_name].get(snr, 1.0)
            relative_tol = base_tol * scaling
        else:
            relative_tol = base_tol

        # compute absolute tolerance
        # use the larger of: (relative_tol × |value|) or absolute_floor
        absolute_from_relative = relative_tol * abs(param_value)
        absolute_floor = self.absolute_tolerance_floor.get(param_name, 0.0)
        absolute_tol = max(absolute_from_relative, absolute_floor)

        return {
            'relative': relative_tol,
            'absolute': absolute_tol,
        }


def check_parameter_recovery(
    recovered: float,
    true_value: float,
    tolerance: Dict[str, float],
    param_name: str,
) -> Tuple[bool, Dict[str, float]]:
    """
    Check if parameter recovered within tolerance.

    Passes if EITHER relative OR absolute criterion is met.

    Parameters
    ----------
    recovered : float
        Recovered parameter value.
    true_value : float
        True parameter value.
    tolerance : dict
        Contains 'relative' and 'absolute' tolerance values.
    param_name : str
        Name of parameter (for logging).

    Returns
    -------
    passed : bool
        True if parameter recovered successfully.
    stats : dict
        Statistics about the recovery.
    """

    abs_error = abs(recovered - true_value)

    # avoid division by zero for parameters exactly at zero
    if abs(true_value) < 1e-10:
        rel_error = abs_error  # treat as absolute error
        passed_relative = False  # can't use relative for zero
    else:
        rel_error = abs_error / abs(true_value)
        passed_relative = rel_error <= tolerance['relative']

    passed_absolute = abs_error <= tolerance['absolute']

    # pass if *either* criterion met
    passed = passed_relative or passed_absolute

    stats = {
        'true': true_value,
        'recovered': recovered,
        'abs_error': abs_error,
        'rel_error': rel_error,
        'abs_tolerance': tolerance['absolute'],
        'rel_tolerance': tolerance['relative'],
        'passed_absolute': passed_absolute,
        'passed_relative': passed_relative,
        'passed': passed,
        'criterion': (
            'relative'
            if passed_relative
            else ('absolute' if passed_absolute else 'none')
        ),
    }

    return passed, stats


def assert_parameter_recovery(
    recovery_stats: Dict[str, Dict[str, float]], snr: float, test_name: str = "Test"
) -> None:
    """
    Assert that all parameters were recovered within tolerance.

    If any parameter failed, formats a detailed error message and fails the test.

    Parameters
    ----------
    recovery_stats : dict
        Recovery statistics from plot_likelihood_slices() or similar.
        Each entry should have 'passed', 'rel_error', 'abs_error', etc.
    snr : float
        Signal-to-noise ratio (for error message).
    test_name : str, optional
        Name of test (for error message). Default is "Test".
    """

    failed_params = []
    for param_name, stats in recovery_stats.items():
        if not stats['passed']:
            failed_params.append(
                f"{param_name}: "
                f"rel {stats['rel_error']*100:.2f}% (tol {stats['rel_tolerance']*100:.1f}%), "
                f"abs {stats['abs_error']:.4f} (tol {stats['abs_tolerance']:.4f}) "
                f"- recovered {stats['recovered']:.4f}, true {stats['true']:.4f}"
            )

    if failed_params:
        msg = f"{test_name} failed for SNR={snr}:\n" + "\n".join(failed_params)
        pytest.fail(msg)

    return


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
    n_points: int = 201,
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
        Number of points in slice. Default is 100.
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
    n_points: int = 201,
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
        Number of points per slice. Default is 100.
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
    variance: Optional[float] = None,
    n_params: Optional[int] = None,
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
    variance : float, optional
        Variance of noise, if you want to report reduced chi-squared.
    n_params : int, optional
        Number of fitted parameters (for reduced chi-squared). Default is None.
    """

    if not config.enable_plots:
        return

    # Create output directory for this test
    test_dir = config.output_dir / test_name
    test_dir.mkdir(parents=True, exist_ok=True)

    # Compute residuals & chi2
    residual_true = np.array(data_noisy - data_true)
    residual_model = np.array(data_noisy - model_eval)
    residual_model_true = np.array(model_eval - data_true)

    chi2_true = None
    chi2_model = None
    if variance is not None:
        chi2_true = np.sum(residual_true**2 / variance)
        chi2_model = np.sum(residual_model**2 / variance)
        if n_params is not None:
            dof = data_noisy.size - n_params
            chi2_true /= dof
            chi2_model /= dof

    # Set up figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Common colorbar limits for data
    data_arrays = [data_noisy, data_true, model_eval]
    vmin_data = min(np.percentile(arr, 1) for arr in data_arrays)
    vmax_data = max(np.percentile(arr, 99) for arr in data_arrays)
    norm_data = MidpointNormalize(vmin=vmin_data, vmax=vmax_data, midpoint=0)

    # Common colorbar limits for residuals
    residual_arrays = [residual_true, residual_model]
    abs_max = max(np.abs(np.percentile(arr, [1, 99])).max() for arr in residual_arrays)
    norm_resid = MidpointNormalize(vmin=-abs_max, vmax=abs_max, midpoint=0)

    # Row 1: noisy | true | noisy - true
    im00 = axes[0, 0].imshow(
        np.array(data_noisy),
        origin='lower',
        cmap='RdBu_r',
        norm=norm_data,
    )
    axes[0, 0].set_title('Noisy Data')
    divider = make_axes_locatable(axes[0, 0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im00, cax=cax)

    im01 = axes[0, 1].imshow(
        np.array(data_true),
        origin='lower',
        cmap='RdBu_r',
        norm=norm_data,
    )
    axes[0, 1].set_title('True (Noiseless)')
    divider = make_axes_locatable(axes[0, 1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im01, cax=cax)

    im02 = axes[0, 2].imshow(
        residual_true, origin='lower', cmap='RdBu_r', norm=norm_resid
    )
    axes[0, 2].set_title('Noisy - True')
    divider = make_axes_locatable(axes[0, 2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im02, cax=cax)
    if variance is not None:
        axes[0, 2].text(
            0.02,
            0.98,
            f'χ² = {chi2_true:.1f}',
            transform=axes[0, 2].transAxes,
            fontsize=10,
            color='white',
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.5),
        )

    # Row 2: model - true | model | noisy - model
    im10 = axes[1, 0].imshow(
        residual_model_true,
        origin='lower',
        cmap='RdBu_r',
        norm=norm_resid,
    )
    axes[1, 0].set_title('Model - True')
    divider = make_axes_locatable(axes[1, 0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im10, cax=cax)

    im11 = axes[1, 1].imshow(
        np.array(model_eval),
        origin='lower',
        cmap='RdBu_r',
        norm=norm_data,
    )
    axes[1, 1].set_title('Model at True Params')
    divider = make_axes_locatable(axes[1, 1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im11, cax=cax)

    im12 = axes[1, 2].imshow(
        residual_model, origin='lower', cmap='RdBu_r', norm=norm_resid
    )
    axes[1, 2].set_title('Noisy - Model')
    divider = make_axes_locatable(axes[1, 2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im12, cax=cax)
    if variance is not None:
        axes[1, 2].text(
            0.02,
            0.98,
            f'χ² = {chi2_model:.1f}',
            transform=axes[1, 2].transAxes,
            fontsize=10,
            color='white',
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.5),
        )

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
    plt.close(fig)


def plot_likelihood_slices(
    slices: Dict[str, Tuple[jnp.ndarray, jnp.ndarray]],
    true_pars: Dict[str, float],
    test_name: str,
    config: TestConfig,
    snr: float,
    data_type: str,
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
    data_type : str
        'velocity', 'intensity', or 'joint'. Used to get tolerances.

    Returns
    -------
    recovery_stats : dict
        Statistics for each parameter: true, recovered, error, rel_error, passed.
    """

    # Always compute recovery stats
    recovery_stats = {}
    for param_name, (param_values, log_probs) in slices.items():
        true_val = true_pars[param_name]

        best_idx = jnp.argmax(log_probs)
        recovered_val = float(param_values[best_idx])
        true_value = true_pars[param_name]

        # Get tolerance (both relative and absolute)
        tolerance = config.get_tolerance(snr, param_name, true_value, data_type)

        # Check recovery
        passed, stats = check_parameter_recovery(
            recovered_val, true_val, tolerance, param_name
        )

        recovery_stats[param_name] = stats

    if not config.enable_plots:
        return recovery_stats

    # create output directory
    test_dir = config.output_dir / test_name
    test_dir.mkdir(parents=True, exist_ok=True)

    # determine grid layout
    n_params = len(slices)
    ncols = 3
    nrows = int(np.ceil(n_params / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows))
    axes = np.atleast_2d(axes).flatten()

    for idx, (param_name, (param_values, log_probs)) in enumerate(slices.items()):
        ax = axes[idx]
        stats = recovery_stats[param_name]

        # compute acceptance bounds (+/- tolerance around true value)
        true_val = stats['true']
        rel_tolerance = stats['rel_tolerance']
        abs_tolerance = stats['abs_tolerance']
        if true_val != 0:
            lower_bound = true_val * (1 - rel_tolerance)
            upper_bound = true_val * (1 + rel_tolerance)
        else:
            # For parameters where true value is 0, use absolute tolerance
            lower_bound = -abs_tolerance
            upper_bound = abs_tolerance

        # plot likelihood slice
        ax.plot(param_values, log_probs, 'b-', linewidth=2)
        ax.axvline(stats['true'], color='k', linestyle='--', linewidth=2, label='True')
        ax.axvline(
            stats['recovered'],
            color='r',
            linestyle=':',
            linewidth=2,
            label=f'Peak: {stats["recovered"]:.4f} ({stats["rel_error"]*100:.3f}%)',
        )

        # styling
        ax.set_xlabel(param_name)
        ax.set_ylabel('Log-Likelihood')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        if stats['criterion'] == 'relative':
            status = f"PASS (rel: {stats['rel_error']*100:.2f}% < {stats['rel_tolerance']*100:.1f}%)"
        elif stats['criterion'] == 'absolute':
            status = (
                f"PASS (abs: {stats['abs_error']:.4f} < {stats['abs_tolerance']:.4f})"
            )
        else:
            status = f"FAIL (rel: {stats['rel_error']*100:.2f}%, abs: {stats['abs_error']:.4f})"

        # add grey acceptance region
        ax.axvspan(
            lower_bound,
            upper_bound,
            alpha=0.15,  # Low opacity
            color='grey',
            label=f'±{rel_tolerance*100:.1f}% tolerance',
            zorder=1,
        )

        # color title based on pass/fail
        title_color = 'green' if stats['passed'] else 'red'
        ax.set_title(f'{param_name} - {status}', color=title_color)

    # hide unused subplots
    for idx in range(n_params, len(axes)):
        axes[idx].axis('off')

    # overall title
    if data_type == 'velocity':
        base_tolerance = config.base_tolerance_velocity[snr]
    else:
        base_tolerance = config.base_tolerance_intensity[snr]
    fig.suptitle(
        f'{test_name} - Likelihood Slices '
        f'(SNR={snr}, base tolerance={base_tolerance*100:.1f}%)',
        fontsize=14,
    )
    plt.tight_layout()

    # save figure
    outfile = test_dir / f"{test_name}_likelihood_slices.png"
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close(fig)

    return recovery_stats
