"""
Shared test utilities for parameter recovery tests.

This module contains common functions used by both likelihood slicing tests
and gradient-based optimizer tests. May expand further in the future.

Note: plot_data_comparison_panels and plot_combined_data_comparison have been
moved to kl_pipe.diagnostics. The versions in this file are deprecated wrappers
for backward compatibility.
"""

import pytest
import sys
import contextlib
import warnings
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Tuple, Optional, Callable, List
from mpl_toolkits.axes_grid1 import make_axes_locatable

from kl_pipe.parameters import ImagePars
from kl_pipe.plotting import MidpointNormalize


# ==============================================================================
# Output Redirection Utilities
# ==============================================================================


class TeeWriter:
    """Write to both a file and optionally to terminal."""

    def __init__(self, file_handle, also_terminal: bool = False, original_stdout=None):
        self.file_handle = file_handle
        self.also_terminal = also_terminal
        self.original_stdout = original_stdout or sys.__stdout__

    def write(self, message):
        self.file_handle.write(message)
        if self.also_terminal:
            self.original_stdout.write(message)

    def flush(self):
        self.file_handle.flush()
        if self.also_terminal:
            self.original_stdout.flush()


@contextlib.contextmanager
def redirect_sampler_output(log_path: Path, also_terminal: bool = False):
    """
    Redirect stdout to a file, optionally also writing to terminal.

    Sampler output is ALWAYS written to the log file. If also_terminal=True,
    output is also displayed in the terminal (useful for debugging).

    Parameters
    ----------
    log_path : Path
        Path to the output log file.
    also_terminal : bool
        If True, output goes to both file and terminal.
        If False (default), output goes to file only.

    Examples
    --------
    >>> with redirect_sampler_output(Path("sampler.log")):
    ...     sampler.run()  # Output captured to file only

    >>> with redirect_sampler_output(Path("sampler.log"), also_terminal=True):
    ...     sampler.run()  # Output to file AND terminal
    """
    # Ensure parent directory exists
    log_path.parent.mkdir(parents=True, exist_ok=True)

    original_stdout = sys.stdout
    with open(log_path, 'w') as f:
        sys.stdout = TeeWriter(
            f, also_terminal=also_terminal, original_stdout=original_stdout
        )
        try:
            yield
        finally:
            sys.stdout = original_stdout


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
        sersic_backend: str = 'scipy',
        verbose_terminal: bool = False,
    ):
        self.output_dir = output_dir
        # plotting control
        self.enable_plots = enable_plots
        # synthetic data generation
        self.include_poisson_noise = include_poisson_noise
        self.seed = seed
        self.sersic_backend = sersic_backend
        # sampler output control
        self.verbose_terminal = verbose_terminal

        # =========================================================================
        # LIKELIHOOD SLICE TEST TOLERANCES
        # Brute-force parameter recovery - strictest validation of forward models
        # =========================================================================
        self.likelihood_slice_tolerance_velocity = {
            1000: 0.001,  # 0.1%
            500: 0.0025,  # 0.25%
            100: 0.005,  # 0.5%
            50: 0.01,  # 1%
            10: 0.05,  # 5%
        }
        self.likelihood_slice_tolerance_intensity = {
            1000: 0.001,  # 0.1%
            500: 0.0025,  # 0.25%
            100: 0.005,  # 0.5%
            50: 0.01,  # 1%
            10: 0.05,  # 5%
        }

        # =========================================================================
        # OPTIMIZER TEST TOLERANCES
        # Gradient-based recovery - looser due to local optima & parameter degeneracies
        # Philosophy: Validate optimization framework works, not perfect recovery.
        # These are 10-20x looser than likelihood slices to account for:
        #   - Local minima in optimization landscape
        #   - Parameter degeneracies (cosi/g1/g2 trade-off)
        #   - Finite sampling noise in gradients
        #   - Initial guess dependence
        # =========================================================================
        self.optimizer_tolerance_velocity = {
            1000: 0.02,  # 2% (20x looser than likelihood slices)
            500: 0.025,  # 2.5%
            100: 0.03,  # 3%
            50: 0.05,  # 5%
            10: 0.20,  # 20%
        }
        self.optimizer_tolerance_intensity = {
            1000: 0.02,  # 2%
            500: 0.025,  # 2.5%
            100: 0.03,  # 3%
            50: 0.05,  # 5%
            10: 0.20,  # 20%
        }

        # =========================================================================
        # PARAMETER-SPECIFIC SCALING (applies to both test types)
        # Accounts for inherently weaker constraints on certain parameters
        # =========================================================================

        # Likelihood slice test scaling (conservative)
        self.likelihood_slice_param_scaling = {
            # Shear is ~4% of main signal - harder to constrain
            'g1': {1000: 1.0, 500: 1.0, 100: 1.0, 50: 1.5, 10: 1.0},
            'g2': {1000: 1.0, 500: 1.0, 100: 1.0, 50: 1.5, 10: 1.0},
            # v0 is small - harder to measure relative error
            'v0': {1000: 1.0, 500: 1.0, 100: 1.0, 50: 1.0, 10: 1.5},
        }

        # Optimizer test scaling (more lenient for weakly constrained params)
        self.optimizer_param_scaling = {
            # Shear degeneracies with other geometric parameters in optimization
            'g1': {1000: 2.0, 500: 2.0, 100: 2.5, 50: 3.0, 10: 3.0},
            'g2': {1000: 2.0, 500: 2.0, 100: 2.5, 50: 3.0, 10: 3.0},
            # v0 can get stuck in local optima
            'v0': {1000: 1.5, 500: 1.5, 100: 1.5, 50: 2.0, 10: 2.5},
            # Offsets can have shallow likelihood surfaces
            'vel_x0': {1000: 1.5, 500: 1.5, 100: 2.0, 50: 2.0, 10: 2.5},
            'vel_y0': {1000: 1.5, 500: 1.5, 100: 2.0, 50: 2.0, 10: 2.5},
            'int_x0': {1000: 1.5, 500: 1.5, 100: 2.0, 50: 2.0, 10: 2.5},
            'int_y0': {1000: 1.5, 500: 1.5, 100: 2.0, 50: 2.0, 10: 2.5},
        }

        # absolute tolerance floor (for parameters near zero)
        # if true value is very small, relative error is misleading
        self.absolute_tolerance_floor = {
            'g1': 0.0025,  # Increased from 0.002 for low SNR cases
            'g2': 0.0025,  # Increased from 0.002 for low SNR cases
            'vel_x0': 0.1,
            'vel_y0': 0.1,
            'int_x0': 0.1,
            'int_y0': 0.1,
        }

        # PSF tolerance multiplier -- PSF smoothing reduces information content
        self.psf_tolerance_multiplier = 2.0

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

    def get_sampler_log_path(self, test_name: str, sampler_name: str) -> Path:
        """
        Get path for sampler output log file.

        Sampler output is always written to files in the test output directory.

        Parameters
        ----------
        test_name : str
            Name of the test (used as subdirectory).
        sampler_name : str
            Name of the sampler (e.g., 'emcee', 'nautilus', 'blackjax').

        Returns
        -------
        Path
            Path to the log file.
        """
        test_dir = self.output_dir / test_name
        test_dir.mkdir(parents=True, exist_ok=True)
        return test_dir / f"{sampler_name}_output.txt"

    def get_tolerance(
        self,
        snr: float,
        param_name: str,
        param_value: float,
        data_type: str,
        test_type: str,
        has_psf: bool = False,
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
        test_type : str
            'likelihood_slice' or 'optimizer'. Determines which tolerance set to use.
            Optimizer tests use looser tolerances due to local optima & degeneracies.
        has_psf : bool
            If True, apply psf_tolerance_multiplier to loosen tolerances.

        Returns
        -------
        dict
            Contains 'relative' and 'absolute' tolerance values.
        """

        # Get base tolerance for this SNR and test type
        if test_type == 'optimizer':
            if data_type == 'velocity':
                base_tol = self.optimizer_tolerance_velocity.get(snr, 0.075)
            else:
                base_tol = self.optimizer_tolerance_intensity.get(snr, 0.075)
            param_scaling = self.optimizer_param_scaling
        else:  # likelihood_slice (default)
            if data_type == 'velocity':
                base_tol = self.likelihood_slice_tolerance_velocity.get(snr, 0.05)
            else:
                base_tol = self.likelihood_slice_tolerance_intensity.get(snr, 0.05)
            param_scaling = self.likelihood_slice_param_scaling

        # Apply parameter-specific scaling
        if param_name in param_scaling:
            scaling = param_scaling[param_name].get(snr, 1.0)
            relative_tol = base_tol * scaling
        else:
            relative_tol = base_tol

        # Apply PSF tolerance multiplier
        if has_psf:
            relative_tol *= self.psf_tolerance_multiplier

        # compute absolute tolerance
        # use the larger of: (relative_tol × |value|) or absolute_floor
        absolute_from_relative = relative_tol * abs(param_value)
        absolute_floor = self.absolute_tolerance_floor.get(param_name, 0.0)
        absolute_tol = max(absolute_from_relative, absolute_floor)

        if has_psf:
            absolute_tol *= self.psf_tolerance_multiplier

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
    recovery_stats: Dict[str, Dict[str, float]],
    snr: float,
    test_name: str = "Test",
    exclude_params: Optional[list] = None,
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
    exclude_params : list, optional
        Parameter names to exclude from pass/fail check (still reported).
        Useful for degenerate parameters like g1/g2.
    """

    if exclude_params is None:
        exclude_params = []

    failed_params = []
    excluded_params = []

    for param_name, stats in recovery_stats.items():
        if not stats['passed']:
            msg = (
                f"{param_name}: "
                f"rel {stats['rel_error']*100:.2f}% (tol {stats['rel_tolerance']*100:.1f}%), "
                f"abs {stats['abs_error']:.4f} (tol {stats['abs_tolerance']:.4f}) "
                f"- recovered {stats['recovered']:.4f}, true {stats['true']:.4f}"
            )

            if param_name in exclude_params:
                excluded_params.append(msg + " [EXCLUDED]")
            else:
                failed_params.append(msg)

    # Report excluded params as warnings (still visible but don't fail)
    if excluded_params:
        print(f"\n⚠️  {test_name} - Excluded parameters outside tolerance:")
        for msg in excluded_params:
            print(f"  {msg}")

    if failed_params:
        msg = f"{test_name} failed for SNR={snr}:\n" + "\n".join(failed_params)
        pytest.fail(msg)

    return


def check_degenerate_product_recovery(
    pars_true: Dict[str, float],
    pars_recovered: Dict[str, float],
    snr: Optional[float] = None,
    tolerance_relative: Optional[float] = None,
) -> Tuple[bool, Dict[str, any]]:
    """
    Check recovery of degenerate parameter products.

    For velocity models, the observed signal depends on vcirc*sini
    rather than vcirc and cosi independently. This function checks if this
    product is recovered even when individual parameters may be off.

    Parameters
    ----------
    pars_true : dict
        True parameter values (must include 'vcirc' and 'cosi').
    pars_recovered : dict
        Recovered parameter values.
    snr : float, optional
        Signal-to-noise ratio. If provided, tolerance is SNR-dependent.
    tolerance_relative : float, optional
        Relative tolerance for the product. If not provided, uses SNR-dependent
        default (3% for SNR=1000, 10% for SNR=10).

    Returns
    -------
    passed : bool
        Whether the product was recovered within tolerance.
    stats : dict
        Statistics about the product recovery.
    """

    # Check if required parameters exist
    if 'vcirc' not in pars_true or 'cosi' not in pars_true:
        return True, {'note': 'vcirc or cosi not in model, skipping product check'}

    # Set tolerance based on SNR if not explicitly provided
    if tolerance_relative is None:
        if snr is not None:
            # SNR-dependent tolerance for degenerate products
            # More lenient than individual parameters since we only care about observable
            snr_tolerances = {
                1000: 0.03,  # 3%
                500: 0.04,  # 4%
                100: 0.05,  # 5%
                50: 0.07,  # 7%
                10: 0.10,  # 10%
            }
            tolerance_relative = snr_tolerances.get(snr, 0.10)
        else:
            tolerance_relative = 0.05  # Default 5%

    # Compute true product
    sini_true = np.sqrt(1 - pars_true['cosi'] ** 2)
    product_true = pars_true['vcirc'] * sini_true

    # Compute recovered product
    sini_recovered = np.sqrt(1 - pars_recovered['cosi'] ** 2)
    product_recovered = pars_recovered['vcirc'] * sini_recovered

    # Check tolerance
    abs_error = abs(product_recovered - product_true)
    rel_error = abs_error / product_true if product_true > 0 else abs_error
    passed = rel_error <= tolerance_relative

    stats = {
        'true': product_true,
        'recovered': product_recovered,
        'abs_error': abs_error,
        'rel_error': rel_error,
        'tolerance': tolerance_relative,
        'passed': passed,
        'formula': 'vcirc * sin(i) = vcirc * sqrt(1 - cosi²)',
    }

    return passed, stats


def compute_parameter_bounds(
    param_name: str,
    true_value: float,
    config: TestConfig,
    image_pars: Optional[ImagePars] = None,
    fraction: float = 0.25,
) -> Tuple[float, float]:
    """
    Compute scan bounds for a parameter.

    Uses +/-fraction around true value by default, respecting physical boundaries.

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
        If None for centroid params, uses conservative default bounds.
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

        # Compute +/-fraction range
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
        if image_pars is not None:
            # Use image extent
            extent = image_pars.shape[0] * image_pars.pixel_scale / 2
        else:
            # Use conservative default based on typical config values
            # For joint models, use the larger of the two grids
            extent_vel = (
                config.image_pars_velocity.shape[0]
                * config.image_pars_velocity.pixel_scale
                / 2
            )
            extent_int = (
                config.image_pars_intensity.shape[0]
                * config.image_pars_intensity.pixel_scale
                / 2
            )
            extent = max(extent_vel, extent_int)

        delta = (
            fraction * abs(true_value) if true_value != 0 else 1.0
        )  # Default to 1 arcsec if centered
        lower = max(true_value - delta, -extent)
        upper = min(true_value + delta, extent)

    # Default: +/-fraction
    else:
        delta = (
            fraction * abs(true_value) if true_value != 0 else 0.1
        )  # Small default for zero values
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
# Diagnostic Plotting (DEPRECATED - use kl_pipe.diagnostics instead)
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
    model_label: str = 'Model',
) -> None:
    """
    Create 2x3 panel diagnostic plot.

    .. deprecated::
        Use kl_pipe.diagnostics.plot_data_comparison_panels instead.
        This function is kept for backward compatibility.

    Row 1: noisy | true | noisy - true
    Row 2: model - true | model | noisy - model

    Parameters
    ----------
    data_noisy : jnp.ndarray
        Noisy synthetic data.
    data_true : jnp.ndarray
        True noiseless data.
    model_eval : jnp.ndarray
        Model evaluation (can be at true or optimized parameters).
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
    model_label : str, optional
        Label for model panel ('Model' or 'Optimized Model'). Default is 'Model'.
    """
    warnings.warn(
        "plot_data_comparison_panels in test_utils.py is deprecated. "
        "Use kl_pipe.diagnostics.plot_data_comparison_panels instead.",
        DeprecationWarning,
        stacklevel=2,
    )

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
    axes[1, 1].set_title(model_label)
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


def plot_parameter_comparison(
    true_pars: Dict[str, float],
    recovered_pars: Dict[str, float],
    recovery_stats: Dict[str, Dict[str, float]],
    test_name: str,
    config: TestConfig,
    snr: float,
    product_stats: Optional[Dict[str, any]] = None,
    exclude_params: Optional[List[str]] = None,
) -> None:
    """
    Create parameter comparison plot showing true vs recovered values.

    Parameters
    ----------
    true_pars : dict
        True parameter values.
    recovered_pars : dict
        Recovered parameter values.
    recovery_stats : dict
        Recovery statistics from check_parameter_recovery (includes tolerances).
    test_name : str
        Name of test (for filename).
    config : TestConfig
        Test configuration.
    snr : float
        Signal-to-noise ratio.
    product_stats : dict, optional
        Statistics from check_degenerate_product_recovery (vcirc*sini).
    exclude_params : list, optional
        List of parameter names excluded from pass/fail checks.
    """

    if not config.enable_plots:
        return

    # Create output directory
    test_dir = config.output_dir / test_name
    test_dir.mkdir(parents=True, exist_ok=True)

    # Add vcirc*sini product to the comparison if available
    if product_stats is not None and 'true' in product_stats:
        param_names = list(true_pars.keys()) + ['vcirc*sini']
        true_vals = np.concatenate(
            [
                np.array([true_pars[p] for p in true_pars.keys()]),
                [product_stats['true']],
            ]
        )
        recovered_vals = np.concatenate(
            [
                np.array([recovered_pars[p] for p in true_pars.keys()]),
                [product_stats['recovered']],
            ]
        )
        # Create pseudo-recovery stats for product
        all_recovery_stats = dict(recovery_stats)
        all_recovery_stats['vcirc*sini'] = {
            'passed': product_stats['passed'],
            'rel_error': product_stats['rel_error'],
            'abs_error': product_stats['abs_error'],
            'rel_tolerance': product_stats['tolerance'],
            'abs_tolerance': product_stats['abs_error'],
            'criterion': 'relative',
            'true': product_stats['true'],
            'recovered': product_stats['recovered'],
        }
        passed = np.array([all_recovery_stats[p]['passed'] for p in param_names])
    else:
        param_names = list(true_pars.keys())
        true_vals = np.array([true_pars[p] for p in param_names])
        recovered_vals = np.array([recovered_pars[p] for p in param_names])
        all_recovery_stats = recovery_stats
        passed = np.array([recovery_stats[p]['passed'] for p in param_names])

    if exclude_params is None:
        exclude_params = []

    # Count pass/fail (excluding excluded params)
    n_total = len(param_names)
    n_excluded = sum(1 for p in param_names if p in exclude_params)
    # Count only non-excluded parameters
    non_excluded_params = [p for p in param_names if p not in exclude_params]
    n_fail = sum(1 for p in non_excluded_params if not all_recovery_stats[p]['passed'])
    n_pass = len(non_excluded_params) - n_fail

    # Create figure with more vertical space
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))

    # Panel 1: Scatter plot with log scale if values span multiple orders of magnitude
    # Use categorical x-axis for better spacing
    x_pos = np.arange(len(param_names))

    # Use absolute values for log scale (to handle negative parameters like g1, g2)
    abs_true_vals = np.abs(true_vals)
    abs_recovered_vals = np.abs(recovered_vals)

    # Color code for top plot: green for passed, red for failed (even if excluded)
    colors_top = []
    for i, p in enumerate(param_names):
        if all_recovery_stats[p]['passed']:
            colors_top.append('green')
        else:
            colors_top.append('red')

    # Color code for bottom plot: green for passed, red for failed (even if excluded)
    colors_bar = []
    for i, p in enumerate(param_names):
        if all_recovery_stats[p]['passed']:
            colors_bar.append('green')
        else:
            colors_bar.append('red')

    # Add tolerance bands around true values (only for relative tolerance)
    for i, param_name in enumerate(param_names):
        criterion = all_recovery_stats[param_name]['criterion']
        true_val = abs_true_vals[i]

        if criterion == 'relative':
            rel_tol = all_recovery_stats[param_name]['rel_tolerance']
            lower = true_val * (1 - rel_tol)
            upper = true_val * (1 + rel_tol)
            ax1.fill_between(
                [x_pos[i] - 0.4, x_pos[i] + 0.4],
                lower,
                upper,
                alpha=0.2,
                color='gray',
                zorder=0,
            )

    # Plot on categorical x-axis
    # Non-excluded params use circles, excluded params use 'x' markers
    for i, param_name in enumerate(param_names):
        if param_name in exclude_params:
            ax1.scatter(
                x_pos[i],
                abs_recovered_vals[i],
                c=colors_top[i],
                marker='x',
                s=120,
                linewidths=2.5,
                alpha=1.0,
                zorder=3,
            )
        else:
            ax1.scatter(
                x_pos[i],
                abs_recovered_vals[i],
                c=colors_top[i],
                s=120,
                alpha=0.7,
                edgecolors='black',
                linewidths=1.5,
                zorder=3,
            )

    ax1.scatter(
        x_pos,
        abs_true_vals,
        marker='_',
        s=500,
        c='black',
        linewidths=3,
        zorder=4,
        label='True value',
    )

    # Connect true to recovered with lines
    for i in range(len(param_names)):
        ax1.plot(
            [x_pos[i], x_pos[i]],
            [abs_true_vals[i], abs_recovered_vals[i]],
            'k-',
            alpha=0.3,
            linewidth=1,
            zorder=1,
        )

    # Add absolute difference text above each circle
    for i in range(len(param_names)):
        abs_diff = abs(abs_recovered_vals[i] - abs_true_vals[i])
        # Position text above the higher of the two values, but not too high
        y_pos = max(abs_recovered_vals[i], abs_true_vals[i]) * 1.5
        ax1.text(
            x_pos[i],
            y_pos,
            f'{abs_diff:.3f}',
            ha='center',
            va='bottom',
            fontsize=9,
            color='black',
            bbox=dict(
                boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none'
            ),
            clip_on=False,
        )

    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(param_names, rotation=45, ha='right', fontsize=10)
    ax1.set_ylabel('|Parameter Value|', fontsize=13, fontweight='bold')
    ax1.set_yscale('log')

    # Adjust y-axis limits to ensure text labels are fully visible
    # Calculate the maximum y position of text labels
    max_text_y = max(
        [
            max(abs_recovered_vals[i], abs_true_vals[i]) * 1.5
            for i in range(len(param_names))
        ]
    )
    # Add extra space for the text box height (approximately 2x for the text box)
    current_ylim = ax1.get_ylim()
    ax1.set_ylim(current_ylim[0], max_text_y * 2.0)

    # More informative title with excluded params info
    title_str = f'Parameter Recovery: {test_name} (SNR={snr})'
    # Always show format: X passed / Y failed / Z excluded
    if n_excluded > 0:
        subtitle_str = f'{n_pass} passed / {n_fail} failed / {n_excluded} excluded ({", ".join(exclude_params)})'
    else:
        subtitle_str = f'{n_pass} passed / {n_fail} failed'

    # Add TEST PASSED indicator if no non-excluded failures
    if n_fail == 0:
        test_status = '✓ TEST PASSED'
        status_color = 'green'
    else:
        test_status = '✗ TEST FAILED'
        status_color = 'red'

    ax1.set_title(
        f'{title_str}\n{subtitle_str}\n{test_status}',
        fontsize=14,
        pad=10,
        color=status_color,
        weight='bold',
    )

    # Add legend with X markers for excluded params
    legend_elements_top = [
        plt.Line2D(
            [0],
            [0],
            marker='o',
            color='w',
            markerfacecolor='green',
            markersize=10,
            alpha=0.7,
            markeredgecolor='black',
            label='Passed (circle)',
        ),
        plt.Line2D(
            [0],
            [0],
            marker='o',
            color='w',
            markerfacecolor='red',
            markersize=10,
            alpha=0.7,
            markeredgecolor='black',
            label='Failed (circle)',
        ),
    ]
    if n_excluded > 0:
        legend_elements_top.append(
            plt.Line2D(
                [0],
                [0],
                marker='x',
                color='green',
                markersize=10,
                linewidth=2.5,
                label='Excluded (X)',
            )
        )
    legend_elements_top.append(
        plt.Line2D(
            [0],
            [0],
            marker='_',
            color='black',
            markersize=15,
            linewidth=3,
            label='True value',
        )
    )
    ax1.legend(handles=legend_elements_top, loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle=':', linewidth=0.5, axis='y')

    # Panel 2: Fractional error bar chart
    x_pos_bar = np.arange(len(param_names))

    # Compute fractional errors: (recovered - true) / true
    frac_errors = []
    for p in param_names:
        true_val = all_recovery_stats[p]['true']
        if true_val != 0:
            frac_errors.append(
                (all_recovery_stats[p]['recovered'] - true_val) / true_val * 100
            )
        else:
            frac_errors.append(0)
    frac_errors = np.array(frac_errors)

    # Use hatching for excluded parameters
    for i, param_name in enumerate(param_names):
        edgecolor = 'black'
        hatch = '///' if param_name in exclude_params else None
        ax2.bar(
            x_pos_bar[i],
            frac_errors[i],
            color=colors_bar[i],
            alpha=0.7,
            edgecolor=edgecolor,
            linewidth=2.5 if param_name in exclude_params else 1.5,
            hatch=hatch,
        )

    # Add tolerance threshold lines (only for relative tolerance)
    has_rel_tol = False
    for i, param_name in enumerate(param_names):
        criterion = all_recovery_stats[param_name]['criterion']

        if criterion == 'relative':
            rel_tol = all_recovery_stats[param_name]['rel_tolerance'] * 100
            ax2.hlines(
                [rel_tol, -rel_tol],
                i - 0.4,
                i + 0.4,
                colors='black',
                linestyles=':',
                linewidths=2,
                zorder=4,
            )
            has_rel_tol = True

    # Add legend
    legend_elements = [
        plt.Rectangle(
            (0, 0), 1, 1, fc='green', alpha=0.7, edgecolor='black', label='Passed'
        ),
        plt.Rectangle(
            (0, 0), 1, 1, fc='red', alpha=0.7, edgecolor='black', label='Failed'
        ),
    ]
    if n_excluded > 0:
        legend_elements.append(
            plt.Rectangle(
                (0, 0),
                1,
                1,
                fc='green',
                alpha=0.7,
                edgecolor='black',
                hatch='///',
                linewidth=2.5,
                label='Excluded (hatched)',
            )
        )
    if has_rel_tol:
        legend_elements.append(
            plt.Line2D(
                [0],
                [0],
                color='black',
                linestyle=':',
                linewidth=2,
                label='Relative tolerance',
            )
        )
    ax2.legend(handles=legend_elements, loc='best', fontsize=10, ncol=2)

    ax2.set_xticks(x_pos_bar)
    ax2.set_xticklabels(param_names, rotation=45, ha='right', fontsize=10)
    ax2.set_ylabel('Fractional Error (%)', fontsize=13, fontweight='bold')
    ax2.set_title('Parameter Recovery Errors\n(Recovered - True) / True', fontsize=12)
    ax2.axhline(0, color='black', linestyle='-', linewidth=1.5, zorder=2)
    ax2.grid(True, alpha=0.3, axis='y', linestyle=':', linewidth=0.5)

    plt.tight_layout()

    # Save
    outfile = test_dir / f"{test_name}_parameter_comparison.png"
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_combined_data_comparison(
    data_vel_noisy: jnp.ndarray,
    data_vel_true: jnp.ndarray,
    model_vel: jnp.ndarray,
    data_int_noisy: jnp.ndarray,
    data_int_true: jnp.ndarray,
    model_int: jnp.ndarray,
    test_name: str,
    config: TestConfig,
    variance_vel: Optional[float] = None,
    variance_int: Optional[float] = None,
    n_params: Optional[int] = None,
    model_label: str = 'MAP Model',
) -> None:
    """
    Create combined 4x3 panel diagnostic plot for velocity + intensity.

    .. deprecated::
        Use kl_pipe.diagnostics.plot_combined_data_comparison instead.
        This function is kept for backward compatibility.

    Stacks velocity (top 2 rows) and intensity (bottom 2 rows) comparisons
    into a single output figure.

    Layout:
        Row 0: Velocity - noisy | true | noisy - true
        Row 1: Velocity - model - true | model | noisy - model
        Row 2: Intensity - noisy | true | noisy - true
        Row 3: Intensity - model - true | model | noisy - model

    Parameters
    ----------
    data_vel_noisy, data_vel_true, model_vel : jnp.ndarray
        Velocity data arrays.
    data_int_noisy, data_int_true, model_int : jnp.ndarray
        Intensity data arrays.
    test_name : str
        Name of test (for title and filename).
    config : TestConfig
        Test configuration.
    variance_vel, variance_int : float, optional
        Variances for chi-squared computation.
    n_params : int, optional
        Number of fitted parameters (for reduced chi-squared).
    model_label : str, optional
        Label for model panels. Default is 'MAP Model'.
    """
    warnings.warn(
        "plot_combined_data_comparison in test_utils.py is deprecated. "
        "Use kl_pipe.diagnostics.plot_combined_data_comparison instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    if not config.enable_plots:
        return

    # Create output directory for this test
    test_dir = config.output_dir / test_name
    test_dir.mkdir(parents=True, exist_ok=True)

    # Set up figure with 4 rows x 3 cols
    fig, axes = plt.subplots(4, 3, figsize=(15, 18))

    # Helper function to plot a 2x3 block
    def plot_data_block(
        axes_block,
        data_noisy,
        data_true,
        model_eval,
        variance,
        data_type,
    ):
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

        # Common colorbar limits for data
        data_arrays = [data_noisy, data_true, model_eval]
        vmin_data = min(np.percentile(arr, 1) for arr in data_arrays)
        vmax_data = max(np.percentile(arr, 99) for arr in data_arrays)
        norm_data = MidpointNormalize(vmin=vmin_data, vmax=vmax_data, midpoint=0)

        # Common colorbar limits for residuals
        residual_arrays = [residual_true, residual_model]
        abs_max = max(
            np.abs(np.percentile(arr, [1, 99])).max() for arr in residual_arrays
        )
        norm_resid = MidpointNormalize(vmin=-abs_max, vmax=abs_max, midpoint=0)

        # Row 0: noisy | true | noisy - true
        im00 = axes_block[0, 0].imshow(
            np.array(data_noisy), origin='lower', cmap='RdBu_r', norm=norm_data
        )
        axes_block[0, 0].set_title(f'{data_type}: Noisy Data')
        divider = make_axes_locatable(axes_block[0, 0])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im00, cax=cax)

        im01 = axes_block[0, 1].imshow(
            np.array(data_true), origin='lower', cmap='RdBu_r', norm=norm_data
        )
        axes_block[0, 1].set_title(f'{data_type}: True')
        divider = make_axes_locatable(axes_block[0, 1])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im01, cax=cax)

        im02 = axes_block[0, 2].imshow(
            residual_true, origin='lower', cmap='RdBu_r', norm=norm_resid
        )
        axes_block[0, 2].set_title(f'{data_type}: Noisy - True')
        divider = make_axes_locatable(axes_block[0, 2])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im02, cax=cax)
        if variance is not None:
            axes_block[0, 2].text(
                0.02,
                0.98,
                f'χ²={chi2_true:.1f}',
                transform=axes_block[0, 2].transAxes,
                fontsize=9,
                color='white',
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.5),
            )

        # Row 1: model - true | model | noisy - model
        im10 = axes_block[1, 0].imshow(
            residual_model_true, origin='lower', cmap='RdBu_r', norm=norm_resid
        )
        axes_block[1, 0].set_title(f'{data_type}: {model_label} - True')
        divider = make_axes_locatable(axes_block[1, 0])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im10, cax=cax)

        im11 = axes_block[1, 1].imshow(
            np.array(model_eval), origin='lower', cmap='RdBu_r', norm=norm_data
        )
        axes_block[1, 1].set_title(f'{data_type}: {model_label}')
        divider = make_axes_locatable(axes_block[1, 1])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im11, cax=cax)

        im12 = axes_block[1, 2].imshow(
            residual_model, origin='lower', cmap='RdBu_r', norm=norm_resid
        )
        axes_block[1, 2].set_title(f'{data_type}: Noisy - {model_label}')
        divider = make_axes_locatable(axes_block[1, 2])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im12, cax=cax)
        if variance is not None:
            axes_block[1, 2].text(
                0.02,
                0.98,
                f'χ²={chi2_model:.1f}',
                transform=axes_block[1, 2].transAxes,
                fontsize=9,
                color='white',
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.5),
            )

        # Labels
        for ax in axes_block.flat:
            ax.set_xlabel('x (pixels)')
            ax.set_ylabel('y (pixels)')

    # Plot velocity block (rows 0-1)
    plot_data_block(
        axes[0:2, :], data_vel_noisy, data_vel_true, model_vel, variance_vel, 'Velocity'
    )

    # Plot intensity block (rows 2-3)
    plot_data_block(
        axes[2:4, :],
        data_int_noisy,
        data_int_true,
        model_int,
        variance_int,
        'Intensity',
    )

    # Overall title
    fig.suptitle(f'{test_name} - Combined Data Comparison', fontsize=14, y=1.01)
    plt.tight_layout()

    # Save
    outfile = test_dir / f"{test_name}_combined_panels.png"
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
        tolerance = config.get_tolerance(
            snr, param_name, true_value, data_type, test_type='likelihood_slice'
        )

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
        base_tolerance = config.likelihood_slice_tolerance_velocity[snr]
    else:
        base_tolerance = config.likelihood_slice_tolerance_intensity[snr]
    fig.suptitle(
        f'{test_name} - Likelihood Slices '
        f'(SNR={snr}, base tolerance={base_tolerance*100:.1f}%)',
        fontsize=14,
        y=1.01,
    )
    plt.tight_layout()

    # save figure
    outfile = test_dir / f"{test_name}_likelihood_slices.png"
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close(fig)

    return recovery_stats
