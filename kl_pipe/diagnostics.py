"""
Shared diagnostic plotting utilities for parameter recovery tests.

This module provides plotting functions used by both optimizer-based and
sampler-based parameter recovery tests. Functions are designed to:
1. Compute and return statistics (never raise exceptions)
2. Create publication-quality visualizations
3. Support both point estimates (optimizers) and posteriors (samplers)

Key Functions
-------------
- plot_data_comparison_panels: 2x3 panel diagnostic plot for velocity/intensity
- plot_combined_data_comparison: 4x3 panel for joint velocity+intensity
- plot_parameter_recovery: Two-panel recovery plot with joint Nσ validation
- compute_joint_nsigma: Statistical validation using covariance matrix
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Tuple, List, Union, TYPE_CHECKING

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import stats as scipy_stats

if TYPE_CHECKING:
    import jax.numpy as jnp
    from kl_pipe.parameters import ImagePars

# Try to import MidpointNormalize, fall back to simple implementation
try:
    from kl_pipe.plotting import MidpointNormalize
except ImportError:
    # Simple fallback
    class MidpointNormalize(plt.Normalize):
        def __init__(self, vmin=None, vmax=None, midpoint=0, clip=False):
            self.midpoint = midpoint
            super().__init__(vmin, vmax, clip)


# ==============================================================================
# Statistical Utilities
# ==============================================================================


def compute_joint_nsigma(
    recovered: Dict[str, float],
    true_values: Dict[str, float],
    covariance: Optional[np.ndarray] = None,
    samples: Optional[np.ndarray] = None,
    param_names: Optional[List[str]] = None,
) -> Dict[str, any]:
    """
    Compute joint Nσ deviation using Mahalanobis distance.

    This provides a statistically rigorous measure of how far the recovered
    parameters are from the true values, accounting for parameter correlations.
    The Mahalanobis distance follows a chi-squared distribution with n_params
    degrees of freedom.

    Parameters
    ----------
    recovered : dict
        Dictionary of recovered parameter values.
    true_values : dict
        Dictionary of true parameter values.
    covariance : np.ndarray, optional
        Covariance matrix. If not provided, computed from samples.
    samples : np.ndarray, optional
        Sample array of shape (n_samples, n_params). Required if covariance
        is not provided.
    param_names : list of str, optional
        Parameter names corresponding to samples columns. Required if samples
        is provided without covariance.

    Returns
    -------
    dict
        Dictionary containing:
        - 'chi2': Mahalanobis distance squared (chi-squared statistic)
        - 'n_params': Number of parameters
        - 'nsigma': Equivalent Gaussian sigma
        - 'pvalue': p-value from chi-squared distribution
        - 'delta_theta': Parameter deviation vector
        - 'covariance': Covariance matrix used

    Notes
    -----
    The Mahalanobis distance is computed as:
        χ² = Δθᵀ Σ⁻¹ Δθ

    This is converted to an equivalent Gaussian sigma using the chi-squared
    survival function:
        p = χ²_sf(χ², n_params)
        Nσ = √2 * erfinv(1 - p)

    Examples
    --------
    >>> stats = compute_joint_nsigma(
    ...     recovered={'vcirc': 198.5, 'cosi': 0.62},
    ...     true_values={'vcirc': 200.0, 'cosi': 0.60},
    ...     samples=posterior_samples,
    ...     param_names=['vcirc', 'cosi']
    ... )
    >>> print(f"Joint deviation: {stats['nsigma']:.2f}σ")
    """
    # Get common parameters
    common_params = [p for p in true_values.keys() if p in recovered]
    n_params = len(common_params)

    if n_params == 0:
        return {
            'chi2': np.nan,
            'n_params': 0,
            'nsigma': np.nan,
            'pvalue': np.nan,
            'delta_theta': np.array([]),
            'covariance': np.array([[]]),
        }

    # Build deviation vector
    delta_theta = np.array([recovered[p] - true_values[p] for p in common_params])

    # Compute or validate covariance
    if covariance is None:
        if samples is None:
            raise ValueError("Either covariance or samples must be provided")
        if param_names is None:
            raise ValueError("param_names required when samples is provided")

        # Extract columns for common parameters
        param_indices = [param_names.index(p) for p in common_params]
        samples_subset = samples[:, param_indices]
        covariance = np.cov(samples_subset.T)

        # Ensure covariance is 2D even for single parameter
        if n_params == 1:
            covariance = covariance.reshape(1, 1)

    # Use pseudo-inverse for potentially singular matrices
    try:
        cov_inv = np.linalg.pinv(covariance)
    except np.linalg.LinAlgError:
        # Fallback: regularize covariance
        cov_regularized = covariance + 1e-10 * np.eye(n_params)
        cov_inv = np.linalg.pinv(cov_regularized)

    # Compute Mahalanobis distance squared (chi-squared statistic)
    chi2 = float(delta_theta @ cov_inv @ delta_theta)

    # Compute p-value from chi-squared distribution
    pvalue = scipy_stats.chi2.sf(chi2, df=n_params)

    # Convert p-value to equivalent Gaussian sigma
    # Using inverse survival function of standard normal
    if pvalue > 0 and pvalue < 1:
        nsigma = scipy_stats.norm.isf(pvalue / 2)  # Two-tailed
    elif pvalue <= 0:
        nsigma = np.inf
    else:
        nsigma = 0.0

    return {
        'chi2': chi2,
        'n_params': n_params,
        'nsigma': nsigma,
        'pvalue': pvalue,
        'delta_theta': delta_theta,
        'covariance': covariance,
        'param_names': common_params,
    }


def nsigma_to_color(nsigma: float) -> str:
    """
    Convert Nσ value to color for visual feedback.

    Parameters
    ----------
    nsigma : float
        Number of sigma deviation.

    Returns
    -------
    str
        Color string: 'green' (<2σ), 'orange' (2-3σ), 'red' (>3σ).
    """
    if np.isnan(nsigma):
        return 'gray'
    elif nsigma < 2.0:
        return 'green'
    elif nsigma < 3.0:
        return 'orange'
    else:
        return 'red'


# ==============================================================================
# Data Comparison Plots
# ==============================================================================


def plot_data_comparison_panels(
    data_noisy: np.ndarray,
    data_true: np.ndarray,
    model_eval: np.ndarray,
    test_name: str,
    output_dir: Path,
    data_type: str = 'velocity',
    variance: Optional[float] = None,
    n_params: Optional[int] = None,
    model_label: str = 'Model',
    enable_plots: bool = True,
) -> Optional[Path]:
    """
    Create 2x3 panel diagnostic plot.

    Row 1: noisy | true | noisy - true
    Row 2: model - true | model | noisy - model

    Parameters
    ----------
    data_noisy : ndarray
        Noisy synthetic data.
    data_true : ndarray
        True noiseless data.
    model_eval : ndarray
        Model evaluation (can be at true or optimized parameters).
    test_name : str
        Name of test (for title and filename).
    output_dir : Path
        Directory to save output.
    data_type : str, optional
        Type of data ('velocity' or 'intensity'). Default is 'velocity'.
    variance : float, optional
        Variance of noise, for reduced chi-squared computation.
    n_params : int, optional
        Number of fitted parameters (for reduced chi-squared).
    model_label : str, optional
        Label for model panel. Default is 'Model'.
    enable_plots : bool, optional
        If False, skip plotting. Default is True.

    Returns
    -------
    Path or None
        Path to saved figure, or None if plotting disabled.
    """
    if not enable_plots:
        return None

    # Ensure output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert to numpy arrays
    data_noisy = np.asarray(data_noisy)
    data_true = np.asarray(data_true)
    model_eval = np.asarray(model_eval)

    # Compute residuals & chi2
    residual_true = data_noisy - data_true
    residual_model = data_noisy - model_eval
    residual_model_true = model_eval - data_true

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
    im00 = axes[0, 0].imshow(data_noisy, origin='lower', cmap='RdBu_r', norm=norm_data)
    axes[0, 0].set_title('Noisy Data')
    divider = make_axes_locatable(axes[0, 0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im00, cax=cax)

    im01 = axes[0, 1].imshow(data_true, origin='lower', cmap='RdBu_r', norm=norm_data)
    axes[0, 1].set_title('True (Noiseless)')
    divider = make_axes_locatable(axes[0, 1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im01, cax=cax)

    im02 = axes[0, 2].imshow(residual_true, origin='lower', cmap='RdBu_r', norm=norm_resid)
    axes[0, 2].set_title('Noisy - True')
    divider = make_axes_locatable(axes[0, 2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im02, cax=cax)
    if variance is not None:
        axes[0, 2].text(
            0.02, 0.98, f'χ² = {chi2_true:.1f}',
            transform=axes[0, 2].transAxes, fontsize=10,
            color='white', verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.5),
        )

    # Row 2: model - true | model | noisy - model
    im10 = axes[1, 0].imshow(residual_model_true, origin='lower', cmap='RdBu_r', norm=norm_resid)
    axes[1, 0].set_title('Model - True')
    divider = make_axes_locatable(axes[1, 0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im10, cax=cax)

    im11 = axes[1, 1].imshow(model_eval, origin='lower', cmap='RdBu_r', norm=norm_data)
    axes[1, 1].set_title(model_label)
    divider = make_axes_locatable(axes[1, 1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im11, cax=cax)

    im12 = axes[1, 2].imshow(residual_model, origin='lower', cmap='RdBu_r', norm=norm_resid)
    axes[1, 2].set_title('Noisy - Model')
    divider = make_axes_locatable(axes[1, 2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im12, cax=cax)
    if variance is not None:
        axes[1, 2].text(
            0.02, 0.98, f'χ² = {chi2_model:.1f}',
            transform=axes[1, 2].transAxes, fontsize=10,
            color='white', verticalalignment='top',
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
    outfile = output_dir / f"{test_name}_{data_type}_panels.png"
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close(fig)

    return outfile


def plot_combined_data_comparison(
    data_vel_noisy: np.ndarray,
    data_vel_true: np.ndarray,
    model_vel: np.ndarray,
    data_int_noisy: np.ndarray,
    data_int_true: np.ndarray,
    model_int: np.ndarray,
    test_name: str,
    output_dir: Path,
    variance_vel: Optional[float] = None,
    variance_int: Optional[float] = None,
    n_params: Optional[int] = None,
    model_label: str = 'MAP Model',
    enable_plots: bool = True,
) -> Optional[Path]:
    """
    Create combined 4x3 panel diagnostic plot for velocity + intensity.

    Stacks velocity (top 2 rows) and intensity (bottom 2 rows) comparisons
    into a single output figure.

    Layout:
        Row 0: Velocity - noisy | true | noisy - true
        Row 1: Velocity - model - true | model | noisy - model
        Row 2: Intensity - noisy | true | noisy - true
        Row 3: Intensity - model - true | model | noisy - model

    Parameters
    ----------
    data_vel_noisy, data_vel_true, model_vel : ndarray
        Velocity data arrays.
    data_int_noisy, data_int_true, model_int : ndarray
        Intensity data arrays.
    test_name : str
        Name of test (for title and filename).
    output_dir : Path
        Directory to save output.
    variance_vel, variance_int : float, optional
        Variances for chi-squared computation.
    n_params : int, optional
        Number of fitted parameters (for reduced chi-squared).
    model_label : str, optional
        Label for model panels. Default is 'MAP Model'.
    enable_plots : bool, optional
        If False, skip plotting. Default is True.

    Returns
    -------
    Path or None
        Path to saved figure, or None if plotting disabled.
    """
    if not enable_plots:
        return None

    # Ensure output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert to numpy
    data_vel_noisy = np.asarray(data_vel_noisy)
    data_vel_true = np.asarray(data_vel_true)
    model_vel = np.asarray(model_vel)
    data_int_noisy = np.asarray(data_int_noisy)
    data_int_true = np.asarray(data_int_true)
    model_int = np.asarray(model_int)

    # Set up figure with 4 rows x 3 cols
    fig, axes = plt.subplots(4, 3, figsize=(15, 18))

    # Helper function to plot a 2x3 block
    def plot_data_block(axes_block, data_noisy, data_true, model_eval, variance, data_type):
        # Compute residuals & chi2
        residual_true = data_noisy - data_true
        residual_model = data_noisy - model_eval
        residual_model_true = model_eval - data_true

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
        abs_max = max(np.abs(np.percentile(arr, [1, 99])).max() for arr in residual_arrays)
        norm_resid = MidpointNormalize(vmin=-abs_max, vmax=abs_max, midpoint=0)

        # Row 0: noisy | true | noisy - true
        im00 = axes_block[0, 0].imshow(data_noisy, origin='lower', cmap='RdBu_r', norm=norm_data)
        axes_block[0, 0].set_title(f'{data_type}: Noisy Data')
        divider = make_axes_locatable(axes_block[0, 0])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im00, cax=cax)

        im01 = axes_block[0, 1].imshow(data_true, origin='lower', cmap='RdBu_r', norm=norm_data)
        axes_block[0, 1].set_title(f'{data_type}: True')
        divider = make_axes_locatable(axes_block[0, 1])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im01, cax=cax)

        im02 = axes_block[0, 2].imshow(residual_true, origin='lower', cmap='RdBu_r', norm=norm_resid)
        axes_block[0, 2].set_title(f'{data_type}: Noisy - True')
        divider = make_axes_locatable(axes_block[0, 2])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im02, cax=cax)
        if variance is not None:
            axes_block[0, 2].text(
                0.02, 0.98, f'χ²={chi2_true:.1f}',
                transform=axes_block[0, 2].transAxes, fontsize=9,
                color='white', verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.5)
            )

        # Row 1: model - true | model | noisy - model
        im10 = axes_block[1, 0].imshow(residual_model_true, origin='lower', cmap='RdBu_r', norm=norm_resid)
        axes_block[1, 0].set_title(f'{data_type}: {model_label} - True')
        divider = make_axes_locatable(axes_block[1, 0])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im10, cax=cax)

        im11 = axes_block[1, 1].imshow(model_eval, origin='lower', cmap='RdBu_r', norm=norm_data)
        axes_block[1, 1].set_title(f'{data_type}: {model_label}')
        divider = make_axes_locatable(axes_block[1, 1])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im11, cax=cax)

        im12 = axes_block[1, 2].imshow(residual_model, origin='lower', cmap='RdBu_r', norm=norm_resid)
        axes_block[1, 2].set_title(f'{data_type}: Noisy - {model_label}')
        divider = make_axes_locatable(axes_block[1, 2])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im12, cax=cax)
        if variance is not None:
            axes_block[1, 2].text(
                0.02, 0.98, f'χ²={chi2_model:.1f}',
                transform=axes_block[1, 2].transAxes, fontsize=9,
                color='white', verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.5)
            )

        # Labels
        for ax in axes_block.flat:
            ax.set_xlabel('x (pixels)')
            ax.set_ylabel('y (pixels)')

    # Plot velocity block (rows 0-1)
    plot_data_block(axes[0:2, :], data_vel_noisy, data_vel_true, model_vel,
                    variance_vel, 'Velocity')

    # Plot intensity block (rows 2-3)
    plot_data_block(axes[2:4, :], data_int_noisy, data_int_true, model_int,
                    variance_int, 'Intensity')

    # Overall title
    fig.suptitle(f'{test_name} - Combined Data Comparison', fontsize=14, y=1.01)
    plt.tight_layout()

    # Save
    outfile = output_dir / f"{test_name}_combined_panels.png"
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close(fig)

    return outfile


# ==============================================================================
# Parameter Recovery Plots
# ==============================================================================


def plot_parameter_recovery(
    true_values: Dict[str, float],
    recovered_values: Dict[str, float],
    output_dir: Path,
    test_name: str,
    samples: Optional[np.ndarray] = None,
    param_names: Optional[List[str]] = None,
    uncertainties: Optional[Dict[str, Tuple[float, float]]] = None,
    derived_params: Optional[Dict[str, Tuple[float, float]]] = None,
    enable_plots: bool = True,
    sampler_name: Optional[str] = None,
) -> Dict[str, any]:
    """
    Create two-panel parameter recovery plot with joint Nσ validation.

    Top panel: Log-scale absolute values (true vs recovered)
    Bottom panel: Fractional error (%) with uncertainty bands

    The joint Nσ statistic measures how far the recovered parameters
    deviate from true values, accounting for correlations. This is
    displayed in the title with color-coding:
    - Green: < 2σ (expected ~95% of the time)
    - Yellow/Orange: 2-3σ (rare but not alarming)
    - Red: > 3σ (potential problem)

    Parameters
    ----------
    true_values : dict
        True parameter values.
    recovered_values : dict
        Recovered parameter values (MAP estimates or medians).
    output_dir : Path
        Directory to save output.
    test_name : str
        Name for file output.
    samples : np.ndarray, optional
        Posterior samples of shape (n_samples, n_params). Required for
        covariance-based Nσ computation. If not provided, uses simple
        per-parameter comparison.
    param_names : list of str, optional
        Parameter names corresponding to samples columns.
    uncertainties : dict, optional
        Dictionary mapping param name to (lower_error, upper_error) tuple.
        Used for error bars in bottom panel.
    derived_params : dict, optional
        Derived parameters to include, e.g., {'vcirc*cosi': (true, recovered)}.
    enable_plots : bool, optional
        If False, skip plotting but still compute statistics.
    sampler_name : str, optional
        Sampler name for title annotation.

    Returns
    -------
    dict
        Dictionary containing:
        - 'joint_nsigma': Joint Nσ deviation statistic
        - 'joint_chi2': Chi-squared statistic
        - 'joint_pvalue': P-value
        - 'per_param': Dict of per-parameter statistics
        - 'title_color': Color based on Nσ thresholds
        - 'output_path': Path to saved figure (if plotting enabled)

    Notes
    -----
    This function never raises exceptions. It computes and returns statistics
    that tests can use for pass/fail decisions.
    """
    # Build parameter lists
    common_params = [p for p in true_values.keys() if p in recovered_values]

    # Add derived parameters
    all_params = list(common_params)
    all_true = {p: true_values[p] for p in common_params}
    all_recovered = {p: recovered_values[p] for p in common_params}

    if derived_params:
        for name, (true_val, rec_val) in derived_params.items():
            all_params.append(name)
            all_true[name] = true_val
            all_recovered[name] = rec_val

    # Compute joint Nσ using covariance if samples provided
    joint_stats = {'nsigma': np.nan, 'chi2': np.nan, 'pvalue': np.nan}
    if samples is not None and param_names is not None:
        joint_stats = compute_joint_nsigma(
            recovered=recovered_values,
            true_values=true_values,
            samples=samples,
            param_names=param_names,
        )

    # Compute per-parameter statistics
    per_param_stats = {}
    for p in all_params:
        true_val = all_true[p]
        rec_val = all_recovered[p]
        abs_error = abs(rec_val - true_val)

        if abs(true_val) > 1e-10:
            frac_error = (rec_val - true_val) / true_val
        else:
            frac_error = rec_val - true_val  # Absolute for near-zero

        per_param_stats[p] = {
            'true': true_val,
            'recovered': rec_val,
            'abs_error': abs_error,
            'frac_error': frac_error,
        }

    # Determine title color
    title_color = nsigma_to_color(joint_stats.get('nsigma', np.nan))

    # Prepare output
    result = {
        'joint_nsigma': joint_stats.get('nsigma', np.nan),
        'joint_chi2': joint_stats.get('chi2', np.nan),
        'joint_pvalue': joint_stats.get('pvalue', np.nan),
        'per_param': per_param_stats,
        'title_color': title_color,
        'output_path': None,
    }

    if not enable_plots:
        return result

    # Create figure
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    x_pos = np.arange(len(all_params))

    # === Panel 1: Log-scale absolute values ===
    true_vals = np.array([abs(all_true[p]) for p in all_params])
    rec_vals = np.array([abs(all_recovered[p]) for p in all_params])

    # Use minimum nonzero value for log scale floor
    all_nonzero = np.concatenate([true_vals[true_vals > 0], rec_vals[rec_vals > 0]])
    if len(all_nonzero) > 0:
        log_floor = all_nonzero.min() * 0.1
    else:
        log_floor = 1e-10

    # Clip zeros to floor for log scale
    true_vals_plot = np.maximum(true_vals, log_floor)
    rec_vals_plot = np.maximum(rec_vals, log_floor)

    # Plot true values as black horizontal lines
    for i in range(len(all_params)):
        ax1.hlines(true_vals_plot[i], i - 0.3, i + 0.3, colors='black',
                   linewidth=3, label='True' if i == 0 else None)

    # Plot recovered values as colored circles
    ax1.scatter(x_pos, rec_vals_plot, s=150, c='C0', edgecolors='black',
                linewidths=1.5, zorder=5, label='Recovered')

    # Connect with lines
    for i in range(len(all_params)):
        ax1.plot([i, i], [true_vals_plot[i], rec_vals_plot[i]],
                 'k-', alpha=0.3, linewidth=1)

    # Add absolute difference annotations
    for i in range(len(all_params)):
        abs_diff = per_param_stats[all_params[i]]['abs_error']
        y_pos = max(true_vals_plot[i], rec_vals_plot[i]) * 1.5
        ax1.text(i, y_pos, f'{abs_diff:.4g}', ha='center', va='bottom',
                 fontsize=9, bbox=dict(boxstyle='round,pad=0.3',
                                       facecolor='white', alpha=0.8))

    ax1.set_yscale('log')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(all_params, rotation=45, ha='right', fontsize=10)
    ax1.set_ylabel('|Parameter Value|', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y', linestyle=':')
    ax1.set_xlim(-0.5, len(all_params) - 0.5)

    # === Panel 2: Fractional error (%) ===
    frac_errors = np.array([per_param_stats[p]['frac_error'] * 100 for p in all_params])

    # Add error bars if uncertainties provided
    if uncertainties:
        yerr_lower = []
        yerr_upper = []
        for p in all_params:
            if p in uncertainties:
                low, high = uncertainties[p]
                true_val = all_true[p]
                if abs(true_val) > 1e-10:
                    yerr_lower.append(low / abs(true_val) * 100)
                    yerr_upper.append(high / abs(true_val) * 100)
                else:
                    yerr_lower.append(0)
                    yerr_upper.append(0)
            else:
                yerr_lower.append(0)
                yerr_upper.append(0)
        ax2.bar(x_pos, frac_errors, color='C0', alpha=0.7, edgecolor='black',
                yerr=[yerr_lower, yerr_upper], capsize=4)
    else:
        ax2.bar(x_pos, frac_errors, color='C0', alpha=0.7, edgecolor='black')

    ax2.axhline(0, color='black', linestyle='-', linewidth=1.5)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(all_params, rotation=45, ha='right', fontsize=10)
    ax2.set_ylabel('Fractional Error (%)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y', linestyle=':')
    ax2.set_xlim(-0.5, len(all_params) - 0.5)

    # === Title with joint Nσ ===
    nsigma_val = joint_stats.get('nsigma', np.nan)
    if not np.isnan(nsigma_val):
        nsigma_str = f'Joint: {nsigma_val:.2f}σ'
    else:
        nsigma_str = 'Joint: N/A'

    title_parts = [f'Parameter Recovery: {test_name}']
    if sampler_name:
        title_parts[0] += f' ({sampler_name})'
    title_parts.append(nsigma_str)

    fig.suptitle('\n'.join(title_parts), fontsize=14, fontweight='bold',
                 color=title_color)

    plt.tight_layout()

    # Save
    outfile = output_dir / f"{test_name}_parameter_recovery.png"
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close(fig)

    result['output_path'] = outfile
    return result
