"""
Diagnostic plotting utilities for MCMC results.

Provides functions for visualizing sampler output:
- Trace plots for convergence assessment
- Corner plots for parameter correlations
- Recovery comparison plots for validation
- Convergence warning detection and annotation
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Dict, Tuple, TYPE_CHECKING

import numpy as np
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from kl_pipe.sampling.base import SamplerResult


# ==============================================================================
# Convergence Warning Detection
# ==============================================================================


def check_convergence_warnings(result: 'SamplerResult') -> Dict[str, any]:
    """
    Check for potential convergence issues in sampling results.

    Checks for:
    - Zero or near-zero variance in parameters (sampler not moving)
    - Very low acceptance rate
    - Very small sample size

    Parameters
    ----------
    result : SamplerResult
        Sampling results to check.

    Returns
    -------
    dict
        Dictionary containing:
        - 'has_warnings': bool, True if any warnings detected
        - 'warnings': list of warning strings
        - 'zero_variance_params': list of parameter names with zero variance
        - 'low_acceptance': bool, True if acceptance rate < 10%

    Examples
    --------
    >>> warnings = check_convergence_warnings(result)
    >>> if warnings['has_warnings']:
    ...     print("Convergence issues detected:", warnings['warnings'])
    """
    warnings_list = []
    zero_variance_params = []

    # Check for zero variance in parameters
    for name in result.param_names:
        chain = result.get_chain(name)
        var = np.var(chain)
        if var < 1e-10:
            zero_variance_params.append(name)
            warnings_list.append(f"Parameter '{name}' has zero variance (sampler not exploring)")

    # Check acceptance rate
    low_acceptance = False
    if result.acceptance_fraction is not None and result.acceptance_fraction < 0.1:
        low_acceptance = True
        warnings_list.append(
            f"Low acceptance rate ({result.acceptance_fraction:.1%}) - "
            "sampler may be rejecting most proposals"
        )

    # Check sample size
    if result.n_samples < 100:
        warnings_list.append(
            f"Small sample size ({result.n_samples}) - results may be unreliable"
        )

    return {
        'has_warnings': len(warnings_list) > 0,
        'warnings': warnings_list,
        'zero_variance_params': zero_variance_params,
        'low_acceptance': low_acceptance,
    }


def add_convergence_annotation(
    fig: plt.Figure,
    warnings: Dict[str, any],
    position: str = 'bottom',
) -> None:
    """
    Add convergence warning annotation to a matplotlib figure.

    Parameters
    ----------
    fig : Figure
        Matplotlib figure to annotate.
    warnings : dict
        Warnings dict from check_convergence_warnings().
    position : str
        Where to place annotation: 'bottom' or 'top'.
    """
    if not warnings['has_warnings']:
        return

    # Build warning text
    warning_lines = []
    if warnings['zero_variance_params']:
        warning_lines.append(
            f"⚠ Zero variance in: {', '.join(warnings['zero_variance_params'])}"
        )
    if warnings['low_acceptance']:
        warning_lines.append("⚠ Low acceptance rate - check sampler configuration")
    if len(warnings['warnings']) > len(warning_lines):
        # Add any other warnings not already covered
        for w in warnings['warnings']:
            if 'zero variance' not in w.lower() and 'acceptance' not in w.lower():
                warning_lines.append(f"⚠ {w}")

    if not warning_lines:
        return

    warning_text = '\n'.join(warning_lines)

    # Add text annotation
    if position == 'bottom':
        y_pos = 0.02
        va = 'bottom'
    else:
        y_pos = 0.98
        va = 'top'

    fig.text(
        0.5, y_pos, warning_text,
        ha='center', va=va,
        fontsize=10,
        color='red',
        weight='bold',
        transform=fig.transFigure,
        bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='red', alpha=0.9),
    )


def plot_trace(
    result: 'SamplerResult',
    params: Optional[List[str]] = None,
    figsize: Tuple[float, float] = (12, 3),
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Create trace plots for sampled parameters.

    Trace plots show the evolution of each parameter over the sampling
    iterations. Useful for assessing convergence and mixing.

    Parameters
    ----------
    result : SamplerResult
        Sampling results.
    params : list of str, optional
        Parameters to plot. If None, plots all sampled parameters.
    figsize : tuple
        Figure size per parameter (width, height).
    output_path : Path, optional
        If provided, save figure to this path.

    Returns
    -------
    Figure
        Matplotlib figure with trace plots.

    Examples
    --------
    >>> from kl_pipe.sampling.diagnostics import plot_trace
    >>> fig = plot_trace(result, output_path='traces.png')
    """
    if params is None:
        params = result.param_names

    n_params = len(params)
    fig, axes = plt.subplots(n_params, 1, figsize=(figsize[0], figsize[1] * n_params))

    if n_params == 1:
        axes = [axes]

    for ax, param in zip(axes, params):
        chain = result.get_chain(param)
        ax.plot(chain, alpha=0.7, linewidth=0.5)
        ax.set_ylabel(param)
        mean_val = np.mean(chain)
        ax.axhline(mean_val, color='red', linestyle='--', alpha=0.7, label=f'Mean: {mean_val:.3f}')
        ax.legend(loc='upper right', fontsize=8)

    axes[-1].set_xlabel('Sample')
    fig.suptitle('Trace Plots', fontsize=14)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')

    return fig


def plot_corner(
    result: 'SamplerResult',
    params: Optional[List[str]] = None,
    true_values: Optional[Dict[str, float]] = None,
    map_values: Optional[Dict[str, float]] = None,
    output_path: Optional[Path] = None,
    annotate_warnings: bool = True,
    sampler_info: Optional[Dict[str, any]] = None,
    include_derived: bool = True,
    color: str = 'C0',
    **corner_kwargs,
) -> plt.Figure:
    """
    Create corner plot showing parameter correlations.

    Corner plots show 1D marginalized distributions on the diagonal
    and 2D correlations off-diagonal. Essential for understanding
    parameter degeneracies.

    Features:
    - Blue (C0) color scheme for consistent visualization
    - Centralized legend in upper-right white space (~0.65, 0.75)
    - MAP summary text block showing median ± std for each parameter
    - ±1σ shaded regions on diagonal histograms
    - True values (black solid) and MAP values (red dashed)

    Parameters
    ----------
    result : SamplerResult
        Sampling results.
    params : list of str, optional
        Parameters to include. If None, uses all sampled parameters.
    true_values : dict, optional
        Dictionary of true parameter values to mark (shown in black).
    map_values : dict, optional
        Dictionary of MAP parameter values to mark (shown in red).
        If not provided but true_values is, MAP is computed from samples.
    output_path : Path, optional
        If provided, save figure to this path.
    annotate_warnings : bool
        If True (default), add annotations for convergence warnings.
    sampler_info : dict, optional
        Dictionary with sampler metadata to display in title. Keys:
        - 'name': sampler name (e.g., 'emcee', 'blackjax')
        - 'runtime': runtime in seconds
        - 'settings': dict of key settings to display
    include_derived : bool
        If True (default), include derived parameters like vcirc*cosi
        when both vcirc and cosi are present.
    color : str
        Color for the corner plot. Default is 'C0' (matplotlib blue).
    **corner_kwargs
        Additional arguments passed to corner.corner().

    Returns
    -------
    Figure
        Matplotlib figure with corner plot.

    Examples
    --------
    >>> from kl_pipe.sampling.diagnostics import plot_corner
    >>> fig = plot_corner(result, true_values={'vcirc': 200, 'cosi': 0.6})
    """
    import corner
    import matplotlib.lines as mlines
    import matplotlib.patches as mpatches

    # Check for convergence warnings before plotting
    warnings = check_convergence_warnings(result)

    if params is None:
        params = list(result.param_names)
    else:
        params = list(params)

    # Extract samples for selected parameters
    samples = np.column_stack([result.get_chain(p) for p in params])

    # Add derived parameter vcirc*cosi if both are present
    derived_params = []
    if include_derived and 'vcirc' in params and 'cosi' in params:
        vcirc_chain = result.get_chain('vcirc')
        cosi_chain = result.get_chain('cosi')
        vcirc_cosi = vcirc_chain * cosi_chain
        samples = np.column_stack([samples, vcirc_cosi])
        params = params + ['vcirc*cosi']
        derived_params.append('vcirc*cosi')

    # Compute MAP if not provided (sample with highest log_prob)
    if map_values is None and result.log_prob is not None:
        max_idx = np.argmax(result.log_prob)
        map_values = {}
        for i, name in enumerate(result.param_names):
            map_values[name] = float(result.samples[max_idx, i])
        # Add derived MAP values
        if 'vcirc*cosi' in params:
            map_values['vcirc*cosi'] = map_values['vcirc'] * map_values['cosi']

    # Setup true values if provided
    truths = None
    if true_values is not None:
        truths = []
        for p in params:
            if p == 'vcirc*cosi' and 'vcirc' in true_values and 'cosi' in true_values:
                truths.append(true_values['vcirc'] * true_values['cosi'])
            else:
                truths.append(true_values.get(p))

    # Default corner kwargs - use C0 blue color, no quantile lines
    # Use larger fonts for publication readability
    # Note: We pass truths=None and draw truth lines manually to control z-order
    defaults = {
        'labels': params,
        'show_titles': True,
        'title_kwargs': {'fontsize': 14},  # Larger title font
        'label_kwargs': {'fontsize': 12},  # Larger axis labels
        'quantiles': None,  # Disable quantile lines - we'll add shaded regions
        'title_fmt': '.3f',
        'truth_color': 'black',
        'color': color,  # Use specified color (default C0 blue)
        'hist_kwargs': {'alpha': 0.7, 'color': color},
        'contour_kwargs': {'colors': color},
    }
    defaults.update(corner_kwargs)

    # Create corner plot WITHOUT truths - we'll draw them manually with explicit z-order
    fig = corner.corner(samples, truths=None, **defaults)

    # Get axes array
    axes = np.array(fig.axes).reshape((len(params), len(params)))

    # Compute quantiles for shaded regions
    q16 = np.percentile(samples, 16, axis=0)
    q84 = np.percentile(samples, 84, axis=0)
    medians = np.percentile(samples, 50, axis=0)
    stds = np.std(samples, axis=0)

    # Add shaded ±1σ regions on diagonal histograms
    for i in range(len(params)):
        ax = axes[i, i]
        ylim = ax.get_ylim()
        ax.axvspan(q16[i], q84[i], alpha=0.2, color='gray', zorder=0)
        ax.set_ylim(ylim)  # Restore ylim after axvspan

    # Build MAP list for plotting
    map_list = []
    if map_values is not None:
        for p in params:
            if p == 'vcirc*cosi' and 'vcirc*cosi' not in map_values:
                if 'vcirc' in map_values and 'cosi' in map_values:
                    map_list.append(map_values['vcirc'] * map_values['cosi'])
                else:
                    map_list.append(None)
            else:
                map_list.append(map_values.get(p))

        # Add MAP values as red dashed lines with zorder=15 (below truth lines)
        for i in range(len(params)):
            if map_list[i] is not None:
                # Diagonal: add vertical line only (no marker - markers go in 2D panels)
                ax_diag = axes[i, i]
                ax_diag.axvline(map_list[i], color='red', linestyle='--', linewidth=2, zorder=15)

            for j in range(i):
                if map_list[i] is not None and map_list[j] is not None:
                    # Off-diagonal: add crosshairs and square marker at intersection
                    axes[i, j].axvline(map_list[j], color='red', linestyle='--', linewidth=2, alpha=0.7, zorder=15)
                    axes[i, j].axhline(map_list[i], color='red', linestyle='--', linewidth=2, alpha=0.7, zorder=15)
                    # Add square marker at (map_x, map_y) intersection
                    axes[i, j].plot(map_list[j], map_list[i], 's', color='red',
                                   markersize=8, markeredgecolor='red', markerfacecolor='red',
                                   zorder=15)

    # Draw truth lines manually with zorder=20 (on top of MAP lines)
    if truths is not None:
        for i in range(len(params)):
            if truths[i] is not None:
                # Diagonal: add vertical line
                ax_diag = axes[i, i]
                ax_diag.axvline(truths[i], color='black', linestyle='-', linewidth=2, zorder=20)

            for j in range(i):
                if truths[i] is not None and truths[j] is not None:
                    # Off-diagonal: add crosshairs and square marker at intersection
                    axes[i, j].axvline(truths[j], color='black', linestyle='-', linewidth=2, zorder=20)
                    axes[i, j].axhline(truths[i], color='black', linestyle='-', linewidth=2, zorder=20)
                    # Add square marker at (truth_x, truth_y) intersection
                    axes[i, j].plot(truths[j], truths[i], 's', color='black',
                                   markersize=6, markeredgecolor='black', markerfacecolor='black',
                                   zorder=20)

    # Build title with sampler info
    title_lines = []
    if sampler_info:
        name = sampler_info.get('name', 'Unknown')
        runtime = sampler_info.get('runtime')
        settings = sampler_info.get('settings', {})

        # First line: sampler name and runtime
        if runtime is not None:
            title_lines.append(f"{name} (runtime: {runtime:.1f}s)")
        else:
            title_lines.append(name)

        # Second line: key settings
        if settings:
            settings_str = ', '.join(f"{k}={v}" for k, v in settings.items())
            title_lines.append(settings_str)

    if title_lines:
        fig.suptitle('\n'.join(title_lines), fontsize=20, y=1.02)

    # === Centralized legend at (0.65, 0.75) in white space ===
    legend_handles = []
    if truths is not None:
        legend_handles.append(mlines.Line2D([], [], color='black', linestyle='-',
                                            linewidth=2, marker='s', markersize=6,
                                            label='True'))
    if map_values is not None:
        legend_handles.append(mlines.Line2D([], [], color='red', linestyle='--',
                                            linewidth=2, marker='s', markersize=6,
                                            label='MAP'))
    # Add ±1σ region to legend
    legend_handles.append(mpatches.Patch(facecolor='gray', alpha=0.2,
                                         label='±1σ (16-84%)'))

    if legend_handles:
        fig.legend(handles=legend_handles, loc='upper right', fontsize=16,
                   bbox_to_anchor=(0.98, 0.98))

    # === MAP summary text block in white space ===
    # Position in the upper-right white space area of the corner plot
    # Format: SAMPLER_NAME (bold header)
    #         param: median ± std (true)  ● +X.Xσ
    # Plus joint Nσ as final row with aligned dot and sigma value
    
    # Import compute_joint_nsigma for joint statistic
    from kl_pipe.diagnostics import compute_joint_nsigma
    
    # Get sampler name and color for header
    sampler_display_name = sampler_info.get('name', 'Sampler') if sampler_info else 'Sampler'
    sampler_color = color  # Use the same color as the corner plot
    
    # Helper to get sigma color (uses absolute value for thresholds)
    def get_sigma_color(nsigma):
        abs_nsigma = abs(nsigma)
        if abs_nsigma < 2.0:
            return 'green'
        elif abs_nsigma < 3.0:
            return 'orange'
        else:
            return 'red'
    
    # Build recovered values dict for joint Nσ computation
    # IMPORTANT: Exclude derived parameters (e.g., vcirc*cosi) from joint Nσ
    # because they create singular covariance matrices
    recovered_for_joint = {}
    true_for_joint = {}
    sigma_info = []  # List of (nsigma, color) or None for each param
    
    # First pass: collect all data and compute column widths
    param_data = []  # List of (param_name, median, std, true_val_or_None, nsigma_or_None)
    for i, p in enumerate(params):
        # Only include non-derived params in joint calculation
        if p not in derived_params:
            recovered_for_joint[p] = medians[i]
        true_val = None
        nsigma_off = None
        
        if truths is not None and truths[i] is not None:
            true_val = truths[i]
            if p not in derived_params:
                true_for_joint[p] = true_val
            if stds[i] > 1e-10:
                nsigma_off = (medians[i] - true_val) / stds[i]
        
        param_data.append((p, medians[i], stds[i], true_val, nsigma_off))
        sigma_info.append((nsigma_off, get_sigma_color(nsigma_off)) if nsigma_off is not None else None)
    
    # Compute joint Nσ using only non-derived parameters
    joint_nsigma = np.nan
    if truths is not None and len(true_for_joint) > 0:
        # Get indices of non-derived params for extracting the right sample columns
        non_derived_indices = [i for i, p in enumerate(params) if p not in derived_params]
        samples_for_joint = samples[:, non_derived_indices]
        non_derived_params = [p for p in params if p not in derived_params]
        
        joint_stats = compute_joint_nsigma(
            recovered=recovered_for_joint,
            true_values=true_for_joint,
            samples=samples_for_joint,
            param_names=non_derived_params,
        )
        joint_nsigma = joint_stats.get('nsigma', np.nan)
    
    # Compute column widths for alignment
    max_param_len = max(len(p) for p in params)
    # Build formatted lines for the unified box
    # Format: "param_name: median ± std (true)  ● +X.Xσ"
    # We'll use ANSI-free text but color the sigma portion separately via fig.text overlay
    
    # For monospace alignment, we need fixed-width columns
    # Col1: param name (left-aligned, max_param_len chars)
    # Col2: "median ± std" (right-aligned)
    # Col3: "(true)" (right-aligned) 
    # Col4: "● +X.Xσ" (right-aligned) - will be overlaid with color
    
    # Build the base text lines (without colored sigma - that's overlaid)
    base_lines = []
    for p, med, std, true_val, nsigma_off in param_data:
        # Format: "param: median ± std (true)"
        base_part = f"{p:<{max_param_len}}: {med:>8.3f} ± {std:<6.3f}"
        if true_val is not None:
            base_part += f" ({true_val:>6.3g})"
        else:
            base_part += f" {'':>8}"  # Placeholder for alignment
        base_lines.append(base_part)
    
    # Add joint line placeholder (no param stats, just label)
    if not np.isnan(joint_nsigma):
        joint_label = f"{'Joint':<{max_param_len}}:"
        # Pad to align with param lines
        joint_label += " " * (8 + 3 + 6 + 1 + 8)  # Spaces for: median + " ± " + std + " " + "(true)"
        base_lines.append(joint_label)
    
    # Box position
    box_x, box_y = 0.55, 0.75
    header_offset = 0.025  # Space below header for stats
    
    # Render the sampler name header (bold, in sampler color)
    fig.text(box_x, box_y, sampler_display_name.upper(),
             transform=fig.transFigure,
             fontsize=16,
             fontweight='bold',
             color=sampler_color,
             verticalalignment='top',
             horizontalalignment='left',
             fontfamily='sans-serif')
    
    # Render the stats text below the header
    stats_y = box_y - header_offset
    summary_text = '\n'.join(base_lines)
    text_obj = fig.text(box_x, stats_y, summary_text,
                        transform=fig.transFigure,
                        fontsize=18,
                        verticalalignment='top',
                        horizontalalignment='left',
                        fontfamily='monospace')
    
    # Force render to get text positions for overlay
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bbox = text_obj.get_window_extent(renderer=renderer)
    
    # Compute line height in figure coordinates
    n_total_lines = len(base_lines)
    if n_total_lines > 1:
        line_height_pixels = bbox.height / n_total_lines
        line_height_fig = line_height_pixels / fig.get_figheight() / fig.dpi
    else:
        line_height_fig = 0.025
    
    # Get the right edge of the text box for aligned sigma overlay
    bbox_fig = bbox.transformed(fig.transFigure.inverted())
    right_edge = bbox_fig.x1 + 0.02  # Small offset past box edge
    sigma_width_est = 0.08  # Approximate width of sigma column
    
    # Draw the combined box around header + stats + sigma column
    import matplotlib.patches as mpatches_box
    box_left = box_x - 0.01
    box_top = box_y + 0.01
    box_bottom = stats_y - n_total_lines * line_height_fig - 0.01
    box_right = right_edge + sigma_width_est
    box_width = box_right - box_left
    box_height = box_top - box_bottom
    
    rect = mpatches_box.FancyBboxPatch(
        (box_left, box_bottom), box_width, box_height,
        boxstyle='round,pad=0.01,rounding_size=0.02',
        transform=fig.transFigure,
        facecolor='white',
        edgecolor=sampler_color,
        linewidth=2,
        alpha=0.9,
        zorder=0,  # Behind text
    )
    fig.patches.append(rect)
    
    # Overlay colored sigma annotations aligned to the right of the box
    # All dots and sigma decimals will be vertically aligned
    for i, info in enumerate(sigma_info):
        if info is not None:
            nsigma_off, sigma_col = info
            y_pos = stats_y - (i + 0.5) * line_height_fig
            sign_str = "+" if nsigma_off >= 0 else ""
            # Format: "● +X.Xσ" with fixed width for alignment
            sigma_text = f"● {sign_str}{nsigma_off:>4.1f}σ"
            fig.text(right_edge, y_pos, sigma_text,
                     transform=fig.transFigure,
                     fontsize=16,
                     color=sigma_col,
                     verticalalignment='center',
                     horizontalalignment='left',
                     fontweight='bold',
                     fontfamily='monospace')
    
    # Add joint Nσ on the last line (aligned with param sigmas)
    if not np.isnan(joint_nsigma):
        joint_color = get_sigma_color(joint_nsigma)
        y_pos_joint = stats_y - (n_total_lines - 0.5) * line_height_fig
        sigma_text = f"● {joint_nsigma:>5.2f}σ"
        fig.text(right_edge, y_pos_joint, sigma_text,
                 transform=fig.transFigure,
                 fontsize=16,
                 color=joint_color,
                 verticalalignment='center',
                 horizontalalignment='left',
                 fontweight='bold',
                 fontfamily='monospace')

    # Add convergence warning annotations if requested
    if annotate_warnings:
        add_convergence_annotation(fig, warnings, position='bottom')

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')

    return fig


def plot_corner_comparison(
    results: Dict[str, 'SamplerResult'],
    params: Optional[List[str]] = None,
    true_values: Optional[Dict[str, float]] = None,
    output_path: Optional[Path] = None,
    colors: Optional[Dict[str, str]] = None,
    labels: Optional[Dict[str, str]] = None,
    timings: Optional[Dict[str, float]] = None,
    include_derived: bool = True,
    baseline_sampler: str = 'numpyro',
    sampler_configs: Optional[Dict[str, Dict[str, any]]] = None,
    **corner_kwargs,
) -> plt.Figure:
    """
    Create overlaid corner plot comparing multiple samplers.

    Overlays posterior distributions from different samplers on the same
    corner plot, using different colors to distinguish them. Useful for
    comparing convergence and constraints across sampler backends.

    Features:
    - Baseline sampler (default: numpyro) shown with filled contours
    - Other samplers shown with unfilled contours for cleaner comparison
    - Sampler configuration summary in upper-right white space
    - True values marked with black lines

    Parameters
    ----------
    results : dict
        Mapping of sampler_name -> SamplerResult.
    params : list of str, optional
        Parameters to include. If None, uses parameters from first result.
    true_values : dict, optional
        True parameter values to mark with vertical/horizontal lines (black).
    output_path : Path, optional
        If provided, save figure to this path.
    colors : dict, optional
        Mapping of sampler_name -> color. Defaults to matplotlib C0, C1, C2.
    labels : dict, optional
        Custom display labels for samplers.
    timings : dict, optional
        Mapping of sampler_name -> runtime in seconds. Displayed in title.
    include_derived : bool
        If True (default), include derived parameters like vcirc*cosi
        when both vcirc and cosi are present.
    baseline_sampler : str
        Name of sampler to use as baseline (filled contours). Default 'numpyro'.
        If not present in results, first sampler is used as baseline.
    sampler_configs : dict, optional
        Mapping of sampler_name -> config dict with key settings to display.
        Example: {'numpyro': {'n_chains': 2, 'n_warmup': 200}}
    **corner_kwargs
        Additional arguments passed to corner.corner().

    Returns
    -------
    Figure
        Matplotlib figure with overlaid corner plots.

    Examples
    --------
    >>> from kl_pipe.sampling.diagnostics import plot_corner_comparison
    >>> results = {'emcee': result1, 'nautilus': result2, 'numpyro': result3}
    >>> fig = plot_corner_comparison(results, true_values={'vcirc': 200})
    """
    import corner
    import matplotlib.patches as mpatches
    import matplotlib.lines as mlines

    if not results:
        raise ValueError("results dict cannot be empty")

    # Determine baseline sampler first (needed for color assignment)
    sampler_names = list(results.keys())
    if baseline_sampler in sampler_names:
        baseline = baseline_sampler
    else:
        baseline = sampler_names[0]

    # Assign colors: baseline gets C0 (blue), others get C1, C2, etc. in order
    if colors is None:
        colors = {baseline: 'C0'}
        non_baseline_names = [n for n in sampler_names if n != baseline]
        for i, name in enumerate(non_baseline_names):
            colors[name] = f'C{i + 1}'

    # Get parameter list from first result if not specified
    first_result = next(iter(results.values()))
    if params is None:
        params = list(first_result.param_names)
    else:
        params = list(params)

    # Check if we should add derived parameter
    add_vcirc_cosi = include_derived and 'vcirc' in params and 'cosi' in params

    # Setup true values if provided
    truths = None
    if true_values is not None:
        truths = [true_values.get(p) for p in params]
        if add_vcirc_cosi:
            if 'vcirc' in true_values and 'cosi' in true_values:
                truths.append(true_values['vcirc'] * true_values['cosi'])
            else:
                truths.append(None)

    # Add derived param to labels
    plot_params = params.copy()
    if add_vcirc_cosi:
        plot_params.append('vcirc*cosi')

    # Pre-check all samplers for zero-variance issues before plotting
    for name, result in results.items():
        samples = np.column_stack([result.get_chain(p) for p in params])
        if add_vcirc_cosi:
            vcirc_chain = result.get_chain('vcirc')
            cosi_chain = result.get_chain('cosi')
            vcirc_cosi = vcirc_chain * cosi_chain
            samples = np.column_stack([samples, vcirc_cosi])

        variances = np.var(samples, axis=0)
        zero_var_cols = np.where(variances < 1e-10)[0]

        if len(zero_var_cols) > 0:
            zero_var_params = [plot_params[i] for i in zero_var_cols]
            raise ValueError(
                f"Sampler '{name}' has zero variance in parameters: {zero_var_params}. "
                f"This indicates the sampler failed to explore the posterior. "
                f"For BlackJAX, this typically means gradient issues during warmup - "
                f"check that priors and likelihood produce finite gradients. "
                f"Run the BlackJAX diagnostic tests for more details: "
                f"pytest tests/test_blackjax.py -v"
            )

    fig = None
    computed_range = None  # Will be computed from first valid sampler

    # Sort samplers: non-baseline first (behind), baseline last (on top)
    # This ensures baseline is most visible
    non_baseline = [n for n in sampler_names if n != baseline]
    sorted_names_for_plotting = non_baseline + [baseline]

    # First pass: plot all samplers with filled contours
    # Order: non-baseline first (behind), baseline last (on top)
    for i, name in enumerate(sorted_names_for_plotting):
        result = results[name]
        is_baseline = (name == baseline)

        # Extract samples for selected parameters
        samples = np.column_stack([result.get_chain(p) for p in params])

        # Add derived parameter
        if add_vcirc_cosi:
            vcirc_chain = result.get_chain('vcirc')
            cosi_chain = result.get_chain('cosi')
            vcirc_cosi = vcirc_chain * cosi_chain
            samples = np.column_stack([samples, vcirc_cosi])

        # Compute range from first valid sampler to use for all
        if computed_range is None:
            computed_range = [[samples[:, j].min(), samples[:, j].max()]
                             for j in range(samples.shape[1])]
            # Add small buffer for any columns with very small range
            for j in range(len(computed_range)):
                if computed_range[j][0] == computed_range[j][1]:
                    mid = computed_range[j][0]
                    computed_range[j] = [mid - 0.1 * abs(mid) - 1e-6,
                                        mid + 0.1 * abs(mid) + 1e-6]

        # Get color for this sampler
        color = colors.get(name, f'C{i}')

        # We'll draw truth lines manually at the end with explicit z-order
        # So don't pass truths to corner

        # All samplers get filled contours, but baseline is plotted last to be on top
        # Baseline gets higher alpha for visibility
        defaults = {
            'labels': plot_params,
            'show_titles': False,  # Don't show titles to avoid clutter
            'quantiles': None,
            'levels': (0.39, 0.68, 0.95),
            'smooth': 1.0,
            'plot_contours': True,
            'plot_density': False,
            'fill_contours': True,  # All samplers filled
            'truth_color': 'black',
            'contour_kwargs': {'linewidths': 2.5 if is_baseline else 1.5,
                              'alpha': 1.0 if is_baseline else 0.6},
        }
        defaults.update(corner_kwargs)

        # Create or update corner plot (truths=None, we draw them manually later)
        fig = corner.corner(
            samples,
            fig=fig,
            truths=None,
            color=color,
            range=computed_range,
            hist_kwargs={
                'alpha': 0.8 if is_baseline else 0.4,
                'density': True,
                'linewidth': 2.0 if is_baseline else 1.0,
            },
            **defaults,
        )

    # Second pass: compute baseline contours and store paths for clipping
    # Then draw dashed non-baseline contours clipped to baseline regions
    axes = np.array(fig.axes).reshape((len(plot_params), len(plot_params)))
    
    # First, compute and store baseline contour paths for each 2D panel
    from scipy.stats import gaussian_kde
    from matplotlib.path import Path as MplPath
    from matplotlib.patches import PathPatch
    
    baseline_result = results[baseline]
    baseline_samples = np.column_stack([baseline_result.get_chain(p) for p in params])
    if add_vcirc_cosi:
        vcirc_chain = baseline_result.get_chain('vcirc')
        cosi_chain = baseline_result.get_chain('cosi')
        vcirc_cosi = vcirc_chain * cosi_chain
        baseline_samples = np.column_stack([baseline_samples, vcirc_cosi])
    
    # Store baseline 95% contour paths for clipping
    baseline_clip_paths = {}  # (i, j) -> MplPath or None
    
    for i in range(len(plot_params)):
        for j in range(i):
            ax = axes[i, j]
            x = baseline_samples[:, j]
            y = baseline_samples[:, i]
            
            try:
                # Subsample if needed
                if len(x) > 10000:
                    idx = np.random.choice(len(x), 10000, replace=False)
                    x_sub, y_sub = x[idx], y[idx]
                else:
                    x_sub, y_sub = x, y
                
                kernel = gaussian_kde(np.vstack([x_sub, y_sub]))
                
                xlim = ax.get_xlim()
                ylim = ax.get_ylim()
                xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 50),
                                    np.linspace(ylim[0], ylim[1], 50))
                positions = np.vstack([xx.ravel(), yy.ravel()])
                zz = kernel(positions).reshape(xx.shape)
                
                # Compute 95% contour level
                z_flat = zz.flatten()
                z_sorted = np.sort(z_flat)[::-1]
                z_cumsum = np.cumsum(z_sorted)
                z_cumsum /= z_cumsum[-1]
                
                idx_95 = np.searchsorted(z_cumsum, 0.95)
                if idx_95 < len(z_sorted):
                    level_95 = z_sorted[idx_95]
                    # Get contour path
                    cs = ax.contour(xx, yy, zz, levels=[level_95], alpha=0)
                    if cs.allsegs and cs.allsegs[0]:
                        # Get the largest contour segment (in case of multiple)
                        largest_seg = max(cs.allsegs[0], key=len)
                        baseline_clip_paths[(i, j)] = MplPath(largest_seg)
                    # Remove invisible contour
                    for coll in cs.collections:
                        coll.remove()
            except Exception:
                pass
    
    # Now draw dashed non-baseline contours, clipped to baseline regions
    for name in non_baseline:
        result = results[name]
        samples = np.column_stack([result.get_chain(p) for p in params])
        if add_vcirc_cosi:
            vcirc_chain = result.get_chain('vcirc')
            cosi_chain = result.get_chain('cosi')
            vcirc_cosi = vcirc_chain * cosi_chain
            samples = np.column_stack([samples, vcirc_cosi])
        
        color = colors.get(name, 'C0')
        
        for i in range(len(plot_params)):
            for j in range(i):
                ax = axes[i, j]
                x = samples[:, j]
                y = samples[:, i]
                
                # Only draw if we have a baseline clip path
                clip_path = baseline_clip_paths.get((i, j))
                if clip_path is None:
                    continue
                
                try:
                    # Subsample if needed
                    if len(x) > 10000:
                        idx = np.random.choice(len(x), 10000, replace=False)
                        x_sub, y_sub = x[idx], y[idx]
                    else:
                        x_sub, y_sub = x, y
                    
                    kernel = gaussian_kde(np.vstack([x_sub, y_sub]))
                    
                    xlim = ax.get_xlim()
                    ylim = ax.get_ylim()
                    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 50),
                                        np.linspace(ylim[0], ylim[1], 50))
                    positions = np.vstack([xx.ravel(), yy.ravel()])
                    zz = kernel(positions).reshape(xx.shape)
                    
                    # Compute contour levels (68%, 95%)
                    z_flat = zz.flatten()
                    z_sorted = np.sort(z_flat)[::-1]
                    z_cumsum = np.cumsum(z_sorted)
                    z_cumsum /= z_cumsum[-1]
                    
                    levels = []
                    for frac in [0.68, 0.95]:
                        idx = np.searchsorted(z_cumsum, frac)
                        if idx < len(z_sorted):
                            levels.append(z_sorted[idx])
                    
                    if levels:
                        cs = ax.contour(xx, yy, zz, levels=sorted(levels), colors=color,
                                       linestyles='--', linewidths=1.5, alpha=0.9)
                        # Apply clipping to baseline region
                        clip_patch = PathPatch(clip_path, transform=ax.transData,
                                              facecolor='none', edgecolor='none')
                        ax.add_patch(clip_patch)
                        for coll in cs.collections:
                            coll.set_clip_path(clip_patch)
                except Exception:
                    pass

    # Draw truth lines manually with zorder=20 (on top of everything)
    if truths is not None:
        for i in range(len(plot_params)):
            if truths[i] is not None:
                # Diagonal: add vertical line
                ax_diag = axes[i, i]
                ax_diag.axvline(truths[i], color='black', linestyle='-', linewidth=2, zorder=20)

            for j in range(i):
                if truths[i] is not None and truths[j] is not None:
                    # Off-diagonal: add crosshairs and square marker at intersection
                    axes[i, j].axvline(truths[j], color='black', linestyle='-', linewidth=2, zorder=20)
                    axes[i, j].axhline(truths[i], color='black', linestyle='-', linewidth=2, zorder=20)
                    # Add square marker at (truth_x, truth_y) intersection
                    axes[i, j].plot(truths[j], truths[i], 's', color='black',
                                   markersize=6, markeredgecolor='black', markerfacecolor='black',
                                   zorder=20)

    # Build title with sampler info, timings, and configs (baseline first in display order)
    sorted_names_display = [baseline] + non_baseline
    title_lines = ["Sampler Comparison"]
    
    # Add per-sampler subtitle lines with timing and config info
    for name in sorted_names_display:
        subtitle_parts = [name]
        if timings and name in timings:
            subtitle_parts.append(f"{timings[name]:.1f}s")
        if sampler_configs and name in sampler_configs:
            cfg = sampler_configs[name]
            cfg_str = ', '.join(f"{k}={v}" for k, v in cfg.items())
            subtitle_parts.append(cfg_str)
        title_lines.append(' | '.join(subtitle_parts))

    fig.suptitle('\n'.join(title_lines), fontsize=18, y=1.02)

    # === Per-sampler recovery info boxes ===
    # Import compute_joint_nsigma for joint statistic
    from kl_pipe.diagnostics import compute_joint_nsigma
    
    # Helper to get sigma color
    def get_sigma_color(nsigma):
        abs_nsigma = abs(nsigma)
        if abs_nsigma < 2.0:
            return 'green'
        elif abs_nsigma < 3.0:
            return 'orange'
        else:
            return 'red'
    
    # Build recovery info for each sampler
    n_samplers = len(sorted_names_display)
    
    # Position boxes in the upper-right white space, mirroring the corner plot layout
    # For 2 samplers: stack vertically in upper-right quadrant
    # For 3 samplers: one top-right, one middle-right, one bottom area
    # Box positions are designed to avoid overlap with corner plot panels
    n_params_plot = len(plot_params)
    
    # Calculate approximate boundary of corner plot panels
    # Corner plot spans roughly (0.0, 0.0) to (0.5, 0.5) in figure coords for the panels
    # Upper-right white space is roughly x > 0.5 and y > 0.5
    
    if n_samplers == 1:
        # Single box in upper-right
        box_positions = [(0.55, 0.75)]
    elif n_samplers == 2:
        # Two boxes: diagonal arrangement in the upper-right triangle
        # Position along the diagonal to avoid corner plot panels
        # Box 1: upper-right corner, Box 2: lower in the triangle
        box_positions = [(0.68, 0.85), (0.68, 0.42)]
    else:
        # Three or more: diagonal arrangement in complementary triangle
        # Position boxes along the diagonal from upper-right to lower-right
        # This mirrors the corner plot structure and avoids overlap
        # Box 1: top-right corner (above diagonal)
        # Box 2: middle-right (along diagonal)
        # Box 3: lower area (below diagonal, still in white space)
        box_positions = [(0.72, 0.88), (0.55, 0.58), (0.72, 0.28)]
        # Extend if more samplers (unlikely but handle gracefully)
        for i in range(3, n_samplers):
            box_positions.append((0.55, 0.28 - (i - 2) * 0.15))
    
    # Compute max param name length for alignment
    max_param_len = max(len(p) for p in plot_params)
    
    for box_idx, name in enumerate(sorted_names_display):
        result = results[name]
        sampler_color = colors.get(name, 'C0')
        
        # Extract samples for this sampler
        samples = np.column_stack([result.get_chain(p) for p in params])
        if add_vcirc_cosi:
            vcirc_chain = result.get_chain('vcirc')
            cosi_chain = result.get_chain('cosi')
            vcirc_cosi = vcirc_chain * cosi_chain
            samples = np.column_stack([samples, vcirc_cosi])
        
        medians = np.percentile(samples, 50, axis=0)
        stds = np.std(samples, axis=0)
        
        # Build recovered values dict for joint Nσ computation
        # IMPORTANT: Exclude derived parameters (e.g., vcirc*cosi) from joint Nσ
        # because they create singular covariance matrices
        recovered_for_joint = {}
        true_for_joint = {}
        sigma_info = []
        
        # Track derived params for this sampler
        derived_params_here = ['vcirc*cosi'] if add_vcirc_cosi else []
        
        for i, p in enumerate(plot_params):
            # Only include non-derived params in joint calculation
            if p not in derived_params_here:
                recovered_for_joint[p] = medians[i]
            true_val = None
            nsigma_off = None
            
            if truths is not None and truths[i] is not None:
                true_val = truths[i]
                if p not in derived_params_here:
                    true_for_joint[p] = true_val
                if stds[i] > 1e-10:
                    nsigma_off = (medians[i] - true_val) / stds[i]
            
            sigma_info.append((nsigma_off, get_sigma_color(nsigma_off)) if nsigma_off is not None else None)
        
        # Compute joint Nσ using only non-derived parameters
        joint_nsigma = np.nan
        if truths is not None and len(true_for_joint) > 0:
            # Get indices of non-derived params for extracting the right sample columns
            non_derived_indices = [i for i, p in enumerate(plot_params) if p not in derived_params_here]
            samples_for_joint = samples[:, non_derived_indices]
            non_derived_params = [p for p in plot_params if p not in derived_params_here]
            
            joint_stats = compute_joint_nsigma(
                recovered=recovered_for_joint,
                true_values=true_for_joint,
                samples=samples_for_joint,
                param_names=non_derived_params,
            )
            joint_nsigma = joint_stats.get('nsigma', np.nan)
        
        # Build formatted lines for the box - include sigmas in the same box
        # Format: "param: median ± std (true)  ● +X.Xσ"
        # We'll use two-column layout: stats | colored sigma
        
        # Get box position from pre-computed positions
        x_pos, y_pos = box_positions[box_idx] if box_idx < len(box_positions) else box_positions[-1]
        
        # Build the complete box content with aligned columns
        # We need to render the sampler name as a header, then parameter rows
        
        # First, render the sampler name header (bold, in sampler color)
        header_y = y_pos
        fig.text(x_pos, header_y, name.upper(),
                 transform=fig.transFigure,
                 fontsize=16,
                 fontweight='bold',
                 color=sampler_color,
                 verticalalignment='top',
                 horizontalalignment='left',
                 fontfamily='sans-serif')
        
        # Build stats lines (without sigma - we'll add that as colored overlay)
        stats_lines = []
        for i, p in enumerate(plot_params):
            true_val = truths[i] if truths is not None else None
            line = f"{p:<{max_param_len}}: {medians[i]:>8.3f} ± {stds[i]:<6.3f}"
            if true_val is not None:
                line += f" ({true_val:>6.3g})"
            else:
                line += f" {'':>8}"
            stats_lines.append(line)
        
        # Add joint line
        if not np.isnan(joint_nsigma):
            joint_line = f"{'Joint':<{max_param_len}}:"
            joint_line += " " * (8 + 3 + 6 + 1 + 8)  # padding to align
            stats_lines.append(joint_line)
        
        # Render the stats text block (offset below header)
        header_offset = 0.025  # Space below header
        stats_y = header_y - header_offset
        stats_text = '\n'.join(stats_lines)
        
        # Create text object to measure for box sizing
        text_obj = fig.text(x_pos, stats_y, stats_text,
                            transform=fig.transFigure,
                            fontsize=14,
                            verticalalignment='top',
                            horizontalalignment='left',
                            fontfamily='monospace')
        
        # Force render to get positions
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        text_bbox = text_obj.get_window_extent(renderer=renderer)
        text_bbox_fig = text_bbox.transformed(fig.transFigure.inverted())
        
        # Compute line height
        n_stats_lines = len(stats_lines)
        if n_stats_lines > 1:
            line_height_fig = (text_bbox_fig.y1 - text_bbox_fig.y0) / n_stats_lines
        else:
            line_height_fig = 0.018
        
        # Build sigma column content and render overlays
        sigma_x = text_bbox_fig.x1 + 0.01  # Just past the stats text
        
        for i, info in enumerate(sigma_info):
            if info is not None:
                nsigma_off, sigma_col = info
                y_sigma = stats_y - (i + 0.5) * line_height_fig
                sign_str = "+" if nsigma_off >= 0 else ""
                sigma_text = f"● {sign_str}{nsigma_off:>4.1f}σ"
                fig.text(sigma_x, y_sigma, sigma_text,
                         transform=fig.transFigure,
                         fontsize=14,
                         color=sigma_col,
                         verticalalignment='center',
                         horizontalalignment='left',
                         fontweight='bold',
                         fontfamily='monospace')
        
        # Add joint Nσ on the last line (bold)
        if not np.isnan(joint_nsigma):
            joint_color = get_sigma_color(joint_nsigma)
            y_joint = stats_y - (n_stats_lines - 0.5) * line_height_fig
            sigma_text = f"● {joint_nsigma:>5.2f}σ"
            fig.text(sigma_x, y_joint, sigma_text,
                     transform=fig.transFigure,
                     fontsize=14,
                     color=joint_color,
                     verticalalignment='center',
                     horizontalalignment='left',
                     fontweight='bold',
                     fontfamily='monospace')
        
        # Now get the full extent including sigma column and draw box
        fig.canvas.draw()
        
        # Find rightmost sigma text extent
        # We'll estimate it based on sigma_x position + typical sigma text width
        sigma_width_est = 0.08  # Approximate width of "● +X.Xσ" in figure coords
        box_right = sigma_x + sigma_width_est
        
        # Box coordinates: from header to bottom of stats, spanning both columns
        box_left = x_pos - 0.01
        box_top = header_y + 0.01
        box_bottom = stats_y - n_stats_lines * line_height_fig - 0.01
        
        # Draw rounded rectangle box
        import matplotlib.patches as mpatches_box
        box_width = box_right - box_left
        box_height = box_top - box_bottom
        
        rect = mpatches_box.FancyBboxPatch(
            (box_left, box_bottom), box_width, box_height,
            boxstyle='round,pad=0.01,rounding_size=0.02',
            transform=fig.transFigure,
            facecolor='white',
            edgecolor=sampler_color,
            linewidth=2,
            alpha=0.9,
            zorder=0,  # Behind text
        )
        fig.patches.append(rect)

    # Add legend with samplers and true values indicator
    handles = []

    # True values indicator
    if truths is not None:
        handles.append(mlines.Line2D([], [], color='black', linestyle='-',
                                     linewidth=2, marker='s', markersize=6,
                                     label='True'))

    # Sampler colors - baseline first, with filled patch
    # Non-baseline get both solid (filled) and dashed (outline) representation
    for name in sorted_names_display:
        color = colors.get(name, 'C0')
        display_name = labels.get(name, name) if labels else name
        if name == baseline:
            display_name += ' (baseline)'
            handles.append(mpatches.Patch(color=color, label=display_name, alpha=0.8))
        else:
            # Show both solid and dashed for non-baseline
            handles.append(mlines.Line2D([], [], color=color, linestyle='-',
                                        linewidth=2, marker='s', markersize=8,
                                        markerfacecolor=color, markeredgecolor=color,
                                        alpha=0.6, label=display_name))

    fig.legend(handles=handles, loc='upper right', fontsize=16,
               bbox_to_anchor=(0.98, 0.98))

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')

    return fig


def plot_recovery(
    result: 'SamplerResult',
    true_values: Dict[str, float],
    output_path: Optional[Path] = None,
    include_derived: bool = True,
    sampler_name: Optional[str] = None,
) -> Tuple[plt.Figure, Dict[str, any]]:
    """
    Create two-panel parameter recovery plot with joint Nσ validation.

    Top panel: Log-scale absolute values (true vs recovered)
    Bottom panel: Fractional error (%) with uncertainty bands

    The joint Nσ statistic measures how far the recovered parameters
    deviate from true values, accounting for correlations via the
    posterior covariance matrix. This is displayed in the title with
    color-coding:
    - Green: < 2σ (expected ~95% of the time)
    - Yellow/Orange: 2-3σ (rare but not alarming)
    - Red: > 3σ (potential problem)

    Parameters
    ----------
    result : SamplerResult
        Sampling results.
    true_values : dict
        True parameter values.
    output_path : Path, optional
        If provided, save figure to this path.
    include_derived : bool
        If True (default), include vcirc*cosi derived parameter.
    sampler_name : str, optional
        Sampler name for title annotation.

    Returns
    -------
    Figure
        Matplotlib figure with recovery plot.
    dict
        Recovery statistics including:
        - 'joint_nsigma': Joint Nσ deviation
        - 'joint_chi2': Chi-squared statistic
        - 'joint_pvalue': P-value
        - 'per_param': Dict of per-parameter statistics
        - 'title_color': Color based on Nσ thresholds

    Notes
    -----
    This function never raises exceptions. It computes and returns statistics
    that tests can use for pass/fail decisions.

    Examples
    --------
    >>> from kl_pipe.sampling.diagnostics import plot_recovery
    >>> fig, stats = plot_recovery(result, true_pars)
    >>> if stats['joint_nsigma'] > 3:
    ...     print("Warning: >3σ deviation from truth")
    """
    from scipy import stats as scipy_stats

    summary = result.get_summary()

    # Get common parameters
    params = [p for p in result.param_names if p in true_values]

    # Build recovered values dict (using median)
    recovered_values = {}
    uncertainties = {}
    for name in params:
        median = summary[name]['quantiles'][0.5]
        q16 = summary[name]['quantiles'][0.16]
        q84 = summary[name]['quantiles'][0.84]
        recovered_values[name] = median
        uncertainties[name] = (median - q16, q84 - median)

    # Add derived parameter vcirc*cosi if both are present
    derived_params = {}
    if include_derived and 'vcirc' in params and 'cosi' in params:
        vcirc_chain = result.get_chain('vcirc')
        cosi_chain = result.get_chain('cosi')
        vcirc_cosi_chain = vcirc_chain * cosi_chain

        true_vcirc_cosi = true_values['vcirc'] * true_values['cosi']
        rec_vcirc_cosi = np.median(vcirc_cosi_chain)
        q16 = np.percentile(vcirc_cosi_chain, 16)
        q84 = np.percentile(vcirc_cosi_chain, 84)

        derived_params['vcirc*cosi'] = (true_vcirc_cosi, rec_vcirc_cosi)
        uncertainties['vcirc*cosi'] = (rec_vcirc_cosi - q16, q84 - rec_vcirc_cosi)

    # Compute joint Nσ using covariance from samples
    samples = result.samples
    param_names = list(result.param_names)

    # Helper to compute joint Nσ
    def compute_joint_nsigma_internal(recovered, true_vals, samples, param_names):
        common_params = [p for p in true_vals.keys() if p in recovered]
        n_params = len(common_params)

        if n_params == 0:
            return {'nsigma': np.nan, 'chi2': np.nan, 'pvalue': np.nan}

        delta_theta = np.array([recovered[p] - true_vals[p] for p in common_params])

        # Extract columns for common parameters
        param_indices = [param_names.index(p) for p in common_params if p in param_names]

        if len(param_indices) != n_params:
            # Some params not in samples, use diagonal approximation
            stds = np.array([np.std(result.get_chain(p)) for p in common_params])
            chi2 = np.sum((delta_theta / stds) ** 2)
        else:
            samples_subset = samples[:, param_indices]
            covariance = np.cov(samples_subset.T)
            if n_params == 1:
                covariance = covariance.reshape(1, 1)

            try:
                cov_inv = np.linalg.pinv(covariance)
            except np.linalg.LinAlgError:
                cov_regularized = covariance + 1e-10 * np.eye(n_params)
                cov_inv = np.linalg.pinv(cov_regularized)

            chi2 = float(delta_theta @ cov_inv @ delta_theta)

        pvalue = scipy_stats.chi2.sf(chi2, df=n_params)
        if pvalue > 0 and pvalue < 1:
            nsigma = scipy_stats.norm.isf(pvalue / 2)
        elif pvalue <= 0:
            nsigma = np.inf
        else:
            nsigma = 0.0

        return {'nsigma': nsigma, 'chi2': chi2, 'pvalue': pvalue}

    joint_stats = compute_joint_nsigma_internal(
        recovered_values, true_values, samples, param_names
    )

    # Determine title color based on Nσ
    nsigma = joint_stats['nsigma']
    if np.isnan(nsigma):
        title_color = 'gray'
    elif nsigma < 2.0:
        title_color = 'green'
    elif nsigma < 3.0:
        title_color = 'orange'
    else:
        title_color = 'red'

    # Compute per-parameter statistics
    per_param_stats = {}
    all_params = list(params)
    all_true = {p: true_values[p] for p in params}
    all_recovered = {p: recovered_values[p] for p in params}

    if derived_params:
        for name, (true_val, rec_val) in derived_params.items():
            all_params.append(name)
            all_true[name] = true_val
            all_recovered[name] = rec_val

    for p in all_params:
        true_val = all_true[p]
        rec_val = all_recovered[p]
        abs_error = abs(rec_val - true_val)

        if abs(true_val) > 1e-10:
            frac_error = (rec_val - true_val) / true_val
        else:
            frac_error = rec_val - true_val

        per_param_stats[p] = {
            'true': true_val,
            'recovered': rec_val,
            'abs_error': abs_error,
            'frac_error': frac_error,
        }

    # Create figure
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

    # Compute error bars for log-scale panel (absolute uncertainties)
    yerr_abs_lower = []
    yerr_abs_upper = []
    for p in all_params:
        if p in uncertainties:
            low, high = uncertainties[p]
            yerr_abs_lower.append(abs(low))
            yerr_abs_upper.append(abs(high))
        else:
            yerr_abs_lower.append(0)
            yerr_abs_upper.append(0)

    # Plot true values as black horizontal lines with square markers
    for i in range(len(all_params)):
        ax1.hlines(true_vals_plot[i], i - 0.3, i + 0.3, colors='black',
                   linewidth=3, label='True' if i == 0 else None)

    # Plot recovered values with error bars
    # Note: on log scale, asymmetric error bars need special handling
    # We plot the error bars in absolute units on the recovered values
    ax1.errorbar(x_pos, rec_vals_plot, 
                 yerr=[yerr_abs_lower, yerr_abs_upper],
                 fmt='o', color='C0', markersize=12, markeredgecolor='black',
                 markeredgewidth=1.5, capsize=6, capthick=2, elinewidth=2,
                 zorder=5, label='Recovered (median)')

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

    # === Panel 2: Fractional error (%) with uncertainty bands ===
    frac_errors = np.array([per_param_stats[p]['frac_error'] * 100 for p in all_params])

    # Add error bars from posterior uncertainties
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

    ax2.axhline(0, color='black', linestyle='-', linewidth=1.5)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(all_params, rotation=45, ha='right', fontsize=10)
    ax2.set_ylabel('Fractional Error (%)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y', linestyle=':')
    ax2.set_xlim(-0.5, len(all_params) - 0.5)

    # === Title with joint Nσ ===
    if not np.isnan(nsigma):
        nsigma_str = f'Joint: {nsigma:.2f}σ'
    else:
        nsigma_str = 'Joint: N/A'

    title_parts = ['Parameter Recovery']
    if sampler_name:
        title_parts[0] += f' ({sampler_name})'
    title_parts.append(nsigma_str)

    fig.suptitle('\n'.join(title_parts), fontsize=14, fontweight='bold',
                 color=title_color)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')

    # Build return stats dict
    stats = {
        'joint_nsigma': nsigma,
        'joint_chi2': joint_stats['chi2'],
        'joint_pvalue': joint_stats['pvalue'],
        'per_param': per_param_stats,
        'title_color': title_color,
    }

    return fig, stats


def print_summary(result: 'SamplerResult', true_values: Optional[Dict[str, float]] = None) -> None:
    """
    Print a summary of the sampling results.

    Parameters
    ----------
    result : SamplerResult
        Sampling results.
    true_values : dict, optional
        True parameter values for comparison.

    Examples
    --------
    >>> from kl_pipe.sampling.diagnostics import print_summary
    >>> print_summary(result, true_values=true_pars)
    """
    summary = result.get_summary()

    print("=" * 70)
    print("SAMPLING SUMMARY")
    print("=" * 70)
    print(f"Backend: {result.metadata.get('backend', 'unknown')}")
    print(f"N samples: {result.n_samples}")
    print(f"N params: {result.n_params}")

    if result.acceptance_fraction is not None:
        print(f"Acceptance: {result.acceptance_fraction:.1%}")

    if result.evidence is not None:
        print(f"Log evidence: {result.evidence:.2f}")
        if result.evidence_error is not None:
            print(f"Evidence error: {result.evidence_error:.2f}")

    print("-" * 70)
    print(f"{'Parameter':<15} {'Median':>12} {'Std':>12} {'[16%, 84%]':>20}", end='')
    if true_values:
        print(f" {'True':>12} {'Error':>10}")
    else:
        print()
    print("-" * 70)

    for name in result.param_names:
        s = summary[name]
        median = s['quantiles'][0.5]
        q16 = s['quantiles'][0.16]
        q84 = s['quantiles'][0.84]
        std = s['std']

        print(f"{name:<15} {median:>12.4f} {std:>12.4f} [{q16:>8.4f}, {q84:>8.4f}]", end='')

        if true_values and name in true_values:
            true_val = true_values[name]
            if abs(true_val) > 1e-10:
                rel_error = abs(median - true_val) / abs(true_val)
                print(f" {true_val:>12.4f} {rel_error:>9.1%}")
            else:
                print(f" {true_val:>12.4f} {'N/A':>10}")
        else:
            print()

    if result.fixed_params:
        print("-" * 70)
        print("Fixed parameters:")
        for name, val in result.fixed_params.items():
            print(f"  {name}: {val}")

    print("=" * 70)
