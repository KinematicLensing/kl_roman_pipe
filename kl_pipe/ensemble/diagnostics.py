"""
Ensemble run diagnostics: quality gates, recovery pulls, shape noise, plots.

Callable per-piece or as one report::

    from kl_pipe.ensemble.diagnostics import run_report
    run_report('runs/<run_name>')      # -> runs/<run_name>/diagnostics/

Outputs (under ``run_dir/diagnostics/``):
- ``quality.csv``     per-fit quality columns + gate flags
- ``pulls.csv``       per-fit (post - truth)/sigma for the headline params
- ``sigma_eps.csv``   per-axis-bin + collapsed sigma_eps (catalog runs with a
  degenerate manifest axis fall back to quantile bins over the truth column)
- ``sigma_eps_line_snr.csv`` + ``sigma_eps_vs_line_snr.png`` when per-fit
  line SNR varies and is not already the plot axis (catalog mode)
- ``recovery_<param>.png``, ``pulls.png``, ``quality_vs_cosi.png``,
  ``sigma_eps_vs_cosi.png``
- ``corner_<fit_id>.png`` + ``datavector_<fit_id>.png`` for every gated-out
  fit and one representative good fit per bin (needs saved chains/mocks)

Gate policy (recorded, applied post hoc; see the ensemble plan): a fit is
``low_quality`` when max_rhat > 1.01 OR min_ess < 400 OR its divergence rate
exceeds 2x the campaign median; ``catastrophic`` when max_rhat > 1.1 (broken
chains -- never usable). sigma_eps is reported both ways.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from kl_pipe.coordinates import galaxy_frame_samples, rotate_to_galaxy_frame
from kl_pipe.ensemble.calibration import (
    compute_shape_noise,
    measure_shear_bias,
    measure_shear_bias_shrinkage_corrected,
)
from kl_pipe.ensemble.collate import analysis_table
from kl_pipe.ensemble.expander import load_run

# shear is reported in the galaxy frame (g+, gx), not sky-frame (g1, g2): g+/gx
# are the interpretable kinematic-lensing quantities and their marginals do not
# rotate fit-to-fit with the intrinsic PA. Both rotation conventions are
# reported side by side: the 'measured' rotation (each posterior sample by its
# own theta_int -- the only estimator available on real data, but it folds the
# shear-PA posterior ridge into g+/gx: census v1 measured a -0.034 additive
# g+ artifact from that covariance) and the fixed truth-PA rotation
# (convention-free recovery check, only possible on simulations). See
# augment_galaxy_frame.
HEADLINE_PARAMS = (
    'g_plus',
    'g_cross',
    'g_plus_truth_pa',
    'g_cross_truth_pa',
    'cosi',
    'theta_int',
    'vel.vcirc',
)
CORNER_PARAMS = (
    'g1',
    'g2',
    'g_plus',
    'g_cross',
    'cosi',
    'theta_int',
    'vel.vcirc',
    'vel.rscale',
    'Halpha.dispersion',
)

# math display labels for plot axes/titles. All plotted params are mapped so the
# typesetting is consistent (no plain-string param beside a math one). Unmapped
# keys fall back to the raw key.
PARAM_LABELS = {
    'g1': r'$g_1$',
    'g2': r'$g_2$',
    'g_plus': r'$g_+$ (measured PA)',
    'g_cross': r'$g_\times$ (measured PA)',
    'g_plus_truth_pa': r'$g_+$ (truth PA)',
    'g_cross_truth_pa': r'$g_\times$ (truth PA)',
    'cosi': r'$\cos i$',
    'theta_int': r'$\theta_{\rm int}$',
    'vel.vcirc': r'$v_{\rm circ}$',
    'vel.rscale': r'$r_v$',
    'Halpha.dispersion': r'$\sigma_{\rm H\alpha}$',
}


def _label(param: str) -> str:
    """Math display label for a parameter key (falls back to the raw key)."""
    return PARAM_LABELS.get(param, param)


# =============================================================================
# tables
# =============================================================================


def measurement_axis(spec) -> Tuple[str, str, str, str]:
    """(group_col, value_col, axis_label, xscale) for the spec's plot axis."""
    if spec.measurement == 'sigma_eps_vs_line_snr':
        return ('sweep_step', 'line_snr', 'emission-line SNR', 'log')
    return ('cosi_bin', 'truth.cosi', 'cos i (bin center)', 'linear')


def quantile_bins(values: pd.Series, n_bins: int = 5) -> pd.Series:
    """Integer quantile-bin labels (0..n_bins-1) over a continuous column.

    Catalog-mode runs draw cosi / line SNR from the row bank instead of a
    sweep grid, so their manifest bin columns are degenerate; quantile bins
    over the continuous truth column restore a usable plot axis. Ties are
    broken by rank so equal-count bins always exist.
    """
    n_bins = min(n_bins, int(values.nunique()))
    if n_bins < 1:
        raise ValueError('quantile_bins needs at least one finite value')
    ranked = values.rank(method='first')
    return pd.qcut(ranked, n_bins, labels=False).astype(int)


def resolve_plot_axis(
    table: pd.DataFrame, group_col: str, value_col: str, axis_label: str
) -> Tuple[pd.DataFrame, str, str]:
    """Fall back to quantile bins when the manifest plot axis is degenerate.

    Sweep runs carry a real grid in ``group_col``; catalog runs write a
    single placeholder bin for every fit, which would collapse the binned
    plots to one point. Returns (table, group_col, axis_label), with an
    ``axis_qbin`` column added when the fallback fires.
    """
    if table[group_col].nunique() == 1 and table[value_col].nunique() > 1:
        table = table.copy()
        table['axis_qbin'] = quantile_bins(table[value_col])
        return table, 'axis_qbin', axis_label.replace('bin center', 'bin mean')
    return table, group_col, axis_label


def quality_table(table: pd.DataFrame) -> pd.DataFrame:
    """Per-fit quality columns + gate flags (low_quality / catastrophic)."""
    q = table[
        [
            'fit_id',
            'cosi_bin',
            'truth.cosi',
            'max_rhat',
            'min_ess',
            'n_divergences',
            'divergence_rate',
            'mean_accept_prob',
            'n_map_starts_converged',
            'precond_condition_number',
            'fit_wallclock_s',
            'precond_wallclock_s',
        ]
    ].copy()
    div_median = q['divergence_rate'].median()
    q['catastrophic'] = q['max_rhat'] > 1.1
    q['low_quality'] = (
        (q['max_rhat'] > 1.01)
        | (q['min_ess'] < 400)
        | (q['divergence_rate'] > 2.0 * div_median)
    )
    return q


def augment_galaxy_frame(
    run_dir: Path, table: pd.DataFrame, angle: str = 'measured'
) -> pd.DataFrame:
    """Add galaxy-frame shear columns (g+, gx) to a collated table.

    Adds ``truth.g_plus``/``truth.g_cross`` (truth shear rotated by truth PA)
    and posterior ``post.g_plus.mean``/``.std`` + ``post.g_cross.mean``/``.std``
    under ``angle`` (default 'measured'), plus a
    second, always-truth-PA set ``post.g_plus_truth_pa.*`` /
    ``post.g_cross_truth_pa.*`` (with ``truth.g_plus_truth_pa`` etc. aliasing
    the same truth values) so recovery/pull reports show both conventions:
    'measured' is the only estimator available on real data but folds the
    shear-PA posterior ridge into g+/gx (census v1: -0.034 additive g+
    artifact); the truth-PA rotation is the convention-free recovery check.

    The primary posterior columns come from, in order of preference: the
    per-sample rotation of the saved chain; the worker-written summary
    columns; or a marginal fallback from the g1/g2 marginals (mean exact at
    fixed angle, std exact only for an uncorrelated posterior; recorded in
    ``galaxy_frame_method``). The truth-PA set is chain-based when chains
    exist and marginal otherwise. The sky-frame g1/g2 columns are left intact
    but are no longer the reported shear.
    """
    t = table.copy()
    gp_t, gx_t = rotate_to_galaxy_frame(
        t['truth.g1'].to_numpy(),
        t['truth.g2'].to_numpy(),
        t['truth.theta_int'].to_numpy(),
    )
    t['truth.g_plus'] = gp_t
    t['truth.g_cross'] = gx_t
    # same truth values under either rotation convention; aliased so generic
    # pull/recovery code can treat g_plus_truth_pa as a normal parameter
    t['truth.g_plus_truth_pa'] = gp_t
    t['truth.g_cross_truth_pa'] = gx_t

    have_summary = {'post.g_plus.mean', 'post.g_cross.mean'}.issubset(t.columns)
    cols: Dict[str, List[float]] = {
        'post.g_plus.mean': [],
        'post.g_plus.std': [],
        'post.g_cross.mean': [],
        'post.g_cross.std': [],
        'post.g_plus_truth_pa.mean': [],
        'post.g_plus_truth_pa.std': [],
        'post.g_cross_truth_pa.mean': [],
        'post.g_cross_truth_pa.std': [],
    }
    methods = []

    def _marginal_truth_pa(r) -> Tuple[float, float, float, float]:
        """(mean_p, std_p, mean_x, std_x) at the fixed truth PA (mean exact)."""
        theta_true = float(r['truth.theta_int'])
        c2, s2 = np.cos(2 * theta_true), np.sin(2 * theta_true)
        gpm, gxm = rotate_to_galaxy_frame(
            r['post.g1.mean'], r['post.g2.mean'], theta_true
        )
        std_p = float(np.hypot(r['post.g1.std'] * c2, r['post.g2.std'] * s2))
        std_x = float(np.hypot(r['post.g1.std'] * s2, r['post.g2.std'] * c2))
        return float(gpm), std_p, float(gxm), std_x

    for _, r in t.iterrows():
        theta_true = float(r['truth.theta_int'])
        chain_path = Path(run_dir) / 'chains' / f"{r['fit_id']}.npz"
        if chain_path.exists():
            npz = np.load(chain_path)
            nm = list(npz['param_names'])
            S = npz['samples']
            g1s, g2s = S[:, nm.index('g1')], S[:, nm.index('g2')]
            ths = S[:, nm.index('theta_int')]
            for conv, prefix in [(angle, ''), ('truth', '_truth_pa')]:
                gps, gxs = galaxy_frame_samples(g1s, g2s, ths, theta_true, conv)
                cols[f'post.g_plus{prefix}.mean'].append(float(np.mean(gps)))
                cols[f'post.g_plus{prefix}.std'].append(float(np.std(gps, ddof=1)))
                cols[f'post.g_cross{prefix}.mean'].append(float(np.mean(gxs)))
                cols[f'post.g_cross{prefix}.std'].append(float(np.std(gxs, ddof=1)))
            methods.append('chain')
        else:
            if have_summary and np.isfinite(r.get('post.g_plus.mean', np.nan)):
                cols['post.g_plus.mean'].append(float(r['post.g_plus.mean']))
                cols['post.g_plus.std'].append(float(r['post.g_plus.std']))
                cols['post.g_cross.mean'].append(float(r['post.g_cross.mean']))
                cols['post.g_cross.std'].append(float(r['post.g_cross.std']))
                methods.append('summary')
            else:
                mp, sp, mx, sx = _marginal_truth_pa(r)
                cols['post.g_plus.mean'].append(mp)
                cols['post.g_plus.std'].append(sp)
                cols['post.g_cross.mean'].append(mx)
                cols['post.g_cross.std'].append(sx)
                methods.append('marginal-approx')
            mp, sp, mx, sx = _marginal_truth_pa(r)
            cols['post.g_plus_truth_pa.mean'].append(mp)
            cols['post.g_plus_truth_pa.std'].append(sp)
            cols['post.g_cross_truth_pa.mean'].append(mx)
            cols['post.g_cross_truth_pa.std'].append(sx)
    for name, values in cols.items():
        t[name] = values
    t['galaxy_frame_method'] = methods
    return t


# parameters whose residual must be wrapped before forming a pull, with
# their period. theta_int is a position angle: it is defined modulo pi, so a
# recovered value just below pi against a truth just above 0 is a match, not
# a ~pi discrepancy. Without wrapping, such fits show enormous spurious
# pulls and inflate the apparent wrong-mode rate.
_PULL_WRAP_PERIODS = {'theta_int': np.pi}


def _wrapped_residual(
    post_mean: pd.Series, truth: pd.Series, period: float
) -> pd.Series:
    """Residual on a periodic parameter, wrapped to (-period/2, period/2]."""
    resid = post_mean - truth
    return resid - period * np.round(resid / period)


def pull_table(
    table: pd.DataFrame, params: Sequence[str] = HEADLINE_PARAMS
) -> pd.DataFrame:
    """Per-fit recovery pulls (post.mean - truth) / post.std.

    Residuals on periodic parameters (see ``_PULL_WRAP_PERIODS``) are wrapped
    into the principal branch first.
    """
    out = table[['fit_id', 'cosi_bin', 'truth.cosi']].copy()
    for p in params:
        period = _PULL_WRAP_PERIODS.get(p)
        if period is None:
            resid = table[f'post.{p}.mean'] - table[f'truth.{p}']
        else:
            resid = _wrapped_residual(
                table[f'post.{p}.mean'], table[f'truth.{p}'], period
            )
        out[f'pull.{p}'] = resid / table[f'post.{p}.std']
    return out


def _galaxy_frame_sigmas(run_dir: Path, row: pd.Series) -> Tuple[float, float, str]:
    """Posterior widths of (g+, gx) for one fit.

    Exact when the fit's chains were saved (rotate the g1/g2 samples by the
    truth position angle); otherwise falls back to the quadrature-invariant
    approximation from the marginal g1/g2 stds (exact only for uncorrelated
    posteriors -- flagged in the returned method string).
    """
    chain_path = Path(run_dir) / 'chains' / f"{row['fit_id']}.npz"
    theta_true = float(row['truth.theta_int'])
    if chain_path.exists():
        npz = np.load(chain_path)
        names = list(npz['param_names'])
        g1s = npz['samples'][:, names.index('g1')]
        g2s = npz['samples'][:, names.index('g2')]
        ths = npz['samples'][:, names.index('theta_int')]
        gp, gx = galaxy_frame_samples(g1s, g2s, ths, theta_true)
        return float(np.std(gp, ddof=1)), float(np.std(gx, ddof=1)), 'chains'
    s1, s2 = float(row['post.g1.std']), float(row['post.g2.std'])
    c2, s2t = np.cos(2 * theta_true), np.sin(2 * theta_true)
    gp = np.sqrt((s1 * c2) ** 2 + (s2 * s2t) ** 2)
    gx = np.sqrt((s1 * s2t) ** 2 + (s2 * c2) ** 2)
    return float(gp), float(gx), 'marginal-approx'


# shear components reported in the calibration table, as
# (truth column suffix, posterior column suffix) pairs
_BIAS_COMPONENTS = ('g1', 'g2', 'g_plus', 'g_cross')


def shear_bias_table(
    table: pd.DataFrame, sigma_prior: float, s_min: Optional[float] = None
) -> pd.DataFrame:
    """Shear calibration (m, c) per component under three estimators.

    Emits the uncorrected posterior-mean regression alongside the
    shrinkage-corrected forms so the size of the prior-shrinkage artifact is
    visible rather than implicit: under a Gaussian(0, sigma_prior) shear
    prior the naive slope is suppressed by the ensemble-mean shrinkage, which
    is a property of the prior and not of the pipeline.

    ``robust`` is the number to report. It is the one that survives the
    wrong-mode fits, which arrive confidently wrong and therefore carry both
    a large residual and a large weight.
    """
    rows: List[dict] = []
    for comp in _BIAS_COMPONENTS:
        truth_col, mean_col, std_col = (
            f'truth.{comp}',
            f'post.{comp}.mean',
            f'post.{comp}.std',
        )
        if not {truth_col, mean_col, std_col}.issubset(table.columns):
            continue
        g_true = table[truth_col].to_numpy(dtype=float)
        mu = table[mean_col].to_numpy(dtype=float)
        sd = table[std_col].to_numpy(dtype=float)
        if np.ptp(g_true) == 0.0:
            # ring-style specs inject one shear value; a slope is undefined
            continue

        naive = measure_shear_bias(g_true, mu)
        rows.append(
            {
                'component': comp,
                'estimator': 'naive',
                'm': naive.m,
                'sigma_m': naive.sigma_m,
                'c': naive.c,
                'sigma_c': naive.sigma_c,
                'n_used': len(g_true),
                'n_clipped': 0,
                'n_outliers': 0,
                'mean_shrinkage': np.nan,
            }
        )
        for est in ('wls', 'robust'):
            r = measure_shear_bias_shrinkage_corrected(
                g_true, mu, sd, sigma_prior, s_min=s_min, estimator=est
            )
            rows.append(
                {
                    'component': comp,
                    'estimator': f'corrected_{est}',
                    'm': r.m,
                    'sigma_m': r.sigma_m,
                    'c': r.c,
                    'sigma_c': r.sigma_c,
                    'n_used': r.n_used,
                    'n_clipped': r.n_clipped,
                    'n_outliers': r.n_outliers,
                    'mean_shrinkage': r.mean_shrinkage,
                }
            )
    return pd.DataFrame(rows)


def sigma_eps_table(
    run_dir: Path,
    table: pd.DataFrame,
    group_col: str = 'cosi_bin',
    value_col: str = 'truth.cosi',
) -> pd.DataFrame:
    """Per-axis-step + collapsed sigma_eps, under both quality gates.

    ``group_col``/``value_col`` define the plot axis (cosi bins or a
    line_snr sweep); ``axis_step = -1`` rows are the collapsed ensemble.
    """
    q = quality_table(table).set_index('fit_id')
    rows: List[dict] = []
    sig = {}
    for _, r in table.iterrows():
        sp, sx, method = _galaxy_frame_sigmas(run_dir, r)
        sig[r['fit_id']] = (sp, sx, method)

    def _bin_rows(mask_name: str, mask: pd.Series):
        keep = table[table['fit_id'].map(mask)]
        for step, group in keep.groupby(group_col):
            sp = np.array([sig[f][0] for f in group['fit_id']])
            sx = np.array([sig[f][1] for f in group['fit_id']])
            se, err = compute_shape_noise(sp, sx)
            rows.append(
                {
                    'gate': mask_name,
                    'axis_step': int(step),
                    # bin mean; identical to the shared value for sweep axes
                    'axis_value': float(group[value_col].mean()),
                    'n_fits': len(group),
                    'sigma_eps': se,
                    'sigma_eps_err': err,
                }
            )
        if len(keep):
            sp = np.array([sig[f][0] for f in keep['fit_id']])
            sx = np.array([sig[f][1] for f in keep['fit_id']])
            se, err = compute_shape_noise(sp, sx)
            rows.append(
                {
                    'gate': mask_name,
                    'axis_step': -1,
                    'axis_value': np.nan,
                    'n_fits': len(keep),
                    'sigma_eps': se,
                    'sigma_eps_err': err,
                }
            )

    _bin_rows('exclude_catastrophic', ~q['catastrophic'])
    _bin_rows('full_gate', ~q['low_quality'])
    return pd.DataFrame(rows)


# =============================================================================
# plots
# =============================================================================


def _mpl():
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    return plt


def plot_recovery(
    table: pd.DataFrame,
    out_dir: Path,
    params: Sequence[str] = HEADLINE_PARAMS,
) -> List[Path]:
    """Truth vs posterior mean (with 1-sigma bars), colored by cosi."""
    plt = _mpl()
    paths = []
    bad = table['max_rhat'] > 1.1
    for p in params:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.errorbar(
            table.loc[~bad, f'truth.{p}'],
            table.loc[~bad, f'post.{p}.mean'],
            yerr=table.loc[~bad, f'post.{p}.std'],
            fmt='o',
            ms=4,
            lw=1,
            capsize=2,
            zorder=2,
            label='converged',
        )
        if bad.any():
            ax.scatter(
                table.loc[bad, f'truth.{p}'],
                table.loc[bad, f'post.{p}.mean'],
                marker='x',
                color='crimson',
                zorder=3,
                label='max_rhat > 1.1',
            )
        lo = min(table[f'truth.{p}'].min(), table[f'post.{p}.mean'].min())
        hi = max(table[f'truth.{p}'].max(), table[f'post.{p}.mean'].max())
        pad = 0.05 * (hi - lo + 1e-12)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], 'k--', lw=0.8)
        ax.set_xlabel(f'truth {_label(p)}')
        ax.set_ylabel(f'posterior mean {_label(p)}')
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = out_dir / f"recovery_{p.replace('.', '_')}.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        paths.append(path)
    return paths


def plot_pulls(
    table: pd.DataFrame,
    out_dir: Path,
    params: Sequence[str] = HEADLINE_PARAMS,
) -> Path:
    """Pull histograms per parameter with the N(0,1) reference."""
    plt = _mpl()
    good = table[table['max_rhat'] <= 1.1]
    pulls = pull_table(good, params)
    fig, axes = plt.subplots(1, len(params), figsize=(3.2 * len(params), 3.2))
    x = np.linspace(-4, 4, 200)
    for ax, p in zip(np.atleast_1d(axes), params):
        vals = pulls[f'pull.{p}'].values
        ax.hist(vals, bins=np.linspace(-4, 4, 17), density=True, alpha=0.7)
        ax.plot(x, np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi), 'k-', lw=1)
        ax.set_title(
            f'{_label(p)}\nmean {np.mean(vals):+.2f}  std {np.std(vals, ddof=1):.2f}',
            fontsize=9,
        )
        ax.set_xlabel('pull')
    fig.suptitle('recovery pulls (converged fits)', fontsize=10)
    fig.tight_layout()
    path = out_dir / 'pulls.png'
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_quality_vs_cosi(table: pd.DataFrame, out_dir: Path) -> Path:
    """Divergence rate, rhat, ess, wallclock vs cosi -- the stiffness map."""
    plt = _mpl()
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.2))
    panels = [
        ('divergence_rate', 'divergence rate', 'log'),
        ('max_rhat', 'max R-hat', 'log'),
        ('min_ess', 'min ESS', 'log'),
        ('fit_wallclock_s', 'fit wallclock [s]', 'linear'),
    ]
    for ax, (col, label, scale) in zip(axes, panels):
        ax.scatter(table['truth.cosi'], table[col], s=18)
        ax.set_xlabel('truth cos i')
        ax.set_ylabel(label)
        ax.set_yscale(scale)
    fig.suptitle('sampler quality vs inclination', fontsize=10)
    fig.tight_layout()
    path = out_dir / 'quality_vs_cosi.png'
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_sigma_eps(
    sigma_table: pd.DataFrame,
    out_dir: Path,
    axis_label: str = 'cos i (bin center)',
    xscale: str = 'linear',
    filename: str = 'sigma_eps.png',
) -> Path:
    """The vehicle plot: sigma_eps vs the campaign axis, + collapsed lines."""
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for gate, marker in [('exclude_catastrophic', 'o'), ('full_gate', 's')]:
        sub = sigma_table[sigma_table['gate'] == gate]
        bins = sub[sub['axis_step'] >= 0]
        head = sub[sub['axis_step'] == -1]
        ax.errorbar(
            bins['axis_value'],
            bins['sigma_eps'],
            yerr=bins['sigma_eps_err'],
            fmt=marker + '-',
            capsize=3,
            label=f'{gate} (per bin)',
        )
        if len(head):
            ax.axhline(
                head['sigma_eps'].iloc[0],
                ls=':',
                lw=1,
                color=ax.lines[-1].get_color(),
            )
    ax.axhline(0.26, color='gray', ls='--', lw=1, label='photometric 0.26')
    ax.set_xlabel(axis_label)
    ax.set_ylabel(r'effective shape noise $\sigma_\epsilon$')
    ax.set_yscale('log')
    ax.set_xscale(xscale)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = out_dir / filename
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_sigma_eps_slide(
    sigma_table: pd.DataFrame,
    out_dir: Path,
    axis_label: str = 'cos i (bin center)',
    gate: str = 'exclude_catastrophic',
    show_photometric: bool = False,
    filename: str = 'sigma_eps_slide.png',
) -> Path:
    """
    Slide-friendly sigma_eps plot: linear axes, high resolution, a single
    measurement line, per-point n_fits annotations. The photometric 0.26
    reference (per-component shape noise) is off by default -- include it
    only when the sigma_eps definition in use is per-component-comparable.
    """
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    sub = sigma_table[sigma_table['gate'] == gate]
    bins = sub[sub['axis_step'] >= 0]
    ax.errorbar(
        bins['axis_value'],
        bins['sigma_eps'],
        yerr=bins['sigma_eps_err'],
        fmt='o-',
        capsize=3,
        color='C0',
        label='kinematic lensing',
    )
    for _, b in bins.iterrows():
        ax.annotate(
            f"n={int(b['n_fits'])}",
            (b['axis_value'], b['sigma_eps']),
            textcoords='offset points',
            xytext=(6, 8),
            fontsize=9,
        )
    if show_photometric:
        ax.axhline(
            0.26,
            color='gray',
            ls='--',
            lw=1,
            label=r'photometric $\sigma_\epsilon$ (per component)',
        )
    ax.set_xlabel(axis_label)
    ax.set_ylabel(r'effective shape noise $\sigma_\epsilon$')
    ax.set_ylim(bottom=0)
    ax.legend()
    fig.tight_layout()
    path = out_dir / filename
    fig.savefig(path, dpi=250)
    plt.close(fig)
    return path


def _fit_title(fit_id: str, row: pd.Series) -> str:
    """Per-fit plot title: id, truth cosi, per-channel SNRs, quality."""
    snr_cols = [('line_snr', 'lSNR'), ('broadband_snr', 'bbSNR')]
    # catalog manifests carry per-band published-depth SNRs instead of the
    # sampled-mode shared scalar
    snr_cols += [
        (c, f"bbSNR[{c.removeprefix('broadband_snr_')}]")
        for c in row.index
        if c.startswith('broadband_snr_')
    ]
    snr_bits = ''.join(
        f"  {label}={float(row[col]):.0f}"
        for col, label in snr_cols
        if col in row.index and pd.notna(row[col])
    )
    return (
        f"{fit_id}  cosi={row['truth.cosi']:.2f}{snr_bits}  "
        f"rhat={row['max_rhat']:.2f}  div={row['divergence_rate']:.0%}"
    )


def plot_corner_fit(
    run_dir: Path,
    table: pd.DataFrame,
    fit_id: str,
    out_dir: Path,
    params: Sequence[str] = CORNER_PARAMS,
) -> Optional[Path]:
    """Corner plot for one fit from its saved chains (None if not saved)."""
    import corner as corner_mod

    plt = _mpl()
    chain_path = Path(run_dir) / 'chains' / f'{fit_id}.npz'
    if not chain_path.exists():
        return None
    npz = np.load(chain_path)
    names = list(npz['param_names'])
    samples = npz['samples']
    row = table.set_index('fit_id').loc[fit_id]
    # append galaxy-frame shear columns so CORNER_PARAMS (g_plus/g_cross) resolve
    if all(n in names for n in ('g1', 'g2', 'theta_int')):
        gp, gx = galaxy_frame_samples(
            samples[:, names.index('g1')],
            samples[:, names.index('g2')],
            samples[:, names.index('theta_int')],
            float(row['truth.theta_int']),
        )
        samples = np.column_stack([samples, gp, gx])
        names = names + ['g_plus', 'g_cross']
    use = [p for p in params if p in names]
    idx = [names.index(p) for p in use]
    fig = corner_mod.corner(
        samples[:, idx],
        labels=[_label(p) for p in use],
        truths=[float(row[f'truth.{p}']) for p in use],
        show_titles=True,
        title_fmt='.3f',
        title_kwargs={'fontsize': 8},
    )
    fig.suptitle(_fit_title(fit_id, row), fontsize=10)
    path = out_dir / f'corner_{fit_id}.png'
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_datavector_fit(
    run_dir: Path, table: pd.DataFrame, fit_id: str, out_dir: Path
) -> Optional[Path]:
    """Data / truth render / MAP render / residual panels from saved mocks."""
    plt = _mpl()
    mock_path = Path(run_dir) / 'mocks' / f'{fit_id}.npz'
    if not mock_path.exists():
        return None
    npz = np.load(mock_path)
    channels = sorted({k.rsplit('.', 1)[0] for k in npz.files})
    fig, axes = plt.subplots(
        len(channels), 4, figsize=(13, 3.1 * len(channels)), squeeze=False
    )
    for i, ch in enumerate(channels):
        data = npz[f'{ch}.data']
        var = float(np.asarray(npz[f'{ch}.variance']).ravel()[0])
        truth = npz[f'{ch}.truth_render']
        map_key = f'{ch}.map_render'
        model = npz[map_key] if map_key in npz.files else truth
        resid = (data - model) / np.sqrt(var)
        chi2 = float(np.sum(resid**2)) / resid.size
        panels = [
            (data, 'data'),
            (truth, 'truth render'),
            (model, 'MAP render'),
            (resid, f'(data-MAP)/sigma  chi2/pt={chi2:.2f}'),
        ]
        for j, (img, title) in enumerate(panels):
            ax = axes[i, j]
            kwargs = {'cmap': 'RdBu_r', 'vmin': -4, 'vmax': 4} if j == 3 else {}
            im = ax.imshow(np.asarray(img), origin='lower', **kwargs)
            ax.set_title(f'{ch}: {title}', fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046)
    row = table.set_index('fit_id').loc[fit_id]
    fig.suptitle(_fit_title(fit_id, row), fontsize=10)
    fig.tight_layout()
    path = out_dir / f'datavector_{fit_id}.png'
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


# =============================================================================
# report driver
# =============================================================================


def run_report(run_dir: Path, out_dir: Optional[Path] = None) -> Dict[str, object]:
    """Full diagnostics report for a collated run.

    Writes tables + plots to ``out_dir`` (default ``run_dir/diagnostics/``)
    and returns the tables. Requires ``collate`` to have been run.
    """
    run_dir = Path(run_dir)
    out_dir = Path(out_dir) if out_dir is not None else run_dir / 'diagnostics'
    out_dir.mkdir(parents=True, exist_ok=True)

    table = analysis_table(run_dir)
    failed = table[table['status'] == 'failed'] if 'status' in table else []
    if len(failed):
        print(f'WARNING: {len(failed)} failed fits excluded from diagnostics')
        table = table[table['status'] == 'succeeded']

    spec, _, _ = load_run(run_dir)
    group_col, value_col, axis_label, xscale = measurement_axis(spec)

    table = augment_galaxy_frame(run_dir, table)
    table, group_col, axis_label = resolve_plot_axis(
        table, group_col, value_col, axis_label
    )
    quality = quality_table(table)
    pulls = pull_table(table)
    sigma = sigma_eps_table(run_dir, table, group_col, value_col)
    bias = shear_bias_table(table, spec.shear_fit_prior_sigma)
    quality.to_csv(out_dir / 'quality.csv', index=False)
    pulls.to_csv(out_dir / 'pulls.csv', index=False)
    sigma.to_csv(out_dir / 'sigma_eps.csv', index=False)
    if len(bias):
        bias.to_csv(out_dir / 'shear_bias.csv', index=False)

    plot_recovery(table, out_dir)
    plot_pulls(table, out_dir)
    plot_quality_vs_cosi(table, out_dir)
    plot_sigma_eps(sigma, out_dir, axis_label=axis_label, xscale=xscale)
    plot_sigma_eps_slide(sigma, out_dir, axis_label=axis_label)

    # per-fit line SNR varies in catalog mode even when the plot axis is
    # cosi; emit the line-SNR-binned sigma_eps alongside
    if value_col != 'line_snr' and table.get('line_snr') is not None:
        if table['line_snr'].nunique() > 1:
            table = table.copy()
            table['line_snr_qbin'] = quantile_bins(table['line_snr'])
            sigma_snr = sigma_eps_table(run_dir, table, 'line_snr_qbin', 'line_snr')
            sigma_snr.to_csv(out_dir / 'sigma_eps_line_snr.csv', index=False)
            plot_sigma_eps(
                sigma_snr,
                out_dir,
                axis_label='emission-line SNR (bin mean)',
                xscale='log',
                filename='sigma_eps_vs_line_snr.png',
            )

    # per-fit deep dives: every gated-out fit + one good fit per axis step
    flagged = quality.loc[quality['low_quality'], 'fit_id'].tolist()
    good = quality[~quality['low_quality']]
    representatives = (
        good.sort_values('max_rhat')
        .groupby(table.set_index('fit_id').loc[good['fit_id'], group_col].values)[
            'fit_id'
        ]
        .first()
        .tolist()
        if len(good)
        else []
    )
    for fit_id in dict.fromkeys(flagged + representatives):
        plot_corner_fit(run_dir, table, fit_id, out_dir)
        plot_datavector_fit(run_dir, table, fit_id, out_dir)

    n_cat = int(quality['catastrophic'].sum())
    n_low = int(quality['low_quality'].sum())
    print(f'run: {run_dir.name}  fits: {len(table)}')
    print(f'  catastrophic (max_rhat > 1.1): {n_cat}')
    print(f'  low_quality  (full gate):      {n_low}')
    for gate in ('exclude_catastrophic', 'full_gate'):
        head = sigma[(sigma['gate'] == gate) & (sigma['axis_step'] == -1)]
        if len(head):
            print(
                f"  sigma_eps [{gate}]: {head['sigma_eps'].iloc[0]:.4f} "
                f"+/- {head['sigma_eps_err'].iloc[0]:.4f} "
                f"({int(head['n_fits'].iloc[0])} fits)"
            )
    for _, r in bias[bias['estimator'] == 'corrected_robust'].iterrows():
        print(
            f"  m[{r['component']}] = {r['m']:+.3f} +/- {r['sigma_m']:.3f}, "
            f"c = {r['c']:+.4f} +/- {r['sigma_c']:.4f} "
            f"(<s> = {r['mean_shrinkage']:.2f}, {int(r['n_outliers'])} outliers)"
        )
    print(f'  outputs: {out_dir}')
    return {
        'quality': quality,
        'pulls': pulls,
        'sigma_eps': sigma,
        'shear_bias': bias,
        'out_dir': out_dir,
    }
