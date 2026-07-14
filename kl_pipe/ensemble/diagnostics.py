"""
Ensemble run diagnostics: quality gates, recovery pulls, shape noise, plots.

Callable per-piece or as one report::

    from kl_pipe.ensemble.diagnostics import run_report
    run_report('runs/<run_name>')      # -> runs/<run_name>/diagnostics/

Outputs (under ``run_dir/diagnostics/``):
- ``quality.csv``     per-fit quality columns + gate flags
- ``pulls.csv``       per-fit (post - truth)/sigma for the headline params
- ``sigma_eps.csv``   per-cosi-bin + collapsed sigma_eps
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

from kl_pipe.calibration import compute_shape_noise, rotate_to_galaxy_frame
from kl_pipe.ensemble.collate import analysis_table
from kl_pipe.ensemble.expander import load_run

HEADLINE_PARAMS = ('g1', 'g2', 'cosi', 'theta_int', 'vel.vcirc')
CORNER_PARAMS = ('g1', 'g2', 'cosi', 'theta_int', 'vel.vcirc', 'Halpha.dispersion')


# =============================================================================
# tables
# =============================================================================


def measurement_axis(spec) -> Tuple[str, str, str, str]:
    """(group_col, value_col, axis_label, xscale) for the spec's plot axis."""
    if spec.measurement == 'sigma_eps_vs_grism_snr':
        return ('sweep_step', 'grism_snr', 'grism SNR', 'log')
    return ('cosi_bin', 'truth.cosi', 'cos i (bin center)', 'linear')


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


def pull_table(
    table: pd.DataFrame, params: Sequence[str] = HEADLINE_PARAMS
) -> pd.DataFrame:
    """Per-fit recovery pulls (post.mean - truth) / post.std."""
    out = table[['fit_id', 'cosi_bin', 'truth.cosi']].copy()
    for p in params:
        out[f'pull.{p}'] = (table[f'post.{p}.mean'] - table[f'truth.{p}']) / table[
            f'post.{p}.std'
        ]
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
        gp, gx = rotate_to_galaxy_frame(g1s, g2s, theta_true)
        return float(np.std(gp, ddof=1)), float(np.std(gx, ddof=1)), 'chains'
    s1, s2 = float(row['post.g1.std']), float(row['post.g2.std'])
    c2, s2t = np.cos(2 * theta_true), np.sin(2 * theta_true)
    gp = np.sqrt((s1 * c2) ** 2 + (s2 * s2t) ** 2)
    gx = np.sqrt((s1 * s2t) ** 2 + (s2 * c2) ** 2)
    return float(gp), float(gx), 'marginal-approx'


def sigma_eps_table(
    run_dir: Path,
    table: pd.DataFrame,
    group_col: str = 'cosi_bin',
    value_col: str = 'truth.cosi',
) -> pd.DataFrame:
    """Per-axis-step + collapsed sigma_eps, under both quality gates.

    ``group_col``/``value_col`` define the plot axis (cosi bins or a
    grism_snr sweep); ``axis_step = -1`` rows are the collapsed ensemble.
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
                    'axis_value': float(group[value_col].iloc[0]),
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
        ax.set_xlabel(f'truth {p}')
        ax.set_ylabel(f'posterior mean {p}')
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
            f'{p}\nmean {np.mean(vals):+.2f}  std {np.std(vals, ddof=1):.2f}',
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
    row = table.set_index('fit_id').loc[fit_id]
    use = [p for p in params if p in names]
    idx = [names.index(p) for p in use]
    fig = corner_mod.corner(
        npz['samples'][:, idx],
        labels=use,
        truths=[float(row[f'truth.{p}']) for p in use],
        show_titles=True,
        title_fmt='.3f',
        title_kwargs={'fontsize': 8},
    )
    fig.suptitle(
        f"{fit_id}  cosi={row['truth.cosi']:.2f}  "
        f"rhat={row['max_rhat']:.2f}  div={row['divergence_rate']:.0%}",
        fontsize=10,
    )
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
    fig.suptitle(
        f"{fit_id}  cosi={row['truth.cosi']:.2f}  "
        f"rhat={row['max_rhat']:.2f}  div={row['divergence_rate']:.0%}",
        fontsize=10,
    )
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

    quality = quality_table(table)
    pulls = pull_table(table)
    sigma = sigma_eps_table(run_dir, table, group_col, value_col)
    quality.to_csv(out_dir / 'quality.csv', index=False)
    pulls.to_csv(out_dir / 'pulls.csv', index=False)
    sigma.to_csv(out_dir / 'sigma_eps.csv', index=False)

    plot_recovery(table, out_dir)
    plot_pulls(table, out_dir)
    plot_quality_vs_cosi(table, out_dir)
    plot_sigma_eps(sigma, out_dir, axis_label=axis_label, xscale=xscale)

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
    print(f'  outputs: {out_dir}')
    return {'quality': quality, 'pulls': pulls, 'sigma_eps': sigma, 'out_dir': out_dir}
