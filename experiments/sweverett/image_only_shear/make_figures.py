"""Figures for the image-only shear test.

fig_width_vs_snr.png   sigma_{g+} and sigma_{gx} vs image SNR, one line per cos i (disk at 45 deg to the pixel grid).
fig_shear_prior.png    how the shear prior changes the posterior width and the response, per image SNR.

Every mark: median over the fits in the cell (4 noise draws x +/-g), all fits whether or not they pass the convergence
gate; open markers where fewer than half pass (unconverged chains under-explore, so those widths are lower bounds).
Shear priors other than the U(-0.2, 0.2) box are obtained by restricting the box samples (exact for nested supports).

Usage: python make_figures.py out/vista/pinned [--label pinned]
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np

# reference sequential blue ramp (dataviz skill palette.md), ordered light -> dark with cos i
COSI_COLORS = {
    0.03: '#86b6ef',
    0.1: '#5598e7',
    0.3: '#2a78d6',
    0.5: '#256abf',
    0.7: '#184f95',
    0.9: '#0d366b',
}
SNR_COLORS = {47.0: '#86b6ef', 224.0: '#3987e5', 1057.0: '#1c5cab', 5000.0: '#0d366b'}
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'
BOX_STD = 0.2 / math.sqrt(3.0)
# (label, membership test on (g1, g2) samples)
# (label, membership test on (g1, g2) samples, per-component prior std)
PRIORS = [
    (
        r'$|g_i|<0.2$',
        lambda g1, g2: (np.abs(g1) < 0.2) & (np.abs(g2) < 0.2),
        0.2 / math.sqrt(3),
    ),
    (r'$|g|<0.2$', lambda g1, g2: np.hypot(g1, g2) < 0.2, 0.2 / 2),
    (
        r'$|g_i|<0.15$',
        lambda g1, g2: (np.abs(g1) < 0.15) & (np.abs(g2) < 0.15),
        0.15 / math.sqrt(3),
    ),
    (
        r'$|g_i|<0.1$',
        lambda g1, g2: (np.abs(g1) < 0.1) & (np.abs(g2) < 0.1),
        0.1 / math.sqrt(3),
    ),
    (
        r'$|g_i|<0.07$',
        lambda g1, g2: (np.abs(g1) < 0.07) & (np.abs(g2) < 0.07),
        0.07 / math.sqrt(3),
    ),
]


def load(arm_dir: Path):
    fits = []
    for jf in sorted(arm_dir.glob('*.json')):
        row = json.load(open(jf))
        npz = np.load(jf.with_suffix('.npz'))
        names = list(npz['names'])
        g1, g2 = (
            npz['samples'][:, names.index('g1')],
            npz['samples'][:, names.index('g2')],
        )
        th = row['truth.theta_int']
        c, s = math.cos(2 * th), math.sin(2 * th)
        row['gt_plus'] = row['truth.g1'] * c + row['truth.g2'] * s
        if 'h_over_r' in names:
            row['std_lnh'] = float(
                np.std(np.log(npz['samples'][:, names.index('h_over_r')]))
            )
        row['gt_cross'] = -row['truth.g1'] * s + row['truth.g2'] * c
        for k, (_, keep_fn, _) in enumerate(PRIORS):
            keep = keep_fn(g1, g2)
            gp, gx = (g1 * c + g2 * s)[keep], (-g1 * s + g2 * c)[keep]
            row[f'n_keep{k}'] = int(keep.sum())
            row[f'mean_gp{k}'], row[f'std_gp{k}'] = (
                (gp.mean(), gp.std()) if keep.sum() > 20 else (np.nan, np.nan)
            )
            row[f'mean_gx{k}'], row[f'std_gx{k}'] = (
                (gx.mean(), gx.std()) if keep.sum() > 20 else (np.nan, np.nan)
            )
        fits.append(row)
    return fits


def style(ax):
    ax.grid(True, which='major', color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    for sp in ('left', 'bottom'):
        ax.spines[sp].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)


ARM_TITLES = {
    'pinned': r'$h/R=0.1$ fixed',
    'sampled': r'$h/R\sim\mathrm{LN}(0.25,\,0.14\,\mathrm{dex})$ sampled',
    'thin': r'$h/R=0.005$ fixed (thin disk)',
}
# multiplicative x offsets so the six cos i series do not overlap on the log SNR axis
COSI_XOFF = dict(zip(COSI_COLORS, np.linspace(-0.06, 0.06, len(COSI_COLORS))))


def fig_width_vs_snr(fits, path, label):
    import matplotlib.pyplot as plt

    sel = [f for f in fits if f['pa'] in ('pa45', 'pa45rot')]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0), sharey=True)
    for ax, comp, ylab in (
        (axes[0], 'gp', r'median posterior $\sigma_{g_+}$'),
        (axes[1], 'gx', r'median posterior $\sigma_{g_\times}$'),
    ):
        for cosi, col in COSI_COLORS.items():
            xs, ys, lo, hi, conv = [], [], [], [], []
            for snr in sorted({f['snr'] for f in sel}):
                cell = [f for f in sel if f['snr'] == snr and f['cosi'] == cosi]
                if not cell:
                    continue
                w = np.array([f[f'std_{comp}0'] for f in cell])
                xs.append(snr * 10 ** COSI_XOFF[cosi])
                ys.append(np.nanmedian(w))
                lo.append(np.nanpercentile(w, 16))
                hi.append(np.nanpercentile(w, 84))
                conv.append(np.mean([f['converged'] for f in cell]) >= 0.5)
            if not xs:
                continue
            ys, lo, hi = np.array(ys), np.array(lo), np.array(hi)
            ax.errorbar(
                xs,
                ys,
                yerr=[ys - lo, hi - ys],
                fmt='none',
                ecolor=col,
                elinewidth=1,
                capsize=0,
            )
            ax.plot(xs, ys, '-', color=col, lw=2, label=rf'$\cos i={cosi:g}$')
            for x, y, ok in zip(xs, ys, conv):
                ax.plot(x, y, 'o', ms=8, mfc=col if ok else 'white', mec=col, mew=2)
        ax.axhline(BOX_STD, color=MUTED, lw=1, ls='--')
        ax.set_xscale('log')
        ax.set_ylim(0, 0.135)
        ax.set_xlabel(r'image $\mathrm{SNR}$ (F158)', color=INK)
        ax.set_ylabel(ylab, color=INK)
        style(ax)
    axes[1].legend(frameon=False, fontsize=8, loc='lower left')
    fig.suptitle(
        'Shear precision from the image alone\n'
        f'{ARM_TITLES[label]}, '
        r'$|g_\mathrm{true}|=0.05$, shear prior $U(-0.2,\,0.2)$',
        fontsize=10,
        color=INK,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f'wrote {path}')


def _response(cell, k, comp, pas):
    """Median over +/- pairs (shared noise) of (ghat(+) - ghat(-)) / 2 g_true for one disk-frame component."""
    pairs = {}
    for f in cell:
        if f['pa'] in pas:
            pairs.setdefault((f['cosi'], f['pa'], f['rep']), {})[f['sign']] = f
    rk = []
    for pr in pairs.values():
        if 1 in pr and -1 in pr:
            gt = pr[1][f'gt_{"plus" if comp == "gp" else "cross"}']
            rk.append((pr[1][f'mean_{comp}{k}'] - pr[-1][f'mean_{comp}{k}']) / (2 * gt))
    return np.nanmedian(rk) if rk else np.nan


def fig_shear_prior(fits, path, label):
    import matplotlib.pyplot as plt

    snrs = sorted({f['snr'] for f in fits if f['snr'] >= 47})
    x = np.arange(len(PRIORS))
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5), sharex=True)
    # pure-g+ truth: pa0 and pa45rot; pure-gx truth: pa45
    rows = (('gp', '+', ('pa0', 'pa45rot')), ('gx', r'\times', ('pa45',)))
    for r, (comp, sym, pas) in enumerate(rows):
        for snr in snrs:
            col = SNR_COLORS.get(snr, MUTED)
            cell = [f for f in fits if f['snr'] == snr]
            widths = [
                np.nanmedian([f[f'std_{comp}{k}'] for f in cell])
                for k in range(len(PRIORS))
            ]
            axes[r, 0].plot(
                x,
                widths,
                'o-',
                color=col,
                lw=2,
                ms=8,
                label=rf'$\mathrm{{SNR}}={snr:g}$',
            )
            axes[r, 1].plot(
                x,
                [_response(cell, k, comp, pas) for k in range(len(PRIORS))],
                'o-',
                color=col,
                lw=2,
                ms=8,
            )
        axes[r, 0].plot(
            x,
            [p[2] for p in PRIORS],
            '--',
            color=INK,
            lw=1.2,
            label='prior std' if r == 0 else None,
        )
        axes[r, 0].set_ylabel(rf'median posterior $\sigma_{{g_{sym}}}$', color=INK)
        axes[r, 1].set_ylabel(
            rf'response $R_{sym}=(\hat g_{sym}^{{(+)}}-\hat g_{sym}^{{(-)}})/2g_{sym}$',
            color=INK,
        )
        axes[r, 1].axhline(1.0, color=MUTED, lw=1, ls='--')
        axes[r, 0].set_ylim(0, 0.12)
        axes[r, 1].set_ylim(-0.1, 1.15)
    for ax in axes.flat:
        ax.set_xticks(x)
        ax.set_xticklabels([p[0] for p in PRIORS], fontsize=9)
        style(ax)
    for ax in axes[1]:
        ax.set_xlabel('shear prior', color=INK)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle(
        'Shear precision and response vs shear prior\n'
        f'{ARM_TITLES[label]}, '
        r'$|g_\mathrm{true}|=0.05$',
        fontsize=10,
        color=INK,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f'wrote {path}')


def fig_compare(pinned, sampled, path):
    import matplotlib.pyplot as plt

    arms = (('pinned', pinned, '#2a78d6'), ('sampled', sampled, '#eb6834'))
    cosis = sorted(COSI_COLORS)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5), sharex=True)
    for j, snr in enumerate((1057.0, 5000.0)):
        for name, fits, col in arms:
            w45 = [
                np.nanmedian(
                    [
                        f['std_gp0']
                        for f in fits
                        if f['snr'] == snr
                        and f['cosi'] == c
                        and f['pa'] in ('pa45', 'pa45rot')
                    ]
                )
                for c in cosis
            ]
            axes[0, j].plot(
                cosis, w45, 'o-', color=col, lw=2, ms=8, label=ARM_TITLES[name]
            )
            resp = [
                _response(
                    [f for f in fits if f['snr'] == snr and f['cosi'] == c],
                    0,
                    'gp',
                    ('pa0', 'pa45rot'),
                )
                for c in cosis
            ]
            axes[1, j].plot(cosis, resp, 'o-', color=col, lw=2, ms=8)
        axes[0, j].axhline(BOX_STD, color=INK, lw=1.2, ls='--', label='prior std')
        axes[0, j].set_ylim(0, 0.135)
        axes[1, j].axhline(1.0, color=MUTED, lw=1, ls='--')
        axes[1, j].set_ylim(-0.3, 1.3)
        axes[0, j].set_title(rf'image $\mathrm{{SNR}}={snr:g}$', color=INK, fontsize=10)
        axes[1, j].set_xlabel(r'$\cos i$', color=INK)
    axes[0, 0].set_ylabel(r'median posterior $\sigma_{g_+}$', color=INK)
    axes[1, 0].set_ylabel(
        r'response $R_+=(\hat g_+^{(+)}-\hat g_+^{(-)})/2g_+$', color=INK
    )
    axes[0, 0].legend(frameon=False, fontsize=8)
    for ax in axes.flat:
        style(ax)
    fig.suptitle(
        'Shear from the image alone: fixed vs free disk thickness\n'
        r'$|g_\mathrm{true}|=0.05$, shear prior $U(-0.2,\,0.2)$',
        fontsize=10,
        color=INK,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f'wrote {path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('arm_dir', type=Path)
    ap.add_argument('--label', default='pinned')
    ap.add_argument(
        '--compare',
        type=Path,
        default=None,
        help='second arm dir (sampled) for the comparison figure',
    )
    args = ap.parse_args()
    if args.compare is not None:
        fig_compare(
            load(args.arm_dir),
            load(args.compare),
            Path(__file__).parent / 'out' / 'fig_compare_thickness.png',
        )
        return
    fits = load(args.arm_dir)
    print(f'{len(fits)} fits')
    out = Path(__file__).parent / 'out'
    fig_width_vs_snr(fits, out / f'fig_width_vs_snr_{args.label}.png', args.label)
    fig_shear_prior(fits, out / f'fig_shear_prior_{args.label}.png', args.label)


if __name__ == '__main__':
    main()
