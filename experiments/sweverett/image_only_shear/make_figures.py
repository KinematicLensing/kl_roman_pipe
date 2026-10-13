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
PRIORS = [
    (r'$|g_i|<0.2$', lambda g1, g2: (np.abs(g1) < 0.2) & (np.abs(g2) < 0.2)),
    (r'$|g|<0.2$', lambda g1, g2: np.hypot(g1, g2) < 0.2),
    (r'$|g_i|<0.15$', lambda g1, g2: (np.abs(g1) < 0.15) & (np.abs(g2) < 0.15)),
    (r'$|g_i|<0.1$', lambda g1, g2: (np.abs(g1) < 0.1) & (np.abs(g2) < 0.1)),
    (r'$|g_i|<0.07$', lambda g1, g2: (np.abs(g1) < 0.07) & (np.abs(g2) < 0.07)),
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
        row['gt_cross'] = -row['truth.g1'] * s + row['truth.g2'] * c
        for k, (_, keep_fn) in enumerate(PRIORS):
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


def fig_width_vs_snr(fits, path, label):
    import matplotlib.pyplot as plt

    sel = [f for f in fits if f['pa'] in ('pa45', 'pa45rot')]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0), sharey=True)
    for ax, comp, ylab in (
        (axes[0], 'gp', r'$\sigma_{g_+}$'),
        (axes[1], 'gx', r'$\sigma_{g_\times}$'),
    ):
        for cosi, col in COSI_COLORS.items():
            xs, ys, conv = [], [], []
            for snr in sorted({f['snr'] for f in sel}):
                cell = [f for f in sel if f['snr'] == snr and f['cosi'] == cosi]
                if not cell:
                    continue
                xs.append(snr)
                ys.append(np.nanmedian([f[f'std_{comp}0'] for f in cell]))
                conv.append(np.mean([f['converged'] for f in cell]) >= 0.5)
            if not xs:
                continue
            ax.plot(xs, ys, '-', color=col, lw=2, label=rf'$\cos i={cosi:g}$')
            for x, y, ok in zip(xs, ys, conv):
                ax.plot(x, y, 'o', ms=8, mfc=col if ok else 'white', mec=col, mew=2)
        ax.axhline(BOX_STD, color=MUTED, lw=1, ls='--')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r'image $\mathrm{SNR}$ (F158, matched filter)', color=INK)
        ax.set_ylabel(ylab, color=INK)
        style(ax)
    axes[0].text(
        12,
        BOX_STD * 1.05,
        r'prior $U(-0.2,0.2)$ std',
        color=MUTED,
        fontsize=8,
        va='bottom',
    )
    axes[1].legend(frameon=False, fontsize=8, loc='lower left')
    fig.suptitle(
        rf'image only, $h/R={0.1:g}$ ({label}), disk at $45^\circ$ to pixels; open = <50% of fits converged',
        fontsize=10,
        color=INK,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f'wrote {path}')


def fig_shear_prior(fits, path, label):
    import matplotlib.pyplot as plt

    snrs = sorted({f['snr'] for f in fits if f['snr'] >= 47})
    x = np.arange(len(PRIORS))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0))
    for snr in snrs:
        col = SNR_COLORS.get(snr, MUTED)
        cell = [f for f in fits if f['snr'] == snr]
        widths = [
            np.nanmedian([f[f'std_gp{k}'] for f in cell]) for k in range(len(PRIORS))
        ]
        axes[0].plot(
            x, widths, 'o-', color=col, lw=2, ms=8, label=rf'$\mathrm{{SNR}}={snr:g}$'
        )
        # response of g+ from +/- pairs with pure-g+ truth (pa0, pa45rot), shared noise
        resp = []
        for k in range(len(PRIORS)):
            rk = []
            pairs = {}
            for f in cell:
                if f['pa'] in ('pa0', 'pa45rot'):
                    pairs.setdefault((f['cosi'], f['pa'], f['rep']), {})[f['sign']] = f
            for pr in pairs.values():
                if 1 in pr and -1 in pr:
                    num = pr[1][f'mean_gp{k}'] - pr[-1][f'mean_gp{k}']
                    rk.append(num / (2 * pr[1]['gt_plus']))
            resp.append(np.nanmedian(rk) if rk else np.nan)
        axes[1].plot(x, resp, 'o-', color=col, lw=2, ms=8)
    axes[0].set_ylabel(r'median $\sigma_{g_+}$', color=INK)
    axes[1].set_ylabel(
        r'median response $R_+=(\hat g_+^{(+)}-\hat g_+^{(-)})/2g_+$', color=INK
    )
    axes[1].axhline(1.0, color=MUTED, lw=1, ls='--')
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([p[0] for p in PRIORS], fontsize=9)
        ax.set_xlabel('shear prior', color=INK)
        style(ax)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(
        rf'image only, $h/R=0.1$ ({label}), all fits; $|g_\mathrm{{true}}|=0.05$; '
        r'$|g|<0.2$ is the disk prior, $|g_i|<b$ per-component boxes',
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
    args = ap.parse_args()
    fits = load(args.arm_dir)
    print(f'{len(fits)} fits')
    out = Path(__file__).parent / 'out'
    fig_width_vs_snr(fits, out / f'fig_width_vs_snr_{args.label}.png', args.label)
    fig_shear_prior(fits, out / f'fig_shear_prior_{args.label}.png', args.label)


if __name__ == '__main__':
    main()
