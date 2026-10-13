"""Read image-only shear fits: per-cell disk-frame shear widths, response, and box-prior reweighting.

Disk frame: g+ along the true major axis, gx at 45 deg (spin-2 rotation by the true theta_int). Narrower shear
priors are obtained by restricting the U(-0.2, 0.2) chains to |g1|, |g2| < b (exact for nested boxes).

Usage: python read_image_only.py <out_dir>/<arm> [--fig path.png]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

BOXES = (0.2, 0.15, 0.1, 0.07)
BOX_CAP = 0.2 / np.sqrt(3.0)


def disk_frame(g1, g2, theta):
    c, s = np.cos(2 * theta), np.sin(2 * theta)
    return g1 * c + g2 * s, -g1 * s + g2 * c


def load(arm_dir: Path) -> pd.DataFrame:
    rows = []
    for jf in sorted(arm_dir.glob('*.json')):
        row = json.load(open(jf))
        npz = np.load(jf.with_suffix('.npz'))
        names = list(npz['names'])
        smp = npz['samples']
        g1, g2 = smp[:, names.index('g1')], smp[:, names.index('g2')]
        th = row['truth.theta_int']
        gp, gx = disk_frame(g1, g2, th)
        row['truth.gp'], row['truth.gx'] = disk_frame(
            row['truth.g1'], row['truth.g2'], th
        )
        for b in BOXES:
            keep = (np.abs(g1) < b) & (np.abs(g2) < b)
            row[f'n_keep.b{b}'] = int(keep.sum())
            row[f'mean.gp.b{b}'] = float(gp[keep].mean()) if keep.any() else np.nan
            row[f'mean.gx.b{b}'] = float(gx[keep].mean()) if keep.any() else np.nan
            row[f'std.gp.b{b}'] = float(gp[keep].std()) if keep.any() else np.nan
            row[f'std.gx.b{b}'] = float(gx[keep].std()) if keep.any() else np.nan
        rows.append(row)
    n_failed = len(list(arm_dir.glob('*.FAILED')))
    print(f'{arm_dir}: {len(rows)} fits, {n_failed} FAILED files')
    return pd.DataFrame(rows)


def cell_table(df: pd.DataFrame, box: float = 0.2) -> pd.DataFrame:
    keys = ['snr', 'cosi', 'pa']
    out = []
    for k, g in df.groupby(keys):
        conv = g[g.converged]
        rec = dict(zip(keys, k), n=len(g), n_conv=len(conv))
        rec['sig_gp'] = conv[f'std.gp.b{box}'].median() if len(conv) else np.nan
        rec['sig_gx'] = conv[f'std.gx.b{box}'].median() if len(conv) else np.nan
        rec['sig_cosi'] = conv['std.cosi'].median() if len(conv) else np.nan
        # all fits regardless of convergence (unconverged chains tend to under-explore, so these are lower bounds)
        rec['sig_gp_all'] = g[f'std.gp.b{box}'].median()
        rec['sig_gx_all'] = g[f'std.gx.b{box}'].median()
        # response from +/- pairs where both members converged
        resp = []
        for _, pair in conv.groupby('rep'):
            if len(pair) != 2:
                continue
            p, m = pair[pair.sign > 0].iloc[0], pair[pair.sign < 0].iloc[0]
            for comp in ('gp', 'gx'):
                gt = p[f'truth.{comp}']
                if abs(gt) > 0.02:
                    resp.append(
                        (p[f'mean.{comp}.b{box}'] - m[f'mean.{comp}.b{box}']) / (2 * gt)
                    )
        rec['R'] = float(np.mean(resp)) if resp else np.nan
        rec['n_pairs'] = len(resp)
        out.append(rec)
    return pd.DataFrame(out)


def plot(df: pd.DataFrame, path: Path):
    import matplotlib.pyplot as plt

    pas = [p for p in ('pa0', 'pa22', 'pa45', 'pa45rot') if p in set(df.pa)]
    cosis = sorted(df.cosi.unique())
    cmap = plt.get_cmap('viridis')
    fig, axes = plt.subplots(
        2, len(pas), figsize=(4 * len(pas), 7), sharex=True, sharey=True, squeeze=False
    )
    tab = cell_table(df)
    for j, pa in enumerate(pas):
        for i, comp in enumerate(('gp', 'gx')):
            ax = axes[i, j]
            for c_i, c in enumerate(cosis):
                t = tab[(tab.pa == pa) & (tab.cosi == c)].sort_values('snr')
                ax.plot(
                    t.snr,
                    t[f'sig_{comp}_all'],
                    ':',
                    color=cmap(c_i / max(1, len(cosis) - 1)),
                )
                ax.plot(
                    t.snr,
                    t[f'sig_{comp}'],
                    'o-',
                    color=cmap(c_i / max(1, len(cosis) - 1)),
                    label=f'cos i = {c:g}',
                )
            ax.axhline(BOX_CAP, color='0.5', ls='--', lw=1)
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_title(f'{pa}')
            if j == 0:
                ax.set_ylabel(
                    r'median $\sigma(g_+)$'
                    if comp == 'gp'
                    else r'median $\sigma(g_\times)$'
                )
            if i == 1:
                ax.set_xlabel('image SNR (F158 matched filter)')
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(
        'image-only posterior shear width: solid = converged fits, dotted = all fits; dashed = U(-0.2, 0.2) box std'
    )
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f'wrote {path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('arm_dir', type=Path)
    ap.add_argument('--fig', type=Path, default=None)
    args = ap.parse_args()
    df = load(args.arm_dir)
    if df.empty:
        raise SystemExit('no results')
    pd.set_option('display.width', 200)
    print(f"converged {df.converged.sum()}/{len(df)}")
    for box in BOXES:
        print(f'\n== shear box |g| < {box} ==')
        print(cell_table(df, box).round(4).to_string(index=False))
    if args.fig:
        plot(df, args.fig)


if __name__ == '__main__':
    main()
