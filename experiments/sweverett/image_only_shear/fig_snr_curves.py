"""Image-only shear width vs image SNR for a thin disk, a thick disk with known thickness, and a thick disk with free
thickness: MCMC cell medians (points) on Fisher curves (lines), one modelling change at a time.

Points: median posterior std over a cell's fits (3-4 noise draws x +/-g), disk aligned with the pixel grid (pa0), all
fits; open markers where fewer than half pass the convergence gate. Arms: thin = h/r 0.005 fixed in truth and fit;
fixed = h/r 0.1 fixed (Jiachuan's fiducial); free = truth h/r drawn from LN(0.25, 0.14 dex) and fit with that prior.
Lines: Fisher width on the noise-free image at the same orientation, combined with the U(-0.2, 0.2) shear prior as a
Gaussian of the same std (approximation; exact only where one term dominates); the free curve uses h/r 0.25 with the
lognormal prior. The vertical line is the census median F158 SNR (107, census v2).
"""

import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import explainer as ex  # noqa: E402
import make_figures as mf  # noqa: E402
import run_image_only as rio  # noqa: E402

COSIS = (0.1, 0.5, 0.9)
SNR_GRID = np.logspace(1, np.log10(6000), 60)
BOX_STD = 0.2 / math.sqrt(3.0)
CENSUS_SNR = 107.0
SIG_LNH = 0.14 * math.log(10.0)
ARMS = (
    ('thin', r'thin disk ($h/R=0.005$ fixed)', '#86b6ef', 0.005, False),
    ('pinned', r'thick disk ($h/R=0.1$ fixed)', '#2a78d6', 0.1, False),
    ('sampled', r'thick disk ($h/R\sim0.25$, free with prior)', '#eb6834', 0.25, True),
)
NAMES = ['gp', 'gx', 'cosi', 'theta_int', 'flux', 'rscale', 'h_over_r']


def fisher_width_at_snr1(h, free_h, cosi, psf):
    """Disk-frame (sigma_g+, sigma_gx) times SNR (the likelihood-only width at SNR 1), plus the h prior as a fixed term."""
    ex.COSI = cosi
    img, d = ex.derivatives(h, psf, 0.11, 24)
    jac = np.stack([d[n].ravel() for n in NAMES], axis=1)
    sig_pix1 = math.sqrt(np.sum(img**2))  # pixel noise at SNR 1

    def widths(snr):
        f = jac.T @ jac * (snr / sig_pix1) ** 2
        keep = list(range(len(NAMES)))
        ih = NAMES.index('h_over_r')
        if free_h:
            f[ih, ih] += 1.0 / (h * SIG_LNH) ** 2
        else:
            keep.remove(ih)
        cov = np.linalg.inv(f[np.ix_(keep, keep)])
        return math.sqrt(cov[0, 0]), math.sqrt(cov[1, 1])

    return widths


def main():
    import matplotlib.pyplot as plt

    from kl_pipe.ensemble.mocks import _build_band_psf
    from kl_pipe.ensemble.spec import ObservationConfig

    config = ObservationConfig.from_yaml(rio.OBS_CONFIG)
    psf = _build_band_psf(config.band_psf[rio.BAND], rio.BAND, rio.Z, mock=True)
    fits = {arm: mf.load(HERE / 'out' / 'vista' / arm) for arm, *_ in ARMS}

    fig, axes = plt.subplots(2, len(COSIS), figsize=(13, 7.5), sharex=True, sharey=True)
    for j, cosi in enumerate(COSIS):
        for arm, label, col, h, free_h in ARMS:
            wfn = fisher_width_at_snr1(h, free_h, cosi, psf)
            fw = np.array([wfn(s) for s in SNR_GRID])
            for r, comp in enumerate(('gp', 'gx')):
                ax = axes[r, j]
                total = 1.0 / np.sqrt(1.0 / fw[:, r] ** 2 + 1.0 / BOX_STD**2)
                ax.plot(
                    SNR_GRID,
                    total,
                    '-',
                    color=col,
                    lw=3 if arm == 'thin' else 1.5,
                    zorder=3 if arm == 'thin' else 2,
                    label=label if (r, j) == (0, 0) else None,
                )
                for snr in sorted({f['snr'] for f in fits[arm]}):
                    cell = [
                        f
                        for f in fits[arm]
                        if f['snr'] == snr and f['cosi'] == cosi and f['pa'] == 'pa0'
                    ]
                    if not cell:
                        continue
                    y = np.nanmedian([f[f'std_{comp}0'] for f in cell])
                    ok = np.mean([f['converged'] for f in cell]) >= 0.5
                    ax.plot(
                        snr, y, 'o', ms=8, mfc=col if ok else 'white', mec=col, mew=2
                    )
        for r in range(2):
            ax = axes[r, j]
            ax.axhline(BOX_STD, color=mf.INK, lw=1, ls='--', zorder=4)
            ax.axvline(CENSUS_SNR, color=mf.MUTED, lw=1, ls=':')
            ax.set_xscale('log')
            ax.set_ylim(0, 0.135)
            mf.style(ax)
        axes[0, j].set_title(rf'$\cos i={cosi:g}$', color=mf.INK, fontsize=11)
        axes[1, j].set_xlabel(r'image $\mathrm{SNR}$ (F158)', color=mf.INK)
    axes[0, 0].set_ylabel(r'median posterior $\sigma_{g_+}$', color=mf.INK)
    axes[1, 0].set_ylabel(r'median posterior $\sigma_{g_\times}$', color=mf.INK)
    axes[0, 0].text(
        CENSUS_SNR * 1.08,
        0.012,
        'census median',
        color=mf.MUTED,
        fontsize=8,
        rotation=90,
        va='bottom',
    )
    axes[0, 0].text(12, BOX_STD + 0.003, 'prior std', color=mf.INK, fontsize=8)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc='lower center',
        ncol=3,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, 0.03),
    )
    fig.suptitle(
        'Shear from the image alone vs image SNR\n'
        r'$|g_\mathrm{true}|=0.05$, shear prior $U(-0.2,\,0.2)$, disk aligned with the pixels',
        fontsize=11,
        color=mf.INK,
    )
    fig.text(
        0.5,
        0.008,
        'Points: median over fits per cell (open: <50% converged). Lines: Fisher width combined with the prior '
        '(Gaussian approximation).',
        ha='center',
        color=mf.MUTED,
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    out = HERE / 'out' / 'fig_snr_curves.png'
    fig.savefig(out, dpi=150)
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
