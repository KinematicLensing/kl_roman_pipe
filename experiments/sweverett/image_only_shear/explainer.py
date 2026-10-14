"""Explainer figure: what the image alone tells the fit about each shear component.

For each shear component (g+ along the major axis, gx at 45 deg) the figure shows, in units of the per-pixel noise at
a stated image SNR, the image change produced by a shear of 0.05, and the part of that change that no other parameter
can mimic (the least-squares residual of the shear derivative after regressing out every other free parameter's
derivative). The quadrature sum of that unique part is the detection significance of a 0.05 shear, i.e.
0.05 / sigma_g from the Fisher matrix.

Columns:
  1. noise-free image of the thick disk (h/r 0.25) with contours
  2. raw change from a 0.05 shear (thick disk)
  3. unique part, thin disk (h/r 0.02), round disk assumed
  4. unique part, thick disk, thickness known and round disk assumed (the mock's assumptions)
  5. unique part, thick disk, thickness and intrinsic ellipticity free (what real disks require)

Scene: Jiachuan fiducial (r1/2 0.3"), Roman F158 PSF, 0.11" pixels, theta 0 so the major axis is along x, g1 = g+ and
g2 = gx.
"""

import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'kinematic_regime'))
import run_image_only as rio  # noqa: E402

COSI = 0.3
SNR = 1000.0
G = 0.05
H_THICK, H_THIN = 0.25, 0.02
STAMP = 24
INK, MUTED = '#0b0b0b', '#52514e'


def derivatives(h_over_r, psf):
    """Image and its derivatives wrt every parameter of the elliptical inclined exponential at e_int = 0."""
    import jax
    import jax.numpy as jnp
    from elliptical_disk import EllipticalInclinedExponentialModel

    from kl_pipe.observation import build_image_obs
    from kl_pipe.parameters import ImagePars
    from kl_pipe.render import RenderConfig
    from kl_pipe.source import SourceModel

    band = rio.BAND
    src = SourceModel(broadband_models={band: EllipticalInclinedExponentialModel()})
    obs = build_image_obs(
        ImagePars(shape=(STAMP, STAMP), pixel_scale=0.11, indexing='ij'),
        psf=psf,
        render_config=RenderConfig(oversample=rio.OVERSAMPLE),
        int_model=src.broadband_models[band],
        broadband_key=band,
    )
    names = [
        'g1',
        'g2',
        'cosi',
        'theta_int',
        'flux',
        'rscale',
        'h_over_r',
        'e1_int',
        'e2_int',
    ]
    x0 = jnp.array([0.0, 0.0, COSI, 0.0, rio.FLUX, rio.RSCALE, h_over_r, 0.0, 0.0])

    def image(x):
        p = {f'{band}.x0': 0.0, f'{band}.y0': 0.0}
        for n, v in zip(names, x):
            p[
                (
                    n
                    if n in ('g1', 'g2', 'cosi', 'theta_int', 'h_over_r')
                    else f'{band}.{n}'
                )
            ] = v
        return src.render_broadband(p, obs, band)

    img = np.asarray(image(x0))
    jac = np.asarray(jax.jacfwd(image)(x0))  # (ny, nx, n_params)
    return img, {n: jac[..., i] for i, n in enumerate(names)}


def unique_part(target, others):
    """Residual of target after least-squares projection onto the span of others (all flattened)."""
    if not others:
        return target
    a = np.stack([o.ravel() for o in others], axis=1)
    coef, *_ = np.linalg.lstsq(a, target.ravel(), rcond=None)
    return (target.ravel() - a @ coef).reshape(target.shape)


def main():
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    from kl_pipe.ensemble.mocks import _build_band_psf
    from kl_pipe.ensemble.spec import ObservationConfig

    config = ObservationConfig.from_yaml(rio.OBS_CONFIG)
    psf = _build_band_psf(config.band_psf[rio.BAND], rio.BAND, rio.Z, mock=True)
    img_thick, d_thick = derivatives(H_THICK, psf)
    img_thin, d_thin = derivatives(H_THIN, psf)
    # per-pixel noise of the matched-filter convention at this image SNR, separately per disk
    sig_thick = math.sqrt(np.sum(img_thick**2)) / SNR
    sig_thin = math.sqrt(np.sum(img_thin**2)) / SNR

    base = ['cosi', 'theta_int', 'flux', 'rscale']
    rows = [
        (
            'g1',
            r'$g_+$: stretch along the disk axes',
            ['g2'],
            ['h_over_r', 'e1_int', 'e2_int'],
        ),
        (
            'g2',
            r'$g_\times$: stretch at $45^\circ$ to the disk axes',
            ['g1'],
            ['h_over_r', 'e1_int', 'e2_int'],
        ),
    ]
    cmap = LinearSegmentedColormap.from_list('div', ['#2a78d6', '#e4e3df', '#eb6834'])
    fig, axes = plt.subplots(
        2, 5, figsize=(17, 7.6), gridspec_kw=dict(wspace=0.08, hspace=0.12)
    )
    ext = np.array([-1, 1, -1, 1]) * STAMP * 0.11 / 2
    for r, (comp, row_label, other_shear, realistic) in enumerate(rows):
        raw = G * d_thick[comp] / sig_thick
        thin = (
            G
            * unique_part(d_thin[comp], [d_thin[n] for n in base + other_shear])
            / sig_thin
        )
        thick = (
            G
            * unique_part(d_thick[comp], [d_thick[n] for n in base + other_shear])
            / sig_thick
        )
        real = (
            G
            * unique_part(
                d_thick[comp], [d_thick[n] for n in base + other_shear + realistic]
            )
            / sig_thick
        )
        vmax = np.max(np.abs(raw))
        # unique-part panels share their own scale (set by the thick, thickness-known case)
        umax = np.max(np.abs(thick))
        ax = axes[r, 0]
        ax.imshow(
            np.log10(np.maximum(img_thick, img_thick.max() * 1e-4)).T,
            origin='lower',
            cmap='Greys',
            extent=ext,
        )
        ax.contour(
            np.linspace(ext[0], ext[1], STAMP),
            np.linspace(ext[2], ext[3], STAMP),
            img_thick.T,
            levels=img_thick.max() * np.array([0.02, 0.08, 0.3, 0.7]),
            colors='#2a78d6',
            linewidths=1,
        )
        ax.set_title(
            rf'thick disk, $\cos i={COSI:g}$' if r == 0 else '', color=INK, fontsize=10
        )
        ax.set_ylabel(row_label, color=INK, fontsize=11)
        panels = [
            (raw, 'change from a 0.05 shear', None, vmax),
            (thin, 'part nothing else can copy:\nthin disk', thin, umax),
            (
                thick,
                'part nothing else can copy:\nthick disk, thickness known,\nround disk assumed (mock)',
                thick,
                umax,
            ),
            (
                real,
                'part nothing else can copy:\nthick disk, thickness and\nintrinsic shape free',
                real,
                umax,
            ),
        ]
        ims = []
        for c, (arr, title, uniq, lim) in enumerate(panels, start=1):
            ax = axes[r, c]
            ims.append(
                ax.imshow(
                    arr.T, origin='lower', cmap=cmap, vmin=-lim, vmax=lim, extent=ext
                )
            )
            if r == 0:
                ax.set_title(title, color=INK, fontsize=9.5)
            if uniq is not None:
                ax.text(
                    0.03,
                    0.04,
                    rf'{math.sqrt(np.sum(uniq**2)):.2g}$\sigma$ detection',
                    transform=ax.transAxes,
                    color=INK,
                    fontsize=10,
                    bbox=dict(fc='white', ec='none', alpha=0.8),
                )
        im = ims[-1]
        fig.colorbar(ims[0], ax=axes[r, 1], shrink=0.85, pad=0.02).ax.tick_params(
            labelsize=8
        )
        fig.colorbar(
            im,
            ax=axes[r, 2:].tolist(),
            shrink=0.85,
            pad=0.01,
            label='pixel change / pixel noise',
        )
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        'What the image alone tells the fit about each shear component\n'
        rf'image $\mathrm{{SNR}}={SNR:g}$, shear 0.05, $r_{{1/2}}=0.3^{{\prime\prime}}$, '
        r'Roman F158 PSF, $0.11^{\prime\prime}$ pixels',
        color=INK,
        fontsize=11,
    )
    out = HERE / 'out' / 'fig_explainer.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
