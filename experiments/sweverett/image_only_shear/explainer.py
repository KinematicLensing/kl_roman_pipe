"""Explainer figure: what the image alone tells the fit about each shear component.

Rows are four modelling cases (thin disk; thick disk with thickness known and a round face-on disk; thickness free;
thickness and intrinsic face-on shape free). Columns: the galaxy; the pixel change from g+ = 0.05 and gx = 0.05 in
pixel-noise units; and each change left after refitting the rest (least-squares residual of the shear derivative after
regressing out every other free parameter's derivative). The quadrature sum of that leftover is the significance N at
which a 0.05 shear is detected, so sigma_g = 0.05 / N (the Fisher marginal width).

Scene: Jiachuan fiducial (half-light radius 0.3", scale length 0.179"), cos i 0.3, Roman F158 PSF, disk horizontal.
Default: 0.11" pixels at image SNR 1000 (the Roman case). --hires: 4x finer pixels (0.0275") at SNR 5000, same PSF.
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
# major axis along the horizontal display axis
THETA = math.pi / 2
G = 0.05
H_THICK, H_THIN = 0.25, 0.005
INK, MUTED = '#0b0b0b', '#52514e'


def derivatives(h_over_r, psf, pixel, stamp):
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
        ImagePars(shape=(stamp, stamp), pixel_scale=pixel, indexing='ij'),
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
    x0 = jnp.array([0.0, 0.0, COSI, THETA, rio.FLUX, rio.RSCALE, h_over_r, 0.0, 0.0])

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
    d = {n: jac[..., i] for i, n in enumerate(names)}
    # disk-frame shear derivatives: g+ along the disk axes, gx at 45 deg
    c, sn = math.cos(2 * THETA), math.sin(2 * THETA)
    d['gp'] = c * d['g1'] + sn * d['g2']
    d['gx'] = -sn * d['g1'] + c * d['g2']
    return img, d


def unique_part(target, others):
    """Residual of target after least-squares projection onto the span of others (all flattened)."""
    if not others:
        return target
    a = np.stack([o.ravel() for o in others], axis=1)
    coef, *_ = np.linalg.lstsq(a, target.ravel(), rcond=None)
    return (target.ravel() - a @ coef).reshape(target.shape)


def main(pixel: float, stamp: int, snr: float, out_name: str):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    from kl_pipe.ensemble.mocks import _build_band_psf
    from kl_pipe.ensemble.spec import ObservationConfig

    config = ObservationConfig.from_yaml(rio.OBS_CONFIG)
    psf = _build_band_psf(config.band_psf[rio.BAND], rio.BAND, rio.Z, mock=True)
    renders = {
        'thin': derivatives(H_THIN, psf, pixel, stamp),
        'thick': derivatives(H_THICK, psf, pixel, stamp),
    }
    # per-pixel noise of the matched-filter convention at this image SNR, per disk
    noise = {k: math.sqrt(np.sum(img**2)) / snr for k, (img, _) in renders.items()}

    base = ['cosi', 'theta_int', 'flux', 'rscale']
    # (disk, row label, extra parameters the fit is allowed to adjust beyond tilt, rotation, size, flux)
    cases = [
        ('thin', f'thin disk ($h/R={H_THIN:g}$)\nthickness known, round', []),
        ('thick', f'thick disk ($h/R={H_THICK:g}$)\nthickness known, round', []),
        (
            'thick',
            f'thick disk ($h/R={H_THICK:g}$)\nthickness free, round',
            ['h_over_r'],
        ),
        (
            'thick',
            f'thick disk ($h/R={H_THICK:g}$)\nthickness free,\nintrinsic shape free',
            ['h_over_r', 'e1_int', 'e2_int'],
        ),
    ]
    comps = (('gp', 'gx', r'$g_+$'), ('gx', 'gp', r'$g_\times$'))
    cmap = LinearSegmentedColormap.from_list('div', ['#2a78d6', '#e4e3df', '#eb6834'])
    ext = np.array([-1, 1, -1, 1]) * stamp * pixel / 2
    grid = np.linspace(ext[0], ext[1], stamp)

    # change and leftover images for every case and component, in pixel-noise units for a 0.05 shear
    change, left = {}, {}
    for r, (disk, _, extra) in enumerate(cases):
        img, d = renders[disk]
        for comp, other, _ in comps:
            change[r, comp] = G * d[comp] / noise[disk]
            left[r, comp] = (
                G
                * unique_part(d[comp], [d[n] for n in base + [other] + extra])
                / noise[disk]
            )
    cmax = max(np.abs(v).max() for v in change.values())
    lmax = max(np.abs(v).max() for v in left.values())

    fig = plt.figure(figsize=(15, 13.5))
    gs = fig.add_gridspec(
        len(cases) + 1,
        5,
        height_ratios=[1] * len(cases) + [0.06],
        wspace=0.06,
        hspace=0.1,
    )
    axes = np.array(
        [[fig.add_subplot(gs[r, c]) for c in range(5)] for r in range(len(cases))]
    )
    cax_change = fig.add_subplot(gs[len(cases), 1:3])
    cax_left = fig.add_subplot(gs[len(cases), 3:5])
    titles = [
        rf'galaxy ($\cos i={COSI:g}$)',
        r'change from $g_+=0.05$',
        r'change from $g_\times=0.05$',
        r'$g_+$ left after refitting the rest',
        r'$g_\times$ left after refitting the rest',
    ]
    for r, (disk, label, _) in enumerate(cases):
        img = renders[disk][0]
        ax = axes[r, 0]
        ax.imshow(
            np.log10(np.maximum(img, img.max() * 1e-4)).T,
            origin='lower',
            cmap='Greys',
            extent=ext,
        )
        other_img = renders['thick' if disk == 'thin' else 'thin'][0]
        lv = np.array([0.02, 0.08, 0.3, 0.7])
        ax.contour(
            grid, grid, img.T, levels=img.max() * lv, colors='#2a78d6', linewidths=1.2
        )
        ax.contour(
            grid,
            grid,
            other_img.T,
            levels=other_img.max() * lv,
            colors='#eb6834',
            linewidths=1,
            linestyles='--',
        )
        ax.set_ylabel(label, color=INK, fontsize=10)
        from matplotlib.lines import Line2D

        this, other = ('thin', 'thick') if disk == 'thin' else ('thick', 'thin')
        ax.legend(
            handles=[
                Line2D([], [], color='#2a78d6', lw=1.2, label=f'{this} disk'),
                Line2D([], [], color='#eb6834', lw=1, ls='--', label=f'{other} disk'),
            ],
            loc='upper right',
            fontsize=7,
            frameon=True,
            framealpha=0.85,
            handlelength=1.6,
            borderpad=0.3,
        )
        for c, (comp, _, sym) in enumerate(comps):
            im_c = axes[r, 1 + c].imshow(
                change[r, comp].T,
                origin='lower',
                cmap=cmap,
                vmin=-cmax,
                vmax=cmax,
                extent=ext,
            )
            arr = left[r, comp]
            im_l = axes[r, 3 + c].imshow(
                arr.T, origin='lower', cmap=cmap, vmin=-lmax, vmax=lmax, extent=ext
            )
            nsig = math.sqrt(np.sum(arr**2))
            axes[r, 3 + c].text(
                0.03,
                0.04,
                rf'{sym}$=0.05$ seen at {nsig:.2f}$\sigma$',
                transform=axes[r, 3 + c].transAxes,
                color=INK,
                fontsize=9.5,
                bbox=dict(fc='white', ec='none', alpha=0.85),
            )
    for c, t in enumerate(titles):
        axes[0, c].set_title(t, color=INK, fontsize=10)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(
        im_c,
        cax=cax_change,
        orientation='horizontal',
        label='pixel change / pixel noise',
    )
    fig.colorbar(
        im_l, cax=cax_left, orientation='horizontal', label='pixel change / pixel noise'
    )
    fig.suptitle(
        'What the image alone tells the fit about shear\n'
        rf'image $\mathrm{{SNR}}={snr:g}$, half-light radius $0.3^{{\prime\prime}}$, Roman F158 PSF, ${pixel:g}^{{\prime\prime}}$ pixels',
        color=INK,
        fontsize=12,
        y=0.955,
    )
    fig.text(
        0.5,
        0.045,
        'Left after refitting the rest: the image change from a 0.05 shear minus the closest copy the other free '
        'parameters (tilt, rotation, size, flux, and anything the row frees) can make.\nOnly this leftover lets the '
        'fit separate shear; its quadrature sum in pixel-noise units is the significance N at which a 0.05 shear is '
        'detected, so the shear uncertainty is 0.05 / N.',
        ha='center',
        va='top',
        color=MUTED,
        fontsize=9.5,
    )
    out = HERE / 'out' / out_name
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f'wrote {out}')


if __name__ == '__main__':
    if '--hires' in sys.argv:
        main(pixel=0.0275, stamp=96, snr=5000.0, out_name='fig_explainer_hires.png')
    else:
        main(pixel=0.11, stamp=24, snr=1000.0, out_name='fig_explainer.png')
