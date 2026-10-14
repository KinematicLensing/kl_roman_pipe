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
H_THICK, H_THIN = 0.25, 0.005
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
    renders = {'thin': derivatives(H_THIN, psf), 'thick': derivatives(H_THICK, psf)}
    # per-pixel noise of the matched-filter convention at this image SNR, per disk
    noise = {k: math.sqrt(np.sum(img**2)) / SNR for k, (img, _) in renders.items()}

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
    comps = (('g1', 'g2', r'$g_+$'), ('g2', 'g1', r'$g_\times$'))
    cmap = LinearSegmentedColormap.from_list('div', ['#2a78d6', '#e4e3df', '#eb6834'])
    ext = np.array([-1, 1, -1, 1]) * STAMP * 0.11 / 2
    grid = np.linspace(ext[0], ext[1], STAMP)

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
        r'$g_+$ change left after' '\n' r'refitting everything else',
        r'$g_\times$ change left after' '\n' r'refitting everything else',
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
        ax.contour(
            grid,
            grid,
            img.T,
            levels=img.max() * np.array([0.02, 0.08, 0.3, 0.7]),
            colors='#2a78d6',
            linewidths=1,
        )
        ax.set_ylabel(label, color=INK, fontsize=10)
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
                (
                    rf'{sym}$=0.05$ seen at {nsig:.2g}$\sigma$'
                    if nsig > 1e-3
                    else rf'{sym}$=0.05$ seen at 0$\sigma$'
                ),
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
        rf'image $\mathrm{{SNR}}={SNR:g}$, half-light radius $0.3^{{\prime\prime}}$, Roman F158 PSF, $0.11^{{\prime\prime}}$ pixels',
        color=INK,
        fontsize=12,
        y=0.955,
    )
    out = HERE / 'out' / 'fig_explainer.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
