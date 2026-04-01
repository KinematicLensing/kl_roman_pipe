#!/usr/bin/env python
"""Compute best-fit Spergel nu for each Sersic n.

Produces two lookup tables:
1. Face-on (cosi=1.0): flux-weighted L2 matching, galsim.Spergel vs galsim.Sersic
2. Inclined: flux-weighted L2 matching, our InclinedSpergelModel vs
   galsim.InclinedSersic, averaged over inclinations. GalSim Sersic renders
   are cached (constant during nu optimization).

Both use half-light-radius-matched profiles at unit flux.

Usage:
    python scripts/compute_nu_n_mapping.py
"""

import numpy as np
import galsim as gs
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from pathlib import Path

import jax

jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

from kl_pipe.intensity import InclinedSpergelModel
from kl_pipe.parameters import ImagePars


# ==============================================================================
# Face-on table (GalSim vs GalSim, fast)
# ==============================================================================


def faceon_mismatch(nu, n, hlr=5.0, npix=512, ps=0.1):
    """Flux-weighted L2: galsim.Spergel vs galsim.Sersic, face-on."""
    if nu <= -1.0 or nu > 10.0:
        return 1e10
    try:
        gsp = gs.GSParams(folding_threshold=1e-3, maxk_threshold=1e-3)
        sp = gs.Spergel(nu=nu, half_light_radius=hlr, flux=1.0, gsparams=gsp)
        se = gs.Sersic(n=n, half_light_radius=hlr, flux=1.0, gsparams=gsp)
        im_sp = sp.drawImage(nx=npix, ny=npix, scale=ps, method='no_pixel')
        im_se = se.drawImage(nx=npix, ny=npix, scale=ps, method='no_pixel')
        diff = im_sp.array - im_se.array
        weight = np.maximum(im_se.array, 0)
        return np.sum(diff**2 * weight)
    except Exception:
        return 1e10


# ==============================================================================
# Inclined table (our model vs GalSim, cached Sersic renders)
# ==============================================================================


def prerender_sersic_grid(n, cosi_values, hlr=2.0, npix=128, ps=0.11):
    """Pre-render GalSim InclinedSersic at all inclinations (cached)."""
    gsp = gs.GSParams(
        folding_threshold=1e-3, maxk_threshold=1e-3, maximum_fft_size=65536
    )
    images = {}
    for cosi in cosi_values:
        inc = gs.Angle(np.arccos(cosi), gs.radians)
        se = gs.InclinedSersic(
            n=n,
            inclination=inc,
            half_light_radius=hlr,
            scale_h_over_r=0.1,
            flux=1.0,
            gsparams=gsp,
        )
        im = se.drawImage(nx=npix, ny=npix, scale=ps, method='no_pixel')
        images[cosi] = im.array / ps**2
    return images


def inclined_mismatch_cached(
    nu, cached_sersic, cosi_values, hlr=2.0, npix=128, ps=0.11
):
    """Flux-weighted L2 using cached Sersic renders. Only our model re-renders."""
    if nu <= -1.0 or nu > 10.0:
        return 1e10

    model = InclinedSpergelModel()
    ip = ImagePars(shape=(npix, npix), pixel_scale=ps, indexing='ij')

    try:
        spergel_rscale = gs.Spergel(nu=nu, half_light_radius=hlr).scale_radius
    except Exception:
        return 1e10

    total = 0.0
    for cosi in cosi_values:
        try:
            gs_sb = cached_sersic[cosi]
            theta = jnp.array(
                [
                    cosi,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    spergel_rscale,
                    0.1,
                    nu,
                    0.0,
                    0.0,
                ]
            )
            our_sb = np.array(model.render_image(theta, image_pars=ip))
            diff = our_sb - gs_sb
            weight = np.maximum(gs_sb, 0)
            total += np.sum(diff**2 * weight)
        except Exception:
            total += 1e10

    return total / len(cosi_values)


# ==============================================================================
# Table computation
# ==============================================================================


def compute_faceon_table(n_grid):
    """Compute face-on nu(n) table."""
    nu_table = np.zeros_like(n_grid)
    print(f'\nComputing face-on mapping ({len(n_grid)} points)...')
    for i, n in enumerate(n_grid):
        result = minimize_scalar(
            faceon_mismatch,
            args=(n,),
            bounds=(-0.95, 8.0),
            method='bounded',
            options={'xatol': 1e-6, 'maxiter': 100},
        )
        nu_table[i] = result.x
        if i % 10 == 0:
            print(f'  [{i:3d}/{len(n_grid)}] n={n:.3f}: nu={result.x:+.6f}')

    idx_n1 = np.argmin(np.abs(n_grid - 1.0))
    nu_table[idx_n1] = 0.5
    return nu_table


def compute_inclined_table(n_grid, cosi_values):
    """Compute inclined nu(n) table with cached Sersic renders."""
    nu_table = np.zeros_like(n_grid)
    print(
        f'\nComputing inclined mapping ({len(n_grid)} points, '
        f'{len(cosi_values)} inclinations)...'
    )

    for i, n in enumerate(n_grid):
        # cache GalSim renders for this n (done ONCE per n)
        cached = prerender_sersic_grid(n, cosi_values)

        result = minimize_scalar(
            inclined_mismatch_cached,
            args=(cached, cosi_values),
            bounds=(-0.95, 8.0),
            method='bounded',
            options={'xatol': 1e-4, 'maxiter': 50},
        )
        nu_table[i] = result.x
        if i % 5 == 0:
            print(
                f'  [{i:3d}/{len(n_grid)}] n={n:.3f}: nu={result.x:+.6f} '
                f'(err={result.fun:.2e})'
            )

    idx_n1 = np.argmin(np.abs(n_grid - 1.0))
    nu_table[idx_n1] = 0.5
    return nu_table


def fmt_array(name, arr):
    """Format numpy array as paste-able Python code."""
    lines = [f'{name} = np.array([']
    row = '    '
    for i, v in enumerate(arr):
        entry = f'{v:.6f}'
        if i < len(arr) - 1:
            entry += ', '
        if len(row) + len(entry) > 85:
            lines.append(row)
            row = '    '
        row += entry
    lines.append(row)
    lines.append('])')
    return '\n'.join(lines)


def main():
    # coarser grid (35 points instead of 85)
    n_grid = np.concatenate(
        [
            np.linspace(0.3, 1.0, 15),
            np.linspace(1.1, 2.0, 10),
            np.linspace(2.2, 4.0, 10),
        ]
    )

    cosi_values = (0.2, 0.4, 0.6, 0.8, 1.0)

    nu_faceon = compute_faceon_table(n_grid)
    nu_inclined = compute_inclined_table(n_grid, cosi_values)

    # verification
    print(f'\nKey values:')
    print(f'  {"n":>5s}  {"nu_faceon":>10s}  {"nu_inclined":>12s}')
    for n_check in [0.5, 1.0, 2.0, 3.0, 4.0]:
        idx = np.argmin(np.abs(n_grid - n_check))
        print(
            f'  {n_grid[idx]:5.2f}  {nu_faceon[idx]:+10.4f}  '
            f'{nu_inclined[idx]:+12.4f}'
        )

    # roundtrip
    print(f'\nRoundtrip (face-on):')
    mask = nu_faceon < 7.9
    for n_check in [0.5, 1.0, 2.0, 4.0]:
        nu_interp = np.interp(n_check, n_grid, nu_faceon)
        n_back = np.interp(nu_interp, nu_faceon[mask][::-1], n_grid[mask][::-1])
        print(f'  n={n_check:.1f} -> nu={nu_interp:+.4f} -> n={n_back:.2f}')

    # plot
    out_dir = Path('tests/out/spergel')
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(n_grid, nu_faceon, 'b-o', label='face-on (2D)', lw=2, ms=4)
    ax.plot(n_grid, nu_inclined, 'r--s', label='inclined (3D, avg cosi)', lw=2, ms=4)
    ax.axhline(-0.6, color='grey', ls=':', alpha=0.4, label='canonical nu=-0.6')
    ax.plot(1.0, 0.5, 'ko', ms=10, zorder=5, label='exact: n=1, nu=0.5')
    ax.set_xlabel('Sersic n')
    ax.set_ylabel('Best-fit Spergel nu')
    ax.set_title(
        'Spergel nu ↔ Sersic n mapping\n' f'(flux-weighted L2, cosi={cosi_values})'
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.2, 4.2)
    plt.tight_layout()
    plt.savefig(out_dir / 'nu_n_mapping.png', dpi=150)
    plt.close()
    print(f'\nSaved plot to {out_dir / "nu_n_mapping.png"}')

    # print arrays
    print(f'\n# paste into intensity.py:\n')
    print(fmt_array('_N_GRID', n_grid))
    print()
    print(fmt_array('_NU_TABLE_FACEON', nu_faceon))
    print()
    print(fmt_array('_NU_TABLE_INCLINED', nu_inclined))


if __name__ == '__main__':
    main()
