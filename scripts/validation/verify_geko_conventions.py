#!/usr/bin/env python
"""
Verify kl_pipe <-> geko parameter conventions before cross-code comparison.

Three checks:
  A. PA mapping: velocity maps at theta_int=pi/4, 3pi/4, pi + sign-flip test
  B. q0 <-> int_h_over_r: projected ellipticity sweep
  C. Grid centering: intensity peak alignment after square->rect crop

Run in the klpipe_validation env (has both kl_pipe and geko):
    conda run -n klpipe_validation python scripts/validation/verify_geko_conventions.py

Saves diagnostic plots to tests/out/convention-checks/.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

# add scripts/validation to path
_SCRIPTS_DIR = Path(__file__).parent
sys.path.insert(0, str(_SCRIPTS_DIR))

C_KMS = 299792.458
OUT_DIR = Path(__file__).parent.parent.parent / 'tests' / 'out' / 'convention-checks'


def _ensure_outdir():
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def _import_geko():
    """Import geko functions; raise helpful error if missing."""
    # _v_core: try geko.models (v1.0+), then legacy geko.galaxy_model
    try:
        from geko.models import _v_core
    except ImportError:
        try:
            from geko.galaxy_model import _v_core
        except ImportError:
            raise ImportError(
                "Cannot import geko velocity function. "
                "Install: pip install astro-geko. "
                "Or run in klpipe_validation env: conda activate klpipe_validation"
            )
    # sersic_profile: lives in geko.utils (v1.0+), re-exported at top-level
    try:
        from geko.utils import sersic_profile
    except ImportError:
        try:
            from geko import sersic_profile
        except ImportError:
            try:
                from geko.models import sersic_profile
            except ImportError:
                sersic_profile = None
    return _v_core, sersic_profile


def _check_pa_single(
    theta_int,
    vel_model,
    image_pars,
    X,
    Y,
    _v_core,
    pixel_scale,
    im_shape,
    cosi,
    vcirc,
    vel_rscale,
):
    """Compare kl_pipe vs geko velocity maps at a single theta_int.

    Returns (max_resid, vmap_kl, vmap_geko, PA_deg).
    """
    import jax.numpy as jnp

    theta_vel = jnp.array([cosi, theta_int, 0.0, 0.0, 0.0, vcirc, vel_rscale])
    vmap_kl = np.asarray(vel_model(theta_vel, 'obs', X, Y))

    center = (im_shape - 1) / 2.0
    x_1d = np.arange(im_shape) - center
    y_1d = np.arange(im_shape) - center
    X_pix, Y_pix = np.meshgrid(x_1d, y_1d, indexing='xy')

    PA_deg = (90.0 - np.degrees(theta_int)) % 360.0
    i_deg = np.degrees(np.arccos(cosi))
    rt_pix = vel_rscale / pixel_scale

    vmap_geko = np.asarray(_v_core(X_pix, Y_pix, PA_deg, i_deg, vcirc, rt_pix))

    peak = max(np.max(np.abs(vmap_kl)), np.max(np.abs(vmap_geko)), 1e-10)
    max_resid = np.max(np.abs(vmap_kl - vmap_geko)) / peak

    return max_resid, vmap_kl, vmap_geko, PA_deg


def check_pa_mapping():
    """A. Compare velocity maps at multiple theta_int values.

    Tests pi/4 (standard), 3pi/4 (third quadrant PA), and pi (wraps to PA=270
    under % 360; would collide with theta_int=0 under the old % 180).
    """
    import jax

    jax.config.update('jax_enable_x64', True)
    from kl_pipe.parameters import ImagePars
    from kl_pipe.utils import build_map_grid_from_image_pars
    from kl_pipe.velocity import CenteredVelocityModel

    _v_core, _ = _import_geko()

    pixel_scale = 0.11
    im_shape = 48
    cosi = 0.5
    vcirc = 200.0
    vel_rscale = 0.5  # arcsec

    image_pars = ImagePars(
        shape=(im_shape, im_shape), indexing='ij', pixel_scale=pixel_scale
    )
    X, Y = build_map_grid_from_image_pars(image_pars)
    vel_model = CenteredVelocityModel()

    test_angles = {
        'pi/4': np.pi / 4,
        '3pi/4': 3 * np.pi / 4,
        'pi': np.pi,
    }

    all_pass = True
    results_for_plot = {}

    print("[A] PA mapping check (multiple angles):")
    for label, theta_int in test_angles.items():
        resid, vkl, vge, pa = _check_pa_single(
            theta_int,
            vel_model,
            image_pars,
            X,
            Y,
            _v_core,
            pixel_scale,
            im_shape,
            cosi,
            vcirc,
            vel_rscale,
        )
        passed = resid < 0.001
        all_pass = all_pass and passed
        results_for_plot[label] = (resid, vkl, vge, pa)

        status = "PASS" if passed else "FAIL (threshold 0.1%)"
        print(
            f"    theta_int={label:6s} -> PA={pa:6.1f} deg | "
            f"kl=[{vkl.min():.1f},{vkl.max():.1f}] "
            f"geko=[{vge.min():.1f},{vge.max():.1f}] | "
            f"resid={resid:.2e} {status}"
        )

    # verify theta_int=0 and theta_int=pi produce different velocity signs
    theta_0 = 0.0
    r0, vkl_0, _, pa0 = _check_pa_single(
        theta_0,
        vel_model,
        image_pars,
        X,
        Y,
        _v_core,
        pixel_scale,
        im_shape,
        cosi,
        vcirc,
        vel_rscale,
    )
    vkl_pi = results_for_plot['pi'][1]

    # at theta_int=pi the velocity field should be sign-flipped vs theta_int=0
    sign_check = np.max(np.abs(vkl_0 + vkl_pi))
    sign_ok = sign_check < 0.001 * max(np.max(np.abs(vkl_0)), 1e-10)
    print(
        f"    sign check: |v(0) + v(pi)| max = {sign_check:.2e} "
        f"{'PASS' if sign_ok else 'FAIL'}"
    )
    if not sign_ok:
        all_pass = False

    # save diagnostic plots
    _ensure_outdir()
    try:
        import matplotlib

        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        n_angles = len(results_for_plot)
        fig, axes = plt.subplots(n_angles, 3, figsize=(15, 4 * n_angles))
        if n_angles == 1:
            axes = axes[np.newaxis, :]

        for row, (label, (resid, vkl, vge, pa)) in enumerate(results_for_plot.items()):
            vmax = max(np.max(np.abs(vkl)), np.max(np.abs(vge)))
            axes[row, 0].imshow(
                vkl, origin='lower', cmap='RdBu_r', vmin=-vmax, vmax=vmax
            )
            axes[row, 0].set_title(f'kl_pipe (theta_int={label})')
            axes[row, 1].imshow(
                vge, origin='lower', cmap='RdBu_r', vmin=-vmax, vmax=vmax
            )
            axes[row, 1].set_title(f'geko (PA={pa:.1f} deg)')
            r = vkl - vge
            im = axes[row, 2].imshow(r, origin='lower', cmap='RdBu_r')
            axes[row, 2].set_title(f'residual (max={resid:.2e})')
            plt.colorbar(im, ax=axes[row, 2])

        plt.tight_layout()
        fig.savefig(OUT_DIR / 'pa_mapping.png', dpi=150)
        plt.close(fig)
        print(f"    plot saved: {OUT_DIR / 'pa_mapping.png'}")
    except ImportError:
        pass

    return all_pass


def check_q0_mapping():
    """B. Compare projected ellipticity: sweep geko q0 to match kl_pipe."""
    import jax

    jax.config.update('jax_enable_x64', True)
    from kl_pipe.intensity import InclinedExponentialModel
    from kl_pipe.parameters import ImagePars

    _, sersic_profile = _import_geko()
    if sersic_profile is None:
        print("[B] q0 mapping check: SKIPPED (geko sersic_profile not found)")
        return True

    pixel_scale = 0.11
    im_shape = 48
    cosi = 0.5
    int_h_over_r = 0.1
    int_rscale = 0.3  # arcsec

    # kl_pipe: render intensity map, measure second moments
    import jax.numpy as jnp

    image_pars = ImagePars(
        shape=(im_shape, im_shape), indexing='ij', pixel_scale=pixel_scale
    )
    int_model = InclinedExponentialModel()
    theta_int_arr = jnp.array(
        [
            cosi,
            0.0,
            0.0,
            0.0,  # cosi, theta_int, g1, g2
            100.0,
            int_rscale,
            int_h_over_r,
            0.0,
            0.0,  # flux, int_rscale, int_h_over_r, x0, y0
        ]
    )
    imap_kl = np.asarray(int_model.render_unconvolved(theta_int_arr, image_pars))

    def _ellip_from_moments(img):
        """Measure ellipticity from unweighted second moments."""
        total = np.sum(img)
        if total == 0:
            return 0.0
        rows, cols = np.indices(img.shape)
        rc = np.sum(rows * img) / total
        cc = np.sum(cols * img) / total
        Qxx = np.sum((cols - cc) ** 2 * img) / total
        Qyy = np.sum((rows - rc) ** 2 * img) / total
        Qxy = np.sum((cols - cc) * (rows - rc) * img) / total
        e1 = (Qxx - Qyy) / (Qxx + Qyy)
        e2 = 2 * Qxy / (Qxx + Qyy)
        return np.sqrt(e1**2 + e2**2)

    ellip_kl = _ellip_from_moments(imap_kl)

    # sweep geko q0 to match
    i_deg = np.degrees(np.arccos(cosi))
    re_pix = 1.678 * int_rscale / pixel_scale
    center = (im_shape - 1) / 2.0
    x_1d = np.arange(im_shape) - center
    X_pix, Y_pix = np.meshgrid(x_1d, x_1d, indexing='xy')

    q0_values = np.linspace(0.01, 0.5, 50)
    ellip_geko = []
    for q0 in q0_values:
        q_obs = np.sqrt(cosi**2 * (1 - q0**2) + q0**2)
        e = 1.0 - q_obs
        Ie = 100.0 / (2 * np.pi * re_pix**2 * q_obs)
        imap_ge = np.asarray(
            sersic_profile(X_pix, Y_pix, Ie, re_pix, 1.0, 0, 0, e, 0.0)
        )
        ellip_geko.append(_ellip_from_moments(imap_ge))
    ellip_geko = np.array(ellip_geko)

    # find best-fit q0
    idx = np.argmin(np.abs(ellip_geko - ellip_kl))
    q0_best = q0_values[idx]
    ellip_err = abs(ellip_geko[idx] - ellip_kl)

    # compare with naive q0 = int_h_over_r
    naive_err = abs(q0_values[np.argmin(np.abs(q0_values - int_h_over_r))] - q0_best)

    print(f"[B] q0 <-> int_h_over_r mapping (cosi={cosi}):")
    print(f"    kl_pipe projected ellipticity: {ellip_kl:.4f}")
    print(f"    best-fit geko q0: {q0_best:.4f} (ellip match err={ellip_err:.4f})")
    print(f"    int_h_over_r: {int_h_over_r:.4f}")
    print(f"    |q0_best - int_h_over_r|: {naive_err:.4f}")
    if naive_err < 0.02:
        print(f"    q0 ~ int_h_over_r is adequate (|diff| < 0.02)")
    else:
        print(f"    WARNING: q0 != int_h_over_r — need exact mapping")

    # save plot
    _ensure_outdir()
    try:
        import matplotlib

        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(q0_values, ellip_geko, 'b-', label='geko')
        ax.axhline(ellip_kl, color='r', ls='--', label=f'kl_pipe (h/r={int_h_over_r})')
        ax.axvline(q0_best, color='g', ls=':', label=f'best q0={q0_best:.3f}')
        ax.axvline(int_h_over_r, color='orange', ls=':', label=f'h/r={int_h_over_r}')
        ax.set_xlabel('geko q0')
        ax.set_ylabel('projected ellipticity')
        ax.legend()
        ax.set_title(f'q0 mapping at cosi={cosi}')
        fig.savefig(OUT_DIR / 'q0_mapping.png', dpi=150)
        plt.close(fig)
        print(f"    plot saved: {OUT_DIR / 'q0_mapping.png'}")
    except ImportError:
        pass

    return naive_err < 0.02


def check_grid_centering():
    """C. Verify intensity peak alignment after square->rect crop."""
    import jax

    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp
    from kl_pipe.intensity import InclinedExponentialModel
    from kl_pipe.parameters import ImagePars

    _, sersic_profile = _import_geko()
    if sersic_profile is None:
        print("[C] Grid centering check: SKIPPED (geko sersic_profile not found)")
        return True

    pixel_scale = 0.11
    Nrow, Ncol = 32, 48
    im_shape = max(Nrow, Ncol)  # 48

    # kl_pipe: 32x48
    image_pars = ImagePars(shape=(Nrow, Ncol), indexing='ij', pixel_scale=pixel_scale)
    int_model = InclinedExponentialModel()
    theta_int_arr = jnp.array(
        [
            0.5,
            0.0,
            0.0,
            0.0,  # cosi, theta_int, g1, g2
            100.0,
            0.3,
            0.1,
            0.0,
            0.0,  # flux, int_rscale, int_h_over_r, x0, y0
        ]
    )
    imap_kl = np.asarray(int_model.render_unconvolved(theta_int_arr, image_pars))
    peak_kl = np.unravel_index(np.argmax(imap_kl), imap_kl.shape)

    # geko: 48x48, then crop
    center = (im_shape - 1) / 2.0
    x_1d = np.arange(im_shape) - center
    X_pix, Y_pix = np.meshgrid(x_1d, x_1d, indexing='xy')
    cosi = 0.5
    q0 = 0.1
    i_deg = np.degrees(np.arccos(cosi))
    q_obs = np.sqrt(cosi**2 * (1 - q0**2) + q0**2)
    e = 1.0 - q_obs
    re_pix = 1.678 * 0.3 / pixel_scale
    Ie = 100.0 / (2 * np.pi * re_pix**2 * q_obs)
    imap_sq = np.asarray(sersic_profile(X_pix, Y_pix, Ie, re_pix, 1.0, 0, 0, e, 0.0))

    # crop: central Nrow rows
    row_start = (im_shape - Nrow) // 2  # (48-32)//2 = 8
    imap_crop = imap_sq[row_start : row_start + Nrow, :]
    peak_crop = np.unravel_index(np.argmax(imap_crop), imap_crop.shape)

    row_offset = abs(peak_kl[0] - peak_crop[0])
    col_offset = abs(peak_kl[1] - peak_crop[1])

    print(f"[C] Grid centering check:")
    print(f"    kl_pipe peak (32x48): row={peak_kl[0]}, col={peak_kl[1]}")
    print(f"    geko cropped peak:    row={peak_crop[0]}, col={peak_crop[1]}")
    print(f"    offset: ({row_offset}, {col_offset}) pixels")

    if row_offset == 0 and col_offset == 0:
        print(f"    PASS (exact alignment)")
    elif row_offset <= 1 and col_offset <= 1:
        print(f"    WARNING: 1-pixel offset — check sub-pixel centering convention")
    else:
        print(f"    FAIL: large offset — crop indices may be wrong")

    # flux comparison
    if np.max(imap_kl) > 0:
        max_resid = np.max(np.abs(imap_crop - imap_kl)) / np.max(imap_kl)
        print(f"    max |residual| / peak = {max_resid:.4f}")
        print(f"    (expected ~1-3% from intensity model differences)")

    # save plot
    _ensure_outdir()
    try:
        import matplotlib

        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        axes[0].imshow(imap_kl, origin='lower')
        axes[0].plot(peak_kl[1], peak_kl[0], 'rx', ms=10)
        axes[0].set_title('kl_pipe (32x48)')
        axes[1].imshow(imap_crop, origin='lower')
        axes[1].plot(peak_crop[1], peak_crop[0], 'rx', ms=10)
        axes[1].set_title(f'geko cropped (32x48)')
        if imap_kl.shape == imap_crop.shape:
            resid = imap_crop - imap_kl
            im = axes[2].imshow(resid, origin='lower', cmap='RdBu_r')
            axes[2].set_title('residual')
            plt.colorbar(im, ax=axes[2])
        plt.tight_layout()
        fig.savefig(OUT_DIR / 'grid_centering.png', dpi=150)
        plt.close(fig)
        print(f"    plot saved: {OUT_DIR / 'grid_centering.png'}")
    except ImportError:
        pass

    return row_offset <= 1 and col_offset <= 1


def main():
    print("=" * 60)
    print("geko convention verification")
    print("=" * 60)
    print()

    results = {}
    results['PA'] = check_pa_mapping()
    print()
    results['q0'] = check_q0_mapping()
    print()
    results['grid'] = check_grid_centering()
    print()

    print("=" * 60)
    all_pass = all(results.values())
    for name, passed in results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL / WARN'}")
    print(f"\nOverall: {'ALL PASS' if all_pass else 'ACTION NEEDED'}")
    return 0 if all_pass else 1


if __name__ == '__main__':
    sys.exit(main())
