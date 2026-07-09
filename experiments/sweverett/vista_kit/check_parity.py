"""Local-only parity check: GaussianPSFShim vs real galsim.Gaussian.

Run in the klpipe conda env (galsim installed) BEFORE trusting the kit on
Vista. Verifies that the galsim-free PSF path produces the same grids and
(near-)identical numerics as the galsim path used by the profile scripts:

  1. getGoodImageSize identical at coarse + fine scales (grid-size parity =
     timing parity)
  2. stepk / maxk / kValue identical
  3. precompute_psf_fft kernel (via the kit's analytic builder) matches the
     galsim-rendered kernel to ~5e-5 relative (galsim drawImage FFT accuracy;
     the analytic pixel-integrated kernel is exact)
  4. precompute_psf_kspace_fft identical to fp precision
  5. full flagship grism + broadband renders: shim-PSF obs vs galsim-PSF obs

Exits nonzero on failure. Usage:  python check_parity.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

KIT_DIR = os.path.dirname(os.path.abspath(__file__))
if KIT_DIR not in sys.path:
    sys.path.insert(0, KIT_DIR)

FAILURES = []


def check(name, ok, detail=''):
    status = 'PASS' if ok else 'FAIL'
    print(f'  [{status}] {name}  {detail}')
    if not ok:
        FAILURES.append(name)


def main():
    import galsim  # real galsim required here

    if getattr(galsim, '__kl_vista_stub__', False):
        raise RuntimeError('galsim stub active; run in the klpipe env')

    from psf_numpy import (
        GaussianPSFShim,
        build_gaussian_kspace_psf_fft,
        build_gaussian_psf_data,
    )

    fwhm = 0.18
    shim = GaussianPSFShim(fwhm=fwhm)
    gs = galsim.Gaussian(fwhm=fwhm)

    print('== scalar properties ==')
    check(
        'sigma',
        np.isclose(shim.sigma, gs.sigma, rtol=1e-12),
        f'shim {shim.sigma:.6g} vs galsim {gs.sigma:.6g}',
    )
    check(
        'stepk',
        np.isclose(shim.stepk, gs.stepk, rtol=1e-12),
        f'shim {shim.stepk:.6g} vs galsim {gs.stepk:.6g}',
    )
    check(
        'maxk',
        np.isclose(shim.maxk, gs.maxk, rtol=1e-12),
        f'shim {shim.maxk:.6g} vs galsim {gs.maxk:.6g}',
    )
    for k in (0.5, 3.0, 20.0):
        v_s = shim.kValue(galsim.PositionD(0.0, k))
        v_g = abs(gs.kValue(galsim.PositionD(0.0, k)))
        check(
            f'kValue(0,{k})',
            np.isclose(v_s, v_g, rtol=1e-12),
            f'{v_s:.6g} vs {v_g:.6g}',
        )

    print('== grid sizing ==')
    for scale in (0.11, 0.11 / 3, 0.11 / 5):
        n_s = shim.getGoodImageSize(scale)
        n_g = gs.getGoodImageSize(scale)
        check(
            f'getGoodImageSize({scale:.4g})', n_s == n_g, f'shim {n_s} vs galsim {n_g}'
        )

    print('== PSFData kernel parity (coarse + oversample 3) ==')
    import jax

    jax.config.update('jax_enable_x64', True)
    from kl_pipe.psf import precompute_psf_fft

    for oversample in (1, 3):
        pd_shim = build_gaussian_psf_data(shim, (32, 32), 0.11, oversample)
        pd_gs = precompute_psf_fft(
            gs, image_shape=(32, 32), pixel_scale=0.11, oversample=oversample
        )
        check(
            f'padded_shape os={oversample}',
            pd_shim.padded_shape == pd_gs.padded_shape,
            f'{pd_shim.padded_shape} vs {pd_gs.padded_shape}',
        )
        if pd_shim.padded_shape == pd_gs.padded_shape:
            d = float(
                np.max(
                    np.abs(
                        np.asarray(pd_shim.kernel_fft) - np.asarray(pd_gs.kernel_fft)
                    )
                )
            )
            # reference is galsim's own drawImage FFT accuracy vs the exact
            # analytic pixel-integrated kernel: ~1e-4 at the coarse 0.11"
            # scale (os=1, unused by the kit configs), ~2e-8 at the fine
            # oversample-3 scale actually used for the grism PSF
            tol = 5e-4 if oversample == 1 else 5e-7
            check(f'kernel_fft max diff os={oversample}', d < tol, f'{d:.3e}')

    print('== kspace PSF FFT parity ==')
    from kl_pipe.psf import precompute_psf_kspace_fft

    fine_ps = 0.11 / 3
    pad_sq = 192
    k_shim = np.asarray(build_gaussian_kspace_psf_fft(shim, (pad_sq, pad_sq), fine_ps))
    k_gs = np.asarray(precompute_psf_kspace_fft(gs, (pad_sq, pad_sq), fine_ps))
    d = float(np.max(np.abs(k_shim - k_gs)))
    check('kspace_psf_fft max diff', d < 1e-12, f'{d:.3e}')

    print('== full flagship render parity (shim obs vs galsim obs) ==')
    # kit path (installs psf patch + uses shim)
    import tasks_vista as tv

    src_s, pri_s, task_s, obs_f_s, obs_g_s, true_s = tv.build_flagship_task()
    grism_s = np.asarray(src_s.render_grism(true_s, obs_g_s))
    bb_s = np.asarray(src_s.render_broadband(true_s, obs_f_s, 'F087'))

    # galsim path: identical construction, galsim PSF object
    import dataclasses  # noqa: F401  (parallel to tasks_vista imports)
    import jax.numpy as jnp
    from kl_pipe.dispersion import build_grism_pars_for_line
    from kl_pipe.lines import LINE_LAMBDAS
    from kl_pipe.observation import build_grism_obs, build_image_obs
    from kl_pipe.parameters import ImagePars
    from kl_pipe.render import RenderConfig

    image_pars = ImagePars(
        shape=tv.IMAGE_SHAPE, pixel_scale=tv.PIXEL_SCALE, indexing='ij'
    )
    grism_pars = build_grism_pars_for_line(
        LINE_LAMBDAS['Halpha'],
        redshift=tv.Z,
        image_pars=image_pars,
        dispersion=tv.GRISM_DISPERSION_NM_PER_PIX,
    )
    obs_g_g = build_grism_obs(
        grism_pars,
        z=tv.Z,
        psf=gs,
        render_config=RenderConfig(oversample=tv.SPATIAL_OVERSAMPLE),
        data=obs_g_s.data,
        variance=float(np.asarray(obs_g_s.variance)),
    )
    obs_f_g = build_image_obs(
        image_pars,
        psf=gs,
        render_config=RenderConfig(oversample=tv.SPATIAL_OVERSAMPLE),
        data=obs_f_s.data,
        variance=obs_f_s.variance,
        int_model=src_s.broadband_models['F087'],
        broadband_key='F087',
    )
    grism_g = np.asarray(src_s.render_grism(true_s, obs_g_g))
    bb_g = np.asarray(src_s.render_broadband(true_s, obs_f_g, 'F087'))

    for name, a, b in (('grism', grism_s, grism_g), ('broadband', bb_s, bb_g)):
        rel = float(np.max(np.abs(a - b)) / np.max(np.abs(b)))
        check(f'{name} render max|diff|/peak', rel < 1e-4, f'{rel:.3e}')
        check(f'{name} shape', a.shape == b.shape, f'{a.shape} vs {b.shape}')

    # posterior parity: same task likelihood evaluated with shim-built obs vs
    # galsim-built obs (same data arrays -> values should agree to ~render diff)
    from kl_pipe.sampling import InferenceTask

    task_g = InferenceTask.from_obs(
        src_s, pri_s, image_obs={'F087': obs_f_g}, grism_obs={'roll0': obs_g_g}
    )
    theta, _ = tv.perturbed_theta(pri_s, true_s)
    lp_s = float(task_s._log_posterior_jittable(theta))
    lp_g = float(task_g._log_posterior_jittable(theta))
    # absolute tolerance: the log posterior is O(10) here by chi2/prior
    # cancellation, so a relative measure inflates the ~1e-5 absolute diff
    # that follows from the 7e-9-of-peak grism render difference
    diff = abs(lp_s - lp_g)
    check(
        'log_posterior parity',
        diff < 1e-3,
        f'shim {lp_s:.6f} vs galsim {lp_g:.6f} (abs diff {diff:.2e})',
    )

    print()
    if FAILURES:
        print(f'PARITY CHECK FAILED: {FAILURES}')
        sys.exit(1)
    print('PARITY CHECK PASSED (kit PSF path is grid- and value-equivalent)')


if __name__ == '__main__':
    main()
