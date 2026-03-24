#!/usr/bin/env python
"""
Render all validation tests using geko (astro-geko).

Part of the grism cross-code validation (see docs/validation/grism_validation_plan.md).
Loads 28 test configurations from test_params.yaml, maps parameters to geko's
native format, and renders noiseless datacube + grism + velocity/intensity maps.
Outputs are saved as .npz files for comparison against kl_pipe.

Each .npz contains keys: cube, grism, vmap, imap, lambda_grid.

geko requires square images (im_shape is a single int). We render max(Nrow, Ncol)
square and crop to (Nrow, Ncol) before saving.

Requires klpipe_validation env: conda activate klpipe_validation
  (has both kl_pipe and astro-geko)

Usage
-----
    # render all 28 tests
    python scripts/validation/render_geko.py

    # render a single test
    python scripts/validation/render_geko.py --test rotating_base

    # custom output directory
    python scripts/validation/render_geko.py --outdir /tmp/val

CLI Arguments
-------------
--test TEST_NAME   Render a single test by name (default: all).
--outdir DIR       Output directory (default: tests/data/validation/geko).
--config PATH      Config YAML path (default: scripts/validation/test_params.yaml).
"""

from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.signal import fftconvolve

from utils import get_geko_params, get_test_params, load_config

C_KMS = 299792.458


# =============================================================================
# geko imports — try multiple paths, give clear errors
# =============================================================================

_GEKO_AVAILABLE = False
_v_core_fn = None
_sersic_fn = None
_axis_ratio_fn = None
_flux_to_Ie_fn = None
_Grism_cls = None


def _try_import_geko():
    """Import geko functions. Called once at module level."""
    global _GEKO_AVAILABLE, _v_core_fn, _sersic_fn
    global _axis_ratio_fn, _flux_to_Ie_fn, _Grism_cls

    # velocity function
    for mod in ('geko.models', 'geko.galaxy_model', 'geko.velocity'):
        try:
            _v_core_fn = getattr(__import__(mod, fromlist=['_v_core']), '_v_core')
            break
        except (ImportError, AttributeError):
            continue
    if _v_core_fn is None:
        raise ImportError(
            "Cannot import geko velocity function (_v_core). "
            "Install: pip install astro-geko"
        )

    # intensity function
    for mod in (
        'geko.utils',
        'geko',
        'geko.models',
        'geko.galaxy_model',
        'geko.intensity',
    ):
        try:
            _sersic_fn = getattr(
                __import__(mod, fromlist=['sersic_profile']), 'sersic_profile'
            )
            break
        except (ImportError, AttributeError):
            continue

    # axis ratio
    for mod in ('geko.galaxy_model', 'geko.models', 'geko.utils'):
        try:
            _axis_ratio_fn = getattr(
                __import__(mod, fromlist=['compute_axis_ratio']), 'compute_axis_ratio'
            )
            break
        except (ImportError, AttributeError):
            continue

    # flux to Ie
    for mod in ('geko.galaxy_model', 'geko.models', 'geko.utils'):
        try:
            _flux_to_Ie_fn = getattr(
                __import__(mod, fromlist=['flux_to_Ie']), 'flux_to_Ie'
            )
            break
        except (ImportError, AttributeError):
            continue

    # Grism class
    try:
        from geko.grism import Grism

        _Grism_cls = Grism
    except ImportError:
        pass

    _GEKO_AVAILABLE = True


# =============================================================================
# Roman-compatible geko Grism wrapper
# =============================================================================


def make_roman_grism(im_shape, pixel_scale, lambda_grid_nm, psf_array=None):
    """Create geko Grism configured for Roman. Bypasses JWST __init__().

    Parameters
    ----------
    im_shape : int
        Square image side length in pixels.
    pixel_scale : float
        Arcsec/pixel.
    lambda_grid_nm : array
        1D wavelength grid in nm, one per pixel column. Length must equal im_shape.
    psf_array : array, optional
        2D PSF kernel (will be normalized). None = no PSF.

    Returns
    -------
    grism : geko.grism.Grism
        Configured for Roman linear dispersion.
    """
    if _Grism_cls is None:
        raise ImportError("geko.grism.Grism not available")

    grism = object.__new__(_Grism_cls)

    wave_space = np.array(lambda_grid_nm) / 1000.0  # nm -> um
    grism.wave_space = wave_space
    grism.wave_scale = float(np.diff(wave_space[:2])[0])  # um/pix
    grism.im_shape = im_shape
    grism.im_scale = pixel_scale
    grism.detector_scale = pixel_scale
    grism.factor = 1
    grism.wavelength = float(np.median(wave_space))

    # 2D wavelength array: each row has the same wavelength profile
    x_center = im_shape // 2
    x_indices = np.arange(im_shape)
    wave_offset = (x_indices - x_center) * grism.wave_scale
    grism.wave_array = grism.wavelength + wave_offset[None, :] * np.ones((im_shape, 1))

    # Roman grism LSF
    R_roman = 461 * grism.wavelength  # R(lambda) for Roman, lambda in um
    grism.sigma_lsf = grism.wavelength / (2.355 * R_roman)

    grism.index_min = 0
    grism.index_max = len(wave_space)

    # PSF
    if psf_array is not None:
        psf = np.asarray(psf_array, dtype=np.float64)
        grism.oversampled_PSF = psf / np.sum(psf)
    else:
        grism.oversampled_PSF = np.ones((1, 1))
    grism.PSF = grism.oversampled_PSF[:, :, np.newaxis]

    return grism


# =============================================================================
# standalone datacube construction
# =============================================================================


def _roman_R(lambda_nm):
    """Roman grism resolving power."""
    return 461.0 * lambda_nm / 1000.0


def build_datacube(vmap, imap, sigma_v_kms, z, lambda_rest_nm, lambda_grid_nm):
    """Build datacube from velocity/intensity maps and spectral parameters.

    Uses Gaussian emission with absorbed LSF (instrumental sigma added in
    quadrature), matching kl_pipe's SpectralModel convention.

    Parameters
    ----------
    vmap : (Nrow, Ncol) array
        LOS velocity map in km/s.
    imap : (Nrow, Ncol) array
        Intensity map (integrated flux per pixel).
    sigma_v_kms : float
        Intrinsic velocity dispersion in km/s.
    z : float
        Redshift.
    lambda_rest_nm : float
        Rest-frame line wavelength in nm.
    lambda_grid_nm : (Nlambda,) array
        Output wavelength grid in nm.

    Returns
    -------
    cube : (Nrow, Ncol, Nlambda) array
    """
    lam = np.asarray(lambda_grid_nm)
    dlam = np.abs(np.diff(lam[:2])[0]) if len(lam) > 1 else 1.0

    # per-pixel observed wavelength
    lambda_obs = lambda_rest_nm * (1.0 + z) * (1.0 + np.asarray(vmap) / C_KMS)

    # instrumental broadening (absorbed LSF)
    R_at_line = _roman_R(lambda_obs)
    sigma_inst_kms = C_KMS / (2.355 * R_at_line)
    sigma_eff_kms = np.sqrt(sigma_v_kms**2 + sigma_inst_kms**2)

    # convert to wavelength units
    sigma_lam = lambda_obs * sigma_eff_kms / C_KMS  # (Nrow, Ncol)

    # build cube: Gaussian profile at each pixel
    # cube[i,j,k] = imap[i,j] * gauss(lam[k], lambda_obs[i,j], sigma_lam[i,j])
    lam_3d = lam[np.newaxis, np.newaxis, :]  # (1, 1, Nlambda)
    lam_obs_3d = lambda_obs[:, :, np.newaxis]  # (Nrow, Ncol, 1)
    sigma_3d = sigma_lam[:, :, np.newaxis]  # (Nrow, Ncol, 1)

    gauss = np.exp(-0.5 * ((lam_3d - lam_obs_3d) / sigma_3d) ** 2)
    gauss /= sigma_3d * np.sqrt(2.0 * np.pi)  # normalize to unit integral

    cube = np.asarray(imap)[:, :, np.newaxis] * gauss

    return cube


def _gaussian_psf_kernel(fwhm_arcsec, pixel_scale, size=21):
    """Create normalized 2D Gaussian PSF kernel."""
    sigma_pix = fwhm_arcsec / (2.355 * pixel_scale)
    x = np.arange(size) - size // 2
    X, Y = np.meshgrid(x, x)
    kernel = np.exp(-(X**2 + Y**2) / (2 * sigma_pix**2))
    return kernel / kernel.sum()


def _convolve_cube_psf(cube, psf_kernel):
    """Convolve each wavelength slice of datacube with PSF."""
    result = np.empty_like(cube)
    for k in range(cube.shape[-1]):
        result[:, :, k] = fftconvolve(cube[:, :, k], psf_kernel, mode='same')
    return result


# =============================================================================
# velocity model helper
# =============================================================================


def _compute_axis_ratio(i_deg, q0):
    """Hubble formula for observed axis ratio. Fallback if geko function unavailable."""
    cosi = np.cos(np.radians(i_deg))
    return np.sqrt(cosi**2 * (1 - q0**2) + q0**2)


def _render_velocity_map(X_pix, Y_pix, geko_pars):
    """Render LOS velocity map using geko's _v_core."""
    return np.asarray(
        _v_core_fn(
            X_pix,
            Y_pix,
            geko_pars['PA'],
            geko_pars['i'],
            geko_pars['Va'],
            geko_pars['rt_pix'],
        )
    )


def _render_intensity_map(X_pix, Y_pix, geko_pars):
    """Render intensity map using geko's sersic_profile."""
    i_deg = geko_pars['i']
    q0 = geko_pars['q0']
    n = geko_pars['n_sersic']
    re_pix = geko_pars['re_pix']
    amplitude = geko_pars['amplitude']

    # observed axis ratio
    if _axis_ratio_fn is not None:
        q_obs = float(_axis_ratio_fn(i_deg, q0))
    else:
        q_obs = _compute_axis_ratio(i_deg, q0)
    ellip = 1.0 - q_obs

    # surface brightness at re
    if _flux_to_Ie_fn is not None:
        Ie = float(_flux_to_Ie_fn(amplitude, n, re_pix, ellip))
    else:
        raise ImportError(
            "geko flux_to_Ie not available. Required for correct Sersic "
            "normalization (naive formula is ~2x wrong for n=1)."
        )

    # sersic_profile(x, y, Ie, re, n, x0, y0, ellip, theta)
    # geko's Sersic angle: angle of major axis in pixel coords
    # for PA measured from +y CCW, the Sersic PA for imshow is different
    theta_sersic = np.radians(geko_pars['PA'])

    if _sersic_fn is None:
        raise ImportError("geko sersic_profile not available. Cannot render intensity.")

    return np.asarray(
        _sersic_fn(X_pix, Y_pix, Ie, re_pix, n, 0, 0, ellip, theta_sersic)
    )


# =============================================================================
# wavelength grid (match kl_pipe exactly)
# =============================================================================


def _build_lambda_grid(params, config):
    """Build datacube wavelength grid matching kl_pipe's convention."""
    obs = params['observation']
    z = params['z']
    lambda_rest = obs['lines'][0]['lambda_rest']
    dispersion = obs['dispersion']
    rend = config['rendering']['kl_pipe']
    velocity_window_kms = rend.get('velocity_window_kms', 3000.0)

    # try kl_pipe's exact grid builder
    try:
        from kl_pipe.dispersion import GrismPars
        from kl_pipe.parameters import ImagePars

        image_pars = ImagePars(
            shape=(obs['Nrow'], obs['Ncol']),
            pixel_scale=obs['pixel_scale'],
        )
        lambda_ref = obs.get('lambda_ref')
        if lambda_ref is None:
            lambda_ref = lambda_rest * (1.0 + z)
        grism_pars = GrismPars(
            image_pars=image_pars,
            dispersion=dispersion,
            lambda_ref=lambda_ref,
            dispersion_angle=obs.get('dispersion_angle', 0.0),
        )
        cube_pars = grism_pars.to_cube_pars(
            z=z,
            velocity_window_kms=velocity_window_kms,
            line_lambdas_rest=(lambda_rest,),
        )
        return np.asarray(cube_pars.lambda_grid), lambda_ref
    except ImportError:
        pass

    # fallback: compute independently
    lambda_obs = lambda_rest * (1.0 + z)
    half_width = lambda_obs * velocity_window_kms / C_KMS
    n_bins = int(np.ceil(2 * half_width / dispersion)) + 1
    if n_bins % 2 == 0:
        n_bins += 1
    lambda_grid = lambda_obs + (np.arange(n_bins) - n_bins // 2) * dispersion
    return lambda_grid, lambda_obs


# =============================================================================
# main render
# =============================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Render geko validation outputs')
    parser.add_argument(
        '--test',
        type=str,
        default=None,
        help='Render single test (default: all)',
    )
    parser.add_argument(
        '--outdir',
        type=str,
        default='tests/data/validation/geko',
        help='Output directory',
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Config YAML path (default: scripts/validation/test_params.yaml)',
    )
    return parser


def render_test(test_name: str, config: dict, outdir: Path) -> None:
    """Render datacube + grism for a single test via geko."""
    geko_pars = get_geko_params(test_name, config)
    params = get_test_params(test_name, config)
    obs = params['observation']

    Nrow = obs['Nrow']
    Ncol = obs['Ncol']
    pixel_scale = obs['pixel_scale']
    dispersion = obs['dispersion']

    # geko requires square images
    im_shape = max(Nrow, Ncol)

    # build centered pixel grid (origin at center)
    center = (im_shape - 1) / 2.0
    x_1d = np.arange(im_shape) - center
    y_1d = np.arange(im_shape) - center
    X_pix, Y_pix = np.meshgrid(x_1d, y_1d, indexing='xy')

    # --- render velocity map ---
    vmap_sq = _render_velocity_map(X_pix, Y_pix, geko_pars)

    # --- render intensity map ---
    imap_sq = _render_intensity_map(X_pix, Y_pix, geko_pars)

    # --- crop square -> rectangular ---
    if Nrow < im_shape:
        row_start = (im_shape - Nrow) // 2
        row_end = row_start + Nrow
    else:
        row_start, row_end = 0, Nrow
    if Ncol < im_shape:
        col_start = (im_shape - Ncol) // 2
        col_end = col_start + Ncol
    else:
        col_start, col_end = 0, Ncol

    vmap = vmap_sq[row_start:row_end, col_start:col_end]
    imap = imap_sq[row_start:row_end, col_start:col_end]

    # --- wavelength grid ---
    lambda_grid, lambda_ref = _build_lambda_grid(params, config)
    lambda_rest = obs['lines'][0]['lambda_rest']

    # --- build datacube ---
    cube = build_datacube(
        vmap,
        imap,
        sigma_v_kms=params['vel_dispersion'],
        z=params['z'],
        lambda_rest_nm=lambda_rest,
        lambda_grid_nm=lambda_grid,
    )

    # --- PSF convolution ---
    psf_kernel = None
    if obs.get('psf_fwhm') is not None:
        psf_kernel = _gaussian_psf_kernel(obs['psf_fwhm'], pixel_scale)
        cube = _convolve_cube_psf(cube, psf_kernel)

    # --- grism image ---
    grism = None

    # try geko's Grism.disperse
    if _Grism_cls is not None:
        try:
            # build wavelength grid for square grism (im_shape columns)
            grism_lam = lambda_ref + (np.arange(im_shape) - im_shape // 2) * dispersion
            psf_arr = psf_kernel if psf_kernel is not None else None
            grism_obj = make_roman_grism(im_shape, pixel_scale, grism_lam, psf_arr)
            sigma_map = np.full((im_shape, im_shape), params['vel_dispersion'])
            grism_sq = np.asarray(grism_obj.disperse(imap_sq, vmap_sq, sigma_map))
            grism = grism_sq[row_start:row_end, col_start:col_end]
            print(f"  grism via geko.Grism.disperse")
        except Exception as e:
            raise RuntimeError(
                f"geko Grism.disperse failed for '{test_name}': {e}"
            ) from e

    if grism is None:
        raise RuntimeError(
            f"geko Grism.disperse() unavailable for '{test_name}'. "
            "Install astro-geko with grism support or fix the error above."
        )

    # --- convert to kl_pipe conventions before saving ---
    # geko renders in pixel coords (pix^-2); kl_pipe uses arcsec^-2.
    # Convention: all .npz outputs use kl_pipe units:
    #   imap: arcsec^-2 (surface brightness)
    #   vmap: km/s (same in both codes)
    #   grism: TBD — may need similar conversion; diagnosed via comparison
    #   cube: TBD — diagnostic only
    imap_arcsec = imap / pixel_scale**2

    # --- save ---
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / f'{test_name}.npz'
    np.savez(
        outpath,
        cube=np.asarray(cube),
        grism=np.asarray(grism),
        vmap=np.asarray(vmap),
        imap=np.asarray(imap_arcsec),
        lambda_grid=np.asarray(lambda_grid),
    )
    print(f"  saved {outpath}")


def main():
    args = build_parser().parse_args()

    # import geko once
    _try_import_geko()

    config = load_config(args.config)
    outdir = Path(args.outdir)

    tests = [args.test] if args.test else sorted(config['tests'].keys())

    print(f"Rendering {len(tests)} tests to {outdir}/")
    t0 = time.time()

    for test_name in tests:
        t1 = time.time()
        print(f"[{test_name}]")
        try:
            render_test(test_name, config, outdir)
            print(f"  {time.time() - t1:.1f}s")
        except Exception as e:
            print(f"  ERROR: {e}")

    print(f"\nDone. {len(tests)} tests in {time.time() - t0:.1f}s")


if __name__ == '__main__':
    main()
