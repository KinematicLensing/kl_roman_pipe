"""
Utilities for grism cross-code validation.

Loads config, maps parameters between codes, computes comparison metrics.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

# speed of light in km/s
C_KMS = 299792.458

CONFIG_PATH = Path(__file__).parent / 'test_params.yaml'


def load_config(path: Optional[str] = None) -> dict:
    """Load full config dict from YAML."""
    p = Path(path) if path else CONFIG_PATH
    with open(p) as f:
        return yaml.safe_load(f)


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into a copy of base.

    For nested dicts, merges keys. For everything else, override wins.
    """
    result = copy.deepcopy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = copy.deepcopy(v)
    return result


def get_test_params(test_name: str, config: Optional[dict] = None) -> dict:
    """Merge base_params + observation with test overrides.

    Returns flat dict with keys:
      - source params (z, vcirc, cosi, ...)
      - 'observation' sub-dict (pixel_scale, Nrow, Ncol, ...)
    """
    if config is None:
        config = load_config()

    if test_name not in config['tests']:
        raise KeyError(
            f"Unknown test '{test_name}'. "
            f"Available: {sorted(config['tests'].keys())}"
        )

    base = copy.deepcopy(config['base_params'])
    obs = copy.deepcopy(config['observation'])

    overrides = config['tests'][test_name] or {}

    # separate observation overrides from source param overrides
    obs_overrides = (
        overrides.pop('observation', None) if isinstance(overrides, dict) else None
    )

    # apply source param overrides
    if overrides:
        base.update(overrides)

    # apply observation overrides
    if obs_overrides:
        obs = _deep_merge(obs, obs_overrides)

    base['observation'] = obs
    return base


def _build_emission_lines(obs_lines):
    """Convert YAML line dicts to kl_pipe EmissionLine objects."""
    from kl_pipe.spectral import (
        EmissionLine,
        LineSpec,
        halpha_line,
    )

    lines = []
    for ldict in obs_lines:
        name = ldict['name']
        lam_rest = ldict['lambda_rest']

        if name == 'Ha':
            lines.append(halpha_line())
        else:
            # generic line with param_prefix = name
            spec = LineSpec(
                lambda_rest=lam_rest,
                name=name,
                param_prefix=name,
            )
            lines.append(EmissionLine(line_spec=spec, own_params=frozenset({'flux'})))

    return tuple(lines)


def get_kl_pipe_params(test_name: str, config: Optional[dict] = None) -> dict:
    """Build kl_pipe-native parameter dicts from config.

    Returns dict with keys:
      vel_pars, int_pars, spectral_pars, image_pars_kwargs, grism_pars_kwargs,
      lines, rendering
    """
    if config is None:
        config = load_config()

    p = get_test_params(test_name, config)
    obs = p['observation']
    rend = config['rendering']['kl_pipe']

    vel_pars = {
        'cosi': p['cosi'],
        'theta_int': p['theta_int'],
        'g1': p['g1'],
        'g2': p['g2'],
        'v0': p['v0'],
        'vcirc': p['vcirc'],
        'vel_rscale': p['vel_rscale'],
    }

    int_pars = {
        'cosi': p['cosi'],
        'theta_int': p['theta_int'],
        'g1': p['g1'],
        'g2': p['g2'],
        'flux': p['flux'],
        'int_rscale': p['int_rscale'],
        'int_h_over_r': p['int_h_over_r'],
        'int_x0': 0.0,
        'int_y0': 0.0,
    }

    # spectral pars: z, vel_dispersion, per-line flux + cont
    lines = _build_emission_lines(obs['lines'])
    spectral_pars = {
        'z': p['z'],
        'vel_dispersion': p['vel_dispersion'],
    }
    for line in lines:
        prefix = line.line_spec.param_prefix
        for own_p in sorted(line.own_params):
            if own_p == 'flux':
                spectral_pars[f'{prefix}_flux'] = p['flux']
            else:
                spectral_pars[f'{prefix}_{own_p}'] = p.get(own_p, 0.0)
        spectral_pars[f'{prefix}_cont'] = p['continuum']

    image_pars_kwargs = {
        'shape': (obs['Nrow'], obs['Ncol']),
        'pixel_scale': obs['pixel_scale'],
        'indexing': 'ij',
    }

    # auto-compute lambda_ref if null
    lambda_ref = obs.get('lambda_ref')
    if lambda_ref is None:
        # use first line's observed wavelength
        lambda_ref = obs['lines'][0]['lambda_rest'] * (1.0 + p['z'])

    grism_pars_kwargs = {
        'dispersion': obs['dispersion'],
        'lambda_ref': lambda_ref,
        'dispersion_angle': obs['dispersion_angle'],
    }

    psf_kwargs = None
    if obs.get('psf_fwhm') is not None:
        psf_kwargs = {
            'fwhm': obs['psf_fwhm'],
            'type': obs.get('psf_type', 'gaussian'),
        }

    # collect rest wavelengths for cube_pars construction
    line_lambdas_rest = tuple(l.line_spec.lambda_rest for l in lines)

    return {
        'vel_pars': vel_pars,
        'int_pars': int_pars,
        'spectral_pars': spectral_pars,
        'image_pars_kwargs': image_pars_kwargs,
        'grism_pars_kwargs': grism_pars_kwargs,
        'psf_kwargs': psf_kwargs,
        'lines': lines,
        'rendering': rend,
        'z': p['z'],
        'line_lambdas_rest': line_lambdas_rest,
    }


def get_geko_params(test_name: str, config: Optional[dict] = None) -> dict:
    """Apply kl_pipe -> geko parameter mapping.

    Returns geko-native dict with model params and observation context.

    Parameter mapping (verified via verify_geko_conventions.py):
      cosi        -> i (deg) = degrees(arccos(cosi))
      theta_int   -> PA (deg) = (90 - degrees(theta_int)) % 360
                     geko PA: CCW from +y (N). kl_pipe theta_int: CCW from +x.
                     Full 360 range needed: theta_int=0 vs pi produce different
                     velocity signs (sin(psi+pi)=-sin(psi)); % 180 collapses them.
      vcirc       -> Va (km/s), direct
      vel_rscale  -> rt (arcsec), direct; rt_pix = rt / pixel_scale
      int_rscale  -> re (arcsec), re = 1.678 * int_rscale for n=1; re_pix = re / pixel_scale
      flux        -> amplitude (integrated flux), direct. geko computes Ie
                     internally via flux_to_Ie(amplitude, n, re, ellip).
      v0          -> v0, direct
      vel_disp    -> sigma0, direct
      int_h_over_r -> q0 (approximate: q0 ~ int_h_over_r). geko uses Hubble
                      oblate-spheroid q0; kl_pipe uses sech^2 h_over_r.
                      Exact mapping derived by verify_geko_conventions.py.
      g1, g2      -> N/A (geko has no shear; fixed to 0 in kl_pipe)
    """
    if config is None:
        config = load_config()

    p = get_test_params(test_name, config)
    obs = p['observation']
    pixel_scale = obs['pixel_scale']

    cosi = p['cosi']
    i_deg = float(np.degrees(np.arccos(cosi)))

    # PA: kl_pipe theta_int (rad, CCW from +x) -> geko PA (deg, CCW from +y)
    PA_deg = float((90.0 - np.degrees(p['theta_int'])) % 360.0)

    # half-light radius for exponential disk: r_hl = 1.678 * r_scale
    re = 1.678 * p['int_rscale']

    # q0: approximate mapping. verify_geko_conventions.py quantifies the error.
    q0 = p['int_h_over_r']

    geko_pars = {
        'Va': p['vcirc'],
        'rt': p['vel_rscale'],
        'rt_pix': p['vel_rscale'] / pixel_scale,
        'i': i_deg,
        'PA': PA_deg,
        're': re,
        're_pix': re / pixel_scale,
        'amplitude': p['flux'],
        'v0': p['v0'],
        'sigma0': p['vel_dispersion'],
        'q0': q0,
        'z': p['z'],
        'n_sersic': p.get('n_sersic', 1.0),
        # observation context for rendering
        'pixel_scale': pixel_scale,
        'Nrow': obs['Nrow'],
        'Ncol': obs['Ncol'],
    }

    return geko_pars


def load_tolerances(tier: str = 'primary', config: Optional[dict] = None) -> dict:
    """Load tolerance dict for comparison tier."""
    if config is None:
        config = load_config()

    tols = config['tolerances']
    if tier not in tols:
        raise KeyError(
            f"Unknown tolerance tier '{tier}'. Available: {sorted(tols.keys())}"
        )

    return dict(tols[tier])


def load_reference_data(
    test_name: str,
    code: str,
    data_dir: str = 'tests/data/validation',
) -> dict:
    """Load .npz outputs for test + code.

    Returns dict with keys: cube, grism, vmap, imap, lambda_grid.
    Missing keys are None.
    """
    p = Path(data_dir) / code / f'{test_name}.npz'
    if not p.exists():
        raise FileNotFoundError(
            f"Reference data not found: {p}\n"
            f"Run render script for '{code}' first, or download from CyVerse."
        )

    data = np.load(p, allow_pickle=False)
    result = {}
    for key in ('cube', 'grism', 'vmap', 'imap', 'lambda_grid'):
        result[key] = data[key] if key in data else None
    return result


def _compute_centroid(img):
    """Compute flux-weighted centroid of 2D image. Returns (row, col)."""
    total = np.sum(img)
    if total == 0:
        return np.array([0.0, 0.0])

    rows = np.arange(img.shape[0])
    cols = np.arange(img.shape[1])
    row_c = np.sum(rows[:, None] * img) / total
    col_c = np.sum(cols[None, :] * img) / total
    return np.array([row_c, col_c])


def compare_images(img_a, img_b, tolerances: dict) -> dict:
    """Compute all comparison metrics between two 2D images.

    Returns {metric_name: (value, passed_bool)}.
    """
    a = np.asarray(img_a, dtype=np.float64)
    b = np.asarray(img_b, dtype=np.float64)

    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")

    results = {}

    # total flux
    sum_a = np.sum(a)
    sum_b = np.sum(b)
    if sum_a != 0:
        flux_rtol = abs(sum_a - sum_b) / abs(sum_a)
    else:
        flux_rtol = abs(sum_b)
    tol = tolerances.get('total_flux_rtol')
    results['total_flux_rtol'] = (
        flux_rtol,
        flux_rtol <= tol if tol is not None else None,
    )

    # peak pixel
    peak_a = np.max(a)
    peak_b = np.max(b)
    if peak_a != 0:
        peak_rtol = abs(peak_a - peak_b) / abs(peak_a)
    else:
        peak_rtol = abs(peak_b)
    tol = tolerances.get('peak_pixel_rtol')
    results['peak_pixel_rtol'] = (
        peak_rtol,
        peak_rtol <= tol if tol is not None else None,
    )

    # max pixel residual (peak-normalized)
    residual = np.abs(a - b)
    if peak_a != 0:
        max_resid = np.max(residual) / abs(peak_a)
    else:
        max_resid = np.max(residual)
    tol = tolerances.get('max_pixel_residual')
    results['max_pixel_residual'] = (
        max_resid,
        max_resid <= tol if tol is not None else None,
    )

    # rms residual
    rms_a = np.sqrt(np.mean(a**2))
    if rms_a != 0:
        rms_resid = np.sqrt(np.mean((a - b) ** 2)) / rms_a
    else:
        rms_resid = np.sqrt(np.mean((a - b) ** 2))
    tol = tolerances.get('rms_residual')
    results['rms_residual'] = (rms_resid, rms_resid <= tol if tol is not None else None)

    # spatial centroid
    c_a = _compute_centroid(a)
    c_b = _compute_centroid(b)
    centroid_dist = np.sqrt(np.sum((c_a - c_b) ** 2))
    tol = tolerances.get('spatial_centroid_pix')
    results['spatial_centroid_pix'] = (
        centroid_dist,
        centroid_dist <= tol if tol is not None else None,
    )

    return results


def compare_datacubes(
    cube_a,
    cube_b,
    lambda_grid,
    tolerances: dict,
) -> dict:
    """Datacube-specific comparison: total flux, peak, per-pixel, spectral centroid.

    cube shape: (Nrow, Ncol, Nlambda).
    Returns {metric_name: (value, passed_bool)}.
    """
    a = np.asarray(cube_a, dtype=np.float64)
    b = np.asarray(cube_b, dtype=np.float64)
    lam = np.asarray(lambda_grid, dtype=np.float64)

    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")

    results = {}

    # collapse to 2D for image-level metrics
    img_a = np.sum(a, axis=-1)
    img_b = np.sum(b, axis=-1)
    img_results = compare_images(img_a, img_b, tolerances)
    results.update(img_results)

    # per-spaxel spectral centroid (first moment)
    tol = tolerances.get('spectral_centroid_nm')
    if tol is not None:
        eps = 1e-30
        # first moment per spaxel: sum(I * lam) / sum(I)
        moment_a = np.sum(a * lam[None, None, :], axis=-1) / (np.sum(a, axis=-1) + eps)
        moment_b = np.sum(b * lam[None, None, :], axis=-1) / (np.sum(b, axis=-1) + eps)

        # only compare spaxels with significant signal
        signal_mask = np.sum(a, axis=-1) > 0.01 * np.max(np.sum(a, axis=-1))
        if np.any(signal_mask):
            centroid_diff = np.max(
                np.abs(moment_a[signal_mask] - moment_b[signal_mask])
            )
        else:
            centroid_diff = 0.0
        results['spectral_centroid_nm'] = (centroid_diff, centroid_diff <= tol)

    return results


def format_comparison_report(test_name: str, results: dict) -> str:
    """Format comparison results as a readable string."""
    lines = [f"Test: {test_name}"]
    for metric, (value, passed) in results.items():
        status = 'PASS' if passed else 'FAIL' if passed is not None else '---'
        lines.append(f"  {metric:30s} = {value:.6e}  [{status}]")
    return '\n'.join(lines)
