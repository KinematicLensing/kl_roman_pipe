"""
Shear calibration statistics for ensemble fits.

Pure post-processing functions consumed by the ensemble analysis step: frame
rotation of shear components, linear shear-bias fits (m, c), and effective
shape-noise aggregation. All functions operate on plain numpy arrays (no JAX,
no dispatch logic) so they can run anywhere the collated results table lives.

Conventions
-----------
- theta_int is the position angle in radians from +x (Cartesian), matching
  the model convention in kl_pipe.transformation.
- Galaxy-frame shear components: g+ is the component along/across the major
  axis, gx at 45 degrees, obtained by rotating (g1, g2) by 2*theta_int.
- Shear bias follows the standard linear model g_meas = (1 + m) * g_true + c.
- Effective shape noise per galaxy follows Pranjal+2022 Eq. 20:
  sigma_eps_j = sqrt[(sigma_g+^2 + sigma_gx^2) / 2]; the ensemble value is the
  inverse-variance-weighted effective per-galaxy noise.
"""

from __future__ import annotations

from typing import NamedTuple, Optional, Tuple

import numpy as np


def rotate_to_galaxy_frame(
    g1: np.ndarray, g2: np.ndarray, theta_int: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rotate sky-frame shear components into the galaxy frame.

    Shear is a spin-2 quantity, so the frame rotation uses twice the position
    angle:

        g+ =  g1 * cos(2*theta) + g2 * sin(2*theta)
        gx = -g1 * sin(2*theta) + g2 * cos(2*theta)

    Parameters
    ----------
    g1, g2 : np.ndarray
        Sky-frame shear components (dimensionless).
    theta_int : np.ndarray
        Galaxy position angle in radians from +x.

    Returns
    -------
    g_plus, g_cross : np.ndarray
        Galaxy-frame tangential and cross shear components.
    """
    g1 = np.asarray(g1, dtype=float)
    g2 = np.asarray(g2, dtype=float)
    theta_int = np.asarray(theta_int, dtype=float)

    cos2t = np.cos(2.0 * theta_int)
    sin2t = np.sin(2.0 * theta_int)
    g_plus = g1 * cos2t + g2 * sin2t
    g_cross = -g1 * sin2t + g2 * cos2t
    return g_plus, g_cross


class ShearBiasResult(NamedTuple):
    """Linear shear-bias fit result: g_meas = (1 + m) * g_true + c."""

    m: float
    c: float
    sigma_m: float
    sigma_c: float


def measure_shear_bias(
    g_true: np.ndarray,
    g_meas: np.ndarray,
    sigma_meas: Optional[np.ndarray] = None,
) -> ShearBiasResult:
    """
    Fit the linear shear-bias model g_meas = (1 + m) * g_true + c.

    Weighted least squares when per-point measurement errors are supplied
    (parameter errors from the exact WLS covariance); ordinary least squares
    otherwise (parameter errors from the residual variance, n - 2 dof).

    Parameters
    ----------
    g_true : np.ndarray
        Injected shear values (one component), shape (n,).
    g_meas : np.ndarray
        Recovered shear values (posterior point estimates), shape (n,).
    sigma_meas : np.ndarray, optional
        Per-point 1-sigma errors on g_meas. If given, used as WLS weights
        1/sigma^2 and propagated into (sigma_m, sigma_c).

    Returns
    -------
    ShearBiasResult
        (m, c, sigma_m, sigma_c) with m the multiplicative and c the additive
        bias.

    Raises
    ------
    ValueError
        If fewer than 3 points, mismatched shapes, non-positive errors, or
        g_true has no spread (slope undefined).
    """
    g_true = np.asarray(g_true, dtype=float).ravel()
    g_meas = np.asarray(g_meas, dtype=float).ravel()
    if g_true.shape != g_meas.shape:
        raise ValueError(f"g_true shape {g_true.shape} != g_meas shape {g_meas.shape}")
    n = g_true.size
    if n < 3:
        raise ValueError(f"need >= 3 points for a 2-parameter fit, got {n}")
    if np.ptp(g_true) == 0.0:
        raise ValueError("g_true values are all identical; slope is undefined")

    if sigma_meas is not None:
        sigma_meas = np.asarray(sigma_meas, dtype=float).ravel()
        if sigma_meas.shape != g_true.shape:
            raise ValueError(
                f"sigma_meas shape {sigma_meas.shape} != g_true shape {g_true.shape}"
            )
        if np.any(sigma_meas <= 0):
            raise ValueError("sigma_meas must be strictly positive")
        w = 1.0 / sigma_meas**2
    else:
        w = np.ones(n)

    # weighted least squares for y = a*x + b via the normal equations
    sw = np.sum(w)
    swx = np.sum(w * g_true)
    swy = np.sum(w * g_meas)
    swxx = np.sum(w * g_true**2)
    swxy = np.sum(w * g_true * g_meas)
    delta = sw * swxx - swx**2
    if delta <= 0:
        raise ValueError("degenerate design matrix in shear-bias fit")

    a = (sw * swxy - swx * swy) / delta
    b = (swxx * swy - swx * swxy) / delta

    if sigma_meas is not None:
        # exact WLS covariance (errors taken at face value)
        var_a = sw / delta
        var_b = swxx / delta
    else:
        # OLS: scale covariance by the residual variance
        resid = g_meas - (a * g_true + b)
        s2 = np.sum(resid**2) / (n - 2)
        var_a = s2 * sw / delta
        var_b = s2 * swxx / delta

    return ShearBiasResult(
        m=float(a - 1.0),
        c=float(b),
        sigma_m=float(np.sqrt(var_a)),
        sigma_c=float(np.sqrt(var_b)),
    )


def per_galaxy_sigma_eps(
    sigma_g_plus: np.ndarray, sigma_g_cross: np.ndarray
) -> np.ndarray:
    """
    Per-galaxy effective shape noise (Pranjal+2022 Eq. 20).

        sigma_eps_j = sqrt[(sigma_g+_j^2 + sigma_gx_j^2) / 2]

    Parameters
    ----------
    sigma_g_plus, sigma_g_cross : np.ndarray
        Per-galaxy posterior 1-sigma widths of the galaxy-frame shear
        components, shape (n,).

    Returns
    -------
    np.ndarray
        Per-galaxy sigma_eps, shape (n,).
    """
    sp = np.asarray(sigma_g_plus, dtype=float)
    sx = np.asarray(sigma_g_cross, dtype=float)
    if sp.shape != sx.shape:
        raise ValueError(f"shape mismatch: {sp.shape} vs {sx.shape}")
    if np.any(sp <= 0) or np.any(sx <= 0):
        raise ValueError("posterior widths must be strictly positive")
    return np.sqrt(0.5 * (sp**2 + sx**2))


def compute_shape_noise(
    sigma_g_plus: np.ndarray, sigma_g_cross: np.ndarray
) -> Tuple[float, float]:
    """
    Ensemble effective shape noise via inverse-variance weighting.

    Combines per-galaxy sigma_eps_j (see per_galaxy_sigma_eps) into the
    effective per-galaxy shape noise of the inverse-variance-weighted
    ensemble:

        sigma_eps = sqrt[ N / sum_j(1 / sigma_eps_j^2) ]

    so that the ensemble shear uncertainty is sigma_eps / sqrt(N).

    Parameters
    ----------
    sigma_g_plus, sigma_g_cross : np.ndarray
        Per-galaxy posterior 1-sigma widths of the galaxy-frame shear
        components, shape (n,).

    Returns
    -------
    sigma_eps : float
        Ensemble effective per-galaxy shape noise.
    sigma_eps_err : float
        Standard error on sigma_eps from the spread of the per-galaxy
        values: std(sigma_eps_j) / sqrt(N).
    """
    sigma_j = per_galaxy_sigma_eps(sigma_g_plus, sigma_g_cross)
    n = sigma_j.size
    if n == 0:
        raise ValueError("empty ensemble")
    sigma_eps = np.sqrt(n / np.sum(1.0 / sigma_j**2))
    sigma_eps_err = float(np.std(sigma_j, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    return float(sigma_eps), sigma_eps_err
