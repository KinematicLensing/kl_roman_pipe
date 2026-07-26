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
- Posterior means are shrunk toward zero by the shear prior, so regressing
  them directly returns a biased m. Use
  ``measure_shear_bias_shrinkage_corrected`` for any reported calibration;
  ``measure_shear_bias`` is the uncorrected control.
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


# Default galaxy-frame rotation angle for diagnostics. 'measured' rotates each
# posterior sample by its own theta_int (propagates position-angle uncertainty
# into g+/gx and decorrelates the intrinsic-PA vs cross-shear ridge); 'truth'
# rotates by the fixed truth theta_int (assumes the angle is known). Flip this
# to 'truth' to revert all g+/gx diagnostics to the fixed-angle convention.
GALAXY_FRAME_ANGLE = 'measured'


def galaxy_frame_samples(
    g1_samples: np.ndarray,
    g2_samples: np.ndarray,
    theta_int_samples: np.ndarray,
    theta_int_truth: float,
    angle: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rotate posterior shear samples into the galaxy frame.

    Parameters
    ----------
    g1_samples, g2_samples, theta_int_samples : np.ndarray
        Per-sample sky-frame shear and position angle from a posterior chain.
    theta_int_truth : float
        Truth position angle, used only when ``angle='truth'``.
    angle : {'measured', 'truth'}, optional
        Rotation-angle convention (defaults to ``GALAXY_FRAME_ANGLE``).
        'measured' uses each sample's own ``theta_int_samples``; 'truth' uses
        the fixed ``theta_int_truth``.

    Returns
    -------
    g_plus, g_cross : np.ndarray
        Galaxy-frame shear samples.
    """
    angle = angle or GALAXY_FRAME_ANGLE
    g1_samples = np.asarray(g1_samples, dtype=float)
    if angle == 'measured':
        theta = theta_int_samples
    elif angle == 'truth':
        theta = np.full_like(g1_samples, float(theta_int_truth))
    else:
        raise ValueError(f"angle must be 'measured' or 'truth', got {angle!r}")
    return rotate_to_galaxy_frame(g1_samples, g2_samples, theta)


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


class ShrinkageDiagnostics(NamedTuple):
    """Per-galaxy prior-shrinkage decomposition of a shear posterior."""

    s: np.ndarray  # shrinkage factor in (0, 1], clipped at s_floor
    sigma_like: np.ndarray  # implied data-only 1-sigma
    n_clipped: int  # posteriors at least as wide as the prior


def shrinkage_factor(
    sigma_post: np.ndarray,
    sigma_prior: float,
    s_floor: float = 1e-3,
) -> ShrinkageDiagnostics:
    """
    Per-galaxy prior-shrinkage factor from the posterior width.

    For a Gaussian prior N(0, sigma_prior^2) and a Gaussian likelihood of
    width sigma_like, the posterior mean is the maximum-likelihood point
    scaled by

        s = sigma_prior^2 / (sigma_prior^2 + sigma_like^2)

    and the posterior width satisfies sigma_post^2 = s * sigma_like^2. The
    two together invert to a form that needs only the reported width:

        s = 1 - sigma_post^2 / sigma_prior^2
        sigma_like^2 = sigma_post^2 / s

    so the shrinkage can be undone per galaxy without knowing sigma_like
    independently. s -> 1 is data-dominated, s -> 0 prior-dominated.

    The isotropic Gaussian shear prior makes this valid for the galaxy-frame
    components as well as (g1, g2): an isotropic prior is invariant under the
    spin-2 rotation, so g+ and gx see the same sigma_prior.

    Parameters
    ----------
    sigma_post : np.ndarray
        Per-galaxy posterior 1-sigma widths, shape (n,).
    sigma_prior : float
        Width of the Gaussian shear prior used in the fit.
    s_floor : float, default 1e-3
        Lower clip on s. Guards the division by s downstream; galaxies
        hitting it are counted in ``n_clipped``.

    Returns
    -------
    ShrinkageDiagnostics

    Notes
    -----
    sigma_post >= sigma_prior implies s <= 0, which a Gaussian conjugate
    update cannot produce -- it signals a non-Gaussian (typically multimodal
    or prior-railed) posterior for which the width correction is not valid.
    Those galaxies are clipped rather than dropped here, and counted so the
    caller can act; ``measure_shear_bias_shrinkage_corrected`` excludes them
    via ``s_min``.
    """
    sp = np.asarray(sigma_post, dtype=float).ravel()
    if sp.size == 0:
        raise ValueError("empty sigma_post")
    if np.any(~np.isfinite(sp)) or np.any(sp <= 0):
        raise ValueError("sigma_post must be finite and strictly positive")
    sigma_prior = float(sigma_prior)
    if sigma_prior <= 0:
        raise ValueError(f"sigma_prior ({sigma_prior}) must be positive")
    if not (0.0 < s_floor < 1.0):
        raise ValueError(f"s_floor ({s_floor}) must lie in (0, 1)")

    s_raw = 1.0 - sp**2 / sigma_prior**2
    n_clipped = int(np.sum(s_raw < s_floor))
    s = np.clip(s_raw, s_floor, 1.0)
    return ShrinkageDiagnostics(s=s, sigma_like=sp / np.sqrt(s), n_clipped=n_clipped)


class ShrinkageCorrectedBiasResult(NamedTuple):
    """Shrinkage-corrected shear-bias fit, with the cuts that produced it."""

    m: float
    c: float
    sigma_m: float
    sigma_c: float
    estimator: str
    n_total: int  # galaxies supplied
    n_used: int  # galaxies surviving s_min
    n_clipped: int  # posteriors at least as wide as the prior
    n_downweighted: int  # Huber weight < 1 at convergence (0 for 'wls')
    n_outliers: int  # whitened residual beyond OUTLIER_NSIGMA robust sigma
    mean_shrinkage: float  # <s> over the used galaxies


# Whitened-residual threshold defining a catastrophic fit. Under Gaussian
# residuals the expected count is 6e-7 * N, so anything above a handful is
# contamination rather than tail. Distinct from n_downweighted, which counts
# Huber's soft transition and sits near 18% of a perfectly clean sample.
OUTLIER_NSIGMA = 5.0


def _solve_wls_line(
    x: np.ndarray, y: np.ndarray, w: np.ndarray
) -> Tuple[float, float, float, float]:
    """Weighted least squares for y = a*x + b; returns (a, b, var_a, var_b)."""
    sw = np.sum(w)
    swx = np.sum(w * x)
    swy = np.sum(w * y)
    swxx = np.sum(w * x**2)
    swxy = np.sum(w * x * y)
    delta = sw * swxx - swx**2
    if delta <= 0:
        raise ValueError("degenerate design matrix in shear-bias fit")
    a = (sw * swxy - swx * swy) / delta
    b = (swxx * swy - swx * swxy) / delta
    return a, b, sw / delta, swxx / delta


def measure_shear_bias_shrinkage_corrected(
    g_true: np.ndarray,
    g_post_mean: np.ndarray,
    sigma_post: np.ndarray,
    sigma_prior: float,
    s_min: Optional[float] = None,
    estimator: str = 'wls',
    tukey_c: float = 4.685,
    max_iter: int = 50,
    tol: float = 1e-10,
) -> ShrinkageCorrectedBiasResult:
    """
    Shear bias (m, c) from posterior means, corrected for prior shrinkage.

    Regressing posterior means on truth directly measures the shrinkage of
    the prior, not the calibration of the pipeline: under a Gaussian(0,
    sigma_prior) shear prior the recovered slope is suppressed by the
    ensemble-mean shrinkage <s>, which for the census posteriors is ~0.67 and
    produces m ~ -0.33 with no instrumental bias present at all.

    This estimator undoes the shrinkage per galaxy before fitting:

        s_j    = 1 - sigma_post_j^2 / sigma_prior^2
        ghat_j = mu_j / s_j                     (width-corrected estimate)
        w_j    = 1 / sigma_like_j^2             (data-only inverse variance)

    then fits ghat = (1 + m) * g_true + c by weighted least squares.

    The weights are deliberately the data-only variance, not the posterior
    variance. Weighting by 1/sigma_post^2 correlates the weight with the same
    noisy width that enters the correction, and the seeded estimator
    experiment behind this function found it diverges under realistic width
    noise (m up to +1.0) while the data-only form stays unbiased
    (|m| < 5e-4 at N = 1e4).

    Parameters
    ----------
    g_true : np.ndarray
        Injected shear, one component, shape (n,).
    g_post_mean : np.ndarray
        Posterior means of the same component, shape (n,).
    sigma_post : np.ndarray
        Posterior 1-sigma widths, shape (n,).
    sigma_prior : float
        Width of the Gaussian shear prior used in the fit.
    s_min : float, optional
        Drop galaxies with shrinkage below this. None keeps everything.
        0.5 is the natural choice (sigma_like > sigma_prior, i.e.
        prior-dominated fits whose corrected estimate is mostly amplified
        prior).
    estimator : {'wls', 'robust'}, default 'wls'
        'wls' is the plain weighted fit. 'robust' runs iteratively reweighted
        least squares with a Tukey biweight on the whitened residuals, seeded
        from the repeated-median (Siegel) slope, which removes catastrophic
        wrong-mode fits from the slope entirely.
    tukey_c : float, default 4.685
        Biweight cutoff in robust-sigma units; 4.685 is the standard choice
        giving 95% efficiency against Gaussian residuals. Residuals beyond it
        get exactly zero weight.
    max_iter, tol : int, float
        IRLS iteration cap and convergence tolerance on (a, b).

    Returns
    -------
    ShrinkageCorrectedBiasResult

    Notes
    -----
    The width correction assumes an approximately Gaussian posterior. On
    deliberately skewed posteriors the experiment measured a residual
    m ~ +0.045, so a strongly non-Gaussian shear posterior is outside this
    estimator's validity -- check ``n_clipped`` and the skew before quoting.
    """
    if estimator not in ('wls', 'robust'):
        raise ValueError(f"estimator must be 'wls' or 'robust', got '{estimator}'")

    gt = np.asarray(g_true, dtype=float).ravel()
    mu = np.asarray(g_post_mean, dtype=float).ravel()
    sp = np.asarray(sigma_post, dtype=float).ravel()
    if not (gt.shape == mu.shape == sp.shape):
        raise ValueError(
            f"shape mismatch: g_true {gt.shape}, g_post_mean {mu.shape}, "
            f"sigma_post {sp.shape}"
        )
    n_total = gt.size

    shrink = shrinkage_factor(sp, sigma_prior)
    ghat = mu / shrink.s
    w = 1.0 / shrink.sigma_like**2

    if s_min is not None:
        if not (0.0 < s_min < 1.0):
            raise ValueError(f"s_min ({s_min}) must lie in (0, 1)")
        keep = shrink.s >= s_min
        gt, ghat, w = gt[keep], ghat[keep], w[keep]

    n_used = gt.size
    if n_used < 3:
        raise ValueError(
            f"need >= 3 galaxies for a 2-parameter fit, got {n_used} of "
            f"{n_total} after the s_min cut"
        )
    if np.ptp(gt) == 0.0:
        raise ValueError("g_true values are all identical; slope is undefined")

    a, b, var_a, var_b = _solve_wls_line(gt, ghat, w)
    n_downweighted = 0

    if estimator == 'robust':
        # A Huber loss does not suffice here. A wrong-mode fit is confidently
        # wrong, so it arrives with both a large residual and a small reported
        # width: on the planted-outlier test such a fit carries 48x a clean
        # fit's data-only weight, and Huber's linear downweighting leaves it
        # at ~4x, enough for 2% contamination to move m by ~0.8. The biweight
        # redescends to zero instead.
        #
        # Redescending losses have local minima, hence the repeated-median
        # (Siegel) seed: 50% breakdown, and it ignores the weights the
        # outliers inflate.
        from scipy.stats import siegelslopes

        a, b = siegelslopes(ghat, gt)
        omega = np.ones_like(gt)
        for _ in range(max_iter):
            # whiten so the robust scale is measured in units of each
            # galaxy's own expected error rather than raw shear
            r = (ghat - (a * gt + b)) * np.sqrt(w)
            scale = 1.4826 * np.median(np.abs(r - np.median(r)))
            if scale <= 0:
                # residuals already exactly on the line for >half the sample
                break
            u = r / scale
            omega = np.where(np.abs(u) <= tukey_c, (1.0 - (u / tukey_c) ** 2) ** 2, 0.0)
            if not np.any(omega > 0):
                raise ValueError(
                    "robust fit rejected every galaxy; the residual scale has "
                    "collapsed, which usually means fewer than half the fits "
                    "share a common linear relation"
                )
            a_new, b_new, var_a, var_b = _solve_wls_line(gt, ghat, w * omega)
            converged = abs(a_new - a) < tol and abs(b_new - b) < tol
            a, b = a_new, b_new
            if converged:
                break
        n_downweighted = int(np.sum(omega < 1.0))

    # residual rescaling: the data-only weights are estimates, so take the
    # covariance from the achieved scatter rather than at face value
    resid = ghat - (a * gt + b)
    chi2_per_dof = np.sum(w * resid**2) / (n_used - 2)
    scale_cov = max(chi2_per_dof, 1.0)

    r_final = resid * np.sqrt(w)
    robust_scale = 1.4826 * np.median(np.abs(r_final - np.median(r_final)))
    n_outliers = (
        int(np.sum(np.abs(r_final) > OUTLIER_NSIGMA * robust_scale))
        if robust_scale > 0
        else 0
    )

    return ShrinkageCorrectedBiasResult(
        m=float(a - 1.0),
        c=float(b),
        sigma_m=float(np.sqrt(var_a * scale_cov)),
        sigma_c=float(np.sqrt(var_b * scale_cov)),
        estimator=estimator,
        n_total=n_total,
        n_used=n_used,
        n_clipped=shrink.n_clipped,
        n_downweighted=n_downweighted,
        n_outliers=n_outliers,
        mean_shrinkage=float(
            np.mean(shrink.s[keep] if s_min is not None else shrink.s)
        ),
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
