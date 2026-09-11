"""
Fit initialization toolkit: optimizer starts, MAP finder, Laplace metric,
chain initial points.

Everything the sampler needs before its first NUTS step is built here from
small composable pieces, so a fitting procedure can be refined one piece at a
time and every piece leaves a record:

- **Start proposals** (``StartSet``): ``prior_starts`` (independent prior
  draws), ``pa_stratified_starts`` (prior draws with the position angle put
  on a grid), ``moment_starts`` (centroid, flux, size, inclination and
  position angle read off the broadband images with adaptive moments; every
  other parameter at its prior median). Sets concatenate.
- **MAP finder** (``find_map``): multi-start L-BFGS in prior-scaled
  coordinates; keeps every start's endpoint, objective and family label,
  clusters the endpoints into basins and reports the margin of the best
  basin over the runner-up.
- **Laplace metric** (``laplace_metric`` / ``EigenFloor``): regularized
  inverse Hessian at the MAP. The floor is either relative to the stiffest
  eigenvalue (``'relative'``, the historical rule) or absolute in prior
  units (``'prior'``: a posterior direction is never made stiffer than a
  fraction of its prior width).
- **Chain initial points** (``chain_inits``): all chains at the MAP with a
  small jitter, or one chain per competing basin so r-hat can see basin
  disagreement.

``Initializer`` runs the whole procedure from an ``InitConfig`` whose fields
are the ensemble spec's ``fit.*`` initialization knobs, and returns one
``InitResult``; the ensemble worker calls nothing else. The pieces stay
public for refining the procedure one at a time.
``InferenceTask.laplace_preconditioner`` is the same procedure behind the
older keyword interface.

Defaults are the measured-robust settings (cosmos25_bank32 A/B jobs 988356,
988824, 990891): bounded L-BFGS-B with an 8-step Newton polish of the three
leading basins and the prior-unit eigenvalue floor at 0.5. The historical
procedure (unbounded search, no polish, relative floor 1e-4) is reproduced by
passing those values explicitly.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np
import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from kl_pipe.observation import ImageObs
    from kl_pipe.priors import PriorDict
    from kl_pipe.sampling.task import InferenceTask, LaplacePreconditioner
    from kl_pipe.sampling.transforms import UnconstrainingTransform


# ==============================================================================
# Start proposals
# ==============================================================================


@dataclass
class StartSet:
    """Optimizer start points with a family label per row.

    Attributes
    ----------
    points : np.ndarray
        ``(n_starts, n_params)`` in sampled-parameter order.
    labels : list of str
        Family of each start (``'prior'``, ``'pa_stratified'``,
        ``'moments'``, ``'extra'`` ...).
    names : tuple of str
        Sampled parameter names (column order of ``points``).
    """

    points: np.ndarray
    labels: List[str]
    names: Tuple[str, ...]

    def __post_init__(self):
        self.points = np.atleast_2d(np.asarray(self.points, dtype=np.float64))
        if self.points.shape[1] != len(self.names):
            raise ValueError(
                f"StartSet points have {self.points.shape[1]} columns for "
                f"{len(self.names)} parameter names"
            )
        if len(self.labels) != self.points.shape[0]:
            raise ValueError(
                f"StartSet has {len(self.labels)} labels for "
                f"{self.points.shape[0]} points"
            )
        if not np.all(np.isfinite(self.points)):
            raise ValueError("StartSet points must be finite")

    def __len__(self) -> int:
        return self.points.shape[0]

    def families(self) -> Dict[str, int]:
        """Number of starts per family label."""
        out: Dict[str, int] = {}
        for lab in self.labels:
            out[lab] = out.get(lab, 0) + 1
        return out


def combine_starts(*start_sets: Optional[StartSet]) -> StartSet:
    """Stack start sets into one (``None`` entries skipped; same parameter
    names required)."""
    kept = [s for s in start_sets if s is not None]
    if not kept:
        raise ValueError("combine_starts needs at least one StartSet")
    names = kept[0].names
    for s in kept[1:]:
        if s.names != names:
            raise ValueError("combine_starts: parameter names differ")
    return StartSet(
        points=np.vstack([s.points for s in kept]),
        labels=sum((list(s.labels) for s in kept), []),
        names=names,
    )


def prior_starts(task: 'InferenceTask', n: int, seed: int = 0) -> StartSet:
    """``n`` independent prior draws (key ``PRNGKey(seed + 1)``, the
    historical preconditioner stream, so existing runs reproduce)."""
    if n < 1:
        raise ValueError(f"prior_starts needs n >= 1, got {n}")
    pts = np.asarray(task.sample_prior(jax.random.PRNGKey(seed + 1), n_samples=n))
    return StartSet(pts, ['prior'] * n, tuple(task.sampled_names))


def pa_stratified_starts(
    priors: 'PriorDict', n_pa: int = 4, seed: int = 0
) -> Optional[StartSet]:
    """Prior draws with ``theta_int`` replaced by a grid over its support.

    Under a periodic (full-circle) position-angle prior the grid has
    ``2 * n_pa`` points over one period so both rotation directions are
    started; under a bounded prior ``n_pa`` points over the bounds. Returns
    ``None`` when ``theta_int`` is not sampled or is unbounded and aperiodic.
    Key ``PRNGKey(seed + 2)`` (historical stream). Takes the ``PriorDict``
    (``task.priors``) rather than the task.
    """
    names = list(priors.sampled_names)
    if 'theta_int' not in names:
        return None
    prior = priors.get_prior('theta_int')
    period = getattr(prior, 'period', None)
    if period is not None:
        lo, hi, n = 0.0, float(period), 2 * n_pa
    else:
        lo, hi = prior.bounds
        if lo is None or hi is None:
            return None
        n = n_pa
    starts = np.array(priors.sample(jax.random.PRNGKey(seed + 2), n))
    centers = lo + (np.arange(n) + 0.5) * (hi - lo) / n
    starts[:, names.index('theta_int')] = centers
    return StartSet(starts, ['pa_stratified'] * n, tuple(names))


def prior_center(
    task: 'InferenceTask', seed: int = 0, n_draws: int = 512
) -> np.ndarray:
    """Per-parameter prior median from ``n_draws`` draws (``PRNGKey(seed)``)."""
    draws = np.asarray(task.sample_prior(jax.random.PRNGKey(seed), n_samples=n_draws))
    return np.median(draws, axis=0)


def clip_into_support(
    task: 'InferenceTask', theta: np.ndarray, margin: float = 1e-3
) -> np.ndarray:
    """Move a point strictly inside every bounded prior support.

    Two-sided bounds: at least ``margin`` of the width from either edge.
    One-sided bounds: at least ``margin`` of |bound| (or ``margin`` itself
    when the bound is 0) from the edge.
    """
    out = np.array(theta, dtype=np.float64)
    for i, (lo, hi) in enumerate(task.get_bounds()):
        if lo is not None and hi is not None:
            eps = margin * (hi - lo)
            out[i] = np.clip(out[i], lo + eps, hi - eps)
        elif lo is not None:
            out[i] = max(out[i], lo + margin * max(abs(lo), 1.0))
        elif hi is not None:
            out[i] = min(out[i], hi - margin * max(abs(hi), 1.0))
    return out


# ==============================================================================
# Image moments
# ==============================================================================


class MomentsError(RuntimeError):
    """Adaptive moments did not converge to a positive-definite ellipse."""


@dataclass
class ImageMoments:
    """Adaptive (elliptical-Gaussian-weighted) moments of one image.

    ``m_xx, m_xy, m_yy`` are the second moments of the matched Gaussian
    (twice the weighted moments, exact for a Gaussian object). Angles follow
    the model convention: ``pa`` is the major-axis direction from +x in
    [0, pi). ``flux`` is the plain masked pixel sum.
    """

    x0: float
    y0: float
    flux: float
    m_xx: float
    m_xy: float
    m_yy: float
    n_iter: int
    # False when the iteration cap was reached while still drifting slowly
    # (accepted estimate); a diverging iteration raises instead
    converged: bool = True

    @property
    def matrix(self) -> np.ndarray:
        return np.array([[self.m_xx, self.m_xy], [self.m_xy, self.m_yy]])

    @property
    def sigma_major(self) -> float:
        return float(np.sqrt(np.linalg.eigvalsh(self.matrix)[1]))

    @property
    def sigma_minor(self) -> float:
        return float(np.sqrt(max(np.linalg.eigvalsh(self.matrix)[0], 0.0)))

    @property
    def axis_ratio(self) -> float:
        return self.sigma_minor / self.sigma_major

    @property
    def pa(self) -> float:
        return float(
            np.mod(0.5 * np.arctan2(2.0 * self.m_xy, self.m_xx - self.m_yy), np.pi)
        )

    def deconvolved(
        self, psf: 'ImageMoments', min_fraction: float = 0.05
    ) -> 'ImageMoments':
        """Subtract the PSF second moments (Gaussian approximation).

        Each deconvolved eigenvalue is floored at ``min_fraction`` of the
        observed one so an unresolved object still yields a small positive
        size instead of a negative moment.
        """
        m = self.matrix - psf.matrix
        w, v = np.linalg.eigh(m)
        w_obs = np.linalg.eigvalsh(self.matrix)
        w = np.maximum(w, min_fraction * w_obs)
        m = (v * w) @ v.T
        return ImageMoments(
            self.x0,
            self.y0,
            self.flux,
            m[0, 0],
            m[0, 1],
            m[1, 1],
            self.n_iter,
            self.converged,
        )


def adaptive_moments(
    data: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    mask: Optional[np.ndarray] = None,
    *,
    x0: Optional[float] = None,
    y0: Optional[float] = None,
    sigma0: Optional[float] = None,
    n_iter: int = 100,
    tol: float = 2e-3,
    min_sigma: Optional[float] = None,
) -> ImageMoments:
    """Elliptical-Gaussian-weighted moments iterated to self-consistency.

    Hirata & Seljak (2003) adaptive moments: the weight is a Gaussian with
    the current centroid and covariance; the weighted second moments of a
    Gaussian object measured with its own matched weight are half the true
    ones, so the covariance is updated to twice the weighted moments each
    iteration. Converges in a few iterations on well-detected objects.

    Parameters
    ----------
    data, X, Y : array
        Image (flux per pixel) and its coordinate grids (arcsec).
    mask : bool array, optional
        True = valid pixel.
    x0, y0 : float, optional
        Starting centroid (default: grid centre).
    sigma0 : float, optional
        Starting circular width (default: a fifth of the grid extent).
    n_iter, tol : int, float
        Iteration cap and relative convergence tolerance on the covariance
        and centroid per iteration. The matched-Gaussian iteration on a
        cuspy exponential with a sub-pixel minor axis drifts slowly toward
        the width floor; percent-level accuracy is what a start point needs,
        and an estimate still drifting below ``10 * tol`` per iteration at
        the cap is returned with ``converged=False``.
    min_sigma : float, optional
        Floor on the weight's width along either axis (default: half the
        grid spacing). A sub-pixel object cannot be measured narrower than
        the pixel, and without the floor the weight collapses on it.

    Raises
    ------
    MomentsError
        Non-positive weighted flux (no object under the weight), a
        non-positive-definite covariance, or no convergence in ``n_iter``.
    """
    data = np.asarray(data, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if data.shape != X.shape or data.shape != Y.shape:
        raise ValueError("data, X, Y must share a shape")
    valid = np.ones(data.shape, dtype=bool) if mask is None else np.asarray(mask, bool)
    if not valid.any():
        raise MomentsError("adaptive_moments: no valid pixels")
    d = np.where(valid, data, 0.0)
    extent = max(X.max() - X.min(), Y.max() - Y.min())
    mu = np.array(
        [
            0.5 * (X.max() + X.min()) if x0 is None else x0,
            0.5 * (Y.max() + Y.min()) if y0 is None else y0,
        ]
    )
    s0 = extent / 5.0 if sigma0 is None else float(sigma0)
    if min_sigma is None:
        spacing = np.median(np.abs(np.diff(np.unique(np.round(X, 12)))))
        min_sigma = 0.5 * float(spacing)
    M = np.eye(2) * s0**2
    for it in range(1, n_iter + 1):
        try:
            Minv = np.linalg.inv(M)
        except np.linalg.LinAlgError as err:
            raise MomentsError(
                f"adaptive_moments: singular covariance at iteration {it}"
            ) from err
        dx, dy = X - mu[0], Y - mu[1]
        arg = Minv[0, 0] * dx * dx + 2 * Minv[0, 1] * dx * dy + Minv[1, 1] * dy * dy
        w = np.exp(-0.5 * arg)
        wd = w * d
        norm = wd.sum()
        if not np.isfinite(norm) or norm <= 0:
            raise MomentsError(
                f"adaptive_moments: non-positive weighted flux ({norm:.3g}) at iteration {it}"
            )
        mu_new = np.array([(wd * X).sum(), (wd * Y).sum()]) / norm
        dx, dy = X - mu_new[0], Y - mu_new[1]
        M_new = (
            2.0
            * np.array(
                [
                    [(wd * dx * dx).sum(), (wd * dx * dy).sum()],
                    [(wd * dx * dy).sum(), (wd * dy * dy).sum()],
                ]
            )
            / norm
        )
        if not np.all(np.isfinite(M_new)):
            raise MomentsError(
                f"adaptive_moments: non-finite covariance at iteration {it}"
            )
        # floor the weight width at min_sigma along each principal axis
        eig, vec = np.linalg.eigh(M_new)
        eig = np.maximum(eig, min_sigma**2)
        M_new = (vec * eig) @ vec.T
        if eig[1] > (2.0 * extent) ** 2:
            raise MomentsError(
                "adaptive_moments: weight grew beyond twice the grid extent "
                "(no compact object under the weight)"
            )
        delta = np.max(np.abs(M_new - M)) / eig[1]
        shift = np.hypot(*(mu_new - mu)) / np.sqrt(eig[1])
        mu, M = mu_new, M_new
        if delta < tol and shift < tol:
            break
    else:
        if not (delta < 10 * tol and shift < 10 * tol):
            raise MomentsError(
                f"adaptive_moments: no convergence in {n_iter} iterations "
                f"(covariance change {delta:.2e}, centroid shift {shift:.2e} "
                f"per iteration)"
            )
        return ImageMoments(
            float(mu[0]),
            float(mu[1]),
            float(d.sum()),
            float(M[0, 0]),
            float(M[0, 1]),
            float(M[1, 1]),
            n_iter,
            False,
        )
    return ImageMoments(
        float(mu[0]),
        float(mu[1]),
        float(d.sum()),
        float(M[0, 0]),
        float(M[0, 1]),
        float(M[1, 1]),
        it,
        True,
    )


def psf_moments(obs: 'ImageObs', weight: Optional[np.ndarray] = None) -> ImageMoments:
    """Second moments of the observation's own PSF kernel.

    Recovers the real-space kernel from the precomputed ``psf_data`` (the
    kernel actually convolved into the model, at the fine pixel scale);
    coordinates are wrap-aware so the FFT-shifted kernel is measured about
    its centre.

    Without ``weight``: the kernel's own adaptive (matched-Gaussian)
    moments -- core-dominated for a PSF with extended wings. With
    ``weight`` (a 2x2 covariance, normally the galaxy's observed matched
    covariance): the Gaussian-equivalent PSF second moments as seen at that
    scale, ``P = ((M_w)^-1 - W^-1)^-1`` where ``M_w`` are the kernel's
    second moments under the fixed Gaussian weight ``W``; exact for a
    Gaussian PSF, and for a real PSF it counts the wings out to the
    galaxy's scale, which is what enters the galaxy's observed moments.
    """
    pd = obs.psf_data
    if pd is None:
        raise ValueError("psf_moments: the observation carries no psf_data")
    kernel = np.asarray(jnp.fft.irfft2(pd.kernel_fft, s=tuple(pd.padded_shape)))
    nrow, ncol = kernel.shape
    fine = float(obs.image_pars.pixel_scale) / int(pd.oversample)
    r = ((np.arange(nrow) + nrow // 2) % nrow - nrow // 2) * fine
    c = ((np.arange(ncol) + ncol // 2) % ncol - ncol // 2) * fine
    R, C = np.meshgrid(r, c, indexing='ij')
    # observation grids vary x along columns and y along rows
    if weight is None:
        return adaptive_moments(kernel, C, R, x0=0.0, y0=0.0, sigma0=3 * fine)
    W = np.asarray(weight, dtype=np.float64)
    Winv = np.linalg.inv(W)
    w = np.exp(
        -0.5 * (Winv[0, 0] * C * C + 2 * Winv[0, 1] * C * R + Winv[1, 1] * R * R)
    )
    wk = w * kernel
    norm = wk.sum()
    if not norm > 0:
        raise MomentsError("psf_moments: non-positive weighted kernel flux")
    mu = np.array([(wk * C).sum(), (wk * R).sum()]) / norm
    dx, dy = C - mu[0], R - mu[1]
    M_w = (
        np.array(
            [
                [(wk * dx * dx).sum(), (wk * dx * dy).sum()],
                [(wk * dx * dy).sum(), (wk * dy * dy).sum()],
            ]
        )
        / norm
    )
    P = np.linalg.inv(np.linalg.inv(M_w) - Winv)
    if np.linalg.eigvalsh(P).min() <= 0:
        raise MomentsError("psf_moments: weighted PSF moments not positive definite")
    return ImageMoments(
        float(mu[0]), float(mu[1]), float(kernel.sum()), P[0, 0], P[0, 1], P[1, 1], 1
    )


# matched-Gaussian sigma of an exponential disk in units of its scale length
# (1.164 on a fine grid, independent of axis ratio and size)
_EXP_SIGMA_OVER_RSCALE = 1.164
# edge-on axis ratio of the sech^2 disk per unit h_over_r: the moment axis
# ratio follows q^2 = cosi^2 + q0^2 (1 - cosi^2) with q0 = 1.15 h_over_r
# (median over the edge-on half of the same renders, h_over_r 0.25)
_THICK_DISK_Q0_PER_H = 1.15


@dataclass
class MomentEstimates:
    """Band-averaged image-moment estimates in model parameter terms."""

    x0: float
    y0: float
    rscale: float
    cosi: float
    pa: float
    flux: Dict[str, float]
    per_band: Dict[str, ImageMoments]
    axis_ratio: float = float('nan')
    q0: float = 0.0


def moment_estimates(
    task: 'InferenceTask',
    image_obs: Dict[str, 'ImageObs'],
    *,
    deconvolve: bool = True,
) -> MomentEstimates:
    """Read centroid, size, inclination, position angle and per-band flux off
    the broadband images.

    Size: the PSF-deconvolved matched-Gaussian major-axis sigma divided by
    ``_EXP_SIGMA_OVER_RSCALE`` (exponential-disk calibration). Inclination
    from the axis ratio ``q`` with the disk-thickness correction
    ``cosi^2 = (q^2 - q0^2) / (1 - q0^2)``, ``q0 = _THICK_DISK_Q0_PER_H *
    h_over_r`` where ``h_over_r`` is the (fixed) mean over the broadband
    components, 0 when no component carries one. Position angle: major axis
    from +x, modulo pi (the model's ``theta_int`` convention). Centroid,
    size, cosi and pa are flux-weighted averages over bands.
    """
    if not image_obs:
        raise ValueError("moment_estimates needs at least one broadband image")
    per_band: Dict[str, ImageMoments] = {}
    for band, obs in image_obs.items():
        if obs.data is None:
            raise ValueError(f"image_obs['{band}'] carries no data")
        m = adaptive_moments(
            np.asarray(obs.data),
            np.asarray(obs.X),
            np.asarray(obs.Y),
            None if obs.mask is None else np.asarray(obs.mask),
        )
        if deconvolve and obs.psf_data is not None:
            # PSF second moments as seen through the galaxy's own weight; the
            # coarse pixel adds its top-hat variance p^2 / 12 per axis
            p2 = float(obs.image_pars.pixel_scale) ** 2 / 12.0
            psf = psf_moments(obs, weight=m.matrix)
            psf = ImageMoments(
                psf.x0,
                psf.y0,
                psf.flux,
                psf.m_xx + p2,
                psf.m_xy,
                psf.m_yy + p2,
                psf.n_iter,
            )
            m = m.deconvolved(psf)
        per_band[band] = m
    weights = np.array([abs(m.flux) for m in per_band.values()])
    if weights.sum() <= 0:
        raise MomentsError("moment_estimates: non-positive total flux in every band")
    weights = weights / weights.sum()
    ms = list(per_band.values())
    x0 = float(sum(w * m.x0 for w, m in zip(weights, ms)))
    y0 = float(sum(w * m.y0 for w, m in zip(weights, ms)))
    # average the moment matrices, not the derived angles (pa wraps)
    M = sum(w * m.matrix for w, m in zip(weights, ms))
    eig = np.linalg.eigvalsh(M)
    sigma_major = float(np.sqrt(eig[1]))
    q = float(np.sqrt(max(eig[0], 0.0)) / sigma_major)
    pa = float(np.mod(0.5 * np.arctan2(2.0 * M[0, 1], M[0, 0] - M[1, 1]), np.pi))
    fixed = task.priors.fixed_values
    h_over_r = [
        fixed[f'{band}.h_over_r'] for band in image_obs if f'{band}.h_over_r' in fixed
    ]
    q0 = _THICK_DISK_Q0_PER_H * float(np.mean(h_over_r)) if h_over_r else 0.0
    cosi = float(np.sqrt(np.clip((q**2 - q0**2) / (1.0 - q0**2), 0.0, 1.0)))
    return MomentEstimates(
        x0=x0,
        y0=y0,
        rscale=sigma_major / _EXP_SIGMA_OVER_RSCALE,
        cosi=cosi,
        pa=pa,
        flux={band: float(m.flux) for band, m in per_band.items()},
        per_band=per_band,
        axis_ratio=q,
        q0=q0,
    )


def moment_starts(
    task: 'InferenceTask',
    image_obs: Dict[str, 'ImageObs'],
    seed: int = 0,
    *,
    pa_offsets: Sequence[float] = (0.0, np.pi),
    estimates: Optional[MomentEstimates] = None,
) -> StartSet:
    """Data-informed optimizer starts from image moments.

    Every sampled parameter starts at its prior median except: ``<comp>.x0``
    / ``<comp>.y0`` at the moment centroid (all components, broadband and
    line), ``<band>.flux`` at the band's pixel sum, every ``<comp>.rscale``
    at the moment size, ``cosi`` at the axis ratio, ``theta_int`` at the
    moment position angle plus each entry of ``pa_offsets`` (default: both
    rotation directions, which the image cannot distinguish). Shear starts
    at 0 when sampled. Everything is clipped into the prior support.

    Raises ``MomentsError`` when the images yield no usable moments; callers
    that must not fail on a bad image catch it and fall back to prior starts.
    """
    est = estimates if estimates is not None else moment_estimates(task, image_obs)
    names = list(task.sampled_names)
    base = prior_center(task, seed)
    for i, name in enumerate(names):
        comp, _, leaf = name.rpartition('.')
        if leaf == 'x0' and comp:
            base[i] = est.x0
        elif leaf == 'y0' and comp:
            base[i] = est.y0
        elif leaf == 'rscale' and comp and comp != 'vel':
            base[i] = est.rscale
        elif leaf == 'flux' and comp in est.flux:
            base[i] = est.flux[comp]
        elif name == 'cosi':
            base[i] = est.cosi
        elif name in ('g1', 'g2'):
            base[i] = 0.0
    base = clip_into_support(task, base)
    if 'theta_int' not in names:
        return StartSet(base[None, :], ['moments'], tuple(names))
    ith = names.index('theta_int')
    prior = task.priors.get_prior('theta_int')
    period = getattr(prior, 'period', None)
    rows = []
    for off in pa_offsets:
        row = base.copy()
        pa = est.pa + off
        if period is not None:
            pa = np.mod(pa, period)
        row[ith] = pa
        rows.append(clip_into_support(task, row))
    return StartSet(np.vstack(rows), ['moments'] * len(rows), tuple(names))


# ==============================================================================
# MAP finder
# ==============================================================================


@dataclass
class MapResult:
    """Everything the multi-start optimizer learned.

    Attributes
    ----------
    theta_map : np.ndarray
        Best endpoint (physical coordinates, periodic parameters wrapped).
    neg_logpost : float
        Negative log-posterior at ``theta_map``.
    start_points, end_points : np.ndarray
        ``(n_finite, n_params)`` where each finite-objective start began and
        settled.
    objectives : np.ndarray
        Negative log-posterior at each endpoint.
    labels : list of str
        Start family per row.
    converged : np.ndarray of bool
        L-BFGS success flag per row.
    basin_ids : np.ndarray of int
        Basin index per row; 0 is the MAP basin, then by increasing objective.
    basin_points, basin_objectives : np.ndarray
        Best endpoint and objective per basin, basin order.
    loc, scale : np.ndarray
        Prior-derived affine scaling used by the optimizer.
    n_evals : int
        Total objective+gradient evaluations.
    wall_s : float
    """

    theta_map: np.ndarray
    neg_logpost: float
    start_points: np.ndarray
    end_points: np.ndarray
    objectives: np.ndarray
    labels: List[str]
    converged: np.ndarray
    basin_ids: np.ndarray
    basin_points: np.ndarray
    basin_objectives: np.ndarray
    loc: np.ndarray
    scale: np.ndarray
    names: Tuple[str, ...]
    n_evals: int = 0
    wall_s: float = 0.0
    # Newton polish bookkeeping: objective before polishing (per basin
    # polished), gradient norm and smallest prior-scaled Hessian eigenvalue at
    # the final MAP (nan when no polish ran)
    polish_gain: float = 0.0
    map_grad_norm: float = float('nan')
    map_min_eigenvalue: float = float('nan')

    @property
    def n_basins(self) -> int:
        return int(len(self.basin_objectives))

    @property
    def basin_margin(self) -> float:
        """Objective gap between the MAP basin and the runner-up (inf if one basin)."""
        if self.n_basins < 2:
            return float('inf')
        return float(self.basin_objectives[1] - self.basin_objectives[0])

    @property
    def best_index(self) -> int:
        return int(np.argmin(self.objectives))

    @property
    def winning_label(self) -> str:
        return self.labels[self.best_index]

    @property
    def n_converged(self) -> int:
        return int(np.sum(self.converged))

    def families_reaching_map_basin(self) -> Dict[str, int]:
        """Starts per family that ended in the MAP basin."""
        out: Dict[str, int] = {}
        for lab, b in zip(self.labels, self.basin_ids):
            if b == 0:
                out[lab] = out.get(lab, 0) + 1
        return out

    def summary_rows(self) -> List[Dict[str, object]]:
        """One record per start: label, converged, objective, gap to MAP, basin."""
        order = np.argsort(self.objectives)
        return [
            {
                'start': int(i),
                'family': self.labels[i],
                'converged': bool(self.converged[i]),
                'neg_logpost': float(self.objectives[i]),
                'gap_to_map': float(self.objectives[i] - self.neg_logpost),
                'basin': int(self.basin_ids[i]),
            }
            for i in order
        ]

    def format_summary(self) -> str:
        lines = [
            f"MAP -logpost {self.neg_logpost:.3f}; {self.n_basins} basin(s), "
            f"margin {self.basin_margin:.2f} nats; winning start family "
            f"'{self.winning_label}'; {self.n_converged}/{len(self.labels)} "
            f"starts converged; {self.n_evals} evaluations in {self.wall_s:.1f} s",
            f"{'start':>5} {'family':>14} {'conv':>5} {'-logpost':>14} {'gap':>12} {'basin':>5}",
        ]
        for r in self.summary_rows():
            lines.append(
                f"{r['start']:>5} {r['family']:>14} {str(r['converged']):>5} "
                f"{r['neg_logpost']:>14.3f} {r['gap_to_map']:>12.3f} {r['basin']:>5}"
            )
        return '\n'.join(lines)


def _wrap_periodic(theta: np.ndarray, periods: Sequence[Optional[float]]) -> np.ndarray:
    out = np.array(theta, dtype=np.float64)
    for j, period in enumerate(periods):
        if period is not None:
            out[..., j] = np.mod(out[..., j], period)
    return out


def cluster_basins(
    end_points: np.ndarray,
    objectives: np.ndarray,
    scale: np.ndarray,
    periods: Sequence[Optional[float]],
    tol: float = 0.25,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Single-linkage clustering of optimizer endpoints in prior-scaled units.

    Two endpoints share a basin when their scaled distance (periodic
    parameters wrapped) is below ``tol`` prior standard deviations, the
    linkage being transitive. Basins are numbered by their best objective.

    Returns ``(basin_ids, basin_points, basin_objectives)``.
    """
    n = len(objectives)
    if n == 0:
        raise ValueError("cluster_basins: no endpoints")
    d = np.empty((n, n))
    for a in range(n):
        diff = end_points - end_points[a]
        for j, period in enumerate(periods):
            if period is not None:
                diff[:, j] = (diff[:, j] + 0.5 * period) % period - 0.5 * period
        d[a] = np.sqrt(np.sum((diff / scale) ** 2, axis=1))
    labels = -np.ones(n, dtype=int)
    current = 0
    for a in np.argsort(objectives):
        if labels[a] >= 0:
            continue
        stack = [a]
        labels[a] = current
        while stack:
            b = stack.pop()
            for c in np.where((d[b] < tol) & (labels < 0))[0]:
                labels[c] = current
                stack.append(c)
        current += 1
    basin_objectives = np.array([objectives[labels == k].min() for k in range(current)])
    basin_points = np.array(
        [
            end_points[labels == k][np.argmin(objectives[labels == k])]
            for k in range(current)
        ]
    )
    return labels, basin_points, basin_objectives


def support_bounds_scaled(
    task: 'InferenceTask', loc: np.ndarray, scale: np.ndarray, margin: float = 1e-6
) -> List[Tuple[Optional[float], Optional[float]]]:
    """Prior support bounds in the optimizer's scaled coordinates, pulled
    inside the support by ``margin`` of the width (two-sided) or of the
    scale (one-sided) so the boundary itself is never evaluated."""
    out = []
    for i, (lo, hi) in enumerate(task.get_bounds()):
        eps = (
            margin * (hi - lo)
            if lo is not None and hi is not None
            else margin * scale[i]
        )
        ulo = None if lo is None else (lo + eps - loc[i]) / scale[i]
        uhi = None if hi is None else (hi - eps - loc[i]) / scale[i]
        out.append((ulo, uhi))
    return out


def newton_polish(
    task: 'InferenceTask',
    theta0: np.ndarray,
    scale: np.ndarray,
    *,
    n_steps: int = 8,
    grad_tol: float = 1e-3,
    hessian_method: str = 'fd',
    fd_rel_step: float = 1e-5,
) -> Tuple[np.ndarray, float, float, float, int]:
    """Regularized Newton descent on the negative log-posterior.

    L-BFGS stalls where the posterior is non-convex (saddles) or where a
    prior wall meets a downhill direction; a Newton step with the absolute
    value of the prior-scaled Hessian spectrum (eigenvalues floored at 1e-6
    of the largest) and a backtracking line search descends through both.
    Stops when the scaled gradient norm falls below ``grad_tol`` or no step
    decreases the objective. Costs ``2 * n_params`` gradient evaluations
    per step with ``hessian_method='fd'``.

    Returns ``(theta, neg_logpost, grad_norm, min_eigenvalue, n_steps_taken)``
    where the eigenvalue is of the unregularized prior-scaled Hessian at the
    returned point (negative: not a local maximum).
    """
    val_and_grad = task.get_log_posterior_and_grad_fn()

    def f_and_g(theta):
        v, g = val_and_grad(jnp.asarray(theta))
        return float(-v), np.asarray(-g, dtype=np.float64) * scale

    def scaled_hessian(theta):
        t = jnp.asarray(np.asarray(theta, dtype=np.float64))
        if hessian_method == 'ad':
            H = np.asarray(task._ad_hessian(t), dtype=np.float64)
        else:
            H = task._fd_hessian(val_and_grad, t, scale, fd_rel_step)
        Hn = (scale[:, None] * H) * scale[None, :]
        return 0.5 * (Hn + Hn.T)

    theta = np.array(theta0, dtype=np.float64)
    f, g = f_and_g(theta)
    w, V = np.linalg.eigh(scaled_hessian(theta))
    taken = 0
    for _ in range(n_steps):
        if np.linalg.norm(g) < grad_tol and w.min() > 0:
            break
        w_reg = np.maximum(np.abs(w), 1e-6 * np.abs(w).max())
        step_u = -(V * (1.0 / w_reg)) @ (V.T @ g)
        alpha = 1.0
        accepted = False
        while alpha > 1e-6:
            cand = theta + alpha * scale * step_u
            f_new, g_new = f_and_g(cand)
            if np.isfinite(f_new) and f_new < f:
                accepted = True
                break
            alpha *= 0.5
        if not accepted:
            break
        theta, f, g = cand, f_new, g_new
        w, V = np.linalg.eigh(scaled_hessian(theta))
        taken += 1
    return theta, f, float(np.linalg.norm(g)), float(w.min()), taken


def find_map(
    task: 'InferenceTask',
    starts: StartSet,
    *,
    maxiter: int = 2000,
    seed: int = 0,
    basin_tol: float = 0.25,
    bounded: bool = True,
    polish_steps: int = 8,
    polish_basins: int = 3,
    hessian_method: str = 'fd',
    fd_rel_step: float = 1e-5,
) -> MapResult:
    """Multi-start L-BFGS-B for the MAP in prior-scaled coordinates.

    The optimizer runs in ``u`` with ``theta = loc + scale * u``
    (``loc``/``scale`` = mean/std of 512 prior draws, ``PRNGKey(seed)``).
    ``bounded=True`` (default): the prior support bounds are handed to
    L-BFGS-B, whose projected gradient slides along a wall.
    ``bounded=False``: unbounded, out-of-support iterates get ``-inf`` from
    the prior, a soft barrier -- a start whose steepest direction points
    through a prior wall stalls there with a large in-support gradient.
    Every start with a finite endpoint is kept, converged or not: the best
    finite objective is the MAP (a high-SNR posterior can need more than
    ``maxiter`` iterations to satisfy L-BFGS-B's test, and a formally
    converged start on a plateau must not win over it).

    ``polish_steps > 0`` (default 8) runs ``newton_polish`` from the best
    endpoint of each of the ``polish_basins`` best basins (regularized Newton with a
    line search descends through saddles and along walls where L-BFGS
    stops); the basin records are then rebuilt from the polished endpoints
    and ``map_grad_norm`` / ``map_min_eigenvalue`` report the final MAP's
    stationarity.

    Raises ``RuntimeError`` when no start reaches a finite objective.
    """
    from scipy.optimize import minimize

    if tuple(starts.names) != tuple(task.sampled_names):
        raise ValueError("find_map: StartSet parameter names do not match the task")
    if polish_steps < 0 or polish_basins < 1:
        raise ValueError("find_map: polish_steps must be >= 0 and polish_basins >= 1")
    val_and_grad = task.get_log_posterior_and_grad_fn()
    prior_batch = np.asarray(task.sample_prior(jax.random.PRNGKey(seed), n_samples=512))
    loc = prior_batch.mean(axis=0)
    scale = prior_batch.std(axis=0)
    scale = np.where(scale > 0, scale, 1.0)
    periods = task.priors.get_periods()
    n_evals = 0

    def neg_u(u):
        nonlocal n_evals
        n_evals += 1
        v, g = val_and_grad(jnp.asarray(loc + scale * u))
        return float(-v), np.asarray(-g, dtype=np.float64) * scale

    u_bounds = support_bounds_scaled(task, loc, scale) if bounded else None
    t0 = time.perf_counter()
    kept_start, kept_end, kept_obj, kept_lab, kept_conv = [], [], [], [], []
    for s0, lab in zip(starts.points, starts.labels):
        u0 = (s0 - loc) / scale
        if u_bounds is not None:
            u0 = np.array(
                [
                    np.clip(
                        u,
                        lo if lo is not None else -np.inf,
                        hi if hi is not None else np.inf,
                    )
                    for u, (lo, hi) in zip(u0, u_bounds)
                ]
            )
        res = minimize(
            neg_u,
            u0,
            jac=True,
            method='L-BFGS-B',
            bounds=u_bounds,
            options={'maxiter': maxiter},
        )
        if not np.isfinite(res.fun):
            continue
        kept_start.append(np.asarray(s0, dtype=np.float64))
        kept_end.append(_wrap_periodic(loc + scale * res.x, periods))
        kept_obj.append(float(res.fun))
        kept_lab.append(lab)
        kept_conv.append(bool(res.success))
    if not kept_obj:
        raise RuntimeError(
            f"find_map: no optimization start reached a finite log-posterior "
            f"(tried {len(starts)}). Check priors/data."
        )
    end_points = np.array(kept_end)
    objectives = np.array(kept_obj)
    basin_ids, basin_points, basin_objectives = cluster_basins(
        end_points, objectives, scale, periods, tol=basin_tol
    )
    polish_gain = 0.0
    grad_norm = float('nan')
    min_eig = float('nan')
    if polish_steps > 0:
        # polish the best endpoint of each leading basin in place, then
        # re-cluster: a polished endpoint may merge basins or reorder them
        for k in range(min(polish_basins, len(basin_objectives))):
            members = np.where(basin_ids == k)[0]
            i = members[np.argmin(objectives[members])]
            theta_p, f_p, gn, me, taken = newton_polish(
                task,
                end_points[i],
                scale,
                n_steps=polish_steps,
                hessian_method=hessian_method,
                fd_rel_step=fd_rel_step,
            )
            # the polish evaluates the gradient and Hessian once before its
            # first step, then once per step taken
            n_evals += (taken + 1) * (2 * len(scale) + 1)
            polish_gain = max(polish_gain, objectives[i] - f_p)
            end_points[i] = _wrap_periodic(theta_p, periods)
            objectives[i] = f_p
            if k == 0 or f_p < objectives.min() + 1e-12:
                grad_norm, min_eig = gn, me
        basin_ids, basin_points, basin_objectives = cluster_basins(
            end_points, objectives, scale, periods, tol=basin_tol
        )
    best = int(np.argmin(objectives))
    if not kept_conv[best]:
        warnings.warn(
            "find_map: best MAP start did not satisfy L-BFGS-B convergence "
            f"(neg_logpost={objectives[best]:.4g}); using it anyway as the "
            f"closest of {len(starts)} starts to the mode. Raise maxiter "
            f"(currently {maxiter}) if sampling diverges.",
            RuntimeWarning,
        )
    return MapResult(
        theta_map=end_points[best].copy(),
        neg_logpost=float(objectives[best]),
        start_points=np.array(kept_start),
        end_points=end_points,
        objectives=objectives,
        labels=kept_lab,
        converged=np.array(kept_conv),
        basin_ids=basin_ids,
        basin_points=basin_points,
        basin_objectives=basin_objectives,
        loc=loc,
        scale=scale,
        names=tuple(starts.names),
        n_evals=n_evals,
        wall_s=time.perf_counter() - t0,
        polish_gain=float(polish_gain),
        map_grad_norm=grad_norm,
        map_min_eigenvalue=min_eig,
    )


# ==============================================================================
# Laplace metric
# ==============================================================================


@dataclass(frozen=True)
class EigenFloor:
    """Eigenvalue floor for the prior-scaled MAP Hessian.

    ``mode='prior'`` (default): eigenvalues below ``value`` are raised to it,
    in units where 1 means the posterior is as wide as the prior along that
    direction (default 0.5: a direction is never made stiffer than sqrt(2)
    prior widths). ``mode='relative'``: eigenvalues below ``value *
    max_eigenvalue`` are raised to it (condition number capped at
    ``1/value``; default 1e-4). The prior floor never touches a direction
    the data constrain; the relative floor clips soft directions whenever
    the stiffest one is more than ``1/value`` times stiffer.
    """

    mode: str = 'prior'
    value: Optional[float] = None

    _DEFAULTS = {'relative': 1e-4, 'prior': 0.5}

    def __post_init__(self):
        if self.mode not in self._DEFAULTS:
            raise ValueError(
                f"EigenFloor mode must be one of {sorted(self._DEFAULTS)}, got {self.mode!r}"
            )
        if self.value is None:
            object.__setattr__(self, 'value', self._DEFAULTS[self.mode])
        if not (np.isfinite(self.value) and self.value > 0):
            raise ValueError(
                f"EigenFloor value must be a positive float, got {self.value!r}"
            )

    def threshold(self, eigenvalues: np.ndarray) -> float:
        if self.mode == 'relative':
            return float(self.value * np.max(eigenvalues))
        return float(self.value)


@dataclass
class MetricResult:
    """Regularized inverse Hessian and its spectrum diagnostics."""

    inverse_mass_matrix: np.ndarray
    eigenvalues: np.ndarray  # prior-scaled, unfloored, ascending
    eigenvectors: np.ndarray  # columns, prior-scaled
    floor: EigenFloor
    floor_threshold: float
    n_floored: int
    n_negative: int
    condition_number: float

    @property
    def min_eigenvalue_ratio(self) -> float:
        return float(self.eigenvalues.min() / self.eigenvalues.max())


def laplace_metric(
    task: 'InferenceTask',
    theta_map: np.ndarray,
    scale: np.ndarray,
    *,
    floor: EigenFloor = EigenFloor(),
    hessian_method: str = 'fd',
    fd_rel_step: float = 1e-5,
) -> MetricResult:
    """Regularized inverse Hessian of the negative log-posterior at the MAP.

    The Hessian is scale-normalized (``diag(scale) H diag(scale)``) before
    flooring so the floor acts on genuine degeneracy, not on the benign
    physical scale spread (vcirc ~ 200 vs g1 ~ 0.02); the inverse is mapped
    back with the scale kept.
    """
    if hessian_method not in ('ad', 'fd'):
        raise ValueError(f"hessian_method must be 'ad' or 'fd', got '{hessian_method}'")
    theta = jnp.asarray(np.asarray(theta_map, dtype=np.float64))
    if hessian_method == 'ad':
        H = np.asarray(task._ad_hessian(theta), dtype=np.float64)
    else:
        H = task._fd_hessian(
            task.get_log_posterior_and_grad_fn(), theta, scale, fd_rel_step
        )
    H = 0.5 * (H + H.T)
    Hn = (scale[:, None] * H) * scale[None, :]
    Hn = 0.5 * (Hn + Hn.T)
    w, V = np.linalg.eigh(Hn)
    thr = floor.threshold(w)
    w_floored = np.maximum(w, thr)
    inv_n = np.linalg.inv((V * w_floored) @ V.T)
    inv_mass = (scale[:, None] * inv_n) * scale[None, :]
    inv_mass = 0.5 * (inv_mass + inv_mass.T)
    return MetricResult(
        inverse_mass_matrix=inv_mass,
        eigenvalues=w,
        eigenvectors=V,
        floor=floor,
        floor_threshold=thr,
        n_floored=int(np.sum(w < thr)),
        n_negative=int(np.sum(w < 0)),
        condition_number=float(w_floored.max() / w_floored.min()),
    )


def build_preconditioner(
    task: 'InferenceTask',
    map_result: MapResult,
    *,
    floor: EigenFloor = EigenFloor(),
    hessian_method: str = 'fd',
    fd_rel_step: float = 1e-5,
    metric: Optional[MetricResult] = None,
) -> 'LaplacePreconditioner':
    """MAP + Laplace metric packaged for ``NumpyroSampler``.

    ``metric`` (a ``laplace_metric`` result at ``map_result.theta_map``)
    skips the Hessian evaluation when the caller already has it; its floor
    then takes precedence over ``floor``.
    """
    from kl_pipe.sampling.task import LaplacePreconditioner

    if metric is None:
        metric = laplace_metric(
            task,
            map_result.theta_map,
            map_result.scale,
            floor=floor,
            hessian_method=hessian_method,
            fd_rel_step=fd_rel_step,
        )
    floor = metric.floor
    return LaplacePreconditioner(
        map_point=np.asarray(map_result.theta_map, dtype=np.float64),
        inverse_mass_matrix=metric.inverse_mass_matrix,
        n_starts_converged=map_result.n_converged,
        condition_number=metric.condition_number,
        n_negative_eigenvalues=metric.n_negative,
        min_eigenvalue_ratio=metric.min_eigenvalue_ratio,
        start_map_points=map_result.end_points,
        start_neg_logposts=map_result.objectives,
        start_labels=list(map_result.labels),
        basin_points=map_result.basin_points,
        basin_neg_logposts=map_result.basin_objectives,
        n_floored_eigenvalues=metric.n_floored,
        eig_floor_mode=floor.mode,
        eig_floor_value=float(floor.value),
        map_grad_norm=map_result.map_grad_norm,
        map_min_eigenvalue=map_result.map_min_eigenvalue,
        polish_gain=map_result.polish_gain,
    )


# ==============================================================================
# Chain initial points
# ==============================================================================

CHAIN_INIT_MODES = ('map_jitter', 'map_basins')


def chain_inits(
    pre: 'LaplacePreconditioner',
    inv_mass: np.ndarray,
    n_chains: int,
    *,
    mode: str = 'map_jitter',
    seed: int = 0,
    transform: Optional['UnconstrainingTransform'] = None,
    jitter_frac: float = 0.01,
    max_margin: float = 20.0,
) -> np.ndarray:
    """Initial point per chain, in sampling coordinates.

    ``'map_jitter'``: every chain at the MAP plus ``jitter_frac`` of the
    per-dimension metric scale (``sqrt(diag(inv_mass))``) times a standard
    normal (``PRNGKey(seed)``); a single chain sits exactly at the MAP.
    ``'map_basins'``: chain 0 at the MAP; the next chains at the best
    endpoint of each competing basin whose objective is within
    ``max_margin`` nats of the MAP (basins the posterior can actually
    weigh); remaining chains jittered about the MAP as above. Lets r-hat
    detect a posterior that disagrees with the MAP basin instead of hiding
    it behind four chains started in the same place.
    """
    if mode not in CHAIN_INIT_MODES:
        raise ValueError(
            f"chain_inits mode must be one of {CHAIN_INIT_MODES}, got {mode!r}"
        )
    if n_chains < 1:
        raise ValueError(f"n_chains must be >= 1, got {n_chains}")
    inv_mass = np.asarray(inv_mass, dtype=np.float64)
    n_params = inv_mass.shape[0]

    def to_sampling(theta: np.ndarray) -> np.ndarray:
        if transform is None:
            return np.asarray(theta, dtype=np.float64)
        eta, _ = transform.forward_clipped(
            np.asarray(theta, dtype=np.float64), u_margin=1e-6
        )
        return np.asarray(eta, dtype=np.float64)

    center = to_sampling(pre.map_point)
    if n_chains == 1:
        return center[None, :]
    post_scale = np.sqrt(np.diag(inv_mass))
    noise = np.asarray(
        jax.random.normal(jax.random.PRNGKey(seed), (n_chains, n_params))
    )
    inits = center[None, :] + jitter_frac * post_scale[None, :] * noise
    if mode == 'map_basins':
        if pre.basin_points is None or pre.basin_neg_logposts is None:
            raise ValueError(
                "chain_inits mode 'map_basins' needs a preconditioner with basin "
                "records (build it with build_preconditioner / find_map)"
            )
        margins = pre.basin_neg_logposts - pre.basin_neg_logposts[0]
        competing = [
            pre.basin_points[k]
            for k in range(1, len(margins))
            if margins[k] <= max_margin
        ]
        inits[0] = center
        for c, theta in enumerate(competing[: n_chains - 1], start=1):
            inits[c] = to_sampling(theta)
    return inits


# ==============================================================================
# One-call procedure: InitConfig -> Initializer.run() -> InitResult
# ==============================================================================


@dataclass(frozen=True)
class InitConfig:
    """Settings of the whole initialization procedure.

    Field names and defaults are those of the ensemble spec's ``fit.*``
    initialization knobs (``InitConfig.from_spec`` copies them), plus the
    position-angle start count and the optimizer internals the spec does not
    expose.

    Attributes
    ----------
    n_map_starts : int
        Prior-draw optimizer starts.
    n_pa_starts : int
        Position-angle-stratified starts per rotation direction (``2 *
        n_pa_starts`` on a periodic prior); 0 = none.
    map_moment_starts : bool
        Add image-moment starts (needs broadband ``image_obs``).
    map_bounded : bool
        Projected L-BFGS-B on the prior support instead of the ``-inf``
        barrier.
    map_polish_steps, map_polish_basins : int
        Regularized Newton polish steps from the best endpoint of each of the
        leading basins (0 steps = off).
    eig_floor_mode, eig_floor : str, float or None
        Laplace-metric eigenvalue floor rule (``'prior'`` | ``'relative'``)
        and value (None = the rule's default: 0.5 prior, 1e-4 relative).
    chain_init, chain_init_max_margin : str, float
        Chain initial points: ``'map_jitter'`` or ``'map_basins'`` (competing
        basins within ``chain_init_max_margin`` nats of the MAP).
    hessian_method : {'fd', 'ad'}
        Hessian for the metric and the polish.
    maxiter : int
        L-BFGS-B iteration cap per start.
    fd_rel_step : float
        Relative step of the finite-difference Hessian (prior-scale units).
    """

    n_map_starts: int = 4
    n_pa_starts: int = 4
    map_moment_starts: bool = False
    map_bounded: bool = True
    map_polish_steps: int = 8
    map_polish_basins: int = 3
    eig_floor_mode: str = 'prior'
    eig_floor: Optional[float] = None
    chain_init: str = 'map_jitter'
    chain_init_max_margin: float = 20.0
    hessian_method: str = 'fd'
    maxiter: int = 2000
    fd_rel_step: float = 1e-5

    def __post_init__(self):
        if self.n_map_starts < 1:
            raise ValueError(f"n_map_starts must be >= 1, got {self.n_map_starts}")
        if self.n_pa_starts < 0:
            raise ValueError(f"n_pa_starts must be >= 0, got {self.n_pa_starts}")
        if self.map_polish_steps < 0 or self.map_polish_basins < 1:
            raise ValueError(
                "map_polish_steps must be >= 0 and map_polish_basins >= 1, got "
                f"{self.map_polish_steps} / {self.map_polish_basins}"
            )
        if self.hessian_method not in ('ad', 'fd'):
            raise ValueError(
                f"hessian_method must be 'ad' or 'fd', got {self.hessian_method!r}"
            )
        if self.chain_init not in CHAIN_INIT_MODES:
            raise ValueError(
                f"chain_init must be one of {CHAIN_INIT_MODES}, got {self.chain_init!r}"
            )
        if not self.chain_init_max_margin > 0:
            raise ValueError(
                f"chain_init_max_margin must be > 0, got {self.chain_init_max_margin}"
            )
        if self.maxiter < 1:
            raise ValueError(f"maxiter must be >= 1, got {self.maxiter}")
        # validates the floor rule and value
        self.floor

    @property
    def floor(self) -> EigenFloor:
        return EigenFloor(self.eig_floor_mode, self.eig_floor)

    @classmethod
    def from_spec(cls, spec, **overrides) -> 'InitConfig':
        """Copy the ``fit.*`` initialization knobs of an ``EnsembleSpec``."""
        fields = dict(
            n_map_starts=spec.n_map_starts,
            map_moment_starts=spec.map_moment_starts,
            map_bounded=spec.map_bounded,
            map_polish_steps=spec.map_polish_steps,
            map_polish_basins=spec.map_polish_basins,
            eig_floor_mode=spec.eig_floor_mode,
            eig_floor=spec.eig_floor,
            chain_init=spec.chain_init,
            chain_init_max_margin=spec.chain_init_max_margin,
            hessian_method=spec.hessian_method,
        )
        fields.update(overrides)
        return cls(**fields)


@dataclass
class InitResult:
    """What one ``Initializer.run()`` produced.

    Attributes
    ----------
    starts : StartSet
        Every optimizer start with its family label.
    map : MapResult
        MAP finder output (endpoints, basins, stationarity).
    metric : MetricResult
        Laplace metric with its spectrum diagnostics.
    preconditioner : LaplacePreconditioner
        MAP + metric packaged for ``NumpyroSampler``.
    moment_starts_ok : bool or None
        None when moment starts were not requested, False when the images
        yielded no usable moments (the procedure went on without them).
    config : InitConfig
    """

    starts: StartSet
    map: MapResult
    metric: MetricResult
    preconditioner: 'LaplacePreconditioner'
    moment_starts_ok: Optional[bool]
    config: InitConfig

    def chain_points(
        self,
        inverse_mass_matrix: np.ndarray,
        n_chains: int,
        *,
        seed: int = 0,
        transform: Optional['UnconstrainingTransform'] = None,
    ) -> np.ndarray:
        """Initial point per chain in sampling coordinates (``chain_inits``
        with this result's ``chain_init`` settings)."""
        return chain_inits(
            self.preconditioner,
            inverse_mass_matrix,
            n_chains,
            mode=self.config.chain_init,
            seed=seed,
            transform=transform,
            max_margin=self.config.chain_init_max_margin,
        )

    def columns(self) -> Dict[str, object]:
        """Per-fit summary columns describing what the initialization did."""
        return initialization_columns(
            self.preconditioner, self.moment_starts_ok, self.config.chain_init
        )


def initialization_columns(
    preconditioner: Optional['LaplacePreconditioner'],
    moment_starts_ok: Optional[bool],
    chain_init: str,
) -> Dict[str, object]:
    """Summary columns from a preconditioner's initialization records.

    ``map_n_basins`` / ``map_basin_margin`` (nats between the MAP basin and
    the runner-up; inf with one basin), ``map_winning_start`` (family of the
    start that produced the MAP), ``map_moment_starts_ok`` ('yes' | 'no' |
    'n/a'), ``map_grad_norm`` / ``map_min_eigenvalue`` / ``map_polish_gain``
    (stationarity of the final MAP), ``precond_n_floored_eigenvalues`` /
    ``precond_eig_floor_mode`` and ``chain_init``. A preconditioner without
    records (or None) yields the sentinels -1 / nan / ''.
    """
    labels = getattr(preconditioner, 'start_labels', None)
    objectives = getattr(preconditioner, 'start_neg_logposts', None)
    basins = getattr(preconditioner, 'basin_neg_logposts', None)
    if basins is None:
        margin = float('nan')
    elif len(basins) > 1:
        margin = float(basins[1] - basins[0])
    else:
        margin = float('inf')
    return {
        'map_n_basins': int(len(basins)) if basins is not None else -1,
        'map_basin_margin': margin,
        'map_winning_start': (
            str(labels[int(np.argmin(objectives))])
            if labels is not None and objectives is not None
            else ''
        ),
        'map_moment_starts_ok': (
            'n/a' if moment_starts_ok is None else ('yes' if moment_starts_ok else 'no')
        ),
        'precond_n_floored_eigenvalues': int(
            getattr(preconditioner, 'n_floored_eigenvalues', -1)
        ),
        'precond_eig_floor_mode': str(getattr(preconditioner, 'eig_floor_mode', '')),
        'map_grad_norm': float(getattr(preconditioner, 'map_grad_norm', np.nan)),
        'map_min_eigenvalue': float(
            getattr(preconditioner, 'map_min_eigenvalue', np.nan)
        ),
        'map_polish_gain': float(getattr(preconditioner, 'polish_gain', np.nan)),
        'chain_init': str(chain_init),
    }


class Initializer:
    """The initialization procedure for one task.

    ``run()`` performs every stage and returns an ``InitResult``; the stages
    are also callable one at a time (``starts``, ``find_map``, ``metric``,
    ``preconditioner``) for inspection.

    Parameters
    ----------
    task : InferenceTask
    config : InitConfig, optional
        Defaults = the robust procedure.
    seed : int
        Seeds the prior-draw and position-angle start streams and the prior
        scaling draws.
    image_obs : dict of ImageObs, optional
        Broadband images for the moment starts (required when
        ``config.map_moment_starts``).
    """

    def __init__(
        self,
        task: 'InferenceTask',
        config: Optional[InitConfig] = None,
        seed: int = 0,
        image_obs: Optional[Dict[str, 'ImageObs']] = None,
    ):
        self.task = task
        self.config = config if config is not None else InitConfig()
        self.seed = int(seed)
        self.image_obs = image_obs
        self.moment_starts_ok: Optional[bool] = None
        if self.config.map_moment_starts and not image_obs:
            raise ValueError("InitConfig.map_moment_starts needs broadband image_obs")

    def starts(self) -> StartSet:
        """Prior-draw, position-angle-stratified and (optionally) image-moment
        starts. A moment failure is a warning: ``moment_starts_ok`` records
        it and the other families carry the search."""
        cfg = self.config
        sets: List[Optional[StartSet]] = [
            prior_starts(self.task, cfg.n_map_starts, seed=self.seed)
        ]
        if cfg.n_pa_starts > 0:
            sets.append(
                pa_stratified_starts(
                    self.task.priors, n_pa=cfg.n_pa_starts, seed=self.seed
                )
            )
        self.moment_starts_ok = None
        if cfg.map_moment_starts:
            try:
                sets.append(moment_starts(self.task, self.image_obs, seed=self.seed))
                self.moment_starts_ok = True
            except MomentsError as err:
                self.moment_starts_ok = False
                warnings.warn(
                    f"image-moment starts unavailable ({err}); MAP search proceeds "
                    "on the prior and position-angle starts only",
                    RuntimeWarning,
                )
        return combine_starts(*sets)

    def find_map(self, starts: Optional[StartSet] = None) -> MapResult:
        cfg = self.config
        return find_map(
            self.task,
            starts if starts is not None else self.starts(),
            maxiter=cfg.maxiter,
            seed=self.seed,
            bounded=cfg.map_bounded,
            polish_steps=cfg.map_polish_steps,
            polish_basins=cfg.map_polish_basins,
            hessian_method=cfg.hessian_method,
            fd_rel_step=cfg.fd_rel_step,
        )

    def metric(self, map_result: MapResult) -> MetricResult:
        cfg = self.config
        return laplace_metric(
            self.task,
            map_result.theta_map,
            map_result.scale,
            floor=cfg.floor,
            hessian_method=cfg.hessian_method,
            fd_rel_step=cfg.fd_rel_step,
        )

    def preconditioner(
        self, map_result: MapResult, metric: Optional[MetricResult] = None
    ) -> 'LaplacePreconditioner':
        cfg = self.config
        return build_preconditioner(
            self.task,
            map_result,
            floor=cfg.floor,
            hessian_method=cfg.hessian_method,
            fd_rel_step=cfg.fd_rel_step,
            metric=metric,
        )

    def run(self) -> InitResult:
        """Starts -> MAP -> metric -> preconditioner."""
        starts = self.starts()
        map_result = self.find_map(starts)
        metric = self.metric(map_result)
        return InitResult(
            starts=starts,
            map=map_result,
            metric=metric,
            preconditioner=self.preconditioner(map_result, metric),
            moment_starts_ok=self.moment_starts_ok,
            config=self.config,
        )
