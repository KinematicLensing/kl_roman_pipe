"""Gradient-based optimization utilities.

Currently provides ``multi_start_minimize`` — a multi-start wrapper around
``scipy.optimize.minimize`` for likelihoods with boundary attractors or
shallow local minima (e.g. galaxy bulge+disk decomposition where the
``bulge_frac=0`` boundary is a documented L-BFGS-B trap).

References
----------
Sheth+ 2010 (S4G survey, 2MASS-based decomposition recommendation).
Erwin 2015 (IMFIT, ApJ 799:226) — explicitly recommends multi-start.
Robotham+ 2017 (ProFit, MNRAS 466:1513) — Bayesian alternative; multi-start
    used as warm-start for MCMC.
"""

from typing import Callable, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import OptimizeResult, minimize


def multi_start_minimize(
    objective: Callable,
    x0: np.ndarray,
    bounds: Optional[Sequence[Tuple[float, float]]] = None,
    n_starts: int = 10,
    perturbation: float = 0.2,
    method: str = 'L-BFGS-B',
    seed: int = 42,
    fixed_indices: Optional[Sequence[int]] = None,
    **minimize_kwargs,
) -> OptimizeResult:
    """Multi-start gradient minimizer.

    Runs ``scipy.optimize.minimize`` from ``n_starts`` perturbed initial
    points and returns the result with the lowest final objective value.
    Multi-start is the standard remedy for likelihoods with boundary
    attractors or shallow local minima — single-start L-BFGS-B reliably
    converges to the wrong basin in galaxy bulge+disk decomposition (the
    ``bulge_frac=0`` attractor; see refs in module docstring).

    Each start is perturbed by ``perturbation * scale * randn``, clipped
    into ``bounds`` (when supplied). Per-parameter ``scale`` is ``|x0[i]|``
    when nonzero; for parameters with ``x0[i] == 0`` it falls back to the
    bound width ``hi - lo`` (when bounded) or a small fixed floor (1e-3,
    when unbounded), so zero-initial parameters still get perturbed across
    starts. The first start uses the unperturbed ``x0`` to preserve the
    single-start solution as a baseline.

    Parameters
    ----------
    objective : callable
        Function ``x -> (value, grad)`` if ``minimize_kwargs['jac']=True``,
        else ``x -> value``. Identical contract to ``scipy.optimize.minimize``.
    x0 : np.ndarray
        Nominal initial parameter vector. Shape ``(n_params,)``.
    bounds : sequence of (lo, hi), optional
        Per-parameter bounds. Same format as ``scipy.optimize.minimize``.
        Perturbed initial points are clipped into these bounds.
    n_starts : int, default 10
        Number of independent optimization runs.
    perturbation : float, default 0.2
        Fractional standard deviation of initial-point perturbation,
        relative to ``|x0|``. Literature recommends 10-20%.
    method : str, default 'L-BFGS-B'
        Optimizer method (passed to ``scipy.optimize.minimize``).
    seed : int, default 42
        Seed for ``np.random.RandomState`` controlling perturbations.
    fixed_indices : sequence of int, optional
        Indices of ``x0`` that should NOT be perturbed (e.g. parameters
        with bounds collapsed to a single point). Useful when the caller
        bounds a parameter to ``(c, c)`` to fix it.
    **minimize_kwargs
        Forwarded to ``scipy.optimize.minimize`` (e.g. ``jac=True``,
        ``options={'maxiter': 2000}``).

    Returns
    -------
    scipy.optimize.OptimizeResult
        The best run's result, augmented with:

        - ``n_starts_attempted`` : int — number of starts that ran.
        - ``start_results`` : list[OptimizeResult] — all runs in order.
        - ``best_start_index`` : int — index into ``start_results`` of the
          returned best run.
        - ``start_objectives`` : np.ndarray — final objective per start.

    Notes
    -----
    Reproducibility: with the same ``seed``, ``x0``, and ``perturbation``,
    the perturbed initial points are deterministic.

    Bound handling: perturbed starts are clipped into bounds with a small
    interior buffer (1e-9 of the bound width) to avoid placing starts
    exactly on a bound, which can degrade L-BFGS-B convergence.
    """
    x0 = np.asarray(x0, dtype=np.float64)
    n_params = x0.size
    if n_starts < 1:
        raise ValueError(f'n_starts must be >= 1, got {n_starts}')
    if perturbation < 0:
        raise ValueError(f'perturbation must be >= 0, got {perturbation}')

    fixed_mask = np.zeros(n_params, dtype=bool)
    if fixed_indices is not None:
        fixed_mask[list(fixed_indices)] = True

    rng = np.random.RandomState(seed)

    if bounds is not None:
        bounds = list(bounds)
        if len(bounds) != n_params:
            raise ValueError(f'bounds length {len(bounds)} != n_params {n_params}')

    # per-parameter perturbation scale: |x0| for nonzero, bound width for
    # x0=0 with bounds, small floor otherwise. Ensures zero-initial params
    # actually get perturbed across starts (otherwise multi-start collapses
    # to single-start for those dims).
    abs_x0 = np.abs(x0)
    scale = abs_x0.copy()
    zero_mask = abs_x0 == 0
    if zero_mask.any():
        if bounds is not None:
            widths = np.array(
                [
                    (
                        (hi - lo)
                        if (lo is not None and hi is not None and hi > lo)
                        else 0.0
                    )
                    for (lo, hi) in bounds
                ]
            )
            scale[zero_mask] = widths[zero_mask]
        # anything still zero (unbounded x0=0): small fixed floor
        scale[scale == 0] = 1e-3

    starts = [x0.copy()]  # first start: unperturbed, preserves baseline
    for _ in range(n_starts - 1):
        noise = rng.randn(n_params)
        delta = perturbation * scale * noise
        x_try = x0 + delta
        x_try[fixed_mask] = x0[fixed_mask]  # never perturb fixed params
        if bounds is not None:
            for i, (lo, hi) in enumerate(bounds):
                if lo is not None and hi is not None:
                    width = hi - lo
                    if width > 0:
                        buffer = 1e-9 * width
                        x_try[i] = np.clip(x_try[i], lo + buffer, hi - buffer)
                    else:
                        # collapsed bound (lo == hi): pin to the bound
                        x_try[i] = lo
                elif lo is not None:
                    x_try[i] = max(x_try[i], lo)
                elif hi is not None:
                    x_try[i] = min(x_try[i], hi)
        starts.append(x_try)

    results = []
    objectives = np.full(n_starts, np.inf)
    for i, x_init in enumerate(starts):
        res = minimize(
            objective,
            x_init,
            method=method,
            bounds=bounds,
            **minimize_kwargs,
        )
        results.append(res)
        if res.success or np.isfinite(res.fun):
            objectives[i] = float(res.fun)

    best_idx = int(np.argmin(objectives))
    best = results[best_idx]
    best.n_starts_attempted = n_starts
    best.start_results = results
    best.best_start_index = best_idx
    best.start_objectives = objectives
    return best
