"""
Posterior-surface slices of a saved ensemble fit.

Rebuilds the exact log-posterior of one fit from its run directory (manifest
row + frozen spec/observation config regenerate the mock deterministically from
the row's noise seed; the saved mocks npz, when present, must match the rebuilt
data to roundoff) and evaluates it on 2D or 3D grids.

Two slice modes:

- ``conditional``: every other sampled parameter is held at the slice center.
  Through the MAP this is the local curvature of the posterior and is much
  narrower than the marginal spread whenever the posterior is a ridge that
  mixes many parameters.
- ``profile``: at every grid point the log-posterior is maximized over the
  remaining parameters (L-BFGS-B on the JAX gradient, warm-started from the
  neighbouring grid point). This follows the ridge and is the surface whose
  width is comparable to the chain's marginal scatter.

Slice axes are either parameter pairs or eigenvectors of the posterior
covariance (chains, or the inverse MAP Hessian when chains were not saved)
expressed in per-parameter posterior-sigma units; eigen-slice coordinates are
in units of the mode's own sigma.

CLI::

    python -m kl_pipe.diagnostics.posterior_slices --run-dir R --fit-id F \\
        --planes g2,theta_int eigen:0,1 --mode profile --n 40 --out DIR \\
        [--iso g2,theta_int,cosi --n3 18]
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from scipy.linalg import null_space
from scipy.optimize import minimize

import kl_pipe  # noqa: F401  (precision setup: float64 by default)
from kl_pipe.sampling.task import InferenceTask

# delta log P contour levels and their Gaussian-equivalent labels (1 dof)
SIGMA_LEVELS: Dict[float, str] = {-0.5: '1 sigma', -2.0: '2 sigma', -4.5: '3 sigma'}

SLICE_MODES = ('conditional', 'profile')


# ==============================================================================
# Fit reconstruction
# ==============================================================================


@dataclass(frozen=True)
class FitPaths:
    """Files needed to rebuild one fit's posterior.

    ``results`` may be the per-fit ``results/<fit_id>.parquet`` or any
    collated summary parquet holding the fit's row (``map.<name>`` columns).
    ``chains`` / ``mocks`` are None when the fit did not save them.
    """

    manifest: Path
    spec: Path
    config: Path
    results: Path
    chains: Optional[Path] = None
    mocks: Optional[Path] = None

    @classmethod
    def from_run_dir(
        cls,
        run_dir: Union[str, Path],
        fit_id: str,
        *,
        results: Optional[Union[str, Path]] = None,
        chains_dir: Optional[Union[str, Path]] = None,
        mocks_dir: Optional[Union[str, Path]] = None,
    ) -> 'FitPaths':
        """Standard ``<run_dir>/{manifest.parquet, provenance/, results/,
        chains/, mocks/}`` layout; explicit ``results`` / ``chains_dir`` /
        ``mocks_dir`` override the run-dir defaults (and must then exist)."""
        run_dir = Path(run_dir)
        if not run_dir.is_dir():
            raise FileNotFoundError(f"run directory {run_dir} does not exist")
        if results is None:
            per_fit = run_dir / 'results' / f'{fit_id}.parquet'
            results = per_fit if per_fit.exists() else run_dir / 'results.parquet'

        def _artifact(explicit, default_dir: Path, kind: str) -> Optional[Path]:
            if explicit is not None:
                path = Path(explicit) / f'{fit_id}.npz'
                if not path.exists():
                    raise FileNotFoundError(f"{kind} file {path} does not exist")
                return path
            path = default_dir / f'{fit_id}.npz'
            return path if path.exists() else None

        return cls(
            manifest=run_dir / 'manifest.parquet',
            spec=run_dir / 'provenance' / 'ensemble_spec.yaml',
            config=run_dir / 'provenance' / 'observation_config.yaml',
            results=Path(results),
            chains=_artifact(chains_dir, run_dir / 'chains', 'chains'),
            mocks=_artifact(mocks_dir, run_dir / 'mocks', 'mocks'),
        )

    def validate(self) -> None:
        required = {
            'manifest': self.manifest,
            'spec': self.spec,
            'config': self.config,
            'results': self.results,
        }
        missing = [f"{k}: {p}" for k, p in required.items() if not Path(p).exists()]
        if missing:
            raise FileNotFoundError("missing fit files: " + "; ".join(missing))


@dataclass
class FitPosterior:
    """Rebuilt posterior of one fit plus the run's MAP, chains, and truth.

    Unpacks as ``task, map_theta, chains, truth, sampled_names``.
    """

    fit_id: str
    task: InferenceTask
    map_theta: np.ndarray
    chains: Optional[np.ndarray]
    truth: Optional[np.ndarray]
    sampled_names: List[str]
    summary: pd.Series
    paths: FitPaths
    rebuild_wall_s: float

    def __iter__(self) -> Iterator:
        yield from (
            self.task,
            self.map_theta,
            self.chains,
            self.truth,
            self.sampled_names,
        )

    def posterior_sigma(self) -> np.ndarray:
        """Per-parameter posterior std: chain std (ddof=1) when chains were
        saved, else the summary row's ``post.<name>.std`` columns."""
        if self.chains is not None:
            return np.std(self.chains, axis=0, ddof=1)
        return np.array(
            [float(self.summary[f'post.{n}.std']) for n in self.sampled_names]
        )

    def index(self, name: str) -> int:
        if name not in self.sampled_names:
            raise KeyError(
                f"'{name}' is not a sampled parameter of fit {self.fit_id}; "
                f"sampled: {self.sampled_names}"
            )
        return self.sampled_names.index(name)


def _load_manifest_row(manifest_path: Path, fit_id: str) -> pd.Series:
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest {manifest_path} does not exist")
    manifest = pd.read_parquet(manifest_path)
    rows = manifest[manifest['fit_id'] == fit_id]
    if len(rows) != 1:
        raise KeyError(
            f"fit_id '{fit_id}' matches {len(rows)} rows in {manifest_path} "
            f"(expected exactly 1)"
        )
    return rows.iloc[0]


def _load_summary_row(results_path: Path, fit_id: str) -> pd.Series:
    results = pd.read_parquet(results_path)
    rows = results[results['fit_id'] == fit_id]
    if len(rows) != 1:
        raise KeyError(
            f"fit_id '{fit_id}' matches {len(rows)} rows in {results_path} "
            f"(expected exactly 1)"
        )
    return rows.iloc[0]


def _verify_mocks(mocks_path: Path, inputs, rtol: float = 1e-10) -> None:
    """Rebuilt data must match the saved mock arrays to floating-point roundoff
    (``rtol`` times the array's peak absolute value; reduction order differs
    across hosts, so bit equality is not required)."""
    saved = np.load(mocks_path)
    pairs = [(f'image.{b}', obs) for b, obs in inputs.image_obs.items()]
    pairs += [(f'grism.{k}', obs) for k, obs in inputs.grism_obs.items()]
    for key, obs in pairs:
        for field_name, value in (('data', obs.data), ('variance', obs.variance)):
            name = f'{key}.{field_name}'
            if name not in saved.files:
                raise RuntimeError(f"{mocks_path} lacks array '{name}'")
            rebuilt = np.asarray(value)
            if saved[name].shape != rebuilt.shape:
                raise RuntimeError(
                    f"rebuilt {name} has shape {rebuilt.shape}, saved mock in "
                    f"{mocks_path} has {saved[name].shape}"
                )
            diff = float(np.max(np.abs(saved[name] - rebuilt)))
            tol = rtol * float(np.max(np.abs(saved[name])))
            if not diff <= tol:
                raise RuntimeError(
                    f"rebuilt {name} differs from the saved mock in {mocks_path} "
                    f"(max abs diff {diff:.3e} > {tol:.3e}): spec/config/code state "
                    f"does not reproduce this fit"
                )


def rebuild_fit_posterior(
    run_dir: Optional[Union[str, Path]] = None,
    fit_id: Optional[str] = None,
    *,
    paths: Optional[FitPaths] = None,
    results: Optional[Union[str, Path]] = None,
    chains_dir: Optional[Union[str, Path]] = None,
    mocks_dir: Optional[Union[str, Path]] = None,
) -> FitPosterior:
    """
    Rebuild the InferenceTask of a saved ensemble fit.

    Parameters
    ----------
    run_dir : path, optional
        Run directory (``manifest.parquet`` + ``provenance/``); ignored when
        ``paths`` is given.
    fit_id : str
        The fit to rebuild.
    paths : FitPaths, optional
        Explicit file locations (alternative to ``run_dir``).
    results, chains_dir, mocks_dir : path, optional
        Overrides for the run-dir defaults (see ``FitPaths.from_run_dir``).

    Returns
    -------
    FitPosterior
        Unpacks as ``task, map_theta, chains, truth, sampled_names``.

    Raises
    ------
    FileNotFoundError
        Missing manifest / spec / config / results file.
    KeyError
        ``fit_id`` absent from the manifest or results table.
    RuntimeError
        Rebuilt mock data or chain parameter order disagree with the saved
        artifacts.
    """
    if fit_id is None:
        raise ValueError("fit_id is required")
    if paths is None:
        if run_dir is None:
            raise ValueError("pass run_dir or paths")
        paths = FitPaths.from_run_dir(
            run_dir, fit_id, results=results, chains_dir=chains_dir, mocks_dir=mocks_dir
        )
    t0 = time.perf_counter()
    row = _load_manifest_row(Path(paths.manifest), fit_id)
    paths.validate()

    from kl_pipe.ensemble.expander import truth_from_row
    from kl_pipe.ensemble.mocks import build_fit_inputs
    from kl_pipe.ensemble.spec import EnsembleSpec, ObservationConfig

    spec = EnsembleSpec.from_yaml(Path(paths.spec))
    config = ObservationConfig.from_yaml(Path(paths.config))
    truth_dict = truth_from_row(row)
    # catalog manifests carry per-band SNR columns; sampled manifests one scalar
    if f'broadband_snr_{config.bands[0]}' in row:
        band_snrs = {b: float(row[f'broadband_snr_{b}']) for b in config.bands}
    else:
        band_snrs = {b: float(row['broadband_snr']) for b in config.bands}
    inputs = build_fit_inputs(
        truth_dict,
        int(row['noise_seed']),
        spec,
        config,
        band_snrs=band_snrs,
        line_snr=float(row['line_snr']),
        row=row,
    )
    task = InferenceTask.from_obs(
        inputs.source,
        inputs.priors,
        image_obs=inputs.image_obs,
        grism_obs=inputs.grism_obs,
    )
    sampled_names = list(task.sampled_names)

    if paths.mocks is not None:
        _verify_mocks(Path(paths.mocks), inputs)

    summary = _load_summary_row(Path(paths.results), fit_id)
    map_theta = np.array([float(summary[f'map.{n}']) for n in sampled_names])
    truth = np.array([float(truth_dict[n]) for n in sampled_names])

    chains = None
    if paths.chains is not None:
        saved = np.load(paths.chains)
        chain_names = [str(n) for n in saved['param_names']]
        if chain_names != sampled_names:
            raise RuntimeError(
                f"chain param_names {chain_names} != rebuilt sampled_names "
                f"{sampled_names}"
            )
        chains = np.asarray(saved['samples'], dtype=np.float64)

    return FitPosterior(
        fit_id=fit_id,
        task=task,
        map_theta=map_theta,
        chains=chains,
        truth=truth,
        sampled_names=sampled_names,
        summary=summary,
        paths=paths,
        rebuild_wall_s=time.perf_counter() - t0,
    )


# ==============================================================================
# Slice axes
# ==============================================================================


@dataclass(frozen=True)
class SliceAxis:
    """One slice direction in the sampled-parameter space.

    ``vector`` is the physical-space step per unit slice coordinate;
    ``projection`` maps an offset from the slice center back to the
    coordinate (``projection @ vector == 1``). ``absolute`` axes (single
    parameters) are plotted as ``center + t``; others (eigen modes) as ``t``.
    """

    label: str
    vector: np.ndarray
    projection: np.ndarray
    absolute: bool
    index: Optional[int] = None


def parameter_axes(
    names: Sequence[str], sampled_names: Sequence[str]
) -> List[SliceAxis]:
    """Coordinate axes for the named sampled parameters (KeyError if unknown)."""
    sampled_names = list(sampled_names)
    axes = []
    for name in names:
        if name not in sampled_names:
            raise KeyError(
                f"'{name}' is not a sampled parameter; sampled: {sampled_names}"
            )
        i = sampled_names.index(name)
        unit = np.zeros(len(sampled_names))
        unit[i] = 1.0
        axes.append(
            SliceAxis(
                label=name, vector=unit, projection=unit.copy(), absolute=True, index=i
            )
        )
    return axes


@dataclass(frozen=True)
class EigenAxes:
    """Eigen-decomposition of the posterior correlation matrix.

    ``directions_sigma[:, k]`` is mode ``k`` in per-parameter sigma units
    (columns orthonormal); ``eigenvalues`` (descending) are the variances
    along the modes in those units, so mode 0 is the softest direction.
    """

    sigma: np.ndarray
    eigenvalues: np.ndarray
    directions_sigma: np.ndarray
    modes: Tuple[int, ...]
    source: str

    def slice_axes(self) -> List[SliceAxis]:
        """Slice axes for ``modes``; one slice-coordinate unit = one mode sigma."""
        axes = []
        for k in self.modes:
            lam = float(self.eigenvalues[k])
            direction = self.directions_sigma[:, k]
            axes.append(
                SliceAxis(
                    label=f'mode {k} (lambda={lam:.2f}) [mode sigma]',
                    vector=self.sigma * direction * np.sqrt(lam),
                    projection=direction / (self.sigma * np.sqrt(lam)),
                    absolute=False,
                )
            )
        return axes

    def loadings(self, names: Sequence[str], k: int, n_top: int = 6) -> str:
        """Largest-|loading| parameters of mode ``k`` (for reports)."""
        v = self.directions_sigma[:, k]
        top = np.argsort(-np.abs(v))[:n_top]
        return ', '.join(f'{names[i]}:{v[i]:+.2f}' for i in top)


def eigen_axes(
    chains: Optional[np.ndarray] = None,
    covariance: Optional[np.ndarray] = None,
    modes: Tuple[int, ...] = (0, 1),
) -> EigenAxes:
    """
    Softest posterior directions in per-parameter sigma units.

    Parameters
    ----------
    chains : ndarray, optional
        Posterior draws, shape (n_draws, n_params).
    covariance : ndarray, optional
        Posterior covariance, shape (n_params, n_params) (e.g. the inverse
        MAP Hessian from ``map_hessian_covariance``). Exactly one of
        ``chains`` / ``covariance`` must be given.
    modes : tuple of int
        Mode indices (0 = softest) to expose as slice axes.
    """
    if (chains is None) == (covariance is None):
        raise ValueError("pass exactly one of chains or covariance")
    if chains is not None:
        chains = np.asarray(chains, dtype=np.float64)
        if chains.ndim != 2 or chains.shape[0] < chains.shape[1] + 1:
            raise ValueError(
                f"chains must be (n_draws, n_params) with n_draws > n_params, "
                f"got {chains.shape}"
            )
        covariance = np.cov(chains, rowvar=False, ddof=1)
        source = 'chains'
    else:
        covariance = np.asarray(covariance, dtype=np.float64)
        source = 'covariance'
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError(f"covariance must be square, got {covariance.shape}")
    variances = np.diag(covariance)
    if not np.all(np.isfinite(variances)) or np.any(variances <= 0):
        raise ValueError(f"covariance diagonal must be positive, got {variances}")
    sigma = np.sqrt(variances)
    corr = covariance / np.outer(sigma, sigma)
    corr = 0.5 * (corr + corr.T)
    eigenvalues, vectors = np.linalg.eigh(corr)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    vectors = vectors[:, order]
    if np.any(eigenvalues <= 0):
        raise ValueError(
            f"correlation matrix is not positive definite (min eigenvalue "
            f"{eigenvalues.min():.3e}); the covariance is degenerate"
        )
    n = len(sigma)
    for k in modes:
        if not (0 <= int(k) < n):
            raise ValueError(f"mode {k} out of range for {n} parameters")
    return EigenAxes(
        sigma=sigma,
        eigenvalues=eigenvalues,
        directions_sigma=vectors,
        modes=tuple(int(k) for k in modes),
        source=source,
    )


def _prior_scale(task: InferenceTask, seed: int = 0) -> np.ndarray:
    """Per-parameter scale from prior draws (the Laplace-preconditioner rule)."""
    draws = np.asarray(task.sample_prior(jax.random.PRNGKey(seed), n_samples=512))
    scale = draws.std(axis=0)
    return np.where(scale > 0, scale, 1.0)


def map_hessian_covariance(
    task: InferenceTask,
    map_theta: np.ndarray,
    hessian_method: str = 'fd',
    fd_rel_step: float = 1e-5,
) -> np.ndarray:
    """
    Inverse Hessian of the negative log-posterior at ``map_theta``.

    Substitute for the chain covariance when a fit saved no chains. Raises
    if the Hessian is not positive definite (``map_theta`` is then not a
    local maximum and no Gaussian covariance exists there).
    """
    theta = jnp.asarray(np.asarray(map_theta, dtype=np.float64))
    if hessian_method == 'ad':
        hessian = np.asarray(task._ad_hessian(theta), dtype=np.float64)
    elif hessian_method == 'fd':
        hessian = task._fd_hessian(
            task.get_log_posterior_and_grad_fn(), theta, _prior_scale(task), fd_rel_step
        )
    else:
        raise ValueError(f"hessian_method must be 'ad' or 'fd', got '{hessian_method}'")
    hessian = 0.5 * (hessian + hessian.T)
    eigenvalues = np.linalg.eigvalsh(hessian)
    if np.any(eigenvalues <= 0):
        raise ValueError(
            f"MAP Hessian has {int(np.sum(eigenvalues <= 0))} non-positive "
            f"eigenvalues (min {eigenvalues.min():.3e}); map_theta is not a "
            f"strict local maximum"
        )
    covariance = np.linalg.inv(hessian)
    return 0.5 * (covariance + covariance.T)


# ==============================================================================
# Grid evaluation
# ==============================================================================


@dataclass
class SliceResult:
    """Log-posterior on a slice grid.

    ``coords[k]`` are the slice coordinates along ``axes[k]`` (offsets from
    ``center`` in units of ``axes[k].vector``); ``logp`` has shape
    ``tuple(len(c) for c in coords)`` (``-inf`` outside the prior support).
    In profile mode ``theta_opt`` holds the maximizing full parameter vector
    at each grid point.
    """

    mode: str
    axes: List[SliceAxis]
    center: np.ndarray
    coords: List[np.ndarray]
    logp: np.ndarray
    logp_center: float
    n_evals: int
    wall_s: float
    theta_opt: Optional[np.ndarray] = None
    n_unconverged: int = 0
    n_below_conditional: int = 0
    meta: Dict[str, object] = field(default_factory=dict)

    @property
    def ndim(self) -> int:
        return len(self.axes)

    @property
    def delta_logp(self) -> np.ndarray:
        """``logp`` relative to the finite grid maximum."""
        finite = np.isfinite(self.logp)
        if not finite.any():
            raise ValueError("no finite log-posterior value on the grid")
        return self.logp - np.max(self.logp[finite])

    def plot_coords(self) -> List[np.ndarray]:
        """Per-axis plotting coordinates (parameter values for absolute axes)."""
        return [
            self.center[ax.index] + c if ax.absolute else c
            for ax, c in zip(self.axes, self.coords)
        ]

    def project(self, theta: np.ndarray) -> np.ndarray:
        """Slice coordinates of full parameter vectors, shape (n, ndim)."""
        theta = np.atleast_2d(np.asarray(theta, dtype=np.float64))
        projection = np.stack([ax.projection for ax in self.axes])
        return (theta - self.center) @ projection.T

    def plot_points(self, theta: np.ndarray) -> np.ndarray:
        """``project`` shifted into plotting coordinates."""
        pts = self.project(theta)
        for k, ax in enumerate(self.axes):
            if ax.absolute:
                pts[:, k] += self.center[ax.index]
        return pts

    def save(self, path: Union[str, Path]) -> Path:
        path = Path(path)
        arrays = {
            'logp': self.logp,
            'center': self.center,
            'logp_center': np.array(self.logp_center),
            'axis_labels': np.array([ax.label for ax in self.axes]),
            'axis_vectors': np.stack([ax.vector for ax in self.axes]),
            'axis_projections': np.stack([ax.projection for ax in self.axes]),
            'mode': np.array(self.mode),
            'n_evals': np.array(self.n_evals),
            'wall_s': np.array(self.wall_s),
        }
        for k, c in enumerate(self.coords):
            arrays[f'coords_{k}'] = c
        if self.theta_opt is not None:
            arrays['theta_opt'] = self.theta_opt
        np.savez(path, **arrays)
        return path


HalfWidth = Union[float, Tuple[float, float]]


def _axis_coords(half_width: HalfWidth, n: int) -> np.ndarray:
    if np.ndim(half_width) == 0:
        lo, hi = -float(half_width), float(half_width)
    else:
        lo, hi = (float(v) for v in half_width)
    if not (lo < 0 < hi):
        raise ValueError(
            f"half_width must be positive or a (lo, hi) pair with lo < 0 < hi, "
            f"got {half_width}"
        )
    return np.linspace(lo, hi, int(n))


def _check_axes(axes: Sequence[SliceAxis], n_params: int) -> None:
    if len(axes) not in (2, 3):
        raise ValueError(f"slice_grid takes 2 or 3 axes, got {len(axes)}")
    for ax in axes:
        if ax.vector.shape != (n_params,) or ax.projection.shape != (n_params,):
            raise ValueError(
                f"axis '{ax.label}' has shape {ax.vector.shape}, task has "
                f"{n_params} sampled parameters"
            )
    projection = np.stack([ax.projection for ax in axes])
    vectors = np.stack([ax.vector for ax in axes])
    gram = projection @ vectors.T
    if not np.allclose(gram, np.eye(len(axes)), atol=1e-8):
        raise ValueError(
            f"axis projections are not dual to axis vectors "
            f"(projection @ vectors.T = {gram})"
        )


def _batched_logp(task: InferenceTask, theta_grid: np.ndarray, batch_size: int):
    """Log-posterior of every row of ``theta_grid`` in fixed-size batches."""
    fn = jax.jit(jax.vmap(task._log_posterior_jittable))
    n = theta_grid.shape[0]
    out = np.empty(n)
    for start in range(0, n, batch_size):
        block = theta_grid[start : start + batch_size]
        if block.shape[0] < batch_size:
            # pad to the compiled batch shape (one compile per grid)
            pad = np.repeat(block[-1:], batch_size - block.shape[0], axis=0)
            block = np.vstack([block, pad])
        values = np.asarray(fn(jnp.asarray(block)))
        out[start : start + batch_size] = values[: min(batch_size, n - start)]
    return out


def _serpentine(shape: Tuple[int, ...]) -> Iterator[Tuple[int, ...]]:
    """Grid multi-indices with the last axis alternating direction, so each
    visited point is adjacent to the previous one."""
    if len(shape) == 1:
        yield from ((i,) for i in range(shape[0]))
        return
    flip = False
    for head in _serpentine(shape[:-1]):
        rng = range(shape[-1] - 1, -1, -1) if flip else range(shape[-1])
        for j in rng:
            yield head + (j,)
        flip = not flip


def slice_grid(
    task: InferenceTask,
    center: np.ndarray,
    axes: Sequence[SliceAxis],
    half_widths: Sequence[HalfWidth],
    n: Union[int, Sequence[int]],
    mode: str = 'conditional',
    *,
    scale: Optional[np.ndarray] = None,
    batch_size: int = 64,
    profile_maxiter: int = 500,
) -> SliceResult:
    """
    Evaluate the log-posterior on a 2D or 3D slice grid.

    Parameters
    ----------
    task : InferenceTask
        Rebuilt fit posterior.
    center : ndarray
        Slice center in sampled-parameter space (normally the MAP); must be
        inside the prior support.
    axes : sequence of SliceAxis
        Two or three slice directions (``parameter_axes`` or
        ``EigenAxes.slice_axes``).
    half_widths : sequence
        Per-axis half-width in slice-coordinate units, or a ``(lo, hi)``
        offset pair (``lo < 0 < hi``) for an asymmetric range.
    n : int or sequence of int
        Grid points per axis.
    mode : {'conditional', 'profile'}
        ``conditional`` holds the other parameters at ``center``;
        ``profile`` maximizes over them at every grid point.
    scale : ndarray, optional
        Per-parameter scale for the profile optimizer's coordinates
        (posterior sigma when known); default: prior-draw std.
    batch_size : int
        vmap batch for the conditional evaluation.
    profile_maxiter : int
        L-BFGS-B iteration cap per grid point.

    Returns
    -------
    SliceResult
    """
    if mode not in SLICE_MODES:
        raise ValueError(f"mode must be one of {SLICE_MODES}, got '{mode}'")
    center = np.asarray(center, dtype=np.float64)
    n_params = task.n_params
    if center.shape != (n_params,):
        raise ValueError(f"center has shape {center.shape}, expected ({n_params},)")
    axes = list(axes)
    _check_axes(axes, n_params)
    if len(half_widths) != len(axes):
        raise ValueError(f"{len(half_widths)} half_widths for {len(axes)} axes")
    shape = tuple([int(n)] * len(axes)) if np.ndim(n) == 0 else tuple(int(v) for v in n)
    if len(shape) != len(axes) or any(v < 2 for v in shape):
        raise ValueError(f"n must give >= 2 points per axis, got {shape}")

    logp_fn = task.get_log_posterior_fn()
    logp_center = float(logp_fn(jnp.asarray(center)))
    if not np.isfinite(logp_center):
        raise ValueError(
            f"slice center {center} has non-finite log-posterior ({logp_center}); "
            f"it lies outside the prior support"
        )

    t0 = time.perf_counter()
    coords = [_axis_coords(hw, m) for hw, m in zip(half_widths, shape)]
    mesh = np.meshgrid(*coords, indexing='ij')
    offsets = sum(m.ravel()[:, None] * ax.vector[None, :] for m, ax in zip(mesh, axes))
    theta_grid = center[None, :] + offsets
    n_points = theta_grid.shape[0]

    logp_cond = _batched_logp(task, theta_grid, min(batch_size, n_points))
    n_evals = n_points
    result = SliceResult(
        mode=mode,
        axes=axes,
        center=center,
        coords=coords,
        logp=logp_cond.reshape(shape),
        logp_center=logp_center,
        n_evals=n_evals,
        wall_s=0.0,
    )
    if mode == 'conditional':
        result.wall_s = time.perf_counter() - t0
        return result

    # profile: theta = base + scale * (M @ v), M spanning the complement of the
    # slice so the slice coordinates of every iterate stay fixed
    scale = _prior_scale(task) if scale is None else np.asarray(scale, dtype=np.float64)
    if scale.shape != (n_params,) or np.any(scale <= 0):
        raise ValueError(f"scale must be positive with shape ({n_params},)")
    projection = np.stack([ax.projection for ax in axes])
    complement = null_space(projection * scale[None, :])
    if complement.shape[1] != n_params - len(axes):
        raise ValueError(
            f"slice complement has dimension {complement.shape[1]}, expected "
            f"{n_params - len(axes)}: axis projections are linearly dependent"
        )
    val_and_grad = task.get_log_posterior_and_grad_fn()

    def objective(v: np.ndarray, base: np.ndarray):
        theta = jnp.asarray(base + scale * (complement @ v))
        value, grad = val_and_grad(theta)
        return float(-value), -(
            complement.T @ (np.asarray(grad, dtype=np.float64) * scale)
        )

    logp_prof = logp_cond.copy()
    theta_opt = theta_grid.copy()
    n_unconverged = 0
    n_below = 0
    v_prev = np.zeros(complement.shape[1])
    for idx in _serpentine(shape):
        flat = int(np.ravel_multi_index(idx, shape))
        base = theta_grid[flat]
        if not np.isfinite(logp_cond[flat]):
            continue
        v0 = v_prev
        n_evals += 1
        if not np.isfinite(objective(v0, base)[0]):
            v0 = np.zeros_like(v_prev)  # warm start left the support here
        res = minimize(
            objective,
            v0,
            args=(base,),
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': profile_maxiter},
        )
        n_evals += int(res.nfev)
        if not res.success:
            n_unconverged += 1
        value = -float(res.fun)
        if np.isfinite(value) and value >= logp_cond[flat]:
            logp_prof[flat] = value
            theta_opt[flat] = base + scale * (complement @ res.x)
            v_prev = res.x
        else:
            # the conditional point is itself feasible for the maximization
            n_below += 1
            v_prev = np.zeros_like(v_prev)

    result.logp = logp_prof.reshape(shape)
    result.theta_opt = theta_opt.reshape(shape + (n_params,))
    result.n_evals = n_evals
    result.n_unconverged = n_unconverged
    result.n_below_conditional = n_below
    result.wall_s = time.perf_counter() - t0
    return result


# ==============================================================================
# Plots
# ==============================================================================


_MODE_NOTE = {
    'conditional': 'conditional slice: other parameters fixed at the center',
    'profile': 'profile slice: other parameters maximized at each point',
}


def _finite_delta(result: SliceResult) -> np.ma.MaskedArray:
    delta = result.delta_logp
    return np.ma.masked_invalid(delta)


def plot_slice_2d(
    result: SliceResult,
    path: Union[str, Path],
    *,
    chains: Optional[np.ndarray] = None,
    map_theta: Optional[np.ndarray] = None,
    truth: Optional[np.ndarray] = None,
    title: str = '',
    vmin: float = -10.0,
    max_chain_draws: int = 4000,
    dpi: int = 150,
) -> Path:
    """
    Save a 2D slice as a PNG: delta log P map, 1/2/3-sigma contours, and
    chain draws / MAP / truth projected onto the slice axes.
    """
    import matplotlib.pyplot as plt

    if result.ndim != 2:
        raise ValueError(f"plot_slice_2d needs a 2D slice, got {result.ndim}D")
    xs, ys = result.plot_coords()
    delta = _finite_delta(result)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    mesh = ax.pcolormesh(
        xs, ys, delta.T, shading='auto', vmin=vmin, vmax=0.0, cmap='viridis'
    )
    levels = sorted(SIGMA_LEVELS)
    if delta.count() and float(delta.min()) < max(levels):
        contours = ax.contour(
            xs, ys, delta.T, levels=levels, colors='white', linewidths=1.0
        )
        ax.clabel(contours, fmt=SIGMA_LEVELS, fontsize=7)
    fig.colorbar(mesh, ax=ax, label=r'$\Delta \log P$ (relative to slice max)')

    if chains is not None:
        chains = np.asarray(chains)
        if chains.shape[0] > max_chain_draws:
            step = int(np.ceil(chains.shape[0] / max_chain_draws))
            chains = chains[::step]
        pts = result.plot_points(chains)
        ax.scatter(
            pts[:, 0], pts[:, 1], s=2, c='crimson', alpha=0.25, label='chain draws'
        )
    if map_theta is not None:
        pt = result.plot_points(map_theta)[0]
        ax.scatter(
            [pt[0]],
            [pt[1]],
            marker='*',
            s=180,
            c='white',
            edgecolors='black',
            linewidths=1.0,
            label='MAP',
            zorder=5,
        )
    if truth is not None:
        pt = result.plot_points(truth)[0]
        ax.scatter(
            [pt[0]],
            [pt[1]],
            marker='D',
            s=60,
            c='orange',
            edgecolors='black',
            linewidths=1.0,
            label='truth',
            zorder=5,
        )
    ax.set_xlim(xs.min(), xs.max())
    ax.set_ylim(ys.min(), ys.max())
    ax.set_xlabel(result.axes[0].label)
    ax.set_ylabel(result.axes[1].label)
    header = f'{title}\n' if title else ''
    ax.set_title(
        f'{header}{_MODE_NOTE[result.mode]}\n'
        f'{result.n_evals} evals, {result.wall_s:.1f} s',
        fontsize=9,
    )
    ax.legend(fontsize=7, loc='upper right')
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def _import_plotly():
    try:
        import plotly.graph_objects as go
    except ImportError as err:
        raise ImportError(
            "plot_isosurface_3d requires the 'plotly' package, which is not in "
            "the klpipe conda-lock; install it with 'pip install plotly'"
        ) from err
    return go


def plot_isosurface_3d(
    result: SliceResult,
    path: Union[str, Path],
    *,
    chains: Optional[np.ndarray] = None,
    map_theta: Optional[np.ndarray] = None,
    truth: Optional[np.ndarray] = None,
    title: str = '',
    max_chain_draws: int = 4000,
) -> Path:
    """Save a 3D grid as an interactive plotly HTML with the 1/2/3-sigma
    delta log P isosurfaces and projected chain draws / MAP / truth."""
    go = _import_plotly()
    if result.ndim != 3:
        raise ValueError(f"plot_isosurface_3d needs a 3D grid, got {result.ndim}D")
    xs, ys, zs = result.plot_coords()
    delta = result.delta_logp
    finite = np.isfinite(delta)
    # isosurface extraction needs finite values; out-of-support cells sit far
    # below the lowest drawn level
    delta = np.where(finite, delta, np.min(delta[finite]) - 10.0)
    XX, YY, ZZ = np.meshgrid(xs, ys, zs, indexing='ij')
    levels = sorted(SIGMA_LEVELS)
    fig = go.Figure(
        data=go.Isosurface(
            x=XX.ravel(),
            y=YY.ravel(),
            z=ZZ.ravel(),
            value=delta.ravel(),
            isomin=min(levels),
            isomax=max(levels),
            surface_count=len(levels),
            colorscale='Viridis',
            caps=dict(x_show=False, y_show=False, z_show=False),
            opacity=0.5,
            name='delta log P',
        )
    )

    def _marker(theta, color, name):
        pt = result.plot_points(theta)[0]
        fig.add_trace(
            go.Scatter3d(
                x=[pt[0]],
                y=[pt[1]],
                z=[pt[2]],
                mode='markers',
                marker=dict(
                    size=6,
                    color=color,
                    symbol='diamond',
                    line=dict(color='black', width=1),
                ),
                name=name,
            )
        )

    if chains is not None:
        chains = np.asarray(chains)
        if chains.shape[0] > max_chain_draws:
            chains = chains[:: int(np.ceil(chains.shape[0] / max_chain_draws))]
        pts = result.plot_points(chains)
        fig.add_trace(
            go.Scatter3d(
                x=pts[:, 0],
                y=pts[:, 1],
                z=pts[:, 2],
                mode='markers',
                marker=dict(size=1.5, color='crimson', opacity=0.25),
                name='chain draws',
            )
        )
    if map_theta is not None:
        _marker(map_theta, 'white', 'MAP')
    if truth is not None:
        _marker(truth, 'orange', 'truth')
    labels = [ax.label for ax in result.axes]
    fig.update_layout(
        scene=dict(xaxis_title=labels[0], yaxis_title=labels[1], zaxis_title=labels[2]),
        title=(
            f'{title} {_MODE_NOTE[result.mode]}; isosurfaces at '
            f'{", ".join(f"{lvl} ({lab})" for lvl, lab in sorted(SIGMA_LEVELS.items()))}; '
            f'{result.n_evals} evals, {result.wall_s:.1f} s'
        ),
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(path))
    return path


# ==============================================================================
# CLI
# ==============================================================================


def _parse_plane(spec: str) -> Tuple[str, List[str]]:
    """``'g2,theta_int'`` -> ('parameter', [...]); ``'eigen:0,1'`` -> ('eigen', [...])."""
    if spec.startswith('eigen:'):
        modes = [s.strip() for s in spec[len('eigen:') :].split(',') if s.strip()]
        if not all(re.fullmatch(r'\d+', m) for m in modes):
            raise ValueError(
                f"eigen plane must be 'eigen:<int>,<int>[,<int>]', got '{spec}'"
            )
        return 'eigen', modes
    names = [s.strip() for s in spec.split(',') if s.strip()]
    if len(names) < 2:
        raise ValueError(
            f"plane needs at least two comma-separated names, got '{spec}'"
        )
    return 'parameter', names


def _clipped_half_width(
    center: float,
    sigma: float,
    bounds: Tuple[Optional[float], Optional[float]],
    n_sigma: float,
) -> Tuple[float, float]:
    """Symmetric +-n_sigma window shrunk on any side that would leave the prior."""
    lo, hi = -n_sigma * sigma, n_sigma * sigma
    b_lo, b_hi = bounds
    if b_lo is not None and center + lo < b_lo:
        lo = b_lo - center
    if b_hi is not None and center + hi > b_hi:
        hi = b_hi - center
    if not (lo < 0 < hi):
        raise ValueError(
            f"center {center} is within roundoff of a prior bound {bounds}; "
            f"cannot build a slice window"
        )
    return lo, hi


def _slug(labels: Sequence[str]) -> str:
    return '_'.join(
        re.sub(r'[^A-Za-z0-9]+', '_', lab.split(' (')[0]).strip('_') for lab in labels
    )


def _build_axes(
    kind: str,
    items: List[str],
    fit: FitPosterior,
    sigma: np.ndarray,
    n_sigma: float,
    eigen: Optional[EigenAxes],
) -> Tuple[List[SliceAxis], List[HalfWidth]]:
    if kind == 'eigen':
        if eigen is None:
            raise RuntimeError("eigen planes requested but no eigen axes were built")
        modes = tuple(int(m) for m in items)
        ea = EigenAxes(
            eigen.sigma, eigen.eigenvalues, eigen.directions_sigma, modes, eigen.source
        )
        return ea.slice_axes(), [n_sigma] * len(modes)
    axes = parameter_axes(items, fit.sampled_names)
    bounds = fit.task.get_bounds()
    half_widths: List[HalfWidth] = [
        _clipped_half_width(
            fit.map_theta[ax.index], sigma[ax.index], bounds[ax.index], n_sigma
        )
        for ax in axes
    ]
    return axes, half_widths


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog='python -m kl_pipe.diagnostics.posterior_slices',
        description='2D/3D log-posterior slices of a saved ensemble fit',
    )
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--fit-id', required=True)
    parser.add_argument(
        '--planes',
        nargs='+',
        default=[],
        help="slice planes: 'g2,theta_int' (parameter pair) or 'eigen:0,1' (modes)",
    )
    parser.add_argument('--mode', choices=SLICE_MODES, default='conditional')
    parser.add_argument('--n', type=int, default=40, help='grid points per 2D axis')
    parser.add_argument(
        '--n-sigma', type=float, default=4.0, help='half-width in sigma'
    )
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument(
        '--iso', default=None, help="3D grid axes, e.g. 'g2,theta_int,cosi'"
    )
    parser.add_argument('--n3', type=int, default=18, help='grid points per 3D axis')
    parser.add_argument(
        '--results', type=Path, default=None, help='summary parquet override'
    )
    parser.add_argument('--chains-dir', type=Path, default=None)
    parser.add_argument('--mocks-dir', type=Path, default=None)
    parser.add_argument(
        '--eigen-source',
        choices=('auto', 'chains', 'hessian'),
        default='auto',
        help='covariance for eigen axes / sigma (auto: chains when saved, else hessian)',
    )
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--profile-maxiter', type=int, default=500)
    args = parser.parse_args(argv)

    import matplotlib

    matplotlib.use('Agg')

    if not args.planes and args.iso is None:
        parser.error('give at least one --planes entry or --iso')
    planes = [_parse_plane(p) for p in args.planes]
    iso = _parse_plane(args.iso) if args.iso is not None else None

    fit = rebuild_fit_posterior(
        args.run_dir,
        args.fit_id,
        results=args.results,
        chains_dir=args.chains_dir,
        mocks_dir=args.mocks_dir,
    )
    print(
        f'rebuilt fit {fit.fit_id}: {fit.task.n_params} sampled params, '
        f'chains={"yes" if fit.chains is not None else "no"}, '
        f'mocks verified={"yes" if fit.paths.mocks is not None else "no file"}, '
        f'{fit.rebuild_wall_s:.1f} s',
        flush=True,
    )

    source = args.eigen_source
    if source == 'auto':
        source = 'chains' if fit.chains is not None else 'hessian'
    if source == 'chains':
        if fit.chains is None:
            raise FileNotFoundError('--eigen-source chains but the fit saved no chains')
        eigen = eigen_axes(chains=fit.chains)
    else:
        t0 = time.perf_counter()
        eigen = eigen_axes(covariance=map_hessian_covariance(fit.task, fit.map_theta))
        print(f'MAP Hessian covariance: {time.perf_counter() - t0:.1f} s', flush=True)
    sigma = eigen.sigma
    print(f'eigen axes from {eigen.source}; softest modes:')
    for k in range(min(4, len(sigma))):
        print(
            f'  mode {k}: lambda={eigen.eigenvalues[k]:.3f}  {eigen.loadings(fit.sampled_names, k)}'
        )

    args.out.mkdir(parents=True, exist_ok=True)
    run_name = str(fit.summary['run_name']) if 'run_name' in fit.summary else ''
    title = f'{fit.fit_id} {run_name}'.strip()
    for kind, items in planes:
        axes, half_widths = _build_axes(kind, items, fit, sigma, args.n_sigma, eigen)
        result = slice_grid(
            fit.task,
            fit.map_theta,
            axes,
            half_widths,
            args.n,
            mode=args.mode,
            scale=sigma,
            batch_size=args.batch_size,
            profile_maxiter=args.profile_maxiter,
        )
        stem = f'slice_{args.mode}_{_slug([ax.label for ax in axes])}'
        png = plot_slice_2d(
            result,
            args.out / f'{stem}.png',
            chains=fit.chains,
            map_theta=fit.map_theta,
            truth=fit.truth,
            title=title,
        )
        result.save(args.out / f'{stem}.npz')
        n_points = int(np.prod(result.logp.shape))
        extra = ''
        if args.mode == 'profile':
            extra = (
                f' unconverged={result.n_unconverged}'
                f' below_conditional={result.n_below_conditional}'
            )
        print(
            f'{stem}: n_evals={result.n_evals} wall={result.wall_s:.1f} s '
            f'({1e3 * result.wall_s / n_points:.1f} ms/point){extra} -> {png}',
            flush=True,
        )

    if iso is not None:
        kind, items = iso
        if len(items) != 3:
            raise ValueError(f"--iso needs exactly three axes, got {items}")
        axes, half_widths = _build_axes(kind, items, fit, sigma, args.n_sigma, eigen)
        result = slice_grid(
            fit.task,
            fit.map_theta,
            axes,
            half_widths,
            args.n3,
            mode=args.mode,
            scale=sigma,
            batch_size=args.batch_size,
            profile_maxiter=args.profile_maxiter,
        )
        stem = f'iso_{args.mode}_{_slug([ax.label for ax in axes])}'
        html = plot_isosurface_3d(
            result,
            args.out / f'{stem}.html',
            chains=fit.chains,
            map_theta=fit.map_theta,
            truth=fit.truth,
            title=title,
        )
        result.save(args.out / f'{stem}.npz')
        n_points = int(np.prod(result.logp.shape))
        print(
            f'{stem}: n_evals={result.n_evals} wall={result.wall_s:.1f} s '
            f'({1e3 * result.wall_s / n_points:.1f} ms/point) -> {html}',
            flush=True,
        )
    return 0


if __name__ == '__main__':
    sys.exit(main())
