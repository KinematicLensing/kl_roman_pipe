"""
Tests for kl_pipe.diagnostics.posterior_slices on a small velocity-only task.

The task has 5 sampled parameters (vel.v0, vel.vcirc, vel.rscale, cosi,
theta_int) on a 16x16 velocity map; every slice here is a few hundred
evaluations at most.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import jax.numpy as jnp
from scipy.optimize import minimize

from kl_pipe.velocity import CenteredVelocityModel
from kl_pipe.parameters import ImagePars
from kl_pipe.synthetic import SyntheticVelocity
from kl_pipe.priors import Uniform, PriorDict
from kl_pipe.source import SourceModel
from kl_pipe.observation import build_velocity_obs
from kl_pipe.sampling import InferenceTask
from kl_pipe.utils import get_test_dir
from kl_pipe.diagnostics.posterior_slices import (
    EigenAxes,
    eigen_axes,
    map_hessian_covariance,
    parameter_axes,
    plot_isosurface_3d,
    plot_slice_2d,
    rebuild_fit_posterior,
    slice_grid,
)


@pytest.fixture(scope='module')
def output_dir() -> Path:
    out = get_test_dir() / 'out' / 'posterior_slices'
    out.mkdir(parents=True, exist_ok=True)
    return out


@pytest.fixture(scope='module')
def small_task():
    """(task, truth) for a 5-parameter velocity-only fit."""
    image_pars = ImagePars(shape=(16, 16), pixel_scale=0.5, indexing='ij')
    true_flat = {
        'v0': 10.0,
        'vcirc': 200.0,
        'rscale': 3.0,
        'cosi': 0.6,
        'theta_int': 0.785,
        'g1': 0.02,
        'g2': -0.01,
    }
    synth = SyntheticVelocity(true_flat, model_type='arctan', seed=42)
    data = synth.generate(image_pars, snr=100)
    priors = PriorDict(
        {
            'vel.v0': Uniform(-50.0, 50.0),
            'vel.vcirc': Uniform(100.0, 300.0),
            'vel.rscale': Uniform(0.5, 10.0),
            'cosi': Uniform(0.05, 0.95),
            'theta_int': Uniform(0.0, np.pi),
            'g1': 0.02,
            'g2': -0.01,
        }
    )
    obs = build_velocity_obs(image_pars, data=jnp.array(data), variance=synth.variance)
    task = InferenceTask.from_obs(
        SourceModel(velocity_model=CenteredVelocityModel()), priors, velocity_obs=obs
    )
    true_dotted = {
        'vel.v0': 10.0,
        'vel.vcirc': 200.0,
        'vel.rscale': 3.0,
        'cosi': 0.6,
        'theta_int': 0.785,
    }
    truth = np.array([true_dotted[n] for n in task.sampled_names])
    return task, truth


@pytest.fixture(scope='module')
def map_theta(small_task) -> np.ndarray:
    """MAP by L-BFGS-B from truth in prior-width-scaled coordinates."""
    task, truth = small_task
    val_and_grad = task.get_log_posterior_and_grad_fn()
    scale = np.array([hi - lo for lo, hi in task.get_bounds()]) / np.sqrt(12.0)

    def neg(u):
        v, g = val_and_grad(jnp.asarray(truth + scale * u))
        return -float(v), -np.asarray(g, dtype=np.float64) * scale

    res = minimize(
        neg,
        np.zeros(len(truth)),
        jac=True,
        method='L-BFGS-B',
        options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-9},
    )
    assert res.success, res.message
    return truth + scale * res.x


@pytest.fixture(scope='module')
def hessian_sigma(small_task, map_theta) -> np.ndarray:
    task, _ = small_task
    return np.sqrt(np.diag(map_hessian_covariance(task, map_theta)))


def _pair_axes(task):
    return parameter_axes(['vel.vcirc', 'cosi'], task.sampled_names)


class TestConditional:
    def test_max_at_map(self, small_task, map_theta, hessian_sigma):
        task, _ = small_task
        axes = _pair_axes(task)
        widths = [3.0 * hessian_sigma[ax.index] for ax in axes]
        result = slice_grid(task, map_theta, axes, widths, n=21, mode='conditional')
        assert result.logp.shape == (21, 21)
        assert result.n_evals == 21 * 21
        # the MAP maximizes the full posterior, so the conditional slice through
        # it peaks at the center cell exactly (any other cell is a different point)
        assert np.unravel_index(np.argmax(result.logp), result.logp.shape) == (10, 10)
        assert np.isclose(result.logp[10, 10], result.logp_center, rtol=1e-9)
        assert np.all(np.isfinite(result.logp))

    def test_center_outside_support_raises(self, small_task, map_theta, hessian_sigma):
        task, _ = small_task
        axes = _pair_axes(task)
        bad = map_theta.copy()
        bad[task.sampled_names.index('cosi')] = 2.0
        with pytest.raises(ValueError, match='outside the prior support'):
            slice_grid(task, bad, axes, [10.0, 0.1], n=5)

    def test_unknown_parameter_raises(self, small_task):
        task, _ = small_task
        with pytest.raises(KeyError, match='not a sampled parameter'):
            parameter_axes(['vel.vcirc', 'nope'], task.sampled_names)

    def test_bad_mode_raises(self, small_task, map_theta):
        task, _ = small_task
        with pytest.raises(ValueError, match='mode must be'):
            slice_grid(task, map_theta, _pair_axes(task), [1.0, 0.1], n=3, mode='x')


class TestProfile:
    def test_profile_dominates_conditional(self, small_task, map_theta, hessian_sigma):
        task, _ = small_task
        axes = _pair_axes(task)
        widths = [3.0 * hessian_sigma[ax.index] for ax in axes]
        cond = slice_grid(task, map_theta, axes, widths, n=5, mode='conditional')
        prof = slice_grid(
            task, map_theta, axes, widths, n=5, mode='profile', scale=hessian_sigma
        )
        assert prof.logp.shape == (5, 5)
        assert np.all(prof.logp >= cond.logp)
        assert prof.n_evals > cond.n_evals
        assert prof.theta_opt.shape == (5, 5, task.n_params)
        # maximizing over the other parameters at the MAP changes nothing: the
        # re-optimization cannot beat the MAP (measured delta 0.0 on this scene,
        # MAP converged to gtol 1e-9); 1e-3 bounds a MAP converged only to ftol
        assert abs(prof.logp[2, 2] - cond.logp[2, 2]) < 1e-3
        # the slice coordinates of the optimized points are preserved
        pts = prof.project(prof.theta_opt.reshape(-1, task.n_params))
        expected = np.stack(np.meshgrid(*prof.coords, indexing='ij'), -1).reshape(-1, 2)
        np.testing.assert_allclose(pts, expected, atol=1e-9)
        # vel.rscale and vel.v0 trade off against vcirc/cosi, so away from the
        # center the ridge sits strictly above the conditional cut (measured
        # corner gaps 2.7 to 351 in log P; 1.0 is below the smallest)
        assert prof.logp[0, 0] > cond.logp[0, 0] + 1.0


class TestEigenAxes:
    def test_orthonormal_in_sigma_units(self):
        rng = np.random.default_rng(3)
        n = 5
        A = rng.standard_normal((n, n))
        cov = A @ A.T + 0.1 * np.eye(n)
        chains = rng.multivariate_normal(np.zeros(n), cov, size=4000)
        ea = eigen_axes(chains=chains, modes=(0, 1))
        gram = ea.directions_sigma.T @ ea.directions_sigma
        np.testing.assert_allclose(gram, np.eye(n), atol=1e-10)
        assert np.all(np.diff(ea.eigenvalues) <= 0)
        np.testing.assert_allclose(ea.eigenvalues.sum(), n, rtol=1e-10)
        np.testing.assert_allclose(ea.sigma, chains.std(axis=0, ddof=1))
        axes = ea.slice_axes()
        proj = np.stack([ax.projection for ax in axes])
        vec = np.stack([ax.vector for ax in axes])
        np.testing.assert_allclose(proj @ vec.T, np.eye(2), atol=1e-10)
        # one slice-coordinate unit is one mode sigma: unit norm in sigma units
        for ax, k in zip(axes, ea.modes):
            np.testing.assert_allclose(
                np.linalg.norm(ax.vector / ea.sigma), np.sqrt(ea.eigenvalues[k])
            )

    def test_hessian_covariance_path(self, small_task, map_theta):
        task, _ = small_task
        cov = map_hessian_covariance(task, map_theta)
        ea = eigen_axes(covariance=cov, modes=(0, 1))
        assert ea.source == 'covariance'
        assert ea.eigenvalues[0] >= ea.eigenvalues[-1] > 0
        assert isinstance(ea, EigenAxes)

    def test_eigen_slice_projection(self, small_task, map_theta):
        task, _ = small_task
        ea = eigen_axes(covariance=map_hessian_covariance(task, map_theta))
        axes = ea.slice_axes()
        result = slice_grid(task, map_theta, axes, [2.0, 2.0], n=7)
        np.testing.assert_allclose(result.project(map_theta), [[0.0, 0.0]], atol=1e-12)
        np.testing.assert_allclose(
            result.project(map_theta + axes[0].vector), [[1.0, 0.0]], atol=1e-10
        )
        assert np.unravel_index(np.argmax(result.logp), result.logp.shape) == (3, 3)

    def test_bad_inputs(self):
        with pytest.raises(ValueError, match='exactly one'):
            eigen_axes()
        with pytest.raises(ValueError, match='mode 7 out of range'):
            eigen_axes(covariance=np.eye(3), modes=(0, 7))


class TestLoader:
    def test_missing_run_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            rebuild_fit_posterior(tmp_path / 'nope', 'abc')

    def test_missing_fit_id_raises(self, tmp_path):
        pd.DataFrame({'fit_id': ['other'], 'noise_seed': [1]}).to_parquet(
            tmp_path / 'manifest.parquet', index=False
        )
        with pytest.raises(KeyError, match="fit_id 'abc' matches 0 rows"):
            rebuild_fit_posterior(tmp_path, 'abc')

    def test_missing_provenance_raises(self, tmp_path):
        pd.DataFrame({'fit_id': ['abc'], 'noise_seed': [1]}).to_parquet(
            tmp_path / 'manifest.parquet', index=False
        )
        with pytest.raises(FileNotFoundError, match='missing fit files'):
            rebuild_fit_posterior(tmp_path, 'abc')


class TestPlots:
    def test_plot_slice_2d_writes_png(
        self, small_task, map_theta, hessian_sigma, output_dir
    ):
        task, truth = small_task
        axes = _pair_axes(task)
        widths = [3.0 * hessian_sigma[ax.index] for ax in axes]
        result = slice_grid(task, map_theta, axes, widths, n=9)
        rng = np.random.default_rng(0)
        chains = map_theta + rng.standard_normal((300, task.n_params)) * hessian_sigma
        path = plot_slice_2d(
            result,
            output_dir / 'slice_conditional_vcirc_cosi.png',
            chains=chains,
            map_theta=map_theta,
            truth=truth,
            title='test scene',
        )
        assert path.exists() and path.stat().st_size > 0

    def test_plot_isosurface_3d_writes_html(
        self, small_task, map_theta, hessian_sigma, output_dir
    ):
        pytest.importorskip('plotly')
        task, truth = small_task
        axes = parameter_axes(['vel.vcirc', 'cosi', 'vel.rscale'], task.sampled_names)
        widths = [3.0 * hessian_sigma[ax.index] for ax in axes]
        result = slice_grid(task, map_theta, axes, widths, n=4)
        assert result.logp.shape == (4, 4, 4)
        path = plot_isosurface_3d(
            result, output_dir / 'iso_test.html', map_theta=map_theta, truth=truth
        )
        assert path.exists() and path.stat().st_size > 0
