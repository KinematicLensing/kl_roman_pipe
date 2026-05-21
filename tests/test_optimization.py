"""Unit tests for kl_pipe.optimization.multi_start_minimize."""

import numpy as np

from kl_pipe.optimization import multi_start_minimize


def _quadratic(x):
    """Simple convex objective: sum((x - target)^2)."""
    target = np.array([1.0, 2.0, 0.5])
    return float(np.sum((x - target) ** 2))


def _double_well(x):
    """Multimodal objective with double-well in x[1]: minima near -1 and +1.

    Used to make multi-start diversity observable: different perturbed
    starts converge to different basins, so final positions encode the
    initial-point diversity.
    """
    return float((x[1] ** 2 - 1.0) ** 2 + (x[0] - 1.0) ** 2 + (x[2] - 0.5) ** 2)


class TestMultiStartPerturbation:
    """Verify multi-start perturbations are nonzero even for x0[i]==0.

    Regression for Copilot review: previously delta = perturbation * |x0|
    * noise meant params with x0=0 were never perturbed, defeating
    multi-start for boundary-attractor cases.
    """

    def test_perturbs_zero_initial_with_bounds(self):
        # x0[1] = 0 sits between two basins of _double_well at +-1.
        # If perturbation is nonzero (regardless of x0=0), starts should
        # land in BOTH basins and the final positions span both minima.
        x0 = np.array([1.0, 0.0, 0.5])
        bounds = [(-2.0, 2.0), (-3.0, 3.0), (-1.0, 1.0)]
        res = multi_start_minimize(
            _double_well, x0, bounds=bounds, n_starts=10, perturbation=0.2, seed=0
        )
        starts = np.array([r.x for r in res.start_results])
        assert res.n_starts_attempted == 10
        # different starts should explore both wells → x[1] spans positive
        # and negative final positions
        assert starts[:, 1].max() > 0.5
        assert starts[:, 1].min() < -0.5

    def test_perturbs_zero_initial_no_bounds(self):
        # x0[1] = 0, unbounded → small floor (1e-3) drives perturbation.
        # Floor is small, so single-start L-BFGS-B may converge to nearest
        # well. With seed=0 and 10 starts × small perturbation we should
        # still see at least some diversity if perturbation > 0.
        x0 = np.array([1.0, 0.0, 0.5])
        res = multi_start_minimize(
            _double_well, x0, bounds=None, n_starts=10, perturbation=0.2, seed=0
        )
        starts = np.array([r.x for r in res.start_results])
        assert res.n_starts_attempted == 10
        # at minimum, final positions at index 1 should not all be identical
        # (perturbation floor 1e-3 × 0.2 = 2e-4 sigma is enough to nudge
        # symmetric saddle behavior in L-BFGS-B)
        assert starts[:, 1].std() > 1e-8

    def test_nonzero_x0_unchanged_behavior(self):
        # All x0 nonzero → behavior unchanged from old |x0|-scaled perturbation
        x0 = np.array([1.0, 2.0, 0.5])
        bounds = [(0.0, 5.0), (0.0, 5.0), (0.0, 5.0)]
        res = multi_start_minimize(
            _quadratic, x0, bounds=bounds, n_starts=5, perturbation=0.2, seed=0
        )
        # convex objective: best should converge to target [1, 2, 0.5]
        np.testing.assert_allclose(res.x, [1.0, 2.0, 0.5], atol=1e-5)
