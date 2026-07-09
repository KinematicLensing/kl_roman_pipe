"""
Shared test utilities for parameter recovery tests.

This module contains common functions used by both likelihood slicing tests
and gradient-based optimizer tests. May expand further in the future.

Note: plot_data_comparison_panels and plot_combined_data_comparison have been
moved to kl_pipe.diagnostics. The versions in this file are deprecated wrappers
for backward compatibility.
"""

import os
import pytest
import sys
import contextlib
import warnings
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Tuple, Optional, Callable, List
from mpl_toolkits.axes_grid1 import make_axes_locatable

from kl_pipe.parameters import ImagePars
from kl_pipe.plotting import MidpointNormalize


# ==============================================================================
# Output Redirection Utilities
# ==============================================================================


class TeeWriter:
    """Write to both a file and optionally to terminal."""

    def __init__(self, file_handle, also_terminal: bool = False, original_stdout=None):
        self.file_handle = file_handle
        self.also_terminal = also_terminal
        self.original_stdout = original_stdout or sys.__stdout__

    def write(self, message):
        self.file_handle.write(message)
        if self.also_terminal:
            self.original_stdout.write(message)

    def flush(self):
        self.file_handle.flush()
        if self.also_terminal:
            self.original_stdout.flush()


@contextlib.contextmanager
def redirect_sampler_output(log_path: Path, also_terminal: bool = False):
    """
    Redirect stdout to a file, optionally also writing to terminal.

    Sampler output is ALWAYS written to the log file. If also_terminal=True,
    output is also displayed in the terminal (useful for debugging).

    Parameters
    ----------
    log_path : Path
        Path to the output log file.
    also_terminal : bool
        If True, output goes to both file and terminal.
        If False (default), output goes to file only.

    Examples
    --------
    >>> with redirect_sampler_output(Path("sampler.log")):
    ...     sampler.run()  # Output captured to file only

    >>> with redirect_sampler_output(Path("sampler.log"), also_terminal=True):
    ...     sampler.run()  # Output to file AND terminal
    """
    # Ensure parent directory exists
    log_path.parent.mkdir(parents=True, exist_ok=True)

    original_stdout = sys.stdout
    with open(log_path, 'w') as f:
        sys.stdout = TeeWriter(
            f, also_terminal=also_terminal, original_stdout=original_stdout
        )
        try:
            yield
        finally:
            sys.stdout = original_stdout


# ==============================================================================
# Mask Generation Helpers
# ==============================================================================


def make_aperture_mask(shape, radius_frac=1.0):
    """
    Elliptical aperture mask. True=valid (inside aperture).

    Semi-axes = radius_frac * dim/2 for each dimension,
    simulating a circular FOV on a rectangular detector.
    Fraction masked ~ 1 - pi/4 * radius_frac**2 (~21.5% at 1.0).
    """
    nrow, ncol = shape
    iy = np.arange(nrow) - (nrow - 1) / 2.0
    ix = np.arange(ncol) - (ncol - 1) / 2.0
    IX, IY = np.meshgrid(ix, iy, indexing='xy')
    a = radius_frac * ncol / 2.0
    b = radius_frac * nrow / 2.0
    return (IX / a) ** 2 + (IY / b) ** 2 <= 1.0


def make_center_mask(shape, radius_frac=0.2):
    """
    Create a mask that excludes a central disk. True=valid.

    Parameters
    ----------
    shape : tuple
        Shape of the mask array (Nx, Ny).
    radius_frac : float
        Fraction of the smaller dimension to use as exclusion radius.
    """
    nx, ny = shape
    ix = np.arange(nx) - (nx - 1) / 2.0
    iy = np.arange(ny) - (ny - 1) / 2.0
    IX, IY = np.meshgrid(ix, iy, indexing='ij')
    r = np.sqrt(IX**2 + IY**2)
    radius = radius_frac * min(nx, ny) / 2.0
    return r > radius


def make_random_mask(shape, fraction_valid=0.75, seed=42):
    """
    Create a random per-pixel mask. True=valid.

    Parameters
    ----------
    shape : tuple
        Shape of the mask array (Nx, Ny).
    fraction_valid : float
        Fraction of pixels to keep valid.
    seed : int
        Random seed for reproducibility.
    """
    rng = np.random.RandomState(seed)
    return rng.random(shape) < fraction_valid


# ==============================================================================
# Test Configuration Data Structures
# ==============================================================================


class TestConfig:
    """
    Configuration container for parameter recovery tests.

    This makes it easy to pass configuration through pytest fixtures
    rather than using global variables.
    """

    __test__ = False  # tell pytest this is not a test class

    def __init__(
        self,
        output_dir: Path,
        enable_plots: bool = True,
        include_poisson_noise: bool = False,
        seed: int = 42,
        sersic_backend: str = 'scipy',
        verbose_terminal: bool = False,
    ):
        self.output_dir = output_dir
        # plotting control
        self.enable_plots = enable_plots
        # synthetic data generation
        self.include_poisson_noise = include_poisson_noise
        self.seed = seed
        self.sersic_backend = sersic_backend
        # sampler output control
        self.verbose_terminal = verbose_terminal

        # =========================================================================
        # LIKELIHOOD SLICE TEST TOLERANCES
        # Brute-force parameter recovery - strictest validation of forward models
        # =========================================================================
        # Active SNR set (matched-filter convention, see kl_pipe/noise.py):
        # 10000 = high, 1000 = mid, 500 = low.
        self.likelihood_slice_tolerance_velocity = {
            10000: 0.001,  # 0.1%  (high)
            1000: 0.005,  # 0.5%  (mid)
            500: 0.01,  # 1%    (low)
        }
        self.likelihood_slice_tolerance_intensity = {
            10000: 0.001,
            1000: 0.005,
            500: 0.01,
        }

        # Optimizer-tier tolerances retired 2026-07-08: optimizer recovery
        # tests now bound |error| by k * sigma with sigma the measured
        # marginal noise floor (frozen tables in test_optimizer_recovery.py)
        # and k from suite_false_alarm_k.

        # =========================================================================
        # COMPOSITE INTENSITY MODEL TOLERANCES
        # Initialized identically to single-component exponential. Per-parameter
        # scaling entries (see composite_param_scaling) added only with explicit
        # sign-off when the experimental observation justifies it (e.g. emulator
        # core bias on bulge_h_over_hlr at high SNR).
        # =========================================================================
        self.likelihood_slice_tolerance_intensity_composite = {
            10000: 0.001,
            1000: 0.005,
            500: 0.01,
        }

        # =========================================================================
        # PARAMETER-SPECIFIC SCALING (legacy likelihood-slice pattern only)
        # Accounts for inherently weaker constraints on certain parameters
        # =========================================================================

        # Likelihood slice test scaling (conservative)
        self.likelihood_slice_param_scaling = {
            # Shear is ~4% of main signal - harder to constrain
            'g1': {1000: 1.0, 500: 1.0, 100: 1.0, 50: 1.5, 10: 1.0},
            'g2': {1000: 1.0, 500: 1.0, 100: 1.0, 50: 1.5, 10: 1.0},
            # v0 is small - harder to measure relative error
            'v0': {1000: 1.0, 500: 1.0, 100: 1.0, 50: 1.0, 10: 1.5},
            # nu-rscale degeneracy makes nu harder to constrain at low SNR
            'nu': {1000: 1.0, 500: 1.0, 100: 1.5, 50: 2.0, 10: 2.5},
        }

        # Composite-specific param scaling. Starts EMPTY: per the project
        # tolerance policy, scaling entries are added only with explicit
        # sign-off accompanied by a documented physical/numerical reason
        # (e.g. emulator-induced bias on n=4 bulge profile parameters).
        self.composite_param_scaling = {}

        # absolute tolerance floor (for parameters near zero).
        # Keys are unprefixed defaults; get_tolerance tries the full dotted name
        # first, then falls back to the suffix after the first '.'. Add a prefixed
        # entry (e.g. 'vel.x0') only when a band / component / channel needs a
        # tolerance distinct from the shared default.
        self.absolute_tolerance_floor = {
            'g1': 0.004,  # noise floor ~0.003 for g~0.02 at SNR=10 on non-square grids
            'g2': 0.004,
            'x0': 0.1,  # shared default for vel + intensity centroids
            'y0': 0.1,
            'h_over_r': 0.01,  # low sensitivity at h/r=0.1
            'disk_h_over_r': 0.01,  # composite disk: same physical param as h_over_r
            'bulge_h_over_hlr': 0.01,  # composite bulge: same physical param as h_over_r
            'nu': 0.05,  # Spergel index absolute floor
        }

        # PSF tolerance multiplier -- with oversampled rendering (N=5), the
        # forward model mismatch is eliminated; small residual from PSF smoothing
        self.psf_tolerance_multiplier = 1.5

        # physical parameter boundaries
        self.param_bounds = {
            'cosi': (0.0, 0.99),
            'theta_int': (0.0, np.pi),
            'g1': (-0.1, 0.1),
            'g2': (-0.1, 0.1),
            'flux': (1e-8, None),  # Strictly positive
        }

        # image parameters - specified in (Nx, Ny) for easy verification
        Nx_vel, Ny_vel = 40, 30
        self.image_pars_velocity = ImagePars(
            shape=(Nx_vel, Ny_vel), pixel_scale=0.3, indexing='xy'  # arcsec/pixel
        )

        # intensity: taller than wide (Ny > Nx) - opposite orientation
        Nx_int, Ny_int = 60, 80
        self.image_pars_intensity = ImagePars(
            shape=(Nx_int, Ny_int), pixel_scale=0.3, indexing='xy'  # arcsec/pixel
        )

        return

    def get_sampler_log_path(self, test_name: str, sampler_name: str) -> Path:
        """
        Get path for sampler output log file.

        Sampler output is always written to files in the test output directory.

        Parameters
        ----------
        test_name : str
            Name of the test (used as subdirectory).
        sampler_name : str
            Name of the sampler (e.g., 'emcee', 'nautilus', 'blackjax').

        Returns
        -------
        Path
            Path to the log file.
        """
        test_dir = self.output_dir / test_name
        test_dir.mkdir(parents=True, exist_ok=True)
        return test_dir / f"{sampler_name}_output.txt"

    def get_tolerance(
        self,
        snr: float,
        param_name: str,
        param_value: float,
        data_type: str,
        test_type: str,
        has_psf: bool = False,
        model_kind: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Get tolerance for parameter at given SNR.

        Returns both relative and absolute tolerances.
        Parameter passes if EITHER criterion is met.

        Parameters
        ----------
        snr : float
            Signal-to-noise ratio of data.
        param_name : str
            Name of parameter being tested.
        param_value : float
            True value of parameter (needed for absolute tolerance).
        data_type : str
            'velocity' or 'intensity'.
        test_type : str
            Must be 'likelihood_slice' (legacy slice-test pattern). Optimizer
            tests use k-sigma bounds instead (k_sigma_recovery_stats).
        has_psf : bool
            If True, apply psf_tolerance_multiplier to loosen tolerances.
        model_kind : str, optional
            'composite' selects the composite tolerance dicts (initialized
            identically to single-component intensity, but extensible per-param
            via composite_param_scaling). None falls back to default dicts.

        Returns
        -------
        dict
            Contains 'relative' and 'absolute' tolerance values.
        """

        # Get base tolerance for this SNR. Composite intensity models route
        # to the dedicated dicts so per-param scaling and tolerances can
        # evolve independently from the exponential baseline.
        is_composite = model_kind == 'composite' and data_type == 'intensity'
        if test_type != 'likelihood_slice':
            raise ValueError(
                f"unknown test_type '{test_type}'; optimizer-tier tolerances "
                f"were retired 2026-07-08 in favor of k-sigma bounds "
                f"(k_sigma_recovery_stats)"
            )
        if data_type == 'velocity':
            base_tol = self.likelihood_slice_tolerance_velocity.get(snr, 0.05)
        elif is_composite:
            base_tol = self.likelihood_slice_tolerance_intensity_composite.get(
                snr, 0.05
            )
        else:
            base_tol = self.likelihood_slice_tolerance_intensity.get(snr, 0.05)
        param_scaling = (
            self.composite_param_scaling
            if is_composite
            else self.likelihood_slice_param_scaling
        )

        # Parameter-name lookup: exact match (full dotted name) first, then
        # suffix after the first '.' (e.g. 'F087.x0' -> 'x0'). Lets the dict
        # carry shared defaults under unprefixed keys while permitting per-
        # namespace overrides (e.g. 'vel.x0') when physics demands them.
        suffix_key = param_name.split('.', 1)[-1] if '.' in param_name else param_name

        # Apply parameter-specific scaling
        if param_name in param_scaling:
            scaling = param_scaling[param_name].get(snr, 1.0)
            relative_tol = base_tol * scaling
        elif suffix_key in param_scaling:
            scaling = param_scaling[suffix_key].get(snr, 1.0)
            relative_tol = base_tol * scaling
        else:
            relative_tol = base_tol

        # Apply PSF tolerance multiplier
        if has_psf:
            relative_tol *= self.psf_tolerance_multiplier

        # compute absolute tolerance
        # use the larger of: (relative_tol × |value|) or absolute_floor.
        # relative_tol already carries the PSF multiplier; multiplying the
        # absolute tolerance by it again (a long-standing bug fixed
        # 2026-07-08) silently loosened every has_psf test's effective
        # tolerance to 2.25x base instead of the documented 1.5x.
        absolute_from_relative = relative_tol * abs(param_value)
        if param_name in self.absolute_tolerance_floor:
            absolute_floor = self.absolute_tolerance_floor[param_name]
        else:
            absolute_floor = self.absolute_tolerance_floor.get(suffix_key, 0.0)
        absolute_tol = max(absolute_from_relative, absolute_floor)

        return {
            'relative': relative_tol,
            'absolute': absolute_tol,
        }


def _quadratic_peak_interp(param_values, log_probs):
    """Sub-grid-point peak finding via 3-point parabolic interpolation.

    Fits a quadratic to the argmax and its two neighbors, returns the
    vertex location. Falls back to discrete argmax at boundaries or
    when curvature is degenerate.
    """
    best_idx = int(jnp.argmax(log_probs))
    n = len(param_values)

    # boundary — can't interpolate
    if best_idx == 0 or best_idx == n - 1:
        return float(param_values[best_idx])

    x0 = float(param_values[best_idx - 1])
    x1 = float(param_values[best_idx])
    x2 = float(param_values[best_idx + 1])
    y0 = float(log_probs[best_idx - 1])
    y1 = float(log_probs[best_idx])
    y2 = float(log_probs[best_idx + 1])

    denom = 2.0 * (y0 - 2.0 * y1 + y2)

    # flat or degenerate curvature
    if abs(denom) < 1e-30:
        return x1

    # vertex of parabola through the 3 points
    dx = (y0 - y2) / denom
    peak = x1 + dx * (x2 - x0) / 2.0

    # clamp to the interval [x0, x2]
    peak = max(x0, min(x2, peak))

    return peak


def check_parameter_recovery(
    recovered: float,
    true_value: float,
    tolerance: Dict[str, float],
    param_name: str,
) -> Tuple[bool, Dict[str, float]]:
    """
    Check if parameter recovered within tolerance.

    Passes if EITHER relative OR absolute criterion is met.

    Parameters
    ----------
    recovered : float
        Recovered parameter value.
    true_value : float
        True parameter value.
    tolerance : dict
        Contains 'relative' and 'absolute' tolerance values.
    param_name : str
        Name of parameter (for logging).

    Returns
    -------
    passed : bool
        True if parameter recovered successfully.
    stats : dict
        Statistics about the recovery.
    """

    abs_error = abs(recovered - true_value)

    # avoid division by zero for parameters exactly at zero
    if abs(true_value) < 1e-10:
        rel_error = abs_error  # treat as absolute error
        passed_relative = False  # can't use relative for zero
    else:
        rel_error = abs_error / abs(true_value)
        passed_relative = rel_error <= tolerance['relative']

    passed_absolute = abs_error <= tolerance['absolute']

    # pass if *either* criterion met
    passed = passed_relative or passed_absolute

    stats = {
        'true': true_value,
        'recovered': recovered,
        'abs_error': abs_error,
        'rel_error': rel_error,
        'abs_tolerance': tolerance['absolute'],
        'rel_tolerance': tolerance['relative'],
        'passed_absolute': passed_absolute,
        'passed_relative': passed_relative,
        'passed': passed,
        'criterion': (
            'relative'
            if passed_relative
            else ('absolute' if passed_absolute else 'none')
        ),
    }

    return passed, stats


def assert_parameter_recovery(
    recovery_stats: Dict[str, Dict[str, float]],
    snr: float,
    test_name: str = "Test",
    exclude_params: Optional[list] = None,
) -> None:
    """
    Assert that all parameters were recovered within tolerance.

    If any parameter failed, formats a detailed error message and fails the test.

    Parameters
    ----------
    recovery_stats : dict
        Recovery statistics from plot_likelihood_slices() or similar.
        Each entry should have 'passed', 'rel_error', 'abs_error', etc.
    snr : float
        Signal-to-noise ratio (for error message).
    test_name : str, optional
        Name of test (for error message). Default is "Test".
    exclude_params : list, optional
        Parameter names to exclude from pass/fail check (still reported).
        Useful for degenerate parameters like g1/g2.
    """

    if exclude_params is None:
        exclude_params = []

    failed_params = []
    excluded_params = []

    for param_name, stats in recovery_stats.items():
        if not stats['passed']:
            msg = (
                f"{param_name}: "
                f"rel {stats['rel_error']*100:.2f}% (tol {stats['rel_tolerance']*100:.1f}%), "
                f"abs {stats['abs_error']:.4f} (tol {stats['abs_tolerance']:.4f}) "
                f"- recovered {stats['recovered']:.4f}, true {stats['true']:.4f}"
            )

            if param_name in exclude_params:
                excluded_params.append(msg + " [EXCLUDED]")
            else:
                failed_params.append(msg)

    # Report excluded params as warnings (still visible but don't fail)
    if excluded_params:
        print(f"\n⚠️  {test_name} - Excluded parameters outside tolerance:")
        for msg in excluded_params:
            print(f"  {msg}")

    if failed_params:
        msg = f"{test_name} failed for SNR={snr}:\n" + "\n".join(failed_params)
        pytest.fail(msg)

    return


def assert_noiseless_recovery(
    recovery_stats: Dict[str, Dict[str, float]],
    budgets: Dict[str, float],
    test_name: str = "Test",
) -> None:
    """
    Assert noise-free recovery errors are within per-parameter budgets.

    On noise-free data a correct forward model recovers the truth exactly
    (up to estimator resolution), so each budget is an accuracy statement,
    not a noise allowance: for self-rendered data it bounds estimator and
    model bias; for data rendered by an independent backend (e.g. GalSim)
    it bounds the agreement between the two renderers. Budgets are
    absolute, in each parameter's own units. There is no random element:
    a budget violation is a real change in behavior.

    Parameters
    ----------
    recovery_stats : dict
        Recovery statistics from plot_likelihood_slices() or similar;
        each entry needs 'abs_error', 'recovered', 'true'.
    budgets : dict
        Parameter name -> allowed absolute recovery error.
    test_name : str, optional
        Name of test (for error message).
    """
    missing = sorted(set(budgets) - set(recovery_stats))
    if missing:
        pytest.fail(f"{test_name}: no recovery stats for budgeted params {missing}")

    failed = []
    for name, budget in budgets.items():
        stats = recovery_stats[name]
        if not stats['abs_error'] < budget:
            failed.append(
                f"{name}: |error| {stats['abs_error']:.3e} >= budget {budget:.3e} "
                f"(recovered {stats['recovered']:.6g}, true {stats['true']:.6g})"
            )
    if failed:
        pytest.fail(
            f"{test_name}: noise-free recovery outside accuracy budget "
            f"(deterministic -- this is a real behavior change):\n" + "\n".join(failed)
        )


def slice_curvature_sigma(param_values, log_probs) -> float:
    """
    1-sigma error bar implied by a likelihood slice's peak sharpness.

    Fits the parabola at the discrete peak of the slice and returns
    1/sqrt of the negative second derivative of the log-likelihood --
    the uncertainty the data could deliver for this parameter with all
    others held at truth. Requires a uniform grid and an interior peak.
    """
    x = np.asarray(param_values, dtype=float)
    lp = np.asarray(log_probs, dtype=float)
    best = int(np.argmax(lp))
    if best == 0 or best == len(x) - 1:
        raise ValueError(
            f"slice peak at grid boundary (index {best}); widen the scan "
            f"before measuring curvature"
        )
    h = x[1] - x[0]
    curvature = -(lp[best - 1] - 2.0 * lp[best] + lp[best + 1]) / h**2
    if not curvature > 0:
        raise ValueError(
            f"non-positive curvature {curvature} at the slice peak; the "
            f"likelihood is flat or corrupted on this slice"
        )
    return float(1.0 / np.sqrt(curvature))


def assert_slice_curvature_references(
    slices: Dict[str, Tuple],
    references: Dict[str, float],
    test_name: str = "Test",
    band: float = 0.20,
) -> None:
    """
    Assert each slice's implied error bar matches its frozen reference.

    Guards the information content of the likelihood: a change that blurs
    the likelihood (constraining power lost) or sharpens it beyond the
    physics (e.g. a discretization artifact faking precision) moves the
    error bar out of the band and fails loudly. References are measured
    once on the frozen test scene at its stated noise convention and are
    updated only deliberately, with the reason recorded next to the value.

    Parameters
    ----------
    slices : dict
        Parameter name -> (param_values, log_probs) from slice_all_parameters.
    references : dict
        Parameter name -> frozen 1-sigma reference value.
    test_name : str, optional
        Name of test (for error message).
    band : float, optional
        Allowed fractional deviation from the reference (default 0.20).
    """
    missing = sorted(set(references) - set(slices))
    if missing:
        pytest.fail(f"{test_name}: no slices for referenced params {missing}")

    failed = []
    for name, ref in references.items():
        sigma = slice_curvature_sigma(*slices[name])
        if not (abs(sigma / ref - 1.0) < band):
            direction = (
                "blurred (information lost)"
                if sigma > ref
                else ("sharpened beyond the reference (fake precision)")
            )
            failed.append(
                f"{name}: slice error bar {sigma:.4g} vs reference {ref:.4g} "
                f"(+/-{band:.0%} band) -- likelihood {direction}"
            )
    if failed:
        pytest.fail(
            f"{test_name}: likelihood information content changed:\n"
            + "\n".join(failed)
        )


def _round_up_1sf(x: float) -> float:
    """Round a positive number up to one significant figure."""
    if x <= 0:
        raise ValueError(f"expected positive value, got {x}")
    exp = np.floor(np.log10(x))
    mant = x / 10**exp
    return float(np.ceil(mant - 1e-12) * 10**exp)


def assert_noiseless_gates(
    recovery_stats: Dict[str, Dict[str, float]],
    slices: Dict[str, Tuple],
    budgets: Dict[str, float],
    sigma_refs: Dict[str, float],
    test_name: str = "Test",
    band: float = 0.20,
) -> None:
    """
    Run both noiseless gates: accuracy budgets and error-bar references.

    Set the environment variable KLPIPE_TEST_MEASURE=1 to print
    ready-to-freeze tables instead of asserting (budget rule: 10x the
    measured error, floored at 1e-5 of the scan half-range, rounded up to
    one significant figure; error bars to 3 significant figures). Used to
    measure a frozen scene once; the printed values are then pasted into
    the test with a provenance comment.
    """
    if os.environ.get('KLPIPE_TEST_MEASURE'):
        budget_lines = []
        sigma_lines = []
        for name in recovery_stats:
            if name not in slices:
                continue
            vals = np.asarray(slices[name][0], dtype=float)
            half_range = (vals[-1] - vals[0]) / 2.0
            err = abs(recovery_stats[name]['abs_error'])
            budget = _round_up_1sf(max(10.0 * err, 1e-5 * half_range))
            sigma = slice_curvature_sigma(*slices[name])
            budget_lines.append(f"        '{name}': {budget:.0e},")
            sigma_lines.append(f"        '{name}': {float(f'{sigma:.3g}')},")
        print(f"\nKLPIPE-MEASURE {test_name} entry:")
        print(f"    'budgets': {{")
        print("\n".join(budget_lines))
        print(f"    }},")
        print(f"    'sigma_refs': {{")
        print("\n".join(sigma_lines))
        print(f"    }},")
        return

    assert_noiseless_recovery(recovery_stats, budgets, test_name)
    assert_slice_curvature_references(slices, sigma_refs, test_name, band=band)


def suite_false_alarm_k(n_checks: int, budget: float = 0.01) -> float:
    """
    Bound multiplier k for noisy recovery checks, from a false-alarm budget.

    A recovery error on noisy data is a random draw: near the maximum-
    likelihood point it is approximately Gaussian with the parameter's
    marginal 1-sigma noise floor as its scale. A check ``|error| < k *
    sigma`` therefore has a two-sided false-alarm probability
    ``2 * Phi(-k)`` per check even when the code is correct. Requiring the
    whole suite of ``n_checks`` such checks to spuriously fail with total
    probability at most ``budget`` (union bound, checks treated as
    independent) gives ``k = Phi^-1(1 - budget / n_checks / 2)``.

    This is the single suite-wide constant used by every k-sigma bound:
    it is derived, not tuned, and it grows only logarithmically with the
    number of checks (17 checks -> 3.44, 50 -> 3.72, 200 -> 4.06 at the
    default 1% budget).

    Parameters
    ----------
    n_checks : int
        Number of bounded checks sharing the budget.
    budget : float, optional
        Total suite false-alarm probability (default 0.01).

    Returns
    -------
    float
        The bound multiplier k.
    """
    from scipy.stats import norm

    if n_checks < 1:
        raise ValueError(f"n_checks must be >= 1, got {n_checks}")
    if not 0.0 < budget < 1.0:
        raise ValueError(f"budget must be in (0, 1), got {budget}")
    return float(norm.isf(budget / n_checks / 2.0))


def measure_marginal_sigmas(
    log_like: Callable,
    theta_truth: jnp.ndarray,
    names: List[str],
    drop: Tuple[str, ...] = (),
    prior_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Dict[str, float]:
    """
    Marginal 1-sigma noise floors from the likelihood curvature at truth.

    Call with a likelihood built on the MODEL'S OWN noise-free render so
    the curvature at truth is exactly the information matrix (a residual
    at truth -- noise, or an independent renderer's version of the scene
    -- adds a second-order term that can even flip curvature signs along
    degenerate directions). The marginal floor -- the error bar for each
    parameter with all others fit jointly -- is the square root of the
    corresponding diagonal element of the inverse.

    When ``prior_bounds`` is given, each parameter's uniform prior box is
    approximated by a Gaussian of equal variance (precision 12/width^2)
    and added to the information matrix. Data-dominated floors are
    unaffected (the prior precision is negligible against them), while
    degenerate directions -- constrained only by the box in the actual
    fit -- get honest prior-scale floors instead of a singular inversion,
    and parameters coupled to a degenerate direction get their true
    marginal including the bounded wander along it.

    Parameters listed in ``drop`` are removed before inversion; use it
    only for exactly-flat directions that are pinned separately.

    Parameters
    ----------
    log_like : callable
        Jitted log-likelihood of the sampled parameter vector, built on
        the model's own noise-free render with the test's noise variance.
    theta_truth : jnp.ndarray
        True sampled-parameter vector.
    names : list of str
        Sampled parameter names, in theta order.
    drop : tuple of str, optional
        Names of exactly-unconstrained parameters to remove.
    prior_bounds : dict, optional
        Parameter name -> (low, high) uniform prior bounds.

    Returns
    -------
    dict
        Parameter name -> marginal 1-sigma floor.
    """
    hessian = np.asarray(jax.hessian(log_like)(jnp.asarray(theta_truth)))
    fisher = -0.5 * (hessian + hessian.T)  # symmetrize away roundoff
    if prior_bounds is not None:
        for i, name in enumerate(names):
            if name not in prior_bounds:
                continue
            low, high = prior_bounds[name]
            width = float(high) - float(low)
            if width <= 0:
                raise ValueError(f"non-positive prior width for {name}")
            fisher[i, i] += 12.0 / width**2
    keep = [i for i, n in enumerate(names) if n not in drop]
    missing_drops = sorted(set(drop) - set(names))
    if missing_drops:
        raise ValueError(f"drop names not in sampled names: {missing_drops}")
    fisher_kept = fisher[np.ix_(keep, keep)]
    eigvals = np.linalg.eigvalsh(fisher_kept)
    if eigvals[0] <= 0:
        bad = [names[keep[i]] for i in np.argsort(eigvals)[:3]]
        raise ValueError(
            f"information matrix not positive definite (min eigenvalue "
            f"{eigvals[0]:.3e}); flattest directions involve {bad} -- an "
            f"exactly unconstrained parameter must be dropped, or the "
            f"likelihood is corrupted"
        )
    cov = np.linalg.inv(fisher_kept)
    sigmas = np.sqrt(np.diag(cov))
    if not np.all(np.isfinite(sigmas)):
        raise ValueError("non-finite marginal sigma; likelihood corrupted")
    return {names[i]: float(s) for i, s in zip(keep, sigmas)}


def print_optimizer_sigma_entry(
    test_key: str,
    log_like: Callable,
    theta_truth: jnp.ndarray,
    names: List[str],
    snr: float,
    snr_ref: float,
    drop: Tuple[str, ...] = (),
    prior_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
) -> None:
    """
    Print a ready-to-freeze marginal-sigma table entry for one test scene.

    A single frozen table serves every SNR the test runs at via a 1/SNR
    rescale of the floors. That rescale is exact only for data-dominated
    floors (the prior term from ``prior_bounds`` does not scale with SNR),
    so always measure at the HIGHEST SNR the test runs -- snr must equal
    snr_ref -- making the frozen bounds exact at the tightest instance and
    conservative (never tighter than honest) at lower SNR.
    """
    if snr != snr_ref:
        raise ValueError(
            f"measure at the reference SNR (snr={snr} != snr_ref={snr_ref}); "
            f"the 1/SNR rescale is not exact for prior-dominated floors"
        )
    sigmas = measure_marginal_sigmas(
        log_like, theta_truth, names, drop, prior_bounds=prior_bounds
    )
    scale = snr / snr_ref
    lines = [
        f"        '{name}': {float(f'{sigma * scale:.3g}')},"
        for name, sigma in sigmas.items()
    ]
    print(f"\nKLPIPE-MEASURE optimizer sigma entry '{test_key}' (snr_ref={snr_ref:g}):")
    print("    'sigmas': {")
    print("\n".join(lines))
    print("    },")


def optimizer_measure_mode() -> bool:
    """True when KLPIPE_TEST_MEASURE=1: measure noise floors, skip recovery."""
    return bool(os.environ.get('KLPIPE_TEST_MEASURE'))


def print_optimizer_bias_entry(
    test_key: str,
    theta_truth: jnp.ndarray,
    theta_fit: jnp.ndarray,
    names: List[str],
) -> None:
    """
    Print a ready-to-freeze bias-allowance table entry.

    Call with the result of fitting an independent backend's NOISE-FREE
    render (e.g. GalSim): the recovered offset from truth is the
    deterministic rendering difference between that backend and the
    model. The allowance rule is 2x the measured offset, rounded up to
    one significant figure -- margin for optimizer path variation while
    still failing loudly if the rendering difference ever doubles.
    """
    lines = []
    for name, t, f in zip(names, np.asarray(theta_truth), np.asarray(theta_fit)):
        offset = 2.0 * abs(float(f) - float(t))
        allowance = _round_up_1sf(offset) if offset > 0 else 0.0
        lines.append(f"        '{name}': {allowance:.0e},")
    print(f"\nKLPIPE-MEASURE optimizer bias entry '{test_key}' (2x measured, 1 s.f.):")
    print("    'biases': {")
    print("\n".join(lines))
    print("    },")


def k_sigma_recovery_stats(
    pars_true: Dict[str, float],
    pars_recovered: Dict[str, float],
    names: List[str],
    sigmas: Dict[str, float],
    k: float,
    snr: float,
    snr_ref: float,
    biases: Optional[Dict[str, float]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Recovery statistics with k-sigma bounds on noisy-data errors.

    Each parameter's bound is ``bias + k * sigma * (snr_ref / snr)`` where
    sigma is its frozen marginal noise floor at the reference SNR (the
    error a correct pipeline can produce on a noisy draw, scaled by the
    shared suite-wide multiplier from ``suite_false_alarm_k``) and bias is
    an optional frozen allowance for a measured systematic offset (e.g.
    data rendered by an independent backend whose disagreement with the
    model exceeds the noise floor at high SNR). Parameters absent from
    ``sigmas`` (exactly unconstrained directions, pinned separately) are
    skipped. Output is compatible with ``assert_parameter_recovery`` and
    the comparison plots.
    """
    missing = sorted(set(sigmas) - set(names))
    if missing:
        raise ValueError(f"sigma table has unknown parameters: {missing}")
    if biases:
        bad_bias = sorted(set(biases) - set(sigmas))
        if bad_bias:
            raise ValueError(f"bias table has parameters without sigmas: {bad_bias}")
    stats = {}
    for name in names:
        if name not in sigmas:
            continue
        true_val = pars_true[name]
        recovered = pars_recovered[name]
        bound = (biases or {}).get(name, 0.0) + k * sigmas[name] * (snr_ref / snr)
        abs_error = abs(recovered - true_val)
        rel_error = abs_error / abs(true_val) if abs(true_val) > 1e-10 else abs_error
        passed = abs_error < bound
        stats[name] = {
            'true': true_val,
            'recovered': recovered,
            'abs_error': abs_error,
            'rel_error': rel_error,
            'abs_tolerance': bound,
            'rel_tolerance': bound / abs(true_val) if abs(true_val) > 1e-10 else bound,
            'passed_absolute': passed,
            'passed_relative': False,
            'passed': passed,
            'criterion': 'absolute' if passed else 'none',
        }
    return stats


def check_degenerate_product_recovery(
    pars_true: Dict[str, float],
    pars_recovered: Dict[str, float],
    snr: Optional[float] = None,
    tolerance_relative: Optional[float] = None,
) -> Tuple[bool, Dict[str, any]]:
    """
    Check recovery of degenerate parameter products.

    For velocity models, the observed signal depends on vcirc*sini
    rather than vcirc and cosi independently. This function checks if this
    product is recovered even when individual parameters may be off.

    Parameters
    ----------
    pars_true : dict
        True parameter values (must include 'vcirc' and 'cosi').
    pars_recovered : dict
        Recovered parameter values.
    snr : float, optional
        Signal-to-noise ratio. If provided, tolerance is SNR-dependent.
    tolerance_relative : float, optional
        Relative tolerance for the product. If not provided, uses SNR-dependent
        default (3% for SNR=1000, 10% for SNR=10).

    Returns
    -------
    passed : bool
        Whether the product was recovered within tolerance.
    stats : dict
        Statistics about the product recovery.
    """

    # Look up vcirc under either the unprefixed name or the dotted
    # ``vel.vcirc`` form (callers may use either). cosi is shared and
    # unprefixed in both conventions.
    def _vcirc(pars):
        if 'vcirc' in pars:
            return pars['vcirc']
        if 'vel.vcirc' in pars:
            return pars['vel.vcirc']
        return None

    vcirc_true = _vcirc(pars_true)
    vcirc_recovered = _vcirc(pars_recovered)
    if vcirc_true is None or 'cosi' not in pars_true:
        return True, {'note': 'vcirc or cosi not in model, skipping product check'}

    # Set tolerance based on SNR if not explicitly provided
    if tolerance_relative is None:
        if snr is not None:
            # SNR-dependent tolerance for degenerate products
            # More lenient than individual parameters since we only care about observable
            snr_tolerances = {
                1000: 0.03,  # 3%
                500: 0.04,  # 4%
                100: 0.05,  # 5%
                50: 0.07,  # 7%
                10: 0.10,  # 10%
            }
            tolerance_relative = snr_tolerances.get(snr, 0.10)
        else:
            tolerance_relative = 0.05  # Default 5%

    # Compute true product
    sini_true = np.sqrt(1 - pars_true['cosi'] ** 2)
    product_true = vcirc_true * sini_true

    # Compute recovered product
    sini_recovered = np.sqrt(1 - pars_recovered['cosi'] ** 2)
    product_recovered = vcirc_recovered * sini_recovered

    # Check tolerance
    abs_error = abs(product_recovered - product_true)
    rel_error = abs_error / product_true if product_true > 0 else abs_error
    passed = rel_error <= tolerance_relative

    stats = {
        'true': product_true,
        'recovered': product_recovered,
        'abs_error': abs_error,
        'rel_error': rel_error,
        'tolerance': tolerance_relative,
        'passed': passed,
        'formula': 'vcirc * sin(i) = vcirc * sqrt(1 - cosi²)',
    }

    return passed, stats


def compute_parameter_bounds(
    param_name: str,
    true_value: float,
    config: TestConfig,
    image_pars: Optional[ImagePars] = None,
    fraction: float = 0.25,
) -> Tuple[float, float]:
    """
    Compute scan bounds for a parameter.

    Uses +/-fraction around true value by default, respecting physical boundaries.

    Parameters
    ----------
    param_name : str
        Parameter name.
    true_value : float
        True parameter value.
    config : TestConfig
        Test configuration containing parameter bounds.
    image_pars : ImagePars, optional
        Image parameters (needed for x0, y0 bounds).
        If None for centroid params, uses conservative default bounds.
    fraction : float, optional
        Fractional range around true value. Default is 0.25 (±25%).

    Returns
    -------
    lower, upper : float
        Bounds for parameter scan.
    """

    # Check if parameter has physical bounds
    if param_name in config.param_bounds:
        lower_phys, upper_phys = config.param_bounds[param_name]

        # Compute +/-fraction range. A zero-truth bounded parameter (e.g.
        # shear at 0) previously produced a zero-width scan, making its
        # slice vacuous; scan a fraction of the physical span instead.
        delta = fraction * abs(true_value)
        if delta == 0.0:
            if lower_phys is not None and upper_phys is not None:
                delta = fraction * (upper_phys - lower_phys) / 2.0
            else:
                delta = 0.1
        lower_pct = true_value - delta
        upper_pct = true_value + delta

        # Respect physical boundaries
        if lower_phys is not None:
            lower = max(lower_pct, lower_phys)
        else:
            lower = lower_pct

        if upper_phys is not None:
            upper = min(upper_pct, upper_phys)
        else:
            upper = upper_pct

    # Special case: centroid offsets (bounded by image)
    elif param_name in ['x0', 'y0', 'x0', 'y0', 'x0', 'y0']:
        if image_pars is not None:
            # Use image extent
            extent = image_pars.shape[0] * image_pars.pixel_scale / 2
        else:
            # Use conservative default based on typical config values
            # For joint models, use the larger of the two grids
            extent_vel = (
                config.image_pars_velocity.shape[0]
                * config.image_pars_velocity.pixel_scale
                / 2
            )
            extent_int = (
                config.image_pars_intensity.shape[0]
                * config.image_pars_intensity.pixel_scale
                / 2
            )
            extent = max(extent_vel, extent_int)

        delta = (
            fraction * abs(true_value) if true_value != 0 else 1.0
        )  # Default to 1 arcsec if centered
        lower = max(true_value - delta, -extent)
        upper = min(true_value + delta, extent)

    # Default: +/-fraction
    else:
        delta = (
            fraction * abs(true_value) if true_value != 0 else 0.1
        )  # Small default for zero values
        lower = true_value - delta
        upper = true_value + delta

    return lower, upper


def slice_likelihood_1d(
    log_like_fn: Callable,
    theta_true: jnp.ndarray,
    param_idx: int,
    param_name: str,
    config: TestConfig,
    n_points: int = 201,
    image_pars: Optional[ImagePars] = None,
    scan_fraction: float = 0.25,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute 1D likelihood slice for a single parameter.

    Parameters
    ----------
    log_like_fn : callable
        JIT-compiled log-likelihood function.
    theta_true : jnp.ndarray
        True parameter array.
    param_idx : int
        Index of parameter to slice.
    param_name : str
        Name of parameter (for bounds computation).
    config : TestConfig
        Test configuration.
    n_points : int, optional
        Number of points in slice. Default is 100.
    image_pars : ImagePars, optional
        Image parameters (for x0, y0 bounds).
    scan_fraction : float, optional
        Fractional scan range. Default is 0.25.

    Returns
    -------
    param_values : jnp.ndarray
        Parameter values scanned.
    log_probs : jnp.ndarray
        Log-likelihood at each parameter value.
    """

    true_value = float(theta_true[param_idx])

    # Compute scan range
    lower, upper = compute_parameter_bounds(
        param_name, true_value, config, image_pars, fraction=scan_fraction
    )
    param_values = jnp.linspace(lower, upper, n_points)

    # Evaluate likelihood at each point
    log_probs = []
    for val in param_values:
        theta_test = theta_true.at[param_idx].set(val)
        log_prob = log_like_fn(theta_test)
        log_probs.append(float(log_prob))

    return param_values, jnp.array(log_probs)


def slice_all_parameters(
    log_like_fn: Callable,
    sampled_names,
    theta_true: jnp.ndarray,
    config: TestConfig,
    n_points: int = 201,
    image_pars: Optional[ImagePars] = None,
    scan_fraction: float = 0.25,
) -> Dict[str, Tuple[jnp.ndarray, jnp.ndarray]]:
    """
    Compute likelihood slices for all parameters.

    Parameters
    ----------
    log_like_fn : callable
        JIT-compiled log-likelihood function.
    sampled_names : sequence of str
        Ordered names of sampled parameters (e.g. ``priors.sampled_names``).
        Defines the index → name mapping used to label slices and align with
        ``theta_true``.
    theta_true : jnp.ndarray
        True parameter array.
    config : TestConfig
        Test configuration.
    n_points : int, optional
        Number of points per slice. Default is 100.
    image_pars : ImagePars, optional
        Image parameters.
    scan_fraction : float, optional
        Fractional scan range. Default is 0.25.

    Returns
    -------
    slices : dict
        Dictionary mapping parameter names to (values, log_probs) tuples.
    """

    slices = {}

    for idx, param_name in enumerate(sampled_names):
        param_values, log_probs = slice_likelihood_1d(
            log_like_fn,
            theta_true,
            idx,
            param_name,
            config,
            n_points,
            image_pars,
            scan_fraction,
        )
        slices[param_name] = (param_values, log_probs)

    return slices


# ==============================================================================
# Diagnostic Plotting (DEPRECATED - use kl_pipe.diagnostics instead)
# ==============================================================================


def plot_data_comparison_panels(
    data_noisy: jnp.ndarray,
    data_true: jnp.ndarray,
    model_eval: jnp.ndarray,
    test_name: str,
    config: TestConfig,
    data_type: str = 'velocity',
    variance: Optional[float] = None,
    n_params: Optional[int] = None,
    model_label: str = 'Model',
) -> None:
    """
    Create 2x3 panel diagnostic plot.

    .. deprecated::
        Use kl_pipe.diagnostics.plot_data_comparison_panels instead.
        This function is kept for backward compatibility.

    Row 1: noisy | true | noisy - true
    Row 2: model - true | model | noisy - model

    Parameters
    ----------
    data_noisy : jnp.ndarray
        Noisy synthetic data.
    data_true : jnp.ndarray
        True noiseless data.
    model_eval : jnp.ndarray
        Model evaluation (can be at true or optimized parameters).
    test_name : str
        Name of test (for title and filename).
    config : TestConfig
        Test configuration (for output dir and plot enable flag).
    data_type : str, optional
        Type of data ('velocity' or 'intensity'). Default is 'velocity'.
    variance : float, optional
        Variance of noise, if you want to report reduced chi-squared.
    n_params : int, optional
        Number of fitted parameters (for reduced chi-squared). Default is None.
    model_label : str, optional
        Label for model panel ('Model' or 'Optimized Model'). Default is 'Model'.
    """
    warnings.warn(
        "plot_data_comparison_panels in test_utils.py is deprecated. "
        "Use kl_pipe.diagnostics.plot_data_comparison_panels instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    if not config.enable_plots:
        return

    # Create output directory for this test
    test_dir = config.output_dir / test_name
    test_dir.mkdir(parents=True, exist_ok=True)

    # Compute residuals & chi2
    residual_true = np.array(data_noisy - data_true)
    residual_model = np.array(data_noisy - model_eval)
    residual_model_true = np.array(model_eval - data_true)

    chi2_true = None
    chi2_model = None
    if variance is not None:
        chi2_true = np.sum(residual_true**2 / variance)
        chi2_model = np.sum(residual_model**2 / variance)
        if n_params is not None:
            dof = data_noisy.size - n_params
            chi2_true /= dof
            chi2_model /= dof

    # Set up figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Common colorbar limits for data
    data_arrays = [data_noisy, data_true, model_eval]
    vmin_data = min(np.percentile(arr, 1) for arr in data_arrays)
    vmax_data = max(np.percentile(arr, 99) for arr in data_arrays)
    norm_data = MidpointNormalize(vmin=vmin_data, vmax=vmax_data, midpoint=0)

    # Common colorbar limits for residuals
    residual_arrays = [residual_true, residual_model]
    abs_max = max(np.abs(np.percentile(arr, [1, 99])).max() for arr in residual_arrays)
    norm_resid = MidpointNormalize(vmin=-abs_max, vmax=abs_max, midpoint=0)

    # Row 1: noisy | true | noisy - true
    im00 = axes[0, 0].imshow(
        np.array(data_noisy),
        origin='lower',
        cmap='RdBu_r',
        norm=norm_data,
    )
    axes[0, 0].set_title('Noisy Data')
    divider = make_axes_locatable(axes[0, 0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im00, cax=cax)

    im01 = axes[0, 1].imshow(
        np.array(data_true),
        origin='lower',
        cmap='RdBu_r',
        norm=norm_data,
    )
    axes[0, 1].set_title('True (Noiseless)')
    divider = make_axes_locatable(axes[0, 1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im01, cax=cax)

    im02 = axes[0, 2].imshow(
        residual_true, origin='lower', cmap='RdBu_r', norm=norm_resid
    )
    axes[0, 2].set_title('Noisy - True')
    divider = make_axes_locatable(axes[0, 2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im02, cax=cax)
    if variance is not None:
        axes[0, 2].text(
            0.02,
            0.98,
            f'χ² = {chi2_true:.1f}',
            transform=axes[0, 2].transAxes,
            fontsize=10,
            color='white',
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.5),
        )

    # Row 2: model - true | model | noisy - model
    im10 = axes[1, 0].imshow(
        residual_model_true,
        origin='lower',
        cmap='RdBu_r',
        norm=norm_resid,
    )
    axes[1, 0].set_title('Model - True')
    divider = make_axes_locatable(axes[1, 0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im10, cax=cax)

    im11 = axes[1, 1].imshow(
        np.array(model_eval),
        origin='lower',
        cmap='RdBu_r',
        norm=norm_data,
    )
    axes[1, 1].set_title(model_label)
    divider = make_axes_locatable(axes[1, 1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im11, cax=cax)

    im12 = axes[1, 2].imshow(
        residual_model, origin='lower', cmap='RdBu_r', norm=norm_resid
    )
    axes[1, 2].set_title('Noisy - Model')
    divider = make_axes_locatable(axes[1, 2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im12, cax=cax)
    if variance is not None:
        axes[1, 2].text(
            0.02,
            0.98,
            f'χ² = {chi2_model:.1f}',
            transform=axes[1, 2].transAxes,
            fontsize=10,
            color='white',
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.5),
        )

    # Labels
    for ax in axes.flat:
        ax.set_xlabel('x (pixels)')
        ax.set_ylabel('y (pixels)')

    # Overall title
    fig.suptitle(f'{test_name} - {data_type.capitalize()} Comparison', fontsize=14)
    plt.tight_layout()

    # Save
    outfile = test_dir / f"{test_name}_{data_type}_panels.png"
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_parameter_comparison(
    true_pars: Dict[str, float],
    recovered_pars: Dict[str, float],
    recovery_stats: Dict[str, Dict[str, float]],
    test_name: str,
    config: TestConfig,
    snr: float,
    product_stats: Optional[Dict[str, any]] = None,
    exclude_params: Optional[List[str]] = None,
) -> None:
    """
    Create parameter comparison plot showing true vs recovered values.

    Parameters
    ----------
    true_pars : dict
        True parameter values.
    recovered_pars : dict
        Recovered parameter values.
    recovery_stats : dict
        Recovery statistics from check_parameter_recovery (includes tolerances).
    test_name : str
        Name of test (for filename).
    config : TestConfig
        Test configuration.
    snr : float
        Signal-to-noise ratio.
    product_stats : dict, optional
        Statistics from check_degenerate_product_recovery (vcirc*sini).
    exclude_params : list, optional
        List of parameter names excluded from pass/fail checks.
    """

    if not config.enable_plots:
        return

    # Create output directory
    test_dir = config.output_dir / test_name
    test_dir.mkdir(parents=True, exist_ok=True)

    # Add vcirc*sini product to the comparison if available
    if product_stats is not None and 'true' in product_stats:
        param_names = list(true_pars.keys()) + ['vcirc*sini']
        true_vals = np.concatenate(
            [
                np.array([true_pars[p] for p in true_pars.keys()]),
                [product_stats['true']],
            ]
        )
        recovered_vals = np.concatenate(
            [
                np.array([recovered_pars[p] for p in true_pars.keys()]),
                [product_stats['recovered']],
            ]
        )
        # Create pseudo-recovery stats for product
        all_recovery_stats = dict(recovery_stats)
        all_recovery_stats['vcirc*sini'] = {
            'passed': product_stats['passed'],
            'rel_error': product_stats['rel_error'],
            'abs_error': product_stats['abs_error'],
            'rel_tolerance': product_stats['tolerance'],
            'abs_tolerance': product_stats['abs_error'],
            'criterion': 'relative',
            'true': product_stats['true'],
            'recovered': product_stats['recovered'],
        }
        passed = np.array([all_recovery_stats[p]['passed'] for p in param_names])
    else:
        param_names = list(true_pars.keys())
        true_vals = np.array([true_pars[p] for p in param_names])
        recovered_vals = np.array([recovered_pars[p] for p in param_names])
        all_recovery_stats = recovery_stats
        passed = np.array([recovery_stats[p]['passed'] for p in param_names])

    if exclude_params is None:
        exclude_params = []

    # Count pass/fail (excluding excluded params)
    n_total = len(param_names)
    n_excluded = sum(1 for p in param_names if p in exclude_params)
    # Count only non-excluded parameters
    non_excluded_params = [p for p in param_names if p not in exclude_params]
    n_fail = sum(1 for p in non_excluded_params if not all_recovery_stats[p]['passed'])
    n_pass = len(non_excluded_params) - n_fail

    # Create figure with more vertical space
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))

    # Panel 1: Scatter plot with log scale if values span multiple orders of magnitude
    # Use categorical x-axis for better spacing
    x_pos = np.arange(len(param_names))

    # Use absolute values for log scale (to handle negative parameters like g1, g2)
    abs_true_vals = np.abs(true_vals)
    abs_recovered_vals = np.abs(recovered_vals)

    # Color code for top plot: green for passed, red for failed (even if excluded)
    colors_top = []
    for i, p in enumerate(param_names):
        if all_recovery_stats[p]['passed']:
            colors_top.append('green')
        else:
            colors_top.append('red')

    # Color code for bottom plot: green for passed, red for failed (even if excluded)
    colors_bar = []
    for i, p in enumerate(param_names):
        if all_recovery_stats[p]['passed']:
            colors_bar.append('green')
        else:
            colors_bar.append('red')

    # Add tolerance bands around true values (only for relative tolerance)
    for i, param_name in enumerate(param_names):
        criterion = all_recovery_stats[param_name]['criterion']
        true_val = abs_true_vals[i]

        if criterion == 'relative':
            rel_tol = all_recovery_stats[param_name]['rel_tolerance']
            lower = true_val * (1 - rel_tol)
            upper = true_val * (1 + rel_tol)
            ax1.fill_between(
                [x_pos[i] - 0.4, x_pos[i] + 0.4],
                lower,
                upper,
                alpha=0.2,
                color='gray',
                zorder=0,
            )

    # Plot on categorical x-axis
    # Non-excluded params use circles, excluded params use 'x' markers
    for i, param_name in enumerate(param_names):
        if param_name in exclude_params:
            ax1.scatter(
                x_pos[i],
                abs_recovered_vals[i],
                c=colors_top[i],
                marker='x',
                s=120,
                linewidths=2.5,
                alpha=1.0,
                zorder=3,
            )
        else:
            ax1.scatter(
                x_pos[i],
                abs_recovered_vals[i],
                c=colors_top[i],
                s=120,
                alpha=0.7,
                edgecolors='black',
                linewidths=1.5,
                zorder=3,
            )

    ax1.scatter(
        x_pos,
        abs_true_vals,
        marker='_',
        s=500,
        c='black',
        linewidths=3,
        zorder=4,
        label='True value',
    )

    # Connect true to recovered with lines
    for i in range(len(param_names)):
        ax1.plot(
            [x_pos[i], x_pos[i]],
            [abs_true_vals[i], abs_recovered_vals[i]],
            'k-',
            alpha=0.3,
            linewidth=1,
            zorder=1,
        )

    # Add absolute difference text above each circle
    for i in range(len(param_names)):
        abs_diff = abs(abs_recovered_vals[i] - abs_true_vals[i])
        # Position text above the higher of the two values, but not too high
        y_pos = max(abs_recovered_vals[i], abs_true_vals[i]) * 1.5
        ax1.text(
            x_pos[i],
            y_pos,
            f'{abs_diff:.3f}',
            ha='center',
            va='bottom',
            fontsize=9,
            color='black',
            bbox=dict(
                boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none'
            ),
            clip_on=False,
        )

    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(param_names, rotation=45, ha='right', fontsize=10)
    ax1.set_ylabel('|Parameter Value|', fontsize=13, fontweight='bold')
    ax1.set_yscale('log')

    # Adjust y-axis limits to ensure text labels are fully visible
    # Calculate the maximum y position of text labels
    max_text_y = max(
        [
            max(abs_recovered_vals[i], abs_true_vals[i]) * 1.5
            for i in range(len(param_names))
        ]
    )
    # Add extra space for the text box height (approximately 2x for the text box)
    current_ylim = ax1.get_ylim()
    ax1.set_ylim(current_ylim[0], max_text_y * 2.0)

    # More informative title with excluded params info
    title_str = f'Parameter Recovery: {test_name} (SNR={snr})'
    # Always show format: X passed / Y failed / Z excluded
    if n_excluded > 0:
        subtitle_str = f'{n_pass} passed / {n_fail} failed / {n_excluded} excluded ({", ".join(exclude_params)})'
    else:
        subtitle_str = f'{n_pass} passed / {n_fail} failed'

    # Add TEST PASSED indicator if no non-excluded failures
    if n_fail == 0:
        test_status = '✓ TEST PASSED'
        status_color = 'green'
    else:
        test_status = '✗ TEST FAILED'
        status_color = 'red'

    ax1.set_title(
        f'{title_str}\n{subtitle_str}\n{test_status}',
        fontsize=14,
        pad=10,
        color=status_color,
        weight='bold',
    )

    # Add legend with X markers for excluded params
    legend_elements_top = [
        plt.Line2D(
            [0],
            [0],
            marker='o',
            color='w',
            markerfacecolor='green',
            markersize=10,
            alpha=0.7,
            markeredgecolor='black',
            label='Passed (circle)',
        ),
        plt.Line2D(
            [0],
            [0],
            marker='o',
            color='w',
            markerfacecolor='red',
            markersize=10,
            alpha=0.7,
            markeredgecolor='black',
            label='Failed (circle)',
        ),
    ]
    if n_excluded > 0:
        legend_elements_top.append(
            plt.Line2D(
                [0],
                [0],
                marker='x',
                color='green',
                markersize=10,
                linewidth=2.5,
                label='Excluded (X)',
            )
        )
    legend_elements_top.append(
        plt.Line2D(
            [0],
            [0],
            marker='_',
            color='black',
            markersize=15,
            linewidth=3,
            label='True value',
        )
    )
    ax1.legend(handles=legend_elements_top, loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle=':', linewidth=0.5, axis='y')

    # Panel 2: Fractional error bar chart
    x_pos_bar = np.arange(len(param_names))

    # Compute fractional errors: (recovered - true) / true
    frac_errors = []
    for p in param_names:
        true_val = all_recovery_stats[p]['true']
        if true_val != 0:
            frac_errors.append(
                (all_recovery_stats[p]['recovered'] - true_val) / true_val * 100
            )
        else:
            frac_errors.append(0)
    frac_errors = np.array(frac_errors)

    # Use hatching for excluded parameters
    for i, param_name in enumerate(param_names):
        edgecolor = 'black'
        hatch = '///' if param_name in exclude_params else None
        ax2.bar(
            x_pos_bar[i],
            frac_errors[i],
            color=colors_bar[i],
            alpha=0.7,
            edgecolor=edgecolor,
            linewidth=2.5 if param_name in exclude_params else 1.5,
            hatch=hatch,
        )

    # Add tolerance threshold lines (only for relative tolerance)
    has_rel_tol = False
    for i, param_name in enumerate(param_names):
        criterion = all_recovery_stats[param_name]['criterion']

        if criterion == 'relative':
            rel_tol = all_recovery_stats[param_name]['rel_tolerance'] * 100
            ax2.hlines(
                [rel_tol, -rel_tol],
                i - 0.4,
                i + 0.4,
                colors='black',
                linestyles=':',
                linewidths=2,
                zorder=4,
            )
            has_rel_tol = True

    # Add legend
    legend_elements = [
        plt.Rectangle(
            (0, 0), 1, 1, fc='green', alpha=0.7, edgecolor='black', label='Passed'
        ),
        plt.Rectangle(
            (0, 0), 1, 1, fc='red', alpha=0.7, edgecolor='black', label='Failed'
        ),
    ]
    if n_excluded > 0:
        legend_elements.append(
            plt.Rectangle(
                (0, 0),
                1,
                1,
                fc='green',
                alpha=0.7,
                edgecolor='black',
                hatch='///',
                linewidth=2.5,
                label='Excluded (hatched)',
            )
        )
    if has_rel_tol:
        legend_elements.append(
            plt.Line2D(
                [0],
                [0],
                color='black',
                linestyle=':',
                linewidth=2,
                label='Relative tolerance',
            )
        )
    ax2.legend(handles=legend_elements, loc='best', fontsize=10, ncol=2)

    ax2.set_xticks(x_pos_bar)
    ax2.set_xticklabels(param_names, rotation=45, ha='right', fontsize=10)
    ax2.set_ylabel('Fractional Error (%)', fontsize=13, fontweight='bold')
    ax2.set_title('Parameter Recovery Errors\n(Recovered - True) / True', fontsize=12)
    ax2.axhline(0, color='black', linestyle='-', linewidth=1.5, zorder=2)
    ax2.grid(True, alpha=0.3, axis='y', linestyle=':', linewidth=0.5)

    plt.tight_layout()

    # Save
    outfile = test_dir / f"{test_name}_parameter_comparison.png"
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_combined_data_comparison(
    data_vel_noisy: jnp.ndarray,
    data_vel_true: jnp.ndarray,
    model_vel: jnp.ndarray,
    data_int_noisy: jnp.ndarray,
    data_int_true: jnp.ndarray,
    model_int: jnp.ndarray,
    test_name: str,
    config: TestConfig,
    variance_vel: Optional[float] = None,
    variance_int: Optional[float] = None,
    n_params: Optional[int] = None,
    model_label: str = 'MAP Model',
) -> None:
    """
    Create combined 4x3 panel diagnostic plot for velocity + intensity.

    .. deprecated::
        Use kl_pipe.diagnostics.plot_combined_data_comparison instead.
        This function is kept for backward compatibility.

    Stacks velocity (top 2 rows) and intensity (bottom 2 rows) comparisons
    into a single output figure.

    Layout:
        Row 0: Velocity - noisy | true | noisy - true
        Row 1: Velocity - model - true | model | noisy - model
        Row 2: Intensity - noisy | true | noisy - true
        Row 3: Intensity - model - true | model | noisy - model

    Parameters
    ----------
    data_vel_noisy, data_vel_true, model_vel : jnp.ndarray
        Velocity data arrays.
    data_int_noisy, data_int_true, model_int : jnp.ndarray
        Intensity data arrays.
    test_name : str
        Name of test (for title and filename).
    config : TestConfig
        Test configuration.
    variance_vel, variance_int : float, optional
        Variances for chi-squared computation.
    n_params : int, optional
        Number of fitted parameters (for reduced chi-squared).
    model_label : str, optional
        Label for model panels. Default is 'MAP Model'.
    """
    warnings.warn(
        "plot_combined_data_comparison in test_utils.py is deprecated. "
        "Use kl_pipe.diagnostics.plot_combined_data_comparison instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    if not config.enable_plots:
        return

    # Create output directory for this test
    test_dir = config.output_dir / test_name
    test_dir.mkdir(parents=True, exist_ok=True)

    # Set up figure with 4 rows x 3 cols
    fig, axes = plt.subplots(4, 3, figsize=(15, 18))

    # Helper function to plot a 2x3 block
    def plot_data_block(
        axes_block,
        data_noisy,
        data_true,
        model_eval,
        variance,
        data_type,
    ):
        # Compute residuals & chi2
        residual_true = np.array(data_noisy - data_true)
        residual_model = np.array(data_noisy - model_eval)
        residual_model_true = np.array(model_eval - data_true)

        chi2_true = None
        chi2_model = None
        if variance is not None:
            chi2_true = np.sum(residual_true**2 / variance)
            chi2_model = np.sum(residual_model**2 / variance)
            if n_params is not None:
                dof = data_noisy.size - n_params
                chi2_true /= dof
                chi2_model /= dof

        # Common colorbar limits for data
        data_arrays = [data_noisy, data_true, model_eval]
        vmin_data = min(np.percentile(arr, 1) for arr in data_arrays)
        vmax_data = max(np.percentile(arr, 99) for arr in data_arrays)
        norm_data = MidpointNormalize(vmin=vmin_data, vmax=vmax_data, midpoint=0)

        # Common colorbar limits for residuals
        residual_arrays = [residual_true, residual_model]
        abs_max = max(
            np.abs(np.percentile(arr, [1, 99])).max() for arr in residual_arrays
        )
        norm_resid = MidpointNormalize(vmin=-abs_max, vmax=abs_max, midpoint=0)

        # Row 0: noisy | true | noisy - true
        im00 = axes_block[0, 0].imshow(
            np.array(data_noisy), origin='lower', cmap='RdBu_r', norm=norm_data
        )
        axes_block[0, 0].set_title(f'{data_type}: Noisy Data')
        divider = make_axes_locatable(axes_block[0, 0])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im00, cax=cax)

        im01 = axes_block[0, 1].imshow(
            np.array(data_true), origin='lower', cmap='RdBu_r', norm=norm_data
        )
        axes_block[0, 1].set_title(f'{data_type}: True')
        divider = make_axes_locatable(axes_block[0, 1])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im01, cax=cax)

        im02 = axes_block[0, 2].imshow(
            residual_true, origin='lower', cmap='RdBu_r', norm=norm_resid
        )
        axes_block[0, 2].set_title(f'{data_type}: Noisy - True')
        divider = make_axes_locatable(axes_block[0, 2])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im02, cax=cax)
        if variance is not None:
            axes_block[0, 2].text(
                0.02,
                0.98,
                f'χ²={chi2_true:.1f}',
                transform=axes_block[0, 2].transAxes,
                fontsize=9,
                color='white',
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.5),
            )

        # Row 1: model - true | model | noisy - model
        im10 = axes_block[1, 0].imshow(
            residual_model_true, origin='lower', cmap='RdBu_r', norm=norm_resid
        )
        axes_block[1, 0].set_title(f'{data_type}: {model_label} - True')
        divider = make_axes_locatable(axes_block[1, 0])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im10, cax=cax)

        im11 = axes_block[1, 1].imshow(
            np.array(model_eval), origin='lower', cmap='RdBu_r', norm=norm_data
        )
        axes_block[1, 1].set_title(f'{data_type}: {model_label}')
        divider = make_axes_locatable(axes_block[1, 1])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im11, cax=cax)

        im12 = axes_block[1, 2].imshow(
            residual_model, origin='lower', cmap='RdBu_r', norm=norm_resid
        )
        axes_block[1, 2].set_title(f'{data_type}: Noisy - {model_label}')
        divider = make_axes_locatable(axes_block[1, 2])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im12, cax=cax)
        if variance is not None:
            axes_block[1, 2].text(
                0.02,
                0.98,
                f'χ²={chi2_model:.1f}',
                transform=axes_block[1, 2].transAxes,
                fontsize=9,
                color='white',
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.5),
            )

        # Labels
        for ax in axes_block.flat:
            ax.set_xlabel('x (pixels)')
            ax.set_ylabel('y (pixels)')

    # Plot velocity block (rows 0-1)
    plot_data_block(
        axes[0:2, :], data_vel_noisy, data_vel_true, model_vel, variance_vel, 'Velocity'
    )

    # Plot intensity block (rows 2-3)
    plot_data_block(
        axes[2:4, :],
        data_int_noisy,
        data_int_true,
        model_int,
        variance_int,
        'Intensity',
    )

    # Overall title
    fig.suptitle(f'{test_name} - Combined Data Comparison', fontsize=14, y=1.01)
    plt.tight_layout()

    # Save
    outfile = test_dir / f"{test_name}_combined_panels.png"
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_likelihood_slices(
    slices: Dict[str, Tuple[jnp.ndarray, jnp.ndarray]],
    true_pars: Dict[str, float],
    test_name: str,
    config: TestConfig,
    snr: float,
    data_type: str,
    has_psf: bool = False,
    model_kind: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Plot likelihood slices for all parameters.

    Parameters
    ----------
    slices : dict
        Dictionary mapping parameter names to (values, log_probs).
    true_pars : dict
        True parameter values.
    test_name : str
        Name of test.
    config : TestConfig
        Test configuration.
    snr : float
        Signal-to-noise ratio used.
    data_type : str
        'velocity', 'intensity', or 'joint'. Used to get tolerances.

    Returns
    -------
    recovery_stats : dict
        Statistics for each parameter: true, recovered, error, rel_error, passed.
    """

    # Always compute recovery stats
    recovery_stats = {}
    for param_name, (param_values, log_probs) in slices.items():
        true_val = true_pars[param_name]

        recovered_val = _quadratic_peak_interp(param_values, log_probs)
        true_value = true_pars[param_name]

        # Get tolerance (both relative and absolute)
        tolerance = config.get_tolerance(
            snr,
            param_name,
            true_value,
            data_type,
            test_type='likelihood_slice',
            has_psf=has_psf,
            model_kind=model_kind,
        )

        # Check recovery
        passed, stats = check_parameter_recovery(
            recovered_val, true_val, tolerance, param_name
        )

        recovery_stats[param_name] = stats

    if not config.enable_plots:
        return recovery_stats

    # create output directory
    test_dir = config.output_dir / test_name
    test_dir.mkdir(parents=True, exist_ok=True)

    # determine grid layout
    n_params = len(slices)
    ncols = 3
    nrows = int(np.ceil(n_params / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows))
    axes = np.atleast_2d(axes).flatten()

    for idx, (param_name, (param_values, log_probs)) in enumerate(slices.items()):
        ax = axes[idx]
        stats = recovery_stats[param_name]

        # compute acceptance bounds (+/- tolerance around true value)
        true_val = stats['true']
        rel_tolerance = stats['rel_tolerance']
        abs_tolerance = stats['abs_tolerance']
        if true_val != 0:
            lower_bound = true_val * (1 - rel_tolerance)
            upper_bound = true_val * (1 + rel_tolerance)
        else:
            # For parameters where true value is 0, use absolute tolerance
            lower_bound = -abs_tolerance
            upper_bound = abs_tolerance

        # plot likelihood slice
        ax.plot(param_values, log_probs, 'b-', linewidth=2)
        ax.axvline(stats['true'], color='k', linestyle='--', linewidth=2, label='True')
        ax.axvline(
            stats['recovered'],
            color='r',
            linestyle=':',
            linewidth=2,
            label=f'Peak: {stats["recovered"]:.4f} ({stats["rel_error"]*100:.3f}%)',
        )

        # styling
        ax.set_xlabel(param_name)
        ax.set_ylabel('Log-Likelihood')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        if stats['criterion'] == 'relative':
            status = f"PASS (rel: {stats['rel_error']*100:.2f}% < {stats['rel_tolerance']*100:.1f}%)"
        elif stats['criterion'] == 'absolute':
            status = (
                f"PASS (abs: {stats['abs_error']:.4f} < {stats['abs_tolerance']:.4f})"
            )
        else:
            status = f"FAIL (rel: {stats['rel_error']*100:.2f}%, abs: {stats['abs_error']:.4f})"

        # add grey acceptance region
        ax.axvspan(
            lower_bound,
            upper_bound,
            alpha=0.15,  # Low opacity
            color='grey',
            label=f'±{rel_tolerance*100:.1f}% tolerance',
            zorder=1,
        )

        # color title based on pass/fail
        title_color = 'green' if stats['passed'] else 'red'
        ax.set_title(f'{param_name} - {status}', color=title_color)

    # hide unused subplots
    for idx in range(n_params, len(axes)):
        axes[idx].axis('off')

    # overall title
    if data_type == 'velocity':
        base_tolerance = config.likelihood_slice_tolerance_velocity[snr]
    else:
        base_tolerance = config.likelihood_slice_tolerance_intensity[snr]
    fig.suptitle(
        f'{test_name} - Likelihood Slices '
        f'(SNR={snr}, base tolerance={base_tolerance*100:.1f}%)',
        fontsize=14,
        y=1.01,
    )
    plt.tight_layout()

    # save figure
    outfile = test_dir / f"{test_name}_likelihood_slices.png"
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close(fig)

    return recovery_stats


# ---------------------------------------------------------------------------
# Unit tests for _quadratic_peak_interp
# ---------------------------------------------------------------------------


class TestQuadraticPeakInterp:
    """Tests for sub-grid parabolic peak interpolation."""

    def test_exact_parabola(self):
        """Exact parabola sampled on integer grid recovers true peak."""
        import numpy as np

        true_peak = 3.7
        x = np.arange(7, dtype=float)
        y = -((x - true_peak) ** 2)
        result = _quadratic_peak_interp(x, y)
        assert abs(result - true_peak) < 1e-12, f"got {result}, expected {true_peak}"

    def test_boundary_argmax_left(self):
        """Peak at idx=0 returns discrete argmax."""
        import numpy as np

        x = np.array([0.0, 1.0, 2.0, 3.0])
        y = np.array([10.0, 5.0, 2.0, 1.0])
        result = _quadratic_peak_interp(x, y)
        assert result == 0.0

    def test_boundary_argmax_right(self):
        """Peak at last idx returns discrete argmax."""
        import numpy as np

        x = np.array([0.0, 1.0, 2.0, 3.0])
        y = np.array([1.0, 2.0, 5.0, 10.0])
        result = _quadratic_peak_interp(x, y)
        assert result == 3.0

    def test_flat_log_probs(self):
        """All equal values returns discrete argmax without crashing."""
        import numpy as np

        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.full(5, -100.0)
        result = _quadratic_peak_interp(x, y)
        assert result in x

    def test_asymmetric_peak(self):
        """Asymmetric peak returns value between neighbors of argmax."""
        import numpy as np

        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        # steeper on left than right — peak should shift right of idx=2
        y = np.array([-10.0, -2.0, 0.0, -0.5, -5.0])
        result = _quadratic_peak_interp(x, y)
        assert 1.0 <= result <= 3.0, f"result {result} outside neighbor range"
        # peak should be right of center since right side is shallower
        assert result > 2.0


# ==============================================================================
# Coordinate Convention Guardrail Tests
# ==============================================================================


class TestGridConvention:
    """Tests asserting X=horizontal=cols, Y=vertical=rows."""

    def test_grid_convention_standard(self):
        """X=horizontal=cols, Y=vertical=rows on non-square image."""
        ip = ImagePars(shape=(60, 100), pixel_scale=1.0, indexing='ij')
        from kl_pipe.utils import build_map_grid_from_image_pars

        X, Y = build_map_grid_from_image_pars(ip)
        assert X.shape == (60, 100)
        # X varies along cols (axis 1), constant along rows (axis 0)
        assert jnp.allclose(X[0, :], X[1, :])
        assert not jnp.allclose(X[:, 0], X[:, 1])
        # Y varies along rows (axis 0), constant along cols (axis 1)
        assert jnp.allclose(Y[:, 0], Y[:, 1])
        assert not jnp.allclose(Y[0, :], Y[1, :])
        # Nx = horizontal = Ncol
        assert len(jnp.unique(X[0, :])) == ip.Nx == 100
        assert len(jnp.unique(Y[:, 0])) == ip.Ny == 60

    def test_grid_convention_non_centered(self):
        """Non-centered grid: X pixel indices 0..Ncol-1, Y 0..Nrow-1."""
        ip = ImagePars(shape=(60, 100), pixel_scale=1.0, indexing='ij')
        from kl_pipe.utils import build_map_grid_from_image_pars

        X, Y = build_map_grid_from_image_pars(ip, unit='pixel', centered=False)
        assert float(X[0, 0]) == 0.0
        assert float(X[0, 99]) == 99.0
        assert float(Y[0, 0]) == 0.0
        assert float(Y[59, 0]) == 59.0

    def test_image_pars_convention(self):
        """Nx=Ncol, Ny=Nrow for both indexing modes."""
        ip_ij = ImagePars(shape=(60, 100), pixel_scale=1.0, indexing='ij')
        assert ip_ij.Ny == ip_ij.Nrow == 60
        assert ip_ij.Nx == ip_ij.Ncol == 100
        ip_xy = ImagePars(shape=(100, 60), pixel_scale=1.0, indexing='xy')
        assert ip_xy.Ny == ip_xy.Nrow == 60
        assert ip_xy.Nx == ip_xy.Ncol == 100

    def test_wcs_pixel_world_roundtrip(self):
        """Center pixel maps to (0, 0) world coords."""
        ip = ImagePars(shape=(60, 100), pixel_scale=0.3, indexing='ij')
        wcs = ip.wcs
        # crpix is 1-indexed (FITS); pixel_to_world_values is 0-indexed
        cx, cy = wcs.wcs.crpix - 1
        ra, dec = wcs.pixel_to_world_values(cx, cy)
        assert abs(float(ra)) < 1e-10
        assert abs(float(dec)) < 1e-10
