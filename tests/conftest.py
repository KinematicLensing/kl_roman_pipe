"""
Pytest configuration and shared fixtures for kl_pipe tests.

This module provides:
- Warning suppression for expected test warnings
- Shared test configuration fixtures
"""

import jax

from kl_pipe._precision import ensure_precision

# precision comes from the central kl_pipe configuration (float64 default,
# KLPIPE_FP32=1 opts the whole test run into float32). Never force x64 here:
# a hard-coded override silently turns intended-fp32 runs into fp64 ones.
ensure_precision()

import pytest
import warnings


def pytest_report_header(config):
    """Make the active float precision visible in every pytest log."""
    import jax.numpy as jnp

    dtype = jnp.asarray(1.0).dtype
    return (
        f"kl_pipe precision: {dtype} (jax_enable_x64={jax.config.jax_enable_x64}, "
        f"matmul_precision={jax.config.jax_default_matmul_precision})"
    )


from galsim.errors import GalSimFFTSizeWarning


# ==============================================================================
# Warning Suppression Fixtures
# ==============================================================================


@pytest.fixture(autouse=True)
def suppress_expected_warnings():
    """
    Suppress expected warnings during tests.

    These warnings are suppressed in tests only - they will still appear
    in production runs. Diagnostic plots will annotate when these warnings
    are triggered.

    Suppressed warnings:
    - emcee autocorrelation warning (chain too short)
    - JAXopt deprecation warning (blackjax dependency)
    - GalSim FFT size warning (expected for large PSF convolutions)
    - matplotlib tight_layout warning (non-compatible axes)
    """
    with warnings.catch_warnings():
        # emcee chain length warning - expected for short test runs
        warnings.filterwarnings(
            "ignore",
            message="The chain is shorter than",
            category=UserWarning,
        )

        # JAXopt deprecation - external dependency, not actionable
        warnings.filterwarnings(
            "ignore",
            message="JAXopt is no longer maintained",
            category=DeprecationWarning,
        )

        # GalSim FFT size warning - expected for large convolution stamps
        warnings.filterwarnings(
            "ignore",
            category=GalSimFFTSizeWarning,
        )

        # matplotlib tight_layout with non-compatible axes
        warnings.filterwarnings(
            "ignore",
            message="This figure includes Axes that are not compatible with tight_layout",
            category=UserWarning,
        )

        yield


# ==============================================================================
# Slow Test Marker
# ==============================================================================


def pytest_addoption(parser):
    """Register custom CLI options."""
    parser.addoption(
        "--flagship-long",
        action="store_true",
        default=False,
        help="run the flagship test with the longer production sampler config "
        "(more samples/chains/tree-depth) for cleaner posteriors",
    )
    parser.addoption(
        "--flagship-production",
        action="store_true",
        default=False,
        help="run the flagship test with the production observing config "
        "(2 broadband F087+F158 + 4 grism rolls) instead of the dev 1x1 "
        "config; orthogonal to --flagship-long (which sets sampler depth)",
    )


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers",
        "grism_validation: cross-code grism validation (requires reference data)",
    )
