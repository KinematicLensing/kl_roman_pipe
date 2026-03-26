"""
Pytest configuration and shared fixtures for kl_pipe tests.

This module provides:
- Warning suppression for expected test warnings
- Shared test configuration fixtures
"""

import jax

jax.config.update("jax_enable_x64", True)

import pytest
import warnings

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


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
