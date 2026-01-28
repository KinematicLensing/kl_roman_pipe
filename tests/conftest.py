"""
Pytest configuration and shared fixtures for kl_pipe tests.

This module provides:
- Warning suppression for sampler-related warnings (test-only)
- Shared test configuration fixtures
"""

import pytest
import warnings


# ==============================================================================
# Warning Suppression Fixtures
# ==============================================================================


@pytest.fixture(autouse=True)
def suppress_sampler_warnings():
    """
    Suppress expected warnings during tests.

    These warnings are suppressed in tests only - they will still appear
    in production runs. Diagnostic plots will annotate when these warnings
    are triggered.

    Suppressed warnings:
    - emcee autocorrelation warning (chain too short)
    - JAXopt deprecation warning (blackjax dependency)
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

        yield


# ==============================================================================
# Slow Test Marker
# ==============================================================================


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
