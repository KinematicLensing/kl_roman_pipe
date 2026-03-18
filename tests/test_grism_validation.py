"""
Cross-code grism validation test suite.

Compares kl_pipe datacube/grism outputs against external codes (geko, kl-tools,
grizli) using reference .npz files in tests/data/validation/<code>/.

Test matrix: 35 tests across 7 test groups.
See scripts/validation/test_params.yaml for full definitions.

Test groups:
  1. Static Galaxy (4): static_base, static_psf, static_compact, static_extended
  2. Rotating Galaxy (6): rotating_base, rotating_edgeon, rotating_edgeon_ortho_pa,
     rotating_psf, rotating_compact, rotating_extended
  3. Inclination Sweep (5): incl_sweep_cosi01..09
  4. PA Sweep (7): pa_sweep_0..180
  5. Dispersion Angle Sweep (7): dispangle_sweep_0..180
  6. Spectral Properties (2): narrow_lines, broad_lines
  7. Redshift Sweep (4): redshift_sweep_05..15

Tags: static, rotating, sweep, psf

Marker: @pytest.mark.grism_validation
Requires: reference data in tests/data/validation/ (download via
  `make download-validation-data` or render via `make render-validation-*`)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

# add scripts/validation to path for utils
_SCRIPTS_DIR = Path(__file__).parent.parent / 'scripts' / 'validation'
sys.path.insert(0, str(_SCRIPTS_DIR))

from utils import (
    compare_datacubes,
    compare_images,
    format_comparison_report,
    load_config,
    load_reference_data,
    load_tolerances,
)

OUT_DIR = os.path.join(os.path.dirname(__file__), 'out', 'grism-validation')
os.makedirs(OUT_DIR, exist_ok=True)

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data', 'validation')

# reference code to compare against; change to 'kl_tools' or 'grizli' as available
REFERENCE_CODE = 'geko'

# tolerance tier: 'primary' for geko/kl-tools, 'secondary' for grizli
TOLERANCE_TIER = 'primary'


def _have_reference(test_name: str, code: str = REFERENCE_CODE) -> bool:
    """Check if reference data exists for this test."""
    p = Path(DATA_DIR) / code / f'{test_name}.npz'
    return p.exists()


def _have_kl_pipe(test_name: str) -> bool:
    """Check if kl_pipe rendered data exists."""
    p = Path(DATA_DIR) / 'kl_pipe' / f'{test_name}.npz'
    return p.exists()


def _skip_if_missing(test_name: str, code: str = REFERENCE_CODE):
    """Skip test if either kl_pipe or reference data is missing."""
    if not _have_kl_pipe(test_name):
        pytest.skip(
            f"kl_pipe data missing for '{test_name}'. "
            "Run: make render-validation-kl-pipe"
        )
    if not _have_reference(test_name, code):
        pytest.skip(
            f"{code} data missing for '{test_name}'. "
            f"Run: make render-validation-{code}"
        )


def _run_comparison(
    test_name: str, code: str = REFERENCE_CODE, tier: str = TOLERANCE_TIER
):
    """Run full comparison for a test. Returns (grism_results, cube_results)."""
    _skip_if_missing(test_name, code)

    kl_data = load_reference_data(test_name, 'kl_pipe', data_dir=DATA_DIR)
    ref_data = load_reference_data(test_name, code, data_dir=DATA_DIR)

    tolerances = load_tolerances(tier)

    # grism comparison
    grism_results = None
    if kl_data['grism'] is not None and ref_data['grism'] is not None:
        grism_results = compare_images(kl_data['grism'], ref_data['grism'], tolerances)

    # datacube comparison
    cube_results = None
    if kl_data['cube'] is not None and ref_data['cube'] is not None:
        cube_results = compare_datacubes(
            kl_data['cube'],
            ref_data['cube'],
            kl_data['lambda_grid'],
            tolerances,
        )

    return grism_results, cube_results


def _assert_results(test_name, results, data_type):
    """Assert all metrics pass. Print report on failure."""
    if results is None:
        pytest.skip(f"No {data_type} data for '{test_name}'")

    failures = {k: v for k, v in results.items() if v[1] is False}
    if failures:
        report = format_comparison_report(test_name, results)
        pytest.fail(f"{data_type} comparison failed for '{test_name}':\n{report}")


def _save_diagnostic_plot(test_name, kl_data, ref_data, code=REFERENCE_CODE):
    """Save side-by-side comparison plot."""
    try:
        import matplotlib

        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'{test_name}: kl_pipe vs {code}', fontsize=14)

    # grism row
    if kl_data['grism'] is not None and ref_data['grism'] is not None:
        vmax = max(np.max(kl_data['grism']), np.max(ref_data['grism']))
        axes[0, 0].imshow(kl_data['grism'], origin='lower', vmin=0, vmax=vmax)
        axes[0, 0].set_title('kl_pipe grism')
        axes[0, 1].imshow(ref_data['grism'], origin='lower', vmin=0, vmax=vmax)
        axes[0, 1].set_title(f'{code} grism')
        resid = kl_data['grism'] - ref_data['grism']
        im = axes[0, 2].imshow(resid, origin='lower', cmap='RdBu_r')
        axes[0, 2].set_title('residual')
        plt.colorbar(im, ax=axes[0, 2])

    # datacube collapsed row
    if kl_data['cube'] is not None and ref_data['cube'] is not None:
        img_kl = np.sum(kl_data['cube'], axis=-1)
        img_ref = np.sum(ref_data['cube'], axis=-1)
        vmax = max(np.max(img_kl), np.max(img_ref))
        axes[1, 0].imshow(img_kl, origin='lower', vmin=0, vmax=vmax)
        axes[1, 0].set_title('kl_pipe cube (collapsed)')
        axes[1, 1].imshow(img_ref, origin='lower', vmin=0, vmax=vmax)
        axes[1, 1].set_title(f'{code} cube (collapsed)')
        resid = img_kl - img_ref
        im = axes[1, 2].imshow(resid, origin='lower', cmap='RdBu_r')
        axes[1, 2].set_title('residual')
        plt.colorbar(im, ax=axes[1, 2])

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f'{test_name}.png'), dpi=150)
    plt.close(fig)


# =============================================================================
# Test group 1: Static Galaxy
# =============================================================================

_STATIC_TESTS = [
    'static_base',
    'static_psf',
    'static_compact',
    'static_extended',
]


@pytest.mark.grism_validation
class TestStaticGalaxy:
    """Non-rotating baselines. Datacube factorizes as I(x,y) * S(lambda)."""

    @pytest.mark.parametrize('test_name', _STATIC_TESTS)
    def test_grism(self, test_name):
        grism_results, _ = _run_comparison(test_name)
        _assert_results(test_name, grism_results, 'grism')

    @pytest.mark.parametrize('test_name', _STATIC_TESTS)
    def test_datacube(self, test_name):
        _, cube_results = _run_comparison(test_name)
        _assert_results(test_name, cube_results, 'datacube')


# =============================================================================
# Test group 2: Rotating Galaxy
# =============================================================================

_ROTATING_TESTS = [
    'rotating_base',
    'rotating_edgeon',
    'rotating_edgeon_ortho_pa',
    'rotating_psf',
    'rotating_compact',
    'rotating_extended',
]


@pytest.mark.grism_validation
class TestRotatingGalaxy:
    """Rotating galaxies with Doppler-shifted line centers."""

    @pytest.mark.parametrize('test_name', _ROTATING_TESTS)
    def test_grism(self, test_name):
        grism_results, _ = _run_comparison(test_name)
        _assert_results(test_name, grism_results, 'grism')

    @pytest.mark.parametrize('test_name', _ROTATING_TESTS)
    def test_datacube(self, test_name):
        _, cube_results = _run_comparison(test_name)
        _assert_results(test_name, cube_results, 'datacube')


# =============================================================================
# Test group 3: Inclination Sweep
# =============================================================================

_INCL_TESTS = [
    'incl_sweep_cosi01',
    'incl_sweep_cosi03',
    'incl_sweep_cosi05',
    'incl_sweep_cosi07',
    'incl_sweep_cosi09',
]


@pytest.mark.grism_validation
class TestInclinationSweep:
    """Inclination sweep from near edge-on (0.1) to near face-on (0.9)."""

    @pytest.mark.parametrize('test_name', _INCL_TESTS)
    def test_grism(self, test_name):
        grism_results, _ = _run_comparison(test_name)
        _assert_results(test_name, grism_results, 'grism')

    @pytest.mark.parametrize('test_name', _INCL_TESTS)
    def test_datacube(self, test_name):
        _, cube_results = _run_comparison(test_name)
        _assert_results(test_name, cube_results, 'datacube')


# =============================================================================
# Test group 4: PA Sweep
# =============================================================================

_PA_TESTS = [
    'pa_sweep_0',
    'pa_sweep_30',
    'pa_sweep_45',
    'pa_sweep_90',
    'pa_sweep_120',
    'pa_sweep_150',
    'pa_sweep_180',
]


@pytest.mark.grism_validation
class TestPASweep:
    """Position angle sweep 0 -> pi relative to dispersion direction."""

    @pytest.mark.parametrize('test_name', _PA_TESTS)
    def test_grism(self, test_name):
        grism_results, _ = _run_comparison(test_name)
        _assert_results(test_name, grism_results, 'grism')

    @pytest.mark.parametrize('test_name', _PA_TESTS)
    def test_datacube(self, test_name):
        _, cube_results = _run_comparison(test_name)
        _assert_results(test_name, cube_results, 'datacube')


# =============================================================================
# Test group 5: Dispersion Angle Sweep
# =============================================================================

_DISPANGLE_TESTS = [
    'dispangle_sweep_0',
    'dispangle_sweep_30',
    'dispangle_sweep_45',
    'dispangle_sweep_90',
    'dispangle_sweep_120',
    'dispangle_sweep_150',
    'dispangle_sweep_180',
]


@pytest.mark.grism_validation
class TestDispAngleSweep:
    """Dispersion angle sweep 0 -> pi. Tests cos/sin decomposition."""

    @pytest.mark.parametrize('test_name', _DISPANGLE_TESTS)
    def test_grism(self, test_name):
        grism_results, _ = _run_comparison(test_name)
        _assert_results(test_name, grism_results, 'grism')

    @pytest.mark.parametrize('test_name', _DISPANGLE_TESTS)
    def test_datacube(self, test_name):
        _, cube_results = _run_comparison(test_name)
        _assert_results(test_name, cube_results, 'datacube')


# =============================================================================
# Test group 6: Spectral Properties
# =============================================================================

_SPECTRAL_TESTS = [
    'narrow_lines',
    'broad_lines',
]


@pytest.mark.grism_validation
class TestSpectralProperties:
    """Velocity dispersion extremes: narrow (30 km/s) and broad (100 km/s)."""

    @pytest.mark.parametrize('test_name', _SPECTRAL_TESTS)
    def test_grism(self, test_name):
        grism_results, _ = _run_comparison(test_name)
        _assert_results(test_name, grism_results, 'grism')

    @pytest.mark.parametrize('test_name', _SPECTRAL_TESTS)
    def test_datacube(self, test_name):
        _, cube_results = _run_comparison(test_name)
        _assert_results(test_name, cube_results, 'datacube')


# =============================================================================
# Test group 7: Redshift Sweep
# =============================================================================

_REDSHIFT_TESTS = [
    'redshift_sweep_05',
    'redshift_sweep_08',
    'redshift_sweep_10',
    'redshift_sweep_15',
]


@pytest.mark.grism_validation
class TestRedshiftSweep:
    """Redshift sweep z=0.5..1.5. Changes R(lambda) and wavelength grid."""

    @pytest.mark.parametrize('test_name', _REDSHIFT_TESTS)
    def test_grism(self, test_name):
        grism_results, _ = _run_comparison(test_name)
        _assert_results(test_name, grism_results, 'grism')

    @pytest.mark.parametrize('test_name', _REDSHIFT_TESTS)
    def test_datacube(self, test_name):
        _, cube_results = _run_comparison(test_name)
        _assert_results(test_name, cube_results, 'datacube')


# =============================================================================
# Diagnostic plots (not pass/fail)
# =============================================================================

_ALL_TESTS = (
    _STATIC_TESTS
    + _ROTATING_TESTS
    + _INCL_TESTS
    + _PA_TESTS
    + _DISPANGLE_TESTS
    + _SPECTRAL_TESTS
    + _REDSHIFT_TESTS
)


@pytest.mark.grism_validation
class TestDiagnosticPlots:
    """Generate side-by-side comparison plots. Not pass/fail."""

    @pytest.mark.parametrize('test_name', _ALL_TESTS)
    def test_plot_comparison(self, test_name):
        _skip_if_missing(test_name)
        kl_data = load_reference_data(test_name, 'kl_pipe', data_dir=DATA_DIR)
        ref_data = load_reference_data(test_name, REFERENCE_CODE, data_dir=DATA_DIR)
        _save_diagnostic_plot(test_name, kl_data, ref_data)
