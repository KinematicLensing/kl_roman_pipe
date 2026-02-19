"""
TNG PSF integration tests.

Tests that PSF convolution works correctly with TNG mock data.
Uses SubhaloID 8 (simplest galaxy).

Requires TNG50 data from CyVerse.
"""

import pytest
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

import galsim as gs

from kl_pipe.parameters import ImagePars
from kl_pipe.utils import get_test_dir

OUTPUT_DIR = get_test_dir() / "out" / "psf_tng"


@pytest.fixture(scope="module")
def output_dir():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


@pytest.fixture(scope="module")
def tng_generator():
    """Load TNG generator for SubhaloID 8."""
    from kl_pipe.tng import TNG50MockData, TNGDataVectorGenerator

    tng_data = TNG50MockData()
    galaxy = tng_data.get_galaxy(subhalo_id=8)
    return TNGDataVectorGenerator(galaxy)


@pytest.fixture(scope="module")
def tng_config():
    """Render config for TNG tests."""
    from kl_pipe.tng.data_vectors import TNGRenderConfig

    image_pars = ImagePars(shape=(64, 64), pixel_scale=0.11, indexing='xy')
    return TNGRenderConfig(
        image_pars=image_pars,
        target_redshift=0.5,
        use_native_orientation=True,
    )


@pytest.fixture(scope="module")
def tng_psf():
    return gs.Gaussian(fwhm=0.625)


# ==============================================================================
# Tests
# ==============================================================================


@pytest.mark.tng50
def test_tng_intensity_with_psf(tng_generator, tng_config, tng_psf, output_dir):
    """
    Generate intensity with and without PSF.
    Verify PSF version smoother, flux preserved.
    """
    from kl_pipe.tng.data_vectors import TNGRenderConfig
    from dataclasses import replace

    # without PSF
    intensity_no_psf, _ = tng_generator.generate_intensity_map(tng_config)

    # with PSF
    config_psf = replace(tng_config, psf=tng_psf)
    intensity_psf, _ = tng_generator.generate_intensity_map(config_psf)

    # flux conservation
    flux_no_psf = np.sum(intensity_no_psf)
    flux_psf = np.sum(intensity_psf)
    rel_flux_diff = abs(flux_psf - flux_no_psf) / abs(flux_no_psf)
    assert rel_flux_diff < 0.01, f"Flux not conserved: {rel_flux_diff:.2%} difference"

    # PSF version should have lower peak (smoother)
    assert np.max(intensity_psf) < np.max(
        intensity_no_psf
    ), "PSF-convolved intensity should have lower peak"

    # diagnostic plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    im0 = axes[0].imshow(intensity_no_psf, origin='lower', cmap='viridis')
    axes[0].set_title('No PSF')
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(intensity_psf, origin='lower', cmap='viridis')
    axes[1].set_title('With PSF')
    plt.colorbar(im1, ax=axes[1])

    diff = intensity_psf - intensity_no_psf
    im2 = axes[2].imshow(diff, origin='lower', cmap='RdBu_r')
    axes[2].set_title('Difference')
    plt.colorbar(im2, ax=axes[2])

    fig.suptitle(f'TNG SubhaloID 8 Intensity (flux diff: {rel_flux_diff:.2e})')
    plt.tight_layout()
    plt.savefig(output_dir / 'tng_intensity_psf_comparison.png', dpi=150)
    plt.close()


@pytest.mark.tng50
def test_tng_velocity_with_psf(tng_generator, tng_config, tng_psf, output_dir):
    """
    Generate velocity with and without PSF.
    Verify PSF version has reduced velocity range.
    """
    from dataclasses import replace

    # without PSF
    velocity_no_psf, _ = tng_generator.generate_velocity_map(tng_config)

    # with PSF -- need intensity for flux weighting
    config_psf = replace(tng_config, psf=tng_psf)
    intensity_psf, _ = tng_generator.generate_intensity_map(config_psf)
    velocity_psf, _ = tng_generator.generate_velocity_map(
        config_psf, intensity_map=intensity_psf
    )

    # PSF should reduce velocity extremes (smoother)
    range_no_psf = np.ptp(velocity_no_psf)
    range_psf = np.ptp(velocity_psf)
    assert (
        range_psf < range_no_psf
    ), f"PSF should reduce velocity range: {range_no_psf:.1f} -> {range_psf:.1f}"

    # diagnostic plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    vmin = min(np.percentile(velocity_no_psf, 2), np.percentile(velocity_psf, 2))
    vmax = max(np.percentile(velocity_no_psf, 98), np.percentile(velocity_psf, 98))

    im0 = axes[0].imshow(
        velocity_no_psf, origin='lower', cmap='RdBu_r', vmin=vmin, vmax=vmax
    )
    axes[0].set_title(f'No PSF (range={range_no_psf:.0f})')
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(
        velocity_psf, origin='lower', cmap='RdBu_r', vmin=vmin, vmax=vmax
    )
    axes[1].set_title(f'With PSF (range={range_psf:.0f})')
    plt.colorbar(im1, ax=axes[1])

    diff = velocity_psf - velocity_no_psf
    im2 = axes[2].imshow(diff, origin='lower', cmap='RdBu_r')
    axes[2].set_title('Difference')
    plt.colorbar(im2, ax=axes[2])

    fig.suptitle('TNG SubhaloID 8 Velocity')
    plt.tight_layout()
    plt.savefig(output_dir / 'tng_velocity_psf_comparison.png', dpi=150)
    plt.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "-m", "tng50"])
