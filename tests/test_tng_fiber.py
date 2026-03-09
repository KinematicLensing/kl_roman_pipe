"""Tests for fiber-based spectral simulation from rendered maps."""

import numpy as np

from kl_pipe.parameters import ImagePars
from kl_pipe.tng.fiber import (
    EmissionConfig,
    FiberObservationConfig,
    FiberPlacement,
    FiberSpectraSimulator,
)


def _build_test_maps(shape=(64, 64), pixel_scale=0.1):
    image_pars = ImagePars(shape=shape, pixel_scale=pixel_scale, indexing='ij')

    # Simple centrally concentrated flux map.
    nrow, ncol = shape
    y, x = np.indices((nrow, ncol), dtype=np.float64)
    xc = (ncol - 1) / 2.0
    yc = (nrow - 1) / 2.0
    rr2 = (x - xc) ** 2 + (y - yc) ** 2
    intensity = np.exp(-rr2 / (2.0 * (0.2 * nrow) ** 2))

    # Left-to-right velocity gradient in km/s.
    vx = (x - xc) / max(xc, 1.0)
    velocity = 200.0 * vx

    return image_pars, intensity, velocity


def _base_configs():
    obs = FiberObservationConfig(
        wave_min=6550.0,
        wave_max=6575.0,
        n_wave=512,
        exposure_time=1000.0,
        spectral_fwhm=0.0,
        aperture_subsampling=5,
    )
    emission = EmissionConfig(
        rest_wavelength=6563.0,
        emission_flux=1.0,
        intrinsic_sigma_kms=20.0,
    )
    return obs, emission


def test_fiber_output_shapes_and_finite_values():
    image_pars, intensity, velocity = _build_test_maps()
    obs, emission = _base_configs()

    fibers = [
        FiberPlacement(0.0, 0.0, 0.4, name='center'),
        FiberPlacement(1.0, 0.5, 0.3, name='offcenter'),
    ]

    sim = FiberSpectraSimulator(image_pars)
    result = sim.simulate_from_maps(intensity, velocity, fibers, obs, emission)

    assert result.wavelengths.shape == (obs.n_wave,)
    assert result.spectra.shape == (len(fibers), obs.n_wave)
    assert result.fiber_masks.shape == (len(fibers),) + image_pars.shape
    assert np.all(np.isfinite(result.spectra))
    assert np.all(result.spectra >= 0)


def test_larger_fiber_collects_more_flux():
    image_pars, intensity, velocity = _build_test_maps()
    obs, emission = _base_configs()

    sim = FiberSpectraSimulator(image_pars)

    small = [FiberPlacement(0.0, 0.0, 0.2, name='small')]
    large = [FiberPlacement(0.0, 0.0, 0.8, name='large')]

    out_small = sim.simulate_from_maps(intensity, velocity, small, obs, emission)
    out_large = sim.simulate_from_maps(intensity, velocity, large, obs, emission)

    flux_small = np.trapz(out_small.spectra[0], out_small.wavelengths)
    flux_large = np.trapz(out_large.spectra[0], out_large.wavelengths)

    assert flux_large > flux_small


def test_velocity_gradient_shifts_line_centroid_between_fibers():
    image_pars, intensity, velocity = _build_test_maps()
    obs, emission = _base_configs()

    fibers = [
        FiberPlacement(-1.5, 0.0, 0.35, name='left'),
        FiberPlacement(1.5, 0.0, 0.35, name='right'),
    ]

    sim = FiberSpectraSimulator(image_pars)
    result = sim.simulate_from_maps(intensity, velocity, fibers, obs, emission)

    left_spec = result.spectra[0]
    right_spec = result.spectra[1]

    left_centroid = np.trapz(
        result.wavelengths * left_spec, result.wavelengths
    ) / np.trapz(left_spec, result.wavelengths)
    right_centroid = np.trapz(
        result.wavelengths * right_spec, result.wavelengths
    ) / np.trapz(right_spec, result.wavelengths)

    assert right_centroid > left_centroid


def test_spectral_resolution_broadening_reduces_peak():
    image_pars, intensity, velocity = _build_test_maps()
    obs, emission = _base_configs()

    fibers = [FiberPlacement(0.0, 0.0, 0.5, name='center')]
    sim = FiberSpectraSimulator(image_pars)

    result_sharp = sim.simulate_from_maps(intensity, velocity, fibers, obs, emission)

    obs_blur = FiberObservationConfig(
        wave_min=obs.wave_min,
        wave_max=obs.wave_max,
        n_wave=obs.n_wave,
        exposure_time=obs.exposure_time,
        spectral_fwhm=1.2,
        aperture_subsampling=obs.aperture_subsampling,
    )
    result_blur = sim.simulate_from_maps(
        intensity, velocity, fibers, obs_blur, emission
    )

    sharp = result_sharp.spectra[0]
    blur = result_blur.spectra[0]

    assert blur.max() < sharp.max()

    flux_sharp = np.trapz(sharp, result_sharp.wavelengths)
    flux_blur = np.trapz(blur, result_blur.wavelengths)

    # Kernel is normalized; allow a small numerical tolerance.
    assert np.isclose(flux_blur, flux_sharp, rtol=2e-2)
