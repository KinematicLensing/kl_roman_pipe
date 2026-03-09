"""
Fiber-based spectral simulation from transformed TNG image products.

This module converts 2D intensity/velocity maps into a set of 1D spectra for
user-defined circular fiber placements.

Design goals:
- Reuse existing TNG transformation/rendering pipeline in ``data_vectors.py``.
- Keep API small and explicit for observation and emission configuration.
- Provide deterministic, testable behavior with optional Gaussian noise.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np

from ..parameters import ImagePars

# Speed of light in km/s
_C_KMS = 299792.458


@dataclass(frozen=True)
class FiberPlacement:
    """A single circular fiber aperture on the image plane.

    Parameters
    ----------
    x_arcsec, y_arcsec : float
        Fiber center in arcsec in the same coordinate frame as the rendered maps.
    radius_arcsec : float
        Fiber radius in arcsec.
    name : str, optional
        Optional label for downstream identification.
    """

    x_arcsec: float
    y_arcsec: float
    radius_arcsec: float
    name: Optional[str] = None


@dataclass(frozen=True)
class FiberObservationConfig:
    """Observation setup for generating 1D spectra.

    Parameters
    ----------
    wave_min, wave_max : float
        Wavelength range in Angstrom.
    n_wave : int
        Number of wavelength samples (uniform grid).
    exposure_time : float
        Exposure time in seconds.
    system_throughput : float | str | np.ndarray, default=1.0
        Throughput model. Supported forms:
        - scalar float in [0, 1]
        - path to two-column ``.dat`` file: wavelength[Angstrom], throughput
        - array-like of length ``n_wave`` with throughput on the simulation grid
    spectral_fwhm : float, default=0.0
        Instrumental Gaussian spectral FWHM in Angstrom.
    systemic_redshift : float, default=0.0
        Optional systemic redshift applied to the emission rest wavelength.
    spectral_dispersion_per_arcsec : float, default=0.0
        Optional linear dispersion term (Angstrom / arcsec) projected along
        ``dispersion_angle``. This can model simple instrument-dependent
        wavelength shifts across a fiber footprint.
    dispersion_angle : float, default=0.0
        Angle in radians for the linear dispersion direction in the image plane.
    noise_sigma : float, default=0.0
        Optional additive Gaussian noise sigma in the same units as output flux.
    rng_seed : int, optional
        Seed for deterministic noise draws.
    aperture_subsampling : int, default=5
        Sub-pixel sampling factor per image pixel for aperture overlap fraction.
    """

    wave_min: float
    wave_max: float
    n_wave: int
    exposure_time: float
    system_throughput: Union[float, str, np.ndarray] = 1.0
    spectral_fwhm: float = 0.0
    systemic_redshift: float = 0.0
    spectral_dispersion_per_arcsec: float = 0.0
    dispersion_angle: float = 0.0
    noise_sigma: float = 0.0
    rng_seed: Optional[int] = None
    aperture_subsampling: int = 5


@dataclass(frozen=True)
class EmissionConfig:
    """Emission-line model configuration.

    Parameters
    ----------
    rest_wavelength : float
        Rest-frame line center in Angstrom.
    emission_flux : float
        Total integrated line flux per second attributed to the full source.
        Fiber spectra capture a fraction of this flux according to aperture
        overlap and intensity weighting.
    intrinsic_sigma_kms : float, default=20.0
        Intrinsic Gaussian velocity dispersion of the line in km/s.
    continuum_type : str, default='none'
        Continuum model mode:
        - 'none': no continuum
        - 'func': evaluate ``continuum_func`` with variable ``wave`` in Angstrom
        - 'temp': interpolate from ``continuum_template_path``
    continuum_func : str, optional
        Expression string for continuum shape in flambda-like units,
        e.g. ``"1.0 - 2e-4*(wave-6500)"``.
    continuum_template_path : str, optional
        Path to two-column template file (wavelength, flux density).
    continuum_template_wave_scale : float, default=1.0
        Multiplicative factor to convert template wavelength column to Angstrom.
        Example: use 10.0 for template wavelengths in nm.
    obs_cont_norm_wave : float, default=6563.0
        Observer-frame wavelength in Angstrom at which the continuum is normalized.
    obs_cont_norm_flam : float, default=0.0
        Continuum flux density normalization at ``obs_cont_norm_wave`` in
        arbitrary flambda-like units per second.
    """

    rest_wavelength: float
    emission_flux: float
    intrinsic_sigma_kms: float = 20.0
    continuum_type: str = "none"
    continuum_func: Optional[str] = None
    continuum_template_path: Optional[str] = None
    continuum_template_wave_scale: float = 1.0
    obs_cont_norm_wave: float = 6563.0
    obs_cont_norm_flam: float = 0.0


@dataclass(frozen=True)
class FiberSimulationResult:
    """Output container for fiber spectral simulations."""

    wavelengths: np.ndarray
    spectra: np.ndarray
    fiber_names: List[str]
    fiber_masks: np.ndarray


class FiberSpectraSimulator:
    """Simulate 1D spectra from transformed image-plane maps."""

    def __init__(self, image_pars: ImagePars):
        self.image_pars = image_pars
        self._throughput_cache: Dict[str, np.ndarray] = {}
        self._continuum_cache: Dict[str, np.ndarray] = {}

    def simulate_from_maps(
        self,
        intensity_map: np.ndarray,
        velocity_map: np.ndarray,
        fibers: Sequence[FiberPlacement],
        obs_config: FiberObservationConfig,
        emission_config: EmissionConfig,
    ) -> FiberSimulationResult:
        """Generate one spectrum per fiber from intensity and LOS velocity maps.

        Notes
        -----
        - ``intensity_map`` defines relative flux weighting of pixels.
        - ``velocity_map`` (km/s) Doppler-shifts the emission line per pixel.
        - The final spectra are integrated line profiles on a shared wavelength grid.
        """

        intensity = np.asarray(intensity_map, dtype=np.float64)
        velocity = np.asarray(velocity_map, dtype=np.float64)

        if intensity.shape != self.image_pars.shape:
            raise ValueError(
                f"intensity_map shape {intensity.shape} != ImagePars shape {self.image_pars.shape}"
            )
        if velocity.shape != self.image_pars.shape:
            raise ValueError(
                f"velocity_map shape {velocity.shape} != ImagePars shape {self.image_pars.shape}"
            )
        if len(fibers) == 0:
            raise ValueError("At least one fiber placement is required")

        wave = np.linspace(obs_config.wave_min, obs_config.wave_max, obs_config.n_wave)
        throughput_curve = self._resolve_throughput_curve(
            obs_config.system_throughput, wave
        )
        dw = wave[1] - wave[0] if wave.size > 1 else 1.0
        spectra = np.zeros((len(fibers), obs_config.n_wave), dtype=np.float64)

        X, Y = self._build_arcsec_grid()

        # Physically meaningful weights are non-negative. Negative pixels can arise
        # after noise/PSF and would break flux partitioning.
        flux_weights = np.clip(intensity, 0.0, None)
        total_weight = flux_weights.sum()
        if total_weight <= 0:
            masks = np.zeros((len(fibers),) + intensity.shape, dtype=np.float64)
            return FiberSimulationResult(
                wavelengths=wave,
                spectra=spectra,
                fiber_names=[f.name or f"fiber_{i}" for i, f in enumerate(fibers)],
                fiber_masks=masks,
            )

        masks = []
        for i, fiber in enumerate(fibers):
            mask = self._build_circular_overlap_mask(
                X,
                Y,
                fiber.x_arcsec,
                fiber.y_arcsec,
                fiber.radius_arcsec,
                obs_config.aperture_subsampling,
            )
            masks.append(mask)

            aperture_weights = flux_weights * mask
            aperture_sum = aperture_weights.sum()
            if aperture_sum <= 0:
                continue

            # Split the configured total source line flux by flux-weighted aperture share.
            fiber_flux_total = (
                emission_config.emission_flux
                * obs_config.exposure_time
                * (aperture_sum / total_weight)
            )
            pixel_flux = fiber_flux_total * (aperture_weights / aperture_sum)

            spec = self._render_line_spectrum(
                wave=wave,
                pixel_flux=pixel_flux,
                velocity_map=velocity,
                x_map=X,
                y_map=Y,
                fiber=fiber,
                obs_config=obs_config,
                emission_config=emission_config,
            )

            continuum = self._render_continuum_spectrum(
                wave=wave,
                aperture_fraction=(aperture_sum / total_weight),
                obs_config=obs_config,
                emission_config=emission_config,
                dw=dw,
            )

            spec = (spec + continuum) * throughput_curve

            if obs_config.spectral_fwhm > 0:
                spec = self._apply_spectral_resolution(
                    spec, wave, obs_config.spectral_fwhm
                )

            spectra[i] = spec

        if obs_config.noise_sigma > 0:
            rng = np.random.default_rng(obs_config.rng_seed)
            spectra = spectra + rng.normal(
                loc=0.0, scale=obs_config.noise_sigma, size=spectra.shape
            )

        return FiberSimulationResult(
            wavelengths=wave,
            spectra=spectra,
            fiber_names=[f.name or f"fiber_{i}" for i, f in enumerate(fibers)],
            fiber_masks=np.asarray(masks, dtype=np.float64),
        )

    def _build_arcsec_grid(self) -> tuple[np.ndarray, np.ndarray]:
        """Build centered coordinate grids in arcsec without JAX dependencies."""
        nrow, ncol = self.image_pars.Nrow, self.image_pars.Ncol
        pixel_scale = float(self.image_pars.pixel_scale)

        x_coords = (np.arange(ncol, dtype=np.float64) - (ncol - 1) / 2.0) * pixel_scale
        y_coords = (np.arange(nrow, dtype=np.float64) - (nrow - 1) / 2.0) * pixel_scale

        grids = np.meshgrid(x_coords, y_coords, indexing="xy")
        return grids[0], grids[1]

    def _resolve_throughput_curve(
        self,
        system_throughput: Union[float, str, np.ndarray],
        wave: np.ndarray,
    ) -> np.ndarray:
        """Resolve throughput into an array sampled on ``wave``."""
        n_wave = wave.size

        if isinstance(system_throughput, (int, float)):
            return np.full(n_wave, float(system_throughput), dtype=np.float64)

        if isinstance(system_throughput, str):
            path = str(Path(system_throughput).expanduser().resolve())
            if path not in self._throughput_cache:
                data = np.loadtxt(path)
                if data.ndim != 2 or data.shape[1] < 2:
                    raise ValueError(
                        "system_throughput .dat file must have at least two columns: wave, throughput"
                    )
                self._throughput_cache[path] = data[:, :2]

            table = self._throughput_cache[path]
            tcurve = np.interp(wave, table[:, 0], table[:, 1], left=0.0, right=0.0)
            return np.clip(tcurve, 0.0, None)

        arr = np.asarray(system_throughput, dtype=np.float64)
        if arr.shape != (n_wave,):
            raise ValueError(
                f"Array throughput must have shape ({n_wave},), got {arr.shape}"
            )
        return np.clip(arr, 0.0, None)

    def simulate_from_generator(
        self,
        generator,
        render_config,
        fibers: Sequence[FiberPlacement],
        obs_config: FiberObservationConfig,
        emission_config: EmissionConfig,
        *,
        intensity_snr: Optional[float] = None,
        velocity_snr: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> FiberSimulationResult:
        """Convenience wrapper using ``TNGDataVectorGenerator`` outputs.

        Parameters
        ----------
        generator : TNGDataVectorGenerator
            Existing generator instance from ``kl_pipe.tng.data_vectors``.
        render_config : TNGRenderConfig
            Existing rendering config for transformations/projection.
        """

        intensity_map, _ = generator.generate_intensity_map(
            render_config, snr=intensity_snr, seed=seed
        )
        velocity_map, _ = generator.generate_velocity_map(
            render_config, snr=velocity_snr, seed=seed, intensity_map=intensity_map
        )

        return self.simulate_from_maps(
            intensity_map=intensity_map,
            velocity_map=velocity_map,
            fibers=fibers,
            obs_config=obs_config,
            emission_config=emission_config,
        )

    @staticmethod
    def _build_circular_overlap_mask(
        X: np.ndarray,
        Y: np.ndarray,
        x0: float,
        y0: float,
        radius: float,
        subsampling: int,
    ) -> np.ndarray:
        """Approximate circular aperture overlap fraction with sub-pixel sampling."""
        if subsampling < 1:
            raise ValueError("aperture_subsampling must be >= 1")
        if radius <= 0:
            raise ValueError("Fiber radius must be > 0")

        if subsampling == 1:
            r2 = (X - x0) ** 2 + (Y - y0) ** 2
            return (r2 <= radius**2).astype(np.float64)

        # Reconstruct pixel size from adjacent centers on either axis.
        if X.shape[1] > 1:
            dx = abs(X[0, 1] - X[0, 0])
        elif Y.shape[0] > 1:
            dx = abs(Y[1, 0] - Y[0, 0])
        else:
            dx = 1.0

        offsets = (np.arange(subsampling) + 0.5) / subsampling - 0.5
        offsets = offsets * dx

        mask = np.zeros_like(X, dtype=np.float64)
        for oy in offsets:
            for ox in offsets:
                r2 = (X + ox - x0) ** 2 + (Y + oy - y0) ** 2
                mask += (r2 <= radius**2).astype(np.float64)

        return mask / float(subsampling**2)

    @staticmethod
    def _render_line_spectrum(
        wave: np.ndarray,
        pixel_flux: np.ndarray,
        velocity_map: np.ndarray,
        x_map: np.ndarray,
        y_map: np.ndarray,
        fiber: FiberPlacement,
        obs_config: FiberObservationConfig,
        emission_config: EmissionConfig,
    ) -> np.ndarray:
        """Render integrated Gaussian line profile from pixelized map inputs."""
        spec = np.zeros_like(wave)

        valid = pixel_flux > 0
        if not np.any(valid):
            return spec

        flux_vals = pixel_flux[valid]
        v_vals = velocity_map[valid]
        x_vals = x_map[valid]
        y_vals = y_map[valid]

        lambda_rest = emission_config.rest_wavelength
        lambda_sys = lambda_rest * (1.0 + obs_config.systemic_redshift)

        # Optional linear instrument dispersion across fiber footprint.
        if obs_config.spectral_dispersion_per_arcsec != 0.0:
            ux = np.cos(obs_config.dispersion_angle)
            uy = np.sin(obs_config.dispersion_angle)
            proj = (x_vals - fiber.x_arcsec) * ux + (y_vals - fiber.y_arcsec) * uy
            delta_lambda_instr = obs_config.spectral_dispersion_per_arcsec * proj
        else:
            delta_lambda_instr = 0.0

        # Doppler shift from LOS velocity.
        lambda_center = lambda_sys * (1.0 + v_vals / _C_KMS) + delta_lambda_instr

        sigma_lambda = np.maximum(
            lambda_sys * emission_config.intrinsic_sigma_kms / _C_KMS,
            1e-6,
        )

        # Vectorized Gaussian evaluation: (Npix, Nwave)
        arg = (wave[None, :] - lambda_center[:, None]) / sigma_lambda
        profiles = np.exp(-0.5 * arg**2) / (sigma_lambda * np.sqrt(2.0 * np.pi))

        spec = np.sum(flux_vals[:, None] * profiles, axis=0)

        return spec

    def _render_continuum_spectrum(
        self,
        wave: np.ndarray,
        aperture_fraction: float,
        obs_config: FiberObservationConfig,
        emission_config: EmissionConfig,
        dw: float,
    ) -> np.ndarray:
        """Render continuum contribution using ObsFrameSED-like options."""
        if emission_config.continuum_type.lower() == "none":
            return np.zeros_like(wave)

        if emission_config.obs_cont_norm_flam <= 0:
            return np.zeros_like(wave)

        shape = self._evaluate_continuum_shape(wave, obs_config, emission_config)
        norm_wave = float(emission_config.obs_cont_norm_wave)
        shape_at_norm = np.interp(norm_wave, wave, shape, left=np.nan, right=np.nan)
        if not np.isfinite(shape_at_norm) or shape_at_norm == 0:
            return np.zeros_like(wave)

        shape = shape / shape_at_norm
        flux_density = emission_config.obs_cont_norm_flam * shape

        # Convert flux density to binned flux by multiplying by delta-lambda.
        continuum = (
            aperture_fraction
            * obs_config.exposure_time
            * np.clip(flux_density, 0.0, None)
            * dw
        )
        return continuum

    def _evaluate_continuum_shape(
        self,
        wave: np.ndarray,
        obs_config: FiberObservationConfig,
        emission_config: EmissionConfig,
    ) -> np.ndarray:
        """Evaluate raw continuum shape on observer-frame wavelength grid."""
        ctype = emission_config.continuum_type.lower()
        z = float(obs_config.systemic_redshift)

        if ctype == "func":
            expr = emission_config.continuum_func
            if expr is None:
                raise ValueError(
                    "continuum_func must be provided when continuum_type='func'"
                )
            local = {"wave": wave, "np": np}
            values = eval(expr, {"__builtins__": {}}, local)
            arr = np.asarray(values, dtype=np.float64)
            if arr.ndim == 0:
                return np.full_like(wave, float(arr), dtype=np.float64)
            if arr.shape != wave.shape:
                raise ValueError(
                    f"continuum_func must evaluate to scalar or shape {wave.shape}, got {arr.shape}"
                )
            return arr

        if ctype == "temp":
            template_path = emission_config.continuum_template_path
            if template_path is None:
                raise ValueError(
                    "continuum_template_path must be provided when continuum_type='temp'"
                )
            path = str(Path(template_path).expanduser().resolve())
            if path not in self._continuum_cache:
                data = np.loadtxt(path)
                if data.ndim != 2 or data.shape[1] < 2:
                    raise ValueError(
                        "continuum template file must have at least two columns: wave, flux"
                    )
                self._continuum_cache[path] = data[:, :2]

            table = self._continuum_cache[path]
            wave_rest = table[:, 0] * float(
                emission_config.continuum_template_wave_scale
            )
            wave_obs = wave_rest * (1.0 + z)
            flux = table[:, 1]
            return np.interp(wave, wave_obs, flux, left=0.0, right=0.0)

        raise ValueError(
            f"Unsupported continuum_type '{emission_config.continuum_type}'"
        )

    @staticmethod
    def _apply_spectral_resolution(
        spectrum: np.ndarray,
        wave: np.ndarray,
        fwhm: float,
    ) -> np.ndarray:
        """Convolve spectrum with a Gaussian LSF kernel in wavelength space."""
        if fwhm <= 0:
            return spectrum
        if wave.size < 2:
            return spectrum.copy()

        dw = float(wave[1] - wave[0])
        sigma_pix = (fwhm / 2.354820045) / max(dw, 1e-12)
        if sigma_pix <= 0:
            return spectrum.copy()

        half_width = int(max(3, np.ceil(4.0 * sigma_pix)))
        idx = np.arange(-half_width, half_width + 1, dtype=np.float64)
        kernel = np.exp(-0.5 * (idx / sigma_pix) ** 2)
        kernel /= kernel.sum()

        return np.convolve(spectrum, kernel, mode="same")
