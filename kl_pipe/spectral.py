"""
Spectral model for 3D datacube assembly.

Builds C(x, y, lambda) from velocity + intensity + emission lines.
Datacube is first-class; grism dispersion is one projection mode.

Key classes:
- LineSpec: rest wavelength + naming for a single emission line
- EmissionLine: line spec + which IntensityModel params it overrides
- SpectralConfig: lines, LSF mode, resolving power, spectral oversample
- CubePars: spatial grid (ImagePars) + wavelength array
- SpectralModel: assembles intrinsic 3D datacube (no PSF)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Tuple, FrozenSet, Optional, Callable
import os

import jax
import jax.numpy as jnp
import numpy as np
import galsim as gs

if TYPE_CHECKING:
    from kl_pipe.model import IntensityModel, VelocityModel

from kl_pipe.parameters import ImagePars
from kl_pipe.utils import build_map_grid_from_image_pars

# speed of light in km/s
C_KMS = 299792.458


def roman_grism_R(lambda_nm: float) -> float:
    """Roman grism resolving power: R = 461 * lambda_um."""
    return 461.0 * lambda_nm / 1000.0


def desi_b_R(lambda_nm: float) -> float:
    """DESI B-channel resolving power: see Eq. 5.1 in https://arxiv.org/pdf/1611.00037."""
    D = 4096 * 1.5e-5  # Detector width (4096 pixels) * pixel size (15 microns)
    d = 1.07e-4  # fiber width
    m = 0.42  # magnification
    lambda_range = (593 - 360) * 1e-9  # wavelength range in meters (360-593 nm)
    return 2 * D * lambda_nm * 1e-9 / (np.sqrt(3) * d * m * lambda_range)


def desi_r_R(lambda_nm: float) -> float:
    """DESI R-channel resolving power: see Eq. 5.1 in https://arxiv.org/pdf/1611.00037."""
    D = 4096 * 1.5e-5  # Detector width (4096 pixels) * pixel size (15 microns)
    d = 1.07e-4  # fiber width
    m = 0.46  # magnification
    lambda_range = (772 - 566) * 1e-9  # wavelength range in meters (566-772 nm)
    return 2 * D * lambda_nm * 1e-9 / (np.sqrt(3) * d * m * lambda_range)


def desi_z_R(lambda_nm: float) -> float:
    """DESI Z-channel resolving power: see Eq. 5.1 in https://arxiv.org/pdf/1611.00037."""
    D = 4096 * 1.5e-5  # Detector width (4096 pixels) * pixel size (15 microns)
    d = 1.07e-4  # fiber width
    m = 0.5  # magnification
    lambda_range = (980 - 747) * 1e-9  # wavelength range in meters (747-980 nm)
    return 2 * D * lambda_nm * 1e-9 / (np.sqrt(3) * d * m * lambda_range)


# =============================================================================
# Emission line specification
# =============================================================================


@dataclass(frozen=True)
class LineSpec:
    """Identifies a single emission line by rest wavelength and naming."""

    lambda_rest: float  # rest-frame wavelength in nm
    name: str  # human-readable, e.g. 'Halpha'
    param_prefix: str  # prefix for PARAMETER_NAMES, e.g. 'Ha'


@dataclass(frozen=True)
class EmissionLine:
    """One emission line in the spectral model.

    own_params: which IntensityModel params this line overrides from broadband.
    {prefix}_cont is ALWAYS included (local continuum level near this line).
    """

    line_spec: LineSpec
    own_params: FrozenSet[str] = frozenset({'flux'})


# =============================================================================
# Configuration
# =============================================================================


@dataclass(frozen=True)
class SpectralConfig:
    """Defines the spectral model: which lines, LSF mode, resolution.

    Created once at model setup. Same across all observations.
    """

    lines: Tuple[EmissionLine, ...]
    lsf_mode: str = 'absorbed'
    R_func: Optional[Callable] = None  # R(lambda_nm) -> float; default: roman_grism_R
    spectral_oversample: int = 5

    def __post_init__(self):
        if not self.lines:
            raise ValueError("SpectralConfig requires at least one EmissionLine")
        if self.lsf_mode not in ('absorbed', 'convolution'):
            raise ValueError(
                f"lsf_mode must be 'absorbed' or 'convolution', got '{self.lsf_mode}'"
            )
        if self.spectral_oversample < 1:
            raise ValueError(
                f"spectral_oversample must be >= 1, got {self.spectral_oversample}"
            )

    @property
    def effective_R_func(self) -> Callable:
        return self.R_func if self.R_func is not None else roman_grism_R


@dataclass(frozen=True)
class CubePars:
    """Defines the numerical grid for a datacube: spatial pixels + wavelength array.

    image_pars is INDEPENDENT of imaging ImagePars — different instruments.
    """

    image_pars: ImagePars
    lambda_grid: jnp.ndarray  # wavelength array in nm, shape (Nlambda,)

    @classmethod
    def from_range(cls, image_pars, lambda_min, lambda_max, delta_lambda):
        """Create CubePars with uniform wavelength spacing."""
        n = int(np.round((lambda_max - lambda_min) / delta_lambda)) + 1
        grid = jnp.linspace(lambda_min, lambda_max, n)
        return cls(image_pars=image_pars, lambda_grid=grid)

    @classmethod
    def from_R(cls, image_pars, lambda_min, lambda_max, R):
        """Create CubePars with spacing matched to resolving power R.

        delta_lambda = lambda_center / R
        """
        lam_c = 0.5 * (lambda_min + lambda_max)
        dl = lam_c / R
        return cls.from_range(image_pars, lambda_min, lambda_max, dl)

    @property
    def n_lambda(self) -> int:
        return len(self.lambda_grid)

    @property
    def delta_lambda(self) -> float:
        if len(self.lambda_grid) < 2:
            raise ValueError("Need at least 2 wavelength points for delta_lambda")
        return float(self.lambda_grid[1] - self.lambda_grid[0])

    @property
    def spatial_shape(self) -> Tuple[int, int]:
        return (self.image_pars.Nrow, self.image_pars.Ncol)


@dataclass(frozen=True)
class FiberPars:
    cube_pars: CubePars
    obs_conf: dict  # meta_pars: MetaPars <-- could do this instead and get obs_conf from there

    @classmethod
    def from_cube_pars(cls, cube_pars, obs_conf):
        cls.is_dispersed = obs_conf['OBSTYPE'] == 1
        cls.pix_scale = cube_pars.image_pars.pixel_scale
        cls.spatial_shape = cube_pars.spatial_shape
        cls.X, cls.Y = build_map_grid_from_image_pars(
            cube_pars.image_pars
        )  # need to figure out the units for this, whether it can be off-center

        if cls.is_dispersed:  # fiber spectrum
            cls.lambda_grid = cube_pars.lambda_grid
            cls.n_lambda = len(cls.lambda_grid)

            # _bid = obs_conf['SEDBLKID'] #what is block ID?? it seems like it can alter the lambda range and resolution?
            ##_lrange = _pars['model_dimension']['lambda_range'][_bid]. #I need to make sure, is the model dimension what I'd expect in cube_pars?
            ##_dlam = _pars['model_dimension']['lambda_res']
            ##if isinstance(_dlam, list):
            ##_dlam = _dlam[_bid]

            cls.throughput = gs.Bandpass(
                obs_conf['BANDPASS'],
                'nm',
                blue_limit=cls.lambda_grid[0],
                red_limit=cls.lambda_grid[-1],
            )
            cls._bp_array = jnp.array(cls.throughput(cls.lambda_grid))
            cls.lambda_eff = cls.throughput.effective_wavelength

        else:  # photometry
            cls.lambda_grid = cube_pars.lambda_grid
            cls.n_lambda = len(cls.lambda_grid)

            _from_file_ = os.path.isfile(obs_conf['BANDPASS'])
            if _from_file_:
                cls.throughput = gs.Bandpass(obs_conf['BANDPASS'], 'nm')
            else:
                _lrange = [np.min(cls.lambda_grid), np.max(cls.lambda_grid)]
                cls.throughput = gs.Bandpass(
                    obs_conf['BANDPASS'],
                    'nm',
                    blue_limit=_lrange[0],
                    red_limit=_lrange[1],
                )
            cls._bp_array = jnp.array(cls.throughput(cls.lambda_grid))
            cls.lambda_eff = cls.throughput.effective_wavelength

            # this is how it's written in kl-tools but I think I need these even for photometry
            # cls.lambda_grid, cls.n_lambda, cls._bp_array = None, 1, None

        return cls(cube_pars=cube_pars, obs_conf=obs_conf)


# =============================================================================
# SpectralModel
# =============================================================================


class SpectralModel:
    """Assembles 3D datacube C(x,y,lambda) from velocity + intensity + emission lines.

    Does NOT know about the unified parameter space (that's KLModel's job).
    Takes pre-sliced component thetas: theta_spec, theta_vel, theta_int.
    Holds references to velocity_model and intensity_model.
    Returns INTRINSIC cube (no PSF). PSF applied by KLModel per-slice.
    """

    def __init__(
        self,
        spectral_config: SpectralConfig,
        intensity_model: 'IntensityModel',
        velocity_model: 'VelocityModel',
    ):
        self._config = spectral_config
        self.intensity_model = intensity_model
        self.velocity_model = velocity_model

        # validate own_params against IntensityModel parameter names
        valid_int_params = set(intensity_model.PARAMETER_NAMES)
        for line in spectral_config.lines:
            invalid = line.own_params - valid_int_params
            if invalid:
                raise ValueError(
                    f"EmissionLine '{line.line_spec.name}' has own_params {invalid} "
                    f"not in IntensityModel.PARAMETER_NAMES {intensity_model.PARAMETER_NAMES}"
                )

        # build PARAMETER_NAMES (component-level only)
        self._build_parameter_names()

        # pre-compute index arrays for _build_line_theta_int (JIT-safe)
        self._precompute_line_indices()

    def _build_parameter_names(self):
        """Build spectral component's PARAMETER_NAMES dynamically."""
        params = ['z', 'vel_dispersion']
        for line in self._config.lines:
            prefix = line.line_spec.param_prefix
            # own_params: sorted for determinism
            for p in sorted(line.own_params):
                params.append(f'{prefix}_{p}')
            # continuum always present
            params.append(f'{prefix}_cont')
        self.PARAMETER_NAMES = tuple(params)
        self._param_indices = {n: i for i, n in enumerate(self.PARAMETER_NAMES)}

    def _precompute_line_indices(self):
        """Pre-compute arrays for JIT-safe theta_int modification per line.

        For each line, stores:
        - _line_int_src_indices[i]: indices into theta_int to copy FROM
        - _line_int_dst_indices[i]: indices into theta_int to write TO
        - _line_spec_src_indices[i]: indices into theta_spec to read overrides FROM
        """
        int_names = list(self.intensity_model.PARAMETER_NAMES)
        int_idx = {n: i for i, n in enumerate(int_names)}

        self._line_int_override_dst = []
        self._line_spec_override_src = []

        for line in self._config.lines:
            prefix = line.line_spec.param_prefix
            dst_indices = []
            src_indices = []
            for p in sorted(line.own_params):
                if p in int_idx:
                    dst_indices.append(int_idx[p])
                    spec_name = f'{prefix}_{p}'
                    src_indices.append(self._param_indices[spec_name])
            self._line_int_override_dst.append(jnp.array(dst_indices, dtype=jnp.int32))
            self._line_spec_override_src.append(jnp.array(src_indices, dtype=jnp.int32))

    @property
    def config(self) -> SpectralConfig:
        return self._config

    def get_param(self, name: str, theta_spec: jnp.ndarray):
        return theta_spec[self._param_indices[name]]

    def build_cube(
        self,
        theta_spec: jnp.ndarray,
        theta_vel: jnp.ndarray,
        theta_int: jnp.ndarray,
        cube_pars: CubePars,
        plane: str = 'obs',
    ) -> jnp.ndarray:
        """Build intrinsic 3D datacube (no PSF).

        Returns shape (Nrow, Ncol, Nlambda).
        """
        z = self.get_param('z', theta_spec)
        vel_disp = self.get_param('vel_dispersion', theta_spec)

        # 1. velocity map in obs frame (includes v0)
        X, Y = build_map_grid_from_image_pars(cube_pars.image_pars)
        v_map = self.velocity_model(theta_vel, plane, X, Y)

        # subtract v0 for rotation-only Doppler
        v0 = self.velocity_model.get_param('v0', theta_vel)
        v_rotation = v_map - v0  # (Nrow, Ncol)

        # 2. broadband intensity (for continuum)
        I_broadband = self.intensity_model.render_unconvolved(
            theta_int, cube_pars.image_pars
        )

        # 3. build oversampled lambda grid
        lambda_coarse = cube_pars.lambda_grid
        n_lam = len(lambda_coarse)
        osf = self._config.spectral_oversample

        if osf > 1 and n_lam >= 2:
            dl = lambda_coarse[1] - lambda_coarse[0]
            # fine grid: each coarse bin split into `osf` sub-bins
            half = dl / 2.0
            fine_offsets = jnp.linspace(-half + half / osf, half - half / osf, osf)
            # (n_lam, osf)
            lambda_fine = lambda_coarse[:, None] + fine_offsets[None, :]
            lambda_fine = lambda_fine.reshape(-1)  # (n_lam * osf,)
        else:
            lambda_fine = lambda_coarse
            osf = 1  # force no binning if single wavelength

        n_fine = len(lambda_fine)

        # 4. accumulate cube on fine grid
        Nrow, Ncol = cube_pars.spatial_shape
        cube_fine = jnp.zeros((Nrow, Ncol, n_fine))

        R_func = self._config.effective_R_func

        for i, line in enumerate(self._config.lines):
            prefix = line.line_spec.param_prefix

            # per-line intensity
            theta_line = self._build_line_theta_int(theta_spec, theta_int, i)
            I_line = self.intensity_model.render_unconvolved(
                theta_line, cube_pars.image_pars
            )

            # per-line continuum
            cont = self.get_param(f'{prefix}_cont', theta_spec)

            # Doppler-shifted observed wavelength per pixel
            lam_rest = line.line_spec.lambda_rest
            lam_obs = lam_rest * (1.0 + z) * (1.0 + v_rotation / C_KMS)  # (Nrow, Ncol)

            # effective sigma: vel_disp + instrument
            R_at_line = R_func(lam_rest * (1.0 + z))
            sigma_inst_kms = C_KMS / (2.355 * R_at_line)
            sigma_eff_kms = jnp.sqrt(vel_disp**2 + sigma_inst_kms**2)

            # convert to wavelength units: sigma_lambda = lam_obs * sigma_eff / c
            sigma_lambda = lam_obs * sigma_eff_kms / C_KMS  # (Nrow, Ncol)

            # normalized Gaussian: 1/(sigma*sqrt(2pi)) * exp(-0.5*((lam - mu)/sigma)^2)
            # shape: (Nrow, Ncol, n_fine)
            dlam = lambda_fine[None, None, :] - lam_obs[:, :, None]
            sig = sigma_lambda[:, :, None]
            gauss = (1.0 / (sig * jnp.sqrt(2.0 * jnp.pi))) * jnp.exp(
                -0.5 * (dlam / sig) ** 2
            )

            # emission line contribution
            cube_fine = cube_fine + I_line[:, :, None] * gauss

            # continuum contribution (flat in wavelength near this line)
            cube_fine = cube_fine + I_broadband[:, :, None] * cont

        # 5. bin fine grid to coarse
        if osf > 1 and n_lam >= 2:
            cube = cube_fine.reshape(Nrow, Ncol, n_lam, osf).mean(axis=-1)
        else:
            cube = cube_fine

        # 6. optional spectral convolution for 'convolution' LSF mode
        if self._config.lsf_mode == 'convolution':
            raise NotImplementedError(
                "LSF convolution mode not yet implemented. Use lsf_mode='absorbed'."
            )

        return cube

    def _build_line_theta_int(
        self,
        theta_spec: jnp.ndarray,
        theta_int: jnp.ndarray,
        line_idx: int,
    ) -> jnp.ndarray:
        """Build modified theta_int with per-line param overrides.

        JIT-safe via pre-computed index arrays + .at[].set().
        """
        result = theta_int
        dst = self._line_int_override_dst[line_idx]
        src = self._line_spec_override_src[line_idx]

        if len(dst) > 0:
            override_values = theta_spec[src]
            result = result.at[dst].set(override_values)

        return result

    def convolve_spectral(
        self,
        cube: jnp.ndarray,
        lsf_sigma_pixels: float,
    ) -> jnp.ndarray:
        """1D Gaussian convolution along lambda axis (for 'convolution' LSF mode)."""
        # build 1D Gaussian kernel
        half_width = int(4 * lsf_sigma_pixels) + 1
        x = jnp.arange(-half_width, half_width + 1, dtype=jnp.float64)
        kernel = jnp.exp(-0.5 * (x / lsf_sigma_pixels) ** 2)
        kernel = kernel / kernel.sum()

        # convolve along last axis
        # use jnp.convolve per spatial pixel (vectorized via vmap)
        def convolve_1d(spectrum):
            return jnp.convolve(spectrum, kernel, mode='same')

        # vmap over (Nrow, Ncol) -> apply convolve_1d to each pixel's spectrum
        return jax.vmap(jax.vmap(convolve_1d))(cube)


# =============================================================================
# Factories
# =============================================================================

# common line specs
HALPHA = LineSpec(lambda_rest=656.28, name='Halpha', param_prefix='Ha')
HALPHA_VAC = LineSpec(lambda_rest=656.46, name='Halpha_vac', param_prefix='Ha')
NII_6548 = LineSpec(lambda_rest=654.80, name='NII_6548', param_prefix='NII_6548')
NII_6583 = LineSpec(lambda_rest=658.34, name='NII_6583', param_prefix='NII_6583')


def halpha_line(own_params=frozenset({'flux'})) -> EmissionLine:
    """H-alpha at 656.28 nm."""
    return EmissionLine(line_spec=HALPHA, own_params=own_params)


def halpha_vac_line(own_params=frozenset({'flux'})) -> EmissionLine:
    """H-alpha at 656.46 nm (vacuum wavelength)."""
    return EmissionLine(line_spec=HALPHA_VAC, own_params=own_params)


def halpha_nii_lines() -> Tuple[EmissionLine, ...]:
    """H-alpha + [NII] 6548, 6583 triplet."""
    return (
        EmissionLine(line_spec=HALPHA, own_params=frozenset({'flux'})),
        EmissionLine(line_spec=NII_6548, own_params=frozenset({'flux'})),
        EmissionLine(line_spec=NII_6583, own_params=frozenset({'flux'})),
    )


def make_spectral_config(
    lines=None,
    lsf_mode='absorbed',
    spectral_oversample=5,
    R_func=None,
) -> SpectralConfig:
    """Convenience factory. Defaults to H-alpha + NII."""
    if lines is None:
        lines = halpha_nii_lines()
    return SpectralConfig(
        lines=lines,
        lsf_mode=lsf_mode,
        R_func=R_func,
        spectral_oversample=spectral_oversample,
    )
