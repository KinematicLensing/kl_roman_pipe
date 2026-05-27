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
from typing import TYPE_CHECKING, Tuple, FrozenSet

import jax.numpy as jnp
import numpy as np

if TYPE_CHECKING:
    from kl_pipe.model import IntensityModel, VelocityModel

from kl_pipe.parameters import ImagePars
from kl_pipe.utils import build_map_grid_from_image_pars

# speed of light in km/s
C_KMS = 299792.458


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
class _LegacyEmissionLine:
    """One emission line in the SpectralModel cube assembly.

    own_params: which IntensityModel params this line overrides from broadband.
    {prefix}_cont is ALWAYS included (local continuum level near this line).

    Superseded by ``kl_pipe.source.EmissionLine``, which carries a different
    field shape (intensity / intensity_key, continuum / continuum_key,
    dispersion_key, lambda_rest). New code should use that class; this one
    is retained for the SpectralModel cube-assembly path.
    """

    line_spec: LineSpec
    own_params: FrozenSet[str] = frozenset({'flux'})


# Backward-compat alias: callers importing ``EmissionLine`` from this module
# resolve to ``_LegacyEmissionLine`` (the SpectralModel-flavored class). New
# code should import ``EmissionLine`` from ``kl_pipe.source`` instead.
EmissionLine = _LegacyEmissionLine


# =============================================================================
# Configuration
# =============================================================================


@dataclass(frozen=True)
class SpectralConfig:
    """Defines the spectral model: which lines, spectral oversample.

    Created once at model setup. Same across all observations.

    Line broadening is intrinsic only (sigma_eff = vel_disp). The instrumental
    LSF for slitless grism is produced by the PSF-per-slice + dispersion
    geometry downstream; no separate sigma_inst term.
    """

    lines: Tuple[EmissionLine, ...]
    spectral_oversample: int = 5

    def __post_init__(self):
        if not self.lines:
            raise ValueError("SpectralConfig requires at least one EmissionLine")
        if self.spectral_oversample < 1:
            raise ValueError(
                f"spectral_oversample must be >= 1, got {self.spectral_oversample}"
            )


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

            # intrinsic line broadening only; the slitless LSF is reproduced by
            # the PSF-per-slice + dispersion geometry downstream (see issue #40).
            sigma_lambda = lam_obs * vel_disp / C_KMS  # (Nrow, Ncol)

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


# =============================================================================
# Factories
# =============================================================================

# common line specs
HALPHA = LineSpec(lambda_rest=656.28, name='Halpha', param_prefix='Ha')
NII_6548 = LineSpec(lambda_rest=654.80, name='NII_6548', param_prefix='NII_6548')
NII_6583 = LineSpec(lambda_rest=658.34, name='NII_6583', param_prefix='NII_6583')


def halpha_line(own_params=frozenset({'flux'})) -> EmissionLine:
    """H-alpha at 656.28 nm."""
    return EmissionLine(line_spec=HALPHA, own_params=own_params)


def halpha_nii_lines() -> Tuple[EmissionLine, ...]:
    """H-alpha + [NII] 6548, 6583 triplet."""
    return (
        EmissionLine(line_spec=HALPHA, own_params=frozenset({'flux'})),
        EmissionLine(line_spec=NII_6548, own_params=frozenset({'flux'})),
        EmissionLine(line_spec=NII_6583, own_params=frozenset({'flux'})),
    )


def make_spectral_config(
    lines=None,
    spectral_oversample=5,
) -> SpectralConfig:
    """Convenience factory. Defaults to H-alpha + NII."""
    if lines is None:
        lines = halpha_nii_lines()
    return SpectralConfig(
        lines=lines,
        spectral_oversample=spectral_oversample,
    )
