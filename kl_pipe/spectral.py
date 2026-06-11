"""
Datacube grid definitions.

Provides ``CubePars`` (spatial grid + wavelength array) used to assemble and
disperse 3D datacubes C(x, y, lambda). Cube assembly itself lives on
``kl_pipe.source.SourceModel.build_cube``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import jax.numpy as jnp
import numpy as np

from kl_pipe.parameters import ImagePars

# speed of light in km/s
C_KMS = 299792.458


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
