"""
Emission line definitions and rest-wavelength registry.

Holds:

- ``LINE_LAMBDAS``: dict mapping canonical line names (vacuum, nm) to
  rest wavelengths. Singlets use their canonical name (``'Halpha'``);
  doublets / multiplets carry a wavelength-integer suffix
  (``'OIII5007'``, ``'NII6584'``).
- ``EmissionLine``: per-line container used by ``SourceModel.emission_lines``.
  Carries the spatial intensity profile (own or shared via key),
  optional stellar continuum (own or shared via key), and optional
  shared dispersion (via ``dispersion_key``). Rest wavelength is
  auto-resolved from ``LINE_LAMBDAS`` by the dict key under which the
  line is registered with the SourceModel.
- ``air_to_vacuum`` / ``vacuum_to_air``: wavelength conversion (nm) for
  line lists quoted in standard air.

To use a line not in the registry, pass an explicit
``EmissionLine(..., lambda_rest=<nm>)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Optional, Union

import jax
import jax.numpy as jnp
import numpy as np

from kl_pipe._precision import ensure_precision

ensure_precision()

if TYPE_CHECKING:
    from kl_pipe.model import IntensityModel

ArrayLike = Union[float, np.ndarray, jnp.ndarray]


# ===========================================================================
# LINE_LAMBDAS registry: vacuum rest wavelengths (nm)
# ===========================================================================

LINE_LAMBDAS: Dict[str, float] = {
    # SDSS vacuum line list (classic.sdss.org/dr7/algorithms/linestable),
    # consistent with air_to_vacuum(NIST ASD air values) to < 0.005 nm
    # singlets — canonical name suffices
    'Lyalpha': 121.567,
    'CIV': 154.948,  # C IV 1548/1551 blend
    'CIII': 190.873,  # C III] 1908.73
    'MgII': 279.949,  # Mg II 2796/2804 blend
    'Hbeta': 486.268,
    'Hgamma': 434.168,
    'Halpha': 656.461,
    # doublets / multiplets — wavelength integer suffix required (air-based names)
    'OII3726': 372.709,  # [O II] doublet, weaker
    'OII3728': 372.988,  # [O II] doublet, stronger
    'OII3727': 372.848,  # [O II] blended (doublet mean, low-R convention)
    'OIII4959': 496.030,  # [O III] doublet, weaker
    'OIII5007': 500.824,  # [O III] doublet, stronger
    'NII6548': 654.986,  # [N II] doublet, weaker
    'NII6584': 658.527,  # [N II] doublet, stronger
    'SII6717': 671.829,  # [S II] doublet, weaker
    'SII6731': 673.267,  # [S II] doublet, stronger
}


# ===========================================================================
# Air <-> vacuum wavelength conversion
# ===========================================================================

# lower validity bound of the air refractive-index formula; below it
# wavelengths are conventionally quoted in vacuum only
_AIR_VACUUM_MIN_NM = 200.0


def _check_concrete_wavelength(lam: ArrayLike, name: str) -> None:
    # traced inputs cannot be inspected inside jit; the bound is documented
    if isinstance(lam, jax.core.Tracer):
        return
    arr = np.asarray(lam, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name}: wavelengths must be finite, got {arr}")
    if np.any(arr < _AIR_VACUUM_MIN_NM):
        raise ValueError(
            f"{name}: wavelengths must be >= {_AIR_VACUUM_MIN_NM} nm (formula "
            f"validity; UV lines are quoted in vacuum), got min {arr.min()} nm"
        )


def _air_refractive_index(lam_vac_nm: ArrayLike) -> jnp.ndarray:
    # Morton (2000, ApJS 130, 403) eq. 8; s = vacuum wavenumber in um^-1
    s2 = (1.0e3 / lam_vac_nm) ** 2
    return 1.0 + 8.34254e-5 + 2.406147e-2 / (130.0 - s2) + 1.5998e-4 / (38.9 - s2)


def vacuum_to_air(lambda_vac_nm: ArrayLike) -> jnp.ndarray:
    """Convert vacuum wavelengths to standard-air wavelengths.

    Uses the IAU-standard refractive index of Morton (2000, ApJS 130, 403),
    lambda_air = lambda_vac / n(lambda_vac), for dry air at 15 C, 101325 Pa.

    Parameters
    ----------
    lambda_vac_nm : float or array
        Vacuum wavelength(s) in nm. Must be >= 200 nm; checked for concrete
        inputs, unchecked when traced under ``jax.jit``.

    Returns
    -------
    jnp.ndarray
        Air wavelength(s) in nm.

    Raises
    ------
    ValueError
        If a concrete input is non-finite or below 200 nm.
    """
    _check_concrete_wavelength(lambda_vac_nm, 'vacuum_to_air')
    lam = jnp.asarray(lambda_vac_nm)
    return lam / _air_refractive_index(lam)


def air_to_vacuum(lambda_air_nm: ArrayLike) -> jnp.ndarray:
    """Convert standard-air wavelengths to vacuum wavelengths.

    Exact inverse of ``vacuum_to_air``: solves lambda_vac = lambda_air *
    n(lambda_vac) by fixed-point iteration on the Morton (2000) index. Each
    iteration contracts the error by |lambda dn/dlambda| < 1e-5 over the
    valid range, so three iterations reach float64 round-off.

    Parameters
    ----------
    lambda_air_nm : float or array
        Air wavelength(s) in nm. Must be >= 200 nm; checked for concrete
        inputs, unchecked when traced under ``jax.jit``.

    Returns
    -------
    jnp.ndarray
        Vacuum wavelength(s) in nm.

    Raises
    ------
    ValueError
        If a concrete input is non-finite or below 200 nm.
    """
    _check_concrete_wavelength(lambda_air_nm, 'air_to_vacuum')
    lam_air = jnp.asarray(lambda_air_nm)
    lam_vac = lam_air * _air_refractive_index(lam_air)
    for _ in range(3):
        lam_vac = lam_air * _air_refractive_index(lam_vac)
    return lam_vac


# ===========================================================================
# EmissionLine
# ===========================================================================


@dataclass
class EmissionLine:
    """One emission line in a SourceModel.

    Parameters
    ----------
    intensity : IntensityModel, optional
        Spatial profile of the ionized-gas emission at this line wavelength.
        Mutually exclusive with ``intensity_key``; exactly one must be set.
    intensity_key : str, optional
        Reference to another emission line's ``intensity`` (e.g.
        ``intensity_key='Halpha'`` to share Halpha's spatial profile).
        ``<line>.flux`` is still per-line.
    continuum : IntensityModel, optional
        Optional stellar continuum at this line's wavelength. Adds a
        broadband-like component under the line in cube assembly. A raw
        ``IntensityModel`` is auto-wrapped in ``ContinuumModel``, exposing its
        amplitude as the spectral density ``<line>.cont.flux_per_nm``
        [flux/arcsec^2/nm] rather than an integrated ``flux``. Mutually
        exclusive with ``continuum_key``.
    continuum_key : str, optional
        Reference to another emission line's ``continuum``. Same sharing
        semantics as ``intensity_key`` but for the continuum component.
        ``<line>.cont.flux_per_nm`` is still per-line.
    dispersion_key : str, optional
        Reference to another emission line's intrinsic kinematic
        velocity dispersion. When set, this line's dispersion is read
        from the referenced line's ``<line>.dispersion`` prior rather
        than from its own. Validated by ``SourceModel.__post_init__``.
    lambda_rest : float, optional
        Rest-frame line wavelength in nm (vacuum). If None at
        construction, SourceModel will auto-resolve from ``LINE_LAMBDAS``
        using this line's dict key.
    """

    intensity: Optional['IntensityModel'] = None
    intensity_key: Optional[str] = None
    continuum: Optional['IntensityModel'] = None
    continuum_key: Optional[str] = None
    dispersion_key: Optional[str] = None
    lambda_rest: Optional[float] = None

    def __post_init__(self):
        # exactly one of intensity / intensity_key must be set
        has_intensity = self.intensity is not None
        has_key = self.intensity_key is not None
        if has_intensity == has_key:
            raise ValueError(
                "EmissionLine: exactly one of 'intensity' or 'intensity_key' "
                "must be set"
            )
        # at most one of continuum / continuum_key
        if self.continuum is not None and self.continuum_key is not None:
            raise ValueError(
                "EmissionLine: at most one of 'continuum' or 'continuum_key' "
                "may be set"
            )
        # wrap a raw IntensityModel continuum in ContinuumModel so its amplitude
        # is exposed as the density parameter '<line>.cont.flux_per_nm'.
        # Idempotent: a pre-wrapped ContinuumModel passes through unchanged.
        if self.continuum is not None:
            from kl_pipe.model import ContinuumModel

            self.continuum = ContinuumModel(self.continuum)
        # dispersion_key cross-reference is validated by SourceModel,
        # since it needs the full emission_lines dict to check.


# ===========================================================================
# JAX pytree registration
# ===========================================================================


def _emission_line_flatten(line):
    return (), line  # empty children, instance as aux


def _emission_line_unflatten(aux, children):
    return aux


jax.tree_util.register_pytree_node(
    EmissionLine, _emission_line_flatten, _emission_line_unflatten
)
