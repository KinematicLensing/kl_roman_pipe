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

To use a line not in the registry, pass an explicit
``EmissionLine(..., lambda_rest=<nm>)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Optional

import jax

jax.config.update("jax_enable_x64", True)

if TYPE_CHECKING:
    from kl_pipe.model import IntensityModel


# ===========================================================================
# LINE_LAMBDAS registry: vacuum rest wavelengths (nm)
# ===========================================================================

LINE_LAMBDAS: Dict[str, float] = {
    # singlets — canonical name suffices
    'Lyalpha': 121.567,
    'CIV': 154.95,
    'CIII': 190.9,
    'MgII': 279.8,
    'Hbeta': 486.13,
    'Hgamma': 434.05,
    'Halpha': 656.28,
    # doublets / multiplets — wavelength integer suffix required
    'OII3726': 372.60,  # [O II] doublet, weaker
    'OII3728': 372.88,  # [O II] doublet, stronger
    'OII3727': 372.74,  # [O II] blended (low-R convention)
    'OIII4959': 495.89,  # [O III] doublet, weaker
    'OIII5007': 500.68,  # [O III] doublet, stronger
    'NII6548': 654.81,  # [N II] doublet, weaker
    'NII6584': 658.35,  # [N II] doublet, stronger
    'SII6717': 671.65,  # [S II] doublet, weaker
    'SII6731': 673.08,  # [S II] doublet, stronger
}


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
        broadband-like component under the line in cube assembly.
        Mutually exclusive with ``continuum_key``.
    continuum_key : str, optional
        Reference to another emission line's ``continuum``. Same sharing
        semantics as ``intensity_key`` but for the continuum component.
        ``<line>.cont.flux`` is still per-line.
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
