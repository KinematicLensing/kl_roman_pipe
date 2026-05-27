"""
SourceModel: unified source description.

A ``SourceModel`` may populate any subset of three component slots:

- ``velocity_model``: a singular ``VelocityModel`` (kinematics).
- ``broadband_models``: a dict ``{filter_name: IntensityModel}`` for broadband
  imaging in one or more filters.
- ``emission_lines``: a dict ``{line_name: EmissionLine}`` for grism / IFU
  emission line cube assembly.

The SourceModel layer is the only place that knows about the dotted-key
parameter namespace (``vel.vcirc``, ``F087.flux``, ``Halpha.flux``, etc.).
The underlying VelocityModel / IntensityModel classes operate on flat
theta arrays matching their own ``PARAMETER_NAMES`` and stay unaware of
the namespace.

``EmissionLine`` and the ``LINE_LAMBDAS`` rest-wavelength registry live
in ``kl_pipe.lines`` (import from there directly).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, Optional

import jax

from kl_pipe.lines import LINE_LAMBDAS, EmissionLine

if TYPE_CHECKING:
    from kl_pipe.model import IntensityModel, VelocityModel


@dataclass
class SourceModel:
    """Unified source description: velocity + broadband + emission line components.

    A SourceModel may populate any subset of {velocity_model,
    broadband_models, emission_lines}. All-empty is a loud error.
    Otherwise each slot is independently optional.

    Parameters
    ----------
    velocity_model : VelocityModel, optional
        Singular velocity model (rotation curve + flux-weight projection).
        Required for grism and velocity-channel inference.
    broadband_models : dict[str, IntensityModel], optional
        Per-filter broadband intensity models. Keys are user-chosen
        filter labels (e.g. ``'F087'``, ``'F184'``).
    emission_lines : dict[str, EmissionLine], optional
        Per-line emission models. Keys are line names. If a key exists
        in ``LINE_LAMBDAS``, the line's ``lambda_rest`` is auto-resolved
        from the registry; otherwise the line must set ``lambda_rest``
        explicitly.

    Validation (in __post_init__):
    - At least one component must be non-empty.
    - ``broadband_models`` and ``emission_lines`` keys must be disjoint
      (so dotted-key priors like ``F087.flux`` are unambiguous).
    - For each emission line:
        * If ``intensity_key`` is set, it must reference an existing
          line key in ``emission_lines``.
        * If ``continuum_key`` is set, the referenced line must have
          a ``continuum`` set (otherwise there is nothing to share).
        * If ``dispersion_key`` is set, it must reference an existing
          line key, and that line must NOT itself reference dispersion
          via its own ``dispersion_key`` (no chained sharing).
        * If ``lambda_rest`` is None, the line's name must be in
          ``LINE_LAMBDAS``; resolved automatically.
    """

    velocity_model: Optional['VelocityModel'] = None
    broadband_models: Dict[str, 'IntensityModel'] = field(default_factory=dict)
    emission_lines: Dict[str, EmissionLine] = field(default_factory=dict)

    def __post_init__(self):
        # require at least one component
        if (
            self.velocity_model is None
            and not self.broadband_models
            and not self.emission_lines
        ):
            raise ValueError(
                "SourceModel requires at least one of velocity_model, "
                "broadband_models, or emission_lines to be non-empty"
            )

        # disjoint broadband / emission keys
        overlap = set(self.broadband_models) & set(self.emission_lines)
        if overlap:
            raise ValueError(
                f"broadband_models and emission_lines keys must be disjoint; "
                f"collision: {sorted(overlap)}"
            )

        # validate emission line *_key cross-references + auto-resolve lambda_rest
        line_names = set(self.emission_lines)
        for name, line in self.emission_lines.items():
            # intensity_key references another line
            if line.intensity_key is not None and line.intensity_key not in line_names:
                raise ValueError(
                    f"emission line '{name}' has intensity_key='{line.intensity_key}' "
                    f"but that line is not in emission_lines"
                )
            # continuum_key references another line that has a continuum
            if line.continuum_key is not None:
                if line.continuum_key not in line_names:
                    raise ValueError(
                        f"emission line '{name}' has continuum_key='{line.continuum_key}' "
                        f"but that line is not in emission_lines"
                    )
                referenced = self.emission_lines[line.continuum_key]
                if referenced.continuum is None and referenced.continuum_key is None:
                    raise ValueError(
                        f"emission line '{name}' has continuum_key='{line.continuum_key}' "
                        f"but that line has no continuum to share"
                    )
            # dispersion_key references another line; no chained sharing
            if line.dispersion_key is not None:
                if line.dispersion_key not in line_names:
                    raise ValueError(
                        f"emission line '{name}' has dispersion_key="
                        f"'{line.dispersion_key}' but that line is not in "
                        f"emission_lines"
                    )
                referenced = self.emission_lines[line.dispersion_key]
                if referenced.dispersion_key is not None:
                    raise ValueError(
                        f"emission line '{name}' has dispersion_key="
                        f"'{line.dispersion_key}', but that line itself "
                        f"references another line via dispersion_key; chained "
                        f"sharing is not allowed"
                    )
            # auto-resolve lambda_rest from LINE_LAMBDAS
            if line.lambda_rest is None:
                if name not in LINE_LAMBDAS:
                    raise ValueError(
                        f"emission line '{name}' has no lambda_rest and is "
                        f"not in LINE_LAMBDAS. Provide lambda_rest=... "
                        f"explicitly or use a registered line name."
                    )
                line.lambda_rest = LINE_LAMBDAS[name]


# ===========================================================================
# JAX pytree registration
# ===========================================================================
#
# SourceModel is treated as an opaque Python object from JAX's perspective:
# no traceable children, the instance itself is the aux. Users pass it via
# ``partial(jit_fn, source=source)``, so JAX never needs to flatten it at
# the leaf level. Registering it as a pytree with this trivial split lets
# ``jax.tree_util.tree_flatten`` succeed and gives users a place to extend
# later if SourceModel ever needs to carry traceable data.


def _source_model_flatten(source):
    return (), source  # empty children, instance as aux


def _source_model_unflatten(aux, children):
    return aux


jax.tree_util.register_pytree_node(
    SourceModel, _source_model_flatten, _source_model_unflatten
)
