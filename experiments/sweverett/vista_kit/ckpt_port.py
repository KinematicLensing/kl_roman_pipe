"""jax.checkpoint monkeypatch variants for build_cube.

Vendored verbatim from
experiments/sweverett/production_speedups/ckpt_probe.py (patch section).
That script cannot be imported directly here because its module imports
pull in galsim-dependent profile modules; the patch pattern itself is
galsim-free. Keep the two in sync if either changes.

Variants:
  base        : shipped path (no patch)
  cube        : jax.checkpoint around SourceModel.build_cube
  cube_dots   : same, policy=dots_saveable (save contractions)
  intensity   : jax.checkpoint around InclinedExponentialModel.__call__
  int_vel     : intensity + CenteredVelocityModel.__call__

Values and gradients are mathematically identical under remat -- this is a
pure scheduling trade (verified to rtol 1e-10 on CPU; bench_matrix.py
re-verifies on every run).
"""

from __future__ import annotations

import jax

from kl_pipe.intensity import InclinedExponentialModel
from kl_pipe.source import SourceModel
from kl_pipe.velocity import CenteredVelocityModel

_ORIG_BUILD_CUBE = SourceModel.build_cube
_ORIG_INT_CALL = InclinedExponentialModel.__call__
_ORIG_VEL_CALL = CenteredVelocityModel.__call__


def _patch_build_cube(policy=None):
    def build_cube_ckpt(self, pars, cube_pars, **kw):
        def f(pars_):
            return _ORIG_BUILD_CUBE(self, pars_, cube_pars, **kw)

        return jax.checkpoint(f, policy=policy)(pars)

    SourceModel.build_cube = build_cube_ckpt


def _patch_intensity():
    def call_ckpt(self, theta, plane, X, Y):
        def f(theta_, X_, Y_):
            return _ORIG_INT_CALL(self, theta_, plane, X_, Y_)

        return jax.checkpoint(f)(theta, X, Y)

    InclinedExponentialModel.__call__ = call_ckpt


def _patch_velocity():
    def call_ckpt(self, theta, plane, X, Y):
        def f(theta_, X_, Y_):
            return _ORIG_VEL_CALL(self, theta_, plane, X_, Y_)

        return jax.checkpoint(f)(theta, X, Y)

    CenteredVelocityModel.__call__ = call_ckpt


def unpatch():
    SourceModel.build_cube = _ORIG_BUILD_CUBE
    InclinedExponentialModel.__call__ = _ORIG_INT_CALL
    CenteredVelocityModel.__call__ = _ORIG_VEL_CALL


VARIANTS = {
    'base': lambda: None,
    'cube': lambda: _patch_build_cube(),
    'cube_dots': lambda: _patch_build_cube(
        policy=jax.checkpoint_policies.dots_saveable
    ),
    'intensity': _patch_intensity,
    'int_vel': lambda: (_patch_intensity(), _patch_velocity()),
}
