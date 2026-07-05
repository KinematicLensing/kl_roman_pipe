"""
Central JAX floating-point precision configuration.

kl_pipe runs in float64 by default (``jax_enable_x64`` on). Setting the
environment variable ``KLPIPE_FP32`` to ``'1'`` or ``'true'`` opts into
JAX's native float32 default instead. Any other non-empty value raises
``ValueError`` -- a typo must never silently change numerical precision.

``ensure_precision()`` is called from ``kl_pipe/__init__.py`` and from the
modules that historically forced x64 at import time (``coordinates``,
``source``, ``lines``), so precision is configured identically regardless
of which submodule is imported first. It is idempotent and never
overrides the env-var choice on repeat calls.

.. warning::
   ``KLPIPE_FP32`` must be set before the first ``kl_pipe`` (or JAX)
   import; changing it afterwards has no effect on already-created arrays
   and is not re-read by ``ensure_precision()``.
"""

from __future__ import annotations

import os

import jax

_TRUTHY = ('1', 'true')
_FALSY = ('', '0', 'false')

# set by the first ensure_precision() call; later calls are no-ops
_configured: bool = False


def fp32_requested() -> bool:
    """Read ``KLPIPE_FP32`` from the environment (case-insensitive).

    Returns
    -------
    bool
        True when float32 mode is requested (``'1'``/``'true'``), False
        when unset or explicitly falsy (``''``/``'0'``/``'false'``).

    Raises
    ------
    ValueError
        For any other value -- precision must never be set by a typo.
    """
    raw = os.environ.get('KLPIPE_FP32', '')
    value = raw.strip().lower()
    if value in _FALSY:
        return False
    if value in _TRUTHY:
        return True
    raise ValueError(
        f"KLPIPE_FP32 must be one of {_TRUTHY} (float32) or {_FALSY} "
        f"(float64 default); got {raw!r}"
    )


def ensure_precision() -> None:
    """Configure the JAX default float precision (idempotent).

    Default: enable ``jax_enable_x64`` (float64 everywhere, the historical
    kl_pipe behavior). With ``KLPIPE_FP32`` truthy, leave JAX at its native
    float32 default. Only the first call does anything, so a later call
    can never override the env-var choice made at first import.

    In both modes the default matmul precision is pinned to ``'highest'``:
    on NVIDIA Ampere+ GPUs JAX otherwise executes float32 matmuls on
    tensorfloat32 tensor cores (10-bit mantissa), a silent precision
    downgrade far below true float32. The pin is a no-op on CPU and for
    float64 operands.
    """
    global _configured
    if _configured:
        return
    if not fp32_requested():
        jax.config.update('jax_enable_x64', True)
    jax.config.update('jax_default_matmul_precision', 'highest')
    _configured = True
