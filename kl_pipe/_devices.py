"""
Central JAX host-device count configuration.

kl_pipe defaults to whatever device count JAX discovers on its own (typically
one CPU device). Setting the environment variable ``KLPIPE_CPU_DEVICES`` to a
positive integer ``N`` forces JAX to expose ``N`` host CPU devices, enabling
``chain_method='parallel'`` (pmap-based) multi-chain MCMC on CPU. Unset or
empty is a no-op. Any other value raises ``ValueError`` -- a typo must never
silently change the device count.

``configure_cpu_devices()`` is called from ``kl_pipe/__init__.py`` so device
count is configured identically regardless of which submodule is imported
first. It is idempotent and never overrides the env-var choice on repeat
calls.

.. warning::
   ``KLPIPE_CPU_DEVICES`` must be set before the first ``kl_pipe`` (or JAX)
   import, and before any JAX op is executed. ``jax.config.update(
   'jax_num_cpu_devices', N)`` raises ``RuntimeError`` if the JAX backend has
   already been initialized (e.g. by an eager array op elsewhere in the
   process); this error is intentionally not caught here -- it is the correct
   loud failure mode for a misordered import.
"""

from __future__ import annotations

import os
from typing import Optional

import jax

# set by the first configure_cpu_devices() call; later calls are no-ops
_configured: bool = False


def cpu_devices_requested() -> Optional[int]:
    """Read ``KLPIPE_CPU_DEVICES`` from the environment.

    Returns
    -------
    int or None
        Requested host CPU device count, or None when unset/empty (no-op).

    Raises
    ------
    ValueError
        If set to anything other than a positive integer -- device count
        must never be set by a typo.
    """
    raw = os.environ.get('KLPIPE_CPU_DEVICES', '')
    value = raw.strip()
    if value == '':
        return None
    try:
        n = int(value)
    except ValueError:
        raise ValueError(
            f"KLPIPE_CPU_DEVICES must be a positive integer or unset/empty; "
            f"got {raw!r}"
        ) from None
    if n <= 0:
        raise ValueError(
            f"KLPIPE_CPU_DEVICES must be a positive integer or unset/empty; "
            f"got {raw!r}"
        )
    return n


def configure_cpu_devices() -> None:
    """Configure the JAX host CPU device count (idempotent).

    Default: leave JAX's device discovery untouched. With
    ``KLPIPE_CPU_DEVICES`` set to a positive integer ``N``, force ``N`` host
    CPU devices via ``jax.config.update('jax_num_cpu_devices', N)``. Only the
    first call does anything, so a later call can never override the
    env-var choice made at first import.
    """
    global _configured
    if _configured:
        return
    n = cpu_devices_requested()
    if n is not None:
        jax.config.update('jax_num_cpu_devices', n)
    _configured = True
