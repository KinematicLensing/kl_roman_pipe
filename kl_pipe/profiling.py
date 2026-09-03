"""Opt-in performance instrumentation. Off unless ``KLPIPE_PROFILE_DIR`` is set.

Two tools:

- ``trace(name)``: context manager wrapping a region in a JAX profiler trace
  (TensorBoard/Perfetto format) written under ``$KLPIPE_PROFILE_DIR/<name>``.
  A no-op returning immediately when the variable is unset, so production
  code can wrap sampler runs unconditionally at zero cost.
- ``compiled_stats(fn, *args)``: compile a jittable function on the given
  arguments and return XLA's static cost analysis (fusion count, flops,
  bytes accessed, transcendentals) plus compile time. Always available;
  the benchmarks under ``experiments/`` and the ensemble worker's optional
  per-fit stats use it.

View a trace with ``tensorboard --logdir $KLPIPE_PROFILE_DIR`` (needs the
``tensorboard-plugin-profile`` package) or open the ``.trace.json.gz`` file
under ``plugins/profile`` at https://ui.perfetto.dev.
"""

from __future__ import annotations

import contextlib
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, Optional

import jax

_ENV_VAR = 'KLPIPE_PROFILE_DIR'


def profile_dir() -> Optional[Path]:
    """Return the profile output directory, or None when profiling is off."""
    raw = os.environ.get(_ENV_VAR, '').strip()
    if not raw:
        return None
    path = Path(raw).expanduser()
    if path.exists() and not path.is_dir():
        raise ValueError(f"{_ENV_VAR}={raw!r} exists and is not a directory")
    return path


@contextlib.contextmanager
def trace(name: str) -> Iterator[Optional[Path]]:
    """Record a JAX profiler trace of the enclosed region when enabled.

    Parameters
    ----------
    name : str
        Subdirectory under the profile directory (e.g. ``'sampler'``,
        ``'precondition'``). Nested traces are not supported by JAX; keep
        regions disjoint.

    Yields
    ------
    Path or None
        The trace output directory, or None when profiling is off.
    """
    root = profile_dir()
    if root is None:
        yield None
        return
    out = root / name
    out.mkdir(parents=True, exist_ok=True)
    jax.profiler.start_trace(str(out))
    try:
        yield out
    finally:
        jax.profiler.stop_trace()


def compiled_stats(fn: Callable, *args: Any) -> Dict[str, float]:
    """Compile ``fn(*args)`` and report XLA's static cost analysis.

    Parameters
    ----------
    fn : callable
        Function to lower; wrapped in ``jax.jit`` if not already jitted.
    *args
        Example arguments (concrete arrays or pytrees).

    Returns
    -------
    dict
        ``compile_s``, ``hlo_fusions`` (top-level fusion kernels in the
        optimized module), ``flops``, ``bytes_accessed``, ``transcendentals``.
        Cost fields are NaN when the backend does not report them.
    """
    jitted = fn if hasattr(fn, 'lower') else jax.jit(fn)
    t0 = time.perf_counter()
    compiled = jitted.lower(*args).compile()
    compile_s = time.perf_counter() - t0
    cost = compiled.cost_analysis()
    if isinstance(cost, list):
        cost = cost[0]
    cost = cost or {}
    nan = float('nan')
    return {
        'compile_s': compile_s,
        'hlo_fusions': compiled.as_text().count(' fusion('),
        'flops': float(cost.get('flops', nan)),
        'bytes_accessed': float(cost.get('bytes accessed', nan)),
        'transcendentals': float(cost.get('transcendentals', nan)),
    }
