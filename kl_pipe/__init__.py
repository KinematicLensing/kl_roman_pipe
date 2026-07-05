"""
kl_pipe: JAX-based kinematic lensing pipeline.

Importing the package configures the JAX default float precision: float64
by default, float32 when the ``KLPIPE_FP32`` environment variable is
truthy (see ``kl_pipe._precision``).
"""

from kl_pipe._precision import ensure_precision

ensure_precision()
