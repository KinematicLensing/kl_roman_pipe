"""
kl_pipe: JAX-based kinematic lensing pipeline.

Importing the package configures the JAX default float precision (float64
by default, float32 when the ``KLPIPE_FP32`` environment variable is
truthy; see ``kl_pipe._precision``) and the JAX host CPU device count
(unchanged by default, forced to N when ``KLPIPE_CPU_DEVICES`` is set to a
positive integer; see ``kl_pipe._devices``).
"""

from kl_pipe._precision import ensure_precision
from kl_pipe._devices import configure_cpu_devices

ensure_precision()
configure_cpu_devices()
