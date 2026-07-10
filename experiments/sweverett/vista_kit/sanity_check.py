"""GPU sanity check for the kl_pipe Vista container.

Run via klrun.sh on a compute node (idev/sbatch), not the login node:
    bash klrun.sh python sanity_check.py

Expects a CUDA device, working float64, and all imports resolvable.
"""

import jax

print("devices:", jax.devices())
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

print("dtype:", jnp.zeros(1).dtype)

import numpyro  # noqa: F401
import astropy  # noqa: F401
import yaml  # noqa: F401
import kl_pipe.source  # noqa: F401

print("sanity OK")
