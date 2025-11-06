import jax
import jax.numpy as jnp

from kl_pipe.velocity import OffsetVelocityModel
from kl_pipe.intensity import InclinedExponentialModel
from kl_pipe.model import KLModel
from kl_pipe.likelihood import log_likelihood

x = jnp.linspace(-10, 10, 50)
y = jnp.linspace(-10, 10, 50)
X, Y = jnp.meshgrid(x, y)
meta = {'X': X, 'Y': Y}

# Build models
vel_model = OffsetVelocityModel(meta_pars=meta)
int_model = InclinedExponentialModel(meta_pars=meta)
kl_model = KLModel(vel_model, int_model, shared_pars={'rscale'}, meta_pars=meta)

print(f"KL Model parameters: {kl_model.PARAMETER_NAMES}")
print(f"Total parameters: {len(kl_model.PARAMETER_NAMES)}")

# Create synthetic data
theta_true = jnp.array([100.0, 200.0, 5.0, 0.8, 45.0, 0.0, 0.0, 0.0, 0.0, 1.0, 3.0])
vel_map, int_map = kl_model(theta_true, 'obs', X, Y)
data = vel_map * int_map + jax.random.normal(jax.random.PRNGKey(42), vel_map.shape) * 0.1

# Evaluate likelihood
log_like = log_likelihood(theta_true, kl_model, data, meta)
print(f"Log-likelihood: {log_like:.2f}")

# Test gradients
gradient = jax.grad(log_likelihood, argnums=0)(theta_true, kl_model, data, meta)
print(f"Gradient shape: {gradient.shape}")
print(f"Gradient (first 3): {gradient[:3]}")

print("\nWorking!")