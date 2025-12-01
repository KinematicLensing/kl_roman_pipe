# Quickstart Tutorial

A (hopefully) practical introduction to the Roman kinematic lensing pipeline. For now, the emphasis is largely on the velocity and intensity map modeling of disky, rotationally-supported galaxies, as well as the available likelihood functions to use.

**NOTE:** If you'd like to read this tutorial as a Jupyter Notebook, just run the following locally:
```bash
jupytext --to ipynb path/to/your-notebook.md
```

## Design Philosophy

The `kl_pipe` library is built around three key principles:

1. **JAX-Compatible**: All core operations use JAX for automatic differentiation and JIT compilation, enabling fast gradient-based inference (HMC, NUTS, L-BFGS).

2. **Functional Core**: Models are immutable objects with pure evaluation functions. Parameters are passed as arrays (`theta`), never stored as mutable state.

3. **Coordinate Plane Abstraction**: Models transform coordinates through multiple reference frames (obs → cen → source → gal → disk) to handle lensing shear, position angles, and inclination systematically.

This design makes the code fast, composable, and safe for use in MCMC samplers.

---

## Key Classes

### Image Parameters (`ImagePars`)
Defines the geometry of your data:

```{code-cell} python
from kl_pipe.parameters import ImagePars

# Define a 64x32 pixel image with 0.1 arcsec/pixel resolution
image_pars = ImagePars(
    shape=(64, 32),        # (Ny, Nx) in 'ij' indexing
    pixel_scale=0.1,       # arcsec/pixel
    indexing='ij'          # numpy convention
)

print(f"Image: {image_pars.Nx} × {image_pars.Ny} pixels")
print(f"Field of view: {image_pars.Nx * image_pars.pixel_scale:.1f} × {image_pars.Ny * image_pars.pixel_scale:.1f} arcsec")

# Alternatively, define in Cartesian Nx/Ny
image_pars = ImagePars(
    shape=(32, 64),        # (Nx, Ny) in 'xy' indexing
    pixel_scale=0.1,       # arcsec/pixel
    indexing='xy'          # Cartesian convention
)

print(f"Image: {image_pars.Nx} × {image_pars.Ny} pixels")
print(f"Field of view: {image_pars.Nx * image_pars.pixel_scale:.1f} × {image_pars.Ny * image_pars.pixel_scale:.1f} arcsec")
```

### Velocity Models

```{code-cell} python
from kl_pipe.velocity import CenteredVelocityModel

# Create a centered rotating disk model
vel_model = CenteredVelocityModel()

print("Model parameters:", vel_model.PARAMETER_NAMES)
```

**Available velocity models:**
- `CenteredVelocityModel`: Arctangent rotation curve with systemic velocity
- `OffsetVelocityModel`: Same as above but includes centroid offsets (x0, y0)

**Key parameters:**
- `v0`: Systemic velocity (km/s)
- `vcirc`: Asymptotic circular velocity (km/s)  
- `vel_rscale`: Turnover radius (arcsec)
- `cosi`: cos(inclination) - 1=face-on, 0=edge-on
- `theta_int`: Position angle (radians)
- `g1, g2`: Lensing shear components
- `vel_x0, vel_y0`: Centroid offsets

### Intensity Models

```{code-cell} python
from kl_pipe.intensity import InclinedExponentialModel

# Create an exponential disk surface brightness model
int_model = InclinedExponentialModel()

print("Model parameters:", int_model.PARAMETER_NAMES)
```

**Key parameters:**
- `flux`: Total integrated flux (conserved quantity)
- `int_rscale`: Exponential scale length (arcsec)
- `cosi`, `theta_int`, `g1`, `g2`: Same as velocity model

---

## Example 1: Generate and Plot Velocity Data

```{code-cell} python
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from kl_pipe.velocity import CenteredVelocityModel
from kl_pipe.synthetic import SyntheticVelocity
from kl_pipe.parameters import ImagePars
from kl_pipe.utils import build_map_grid_from_image_pars

# Define true parameters
true_params = {
    'v0': 10.0,           # km/s systemic velocity
    'vcirc': 200.0,       # km/s asymptotic velocity
    'vel_rscale': 5.0,    # arcsec turnover radius
    'cosi': 0.6,          # ~53 deg inclination
    'theta_int': 0.785,   # ~45 deg position angle
    'g1': 0.0,
    'g2': 0.0,
}

# Setup image geometry
image_pars = ImagePars(shape=(64, 64), pixel_scale=0.15, indexing='ij')

# Generate synthetic data with noise
# NOTE: Uses simple backend model; independent of model class
synth = SyntheticVelocity(true_params, model_type='arctan', seed=42)
data_noisy = synth.generate(image_pars, snr=50)

# Also evaluate the model directly for comparison
model = CenteredVelocityModel()
theta_true = model.pars2theta(true_params)
X, Y = build_map_grid_from_image_pars(image_pars, unit='arcsec', centered=True)
model_map = model(theta_true, 'obs', X, Y)

# Plot
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

im0 = axes[0].imshow(
    synth.data_true.T, origin='lower', cmap='RdBu_r', vmin=-150, vmax=150
    )
axes[0].set_title('True Velocity Field')
axes[0].set_xlabel('x (pixels)')
axes[0].set_ylabel('y (pixels)')
plt.colorbar(im0, ax=axes[0], label='km/s')

im1 = axes[1].imshow(
    data_noisy.T, origin='lower', cmap='RdBu_r', vmin=-150, vmax=150
    )
axes[1].set_title(f'Noisy Data (SNR={50})')
axes[1].set_xlabel('x (pixels)')
axes[1].set_ylabel('y (pixels)')
plt.colorbar(im1, ax=axes[1], label='km/s')

residual = data_noisy - synth.data_true
im2 = axes[2].imshow(residual.T, origin='lower', cmap='RdBu_r')
axes[2].set_title('Noise Realization')
axes[2].set_xlabel('x (pixels)')
axes[2].set_ylabel('y (pixels)')
plt.colorbar(im2, ax=axes[2], label='km/s')

plt.tight_layout()
plt.show()

print(f"Data shape: {data_noisy.shape}")
print(f"Noise variance: {synth.variance:.2f} (km/s)²")
```

---

## Example 2: Build a Likelihood Function

The library provides helper functions to create JIT-compiled likelihood functions optimized for MCMC/optimization:

```{code-cell} python
from kl_pipe.likelihood import create_jitted_likelihood_velocity

# Create a JIT-compiled likelihood function
# This compiles once, then runs very fast on subsequent calls
# NOTE: This allows for arbitrary `meta_pars` passed at `model` instantiation
log_likelihood = create_jitted_likelihood_velocity(
    vel_model=model,
    image_pars_vel=image_pars,
    variance_vel=synth.variance,
    data_vel=data_noisy
)

# Evaluate at true parameters
log_prob_true = log_likelihood(theta_true)
print(f"Log-likelihood at true params: {log_prob_true:.2f}")

# Evaluate at slightly wrong parameters (lower vcirc)
wrong_params = true_params.copy()
wrong_params['vcirc'] = 150.0  # Should be 200
theta_wrong = model.pars2theta(wrong_params)
log_prob_wrong = log_likelihood(theta_wrong)

print(f"Log-likelihood at wrong params: {log_prob_wrong:.2f}")
print(f"Δ log-likelihood: {log_prob_true - log_prob_wrong:.2f}")
```

**Key features:**
- Returns a function that takes only `theta` as input
- All other arguments (data, variance, grids) are "frozen" via partial application
- JIT-compiled for speed (~microseconds per evaluation)
- Compatible with JAX transformations (grad, vmap, etc.)

---

## Example 3: Parameter Recovery with Optimization

Use JAX gradients for fast parameter fitting:

```{code-cell} python
import jax
from scipy.optimize import minimize

# Create gradient function using JAX
grad_fn = jax.jit(jax.grad(log_likelihood))

# Define objective for scipy (negative log-likelihood)
def objective(theta):
    return -float(log_likelihood(jnp.array(theta)))

def gradient(theta):
    return -np.array(grad_fn(jnp.array(theta)))

# Initial guess (perturb true values slightly)
theta_init = theta_true * jnp.array([1.0, 0.9, 1.1, 0.95, 1.05, 1.0, 1.0])

print("Initial guess:")
print(model.theta2pars(theta_init))

# Optimize using L-BFGS-B with analytical gradients
result = minimize(
    objective,
    theta_init,
    method='L-BFGS-B',
    jac=gradient,
    options={'maxiter': 100, 'disp': False}
)

print(f"\nOptimization converged: {result.success}")
print(f"Final log-likelihood: {-result.fun:.2f}")

# Compare recovered parameters to true values
theta_fit = jnp.array(result.x)
pars_fit = model.theta2pars(theta_fit)

print("\nRecovered parameters:")
for key in true_params.keys():
    true_val = true_params[key]
    fit_val = pars_fit[key]
    error = 100 * abs(fit_val - true_val) / abs(true_val)
    print(f"  {key:12s}: {fit_val:8.4f}  (true: {true_val:8.4f}, error: {error:5.2f}%)")
```

---

## Example 4: Joint Velocity + Intensity Modeling

Combine velocity and intensity observations:

```{code-cell} python
from kl_pipe.model import KLModel
from kl_pipe.intensity import InclinedExponentialModel
from kl_pipe.synthetic import SyntheticIntensity
from kl_pipe.likelihood import create_jitted_likelihood_joint

# Define intensity parameters (shares geometric parameters with velocity)
int_params = {
    'flux': 1.0,
    'int_rscale': 3.0,
    'cosi': 0.6,          # Same as velocity
    'theta_int': 0.785,   # Same as velocity
    'g1': 0.0,            # Same as velocity
    'g2': 0.0,            # Same as velocity
    'int_x0': 0.0,
    'int_y0': 0.0,
}

# Generate synthetic intensity data
synth_int = SyntheticIntensity(int_params, model_type='exponential', seed=43)
data_int = synth_int.generate(image_pars, snr=100)

# Create joint model
vel_model = CenteredVelocityModel()
int_model = InclinedExponentialModel()

kl_model = KLModel(
    velocity_model=vel_model,
    intensity_model=int_model,
    shared_pars={'cosi', 'theta_int', 'g1', 'g2'}  # Share geometric parameters
)

print("Joint model parameters:", kl_model.PARAMETER_NAMES)
print(f"Total parameters: {len(kl_model.PARAMETER_NAMES)}")

# Build joint likelihood
joint_true_pars = {**true_params, **int_params}
theta_joint = kl_model.pars2theta(joint_true_pars)

log_like_joint = create_jitted_likelihood_joint(
    kl_model=kl_model,
    image_pars_vel=image_pars,
    image_pars_int=image_pars,
    variance_vel=synth.variance,
    variance_int=synth_int.variance,
    data_vel=data_noisy,
    data_int=data_int
)

log_prob_joint = log_like_joint(theta_joint)
print(f"\nJoint log-likelihood: {log_prob_joint:.2f}")

# Plot both datasets
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

im0 = axes[0].imshow(data_noisy.T, origin='lower', cmap='RdBu_r')
axes[0].set_title('Velocity Data')
axes[0].set_xlabel('x (pixels)')
axes[0].set_ylabel('y (pixels)')
plt.colorbar(im0, ax=axes[0], label='km/s')

im1 = axes[1].imshow(data_int.T, origin='lower', cmap='viridis')
axes[1].set_title('Intensity Data')
axes[1].set_xlabel('x (pixels)')
axes[1].set_ylabel('y (pixels)')
plt.colorbar(im1, ax=axes[1], label='flux')

plt.tight_layout()
plt.show()
```

**Joint modeling benefits:**
- Shares geometric parameters (inclination, PA, shear) between velocity and intensity
- Breaks degeneracies (e.g., inclination better constrained with both datasets)
- Natural framework for full kinematic-lensing analysis

---

## Example 5: Likelihood Slicing for Validation

Visualize likelihood landscape to validate model and check parameter constraints:

```{code-cell} python
# Slice likelihood along vcirc dimension
vcirc_range = np.linspace(150, 250, 50)
log_probs = []

for vcirc in vcirc_range:
    test_params = true_params.copy()
    test_params['vcirc'] = vcirc
    theta_test = model.pars2theta(test_params)
    log_probs.append(log_likelihood(theta_test))

log_probs = np.array(log_probs)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(vcirc_range, log_probs, 'b-', linewidth=2)
plt.axvline(
    true_params['vcirc'],
    color='r',
    linestyle='--', 
    label=f"True value: {true_params['vcirc']:.0f} km/s"
    )

# Mark peak
peak_idx = np.argmax(log_probs)
peak_vcirc = vcirc_range[peak_idx]
plt.axvline(
    peak_vcirc, color='g', linestyle='--', label=f"Peak: {peak_vcirc:.1f} km/s"
    )

plt.xlabel('vcirc (km/s)', fontsize=12)
plt.ylabel('Log-Likelihood', fontsize=12)
plt.title('Likelihood Slice Along vcirc', fontsize=14)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print(f"True vcirc: {true_params['vcirc']:.1f} km/s")
print(f"Peak vcirc: {peak_vcirc:.1f} km/s")
print(
    f"Error: {abs(peak_vcirc - true_params['vcirc']):.2f} km/s "
    f"({100*abs(peak_vcirc - true_params['vcirc'])/true_params['vcirc']:.1f}%)"
    )
```

This technique is used extensively in the test suite (`tests/test_likelihood_slices.py`) to validate that:
1. Likelihoods peak at true parameter values
2. Parameter constraints are well-behaved
3. Forward models are implemented correctly

---

## TODOs:

- Add eample for MCMC inference

**For more detailed examples:**
- `tests/test_likelihood_slices.py` - Comprehensive parameter recovery tests
- `tests/test_optimizer_recovery.py` - Gradient-based fitting examples