# Quickstart Tutorial

A (hopefully) practical introduction to the Roman kinematic lensing pipeline. For now, the emphasis is largely on the velocity and intensity map modeling of disky, rotationally-supported galaxies, as well as the available likelihood functions to use.

**NOTE:** If you'd like to read this tutorial as a Jupyter Notebook, just run the following locally:
```bash
jupytext --to ipynb .../kl_roman_pipe/docs/tutorials/quickstart.md
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

# Create an exponential disk intensity model (analytic eval is in surface
# brightness; render_image returns flux/pixel)
int_model = InclinedExponentialModel()

print("Model parameters:", int_model.PARAMETER_NAMES)
```

**Key parameters:**
- `flux`: Total integrated flux (conserved quantity)
- `int_rscale`: Exponential scale length (arcsec)
- `int_h_over_r`: Disk scale height-to-radius ratio (dimensionless)
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

# Generate synthetic data with noise.
# NOTE: Uses simple backend model; independent of model class.
# Velocity is a flux-weighted moment, not a count map -- noise is Gaussian
# only (Poisson on velocity at the 2D-image layer is meaningless and is
# rejected at the synthetic-data level).
synth = SyntheticVelocity(true_params, model_type='arctan', seed=42)
data_noisy = synth.generate(image_pars, snr=1000)

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
axes[1].set_title(f'Noisy Data (SNR={1000})')
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
# variance is a per-pixel array; for Gaussian-only velocity noise it is uniform
print(f"Noise std: {float(np.sqrt(np.mean(synth.variance))):.2f} km/s")
```

---

## Example 2: Build a Likelihood Function

The library provides helper functions to create JIT-compiled likelihood functions optimized for MCMC/optimization:

```{code-cell} python
from kl_pipe.likelihood import create_jitted_likelihood_velocity
from kl_pipe.observation import build_velocity_obs

# Build an observation object bundling grids + data + variance
obs_vel = build_velocity_obs(image_pars, data=data_noisy, variance=synth.variance)

# Create a JIT-compiled likelihood function
# This compiles once, then runs very fast on subsequent calls
log_likelihood = create_jitted_likelihood_velocity(model, obs_vel)

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
# Parameter order: cosi, theta_int, g1, g2, v0, vcirc, vel_rscale
theta_init = theta_true + jnp.array([0.05, -0.1, 0.01, -0.01, 1.0, -20.0, 0.5])

print("Initial guess:")
print(model.theta2pars(theta_init))

# Optimize using L-BFGS-B with analytical gradients and bounds
bounds = [
    (0.05, 0.99),    # cosi
    (0.0, np.pi),    # theta_int
    (-0.5, 0.5),     # g1
    (-0.5, 0.5),     # g2
    (-50, 50),        # v0
    (50, 500),        # vcirc
    (0.1, 20.0),      # vel_rscale
]

result = minimize(
    objective,
    theta_init,
    method='L-BFGS-B',
    jac=gradient,
    bounds=bounds,
    options={'maxiter': 200}
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
    if abs(true_val) > 0:
        error = 100 * abs(fit_val - true_val) / abs(true_val)
        print(f"  {key:12s}: {fit_val:8.4f}  (true: {true_val:8.4f}, error: {error:5.2f}%)")
    else:
        print(f"  {key:12s}: {fit_val:8.4f}  (true: {true_val:8.4f}, abs err: {abs(fit_val - true_val):.4f})")
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
    'int_h_over_r': 0.1,
    'cosi': 0.6,          # Same as velocity
    'theta_int': 0.785,   # Same as velocity
    'g1': 0.0,            # Same as velocity
    'g2': 0.0,            # Same as velocity
    'int_x0': 0.0,
    'int_y0': 0.0,
}

# Generate synthetic intensity data.
# include_poisson=False matches the test-suite convention (TestConfig default).
# flux=1.0 here is an arbitrary unit normalization, not a photon count, so
# Poisson statistics are not meaningfully scaled at this stage.
synth_int = SyntheticIntensity(int_params, model_type='exponential', seed=43)
data_int = synth_int.generate(image_pars, snr=1000, include_poisson=False)

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

from kl_pipe.observation import build_joint_obs
obs_vel, obs_int = build_joint_obs(
    image_pars, image_pars, int_model,
    data_vel=data_noisy, variance_vel=synth.variance,
    data_int=data_int, variance_int=synth_int.variance,
)
log_like_joint = create_jitted_likelihood_joint(kl_model, obs_vel, obs_int)

log_prob_joint = log_like_joint(theta_joint)
print(f"\nJoint log-likelihood: {log_prob_joint:.2f}")

# Plot truth + noisy data side-by-side. With flux=1.0 spread over many pixels,
# per-pixel intensity (~1e-4) is far below the per-pixel noise std implied by
# total-flux SNR=100, so the noisy panel is noise-dominated by design; the
# truth panel makes the underlying disk legible.
fig, axes = plt.subplots(2, 2, figsize=(11, 9))

im00 = axes[0, 0].imshow(synth.data_true.T, origin='lower', cmap='RdBu_r')
axes[0, 0].set_title('Velocity Truth')
plt.colorbar(im00, ax=axes[0, 0], label='km/s')

im01 = axes[0, 1].imshow(synth_int.data_true.T, origin='lower', cmap='viridis')
axes[0, 1].set_title('Intensity Truth')
plt.colorbar(im01, ax=axes[0, 1], label='flux')

im10 = axes[1, 0].imshow(data_noisy.T, origin='lower', cmap='RdBu_r')
axes[1, 0].set_title('Velocity Data (noisy)')
plt.colorbar(im10, ax=axes[1, 0], label='km/s')

im11 = axes[1, 1].imshow(data_int.T, origin='lower', cmap='viridis')
axes[1, 1].set_title('Intensity Data (noisy)')
plt.colorbar(im11, ax=axes[1, 1], label='flux')

for ax in axes.flat:
    ax.set_xlabel('x (pixels)')
    ax.set_ylabel('y (pixels)')

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

**For more detailed examples:**
- `tests/test_likelihood_slices.py` - Comprehensive parameter recovery tests
- `tests/test_optimizer_recovery.py` - Gradient-based fitting examples

---

## Multi-Component Intensity Models

Real galaxies have distinct morphological components (disk, bulge, bar). `CompositeIntensityModel` sums N components in k-space (one IFFT), and `BulgeDiskModel` is a convenience subclass for the common disk+bulge case.

### BulgeDiskModel

Exponential disk (n=1, exact FT) + Sersic bulge (n=4, fixed). The disk uses the analytic exponential FT; the bulge uses the Miller & Pasha (2025) symbolic regression emulator.

```{code-cell} python
from kl_pipe.intensity import BulgeDiskModel

model = BulgeDiskModel()
print("Parameters:", model.PARAMETER_NAMES)
print(f"Count: {len(model.PARAMETER_NAMES)}")
```

Flux is parameterized as `total_flux` + `bulge_frac` (B/T ratio). Component fluxes are derived internally: `disk_flux = total_flux * (1 - bulge_frac)`.

Render through the public `build_image_obs` + `render_image` path, with a
PSF. PSF convolution is non-negotiable for a bulge: the de Vaucouleurs n=4
profile has a central cusp that aliases at any finite oversample without it,
and is also unphysical (every real observation is band-limited by the
instrument PSF).

```{code-cell} python
import jax
jax.config.update('jax_enable_x64', True)   # required by kl_pipe.psf

import galsim
import matplotlib.pyplot as plt
from kl_pipe.parameters import ImagePars
from kl_pipe.observation import build_image_obs

pars = {
    'cosi': 0.5, 'theta_int': 0.3, 'g1': 0.02, 'g2': -0.01,
    'total_flux': 1e4, 'bulge_frac': 0.25,
    'disk_rscale': 1.5, 'disk_h_over_r': 0.1,
    'disk_x0': 0.0, 'disk_y0': 0.0,
    'bulge_hlr': 0.4, 'bulge_h_over_hlr': 0.3,
    'bulge_x0': 0.0, 'bulge_y0': 0.0,
}
theta = model.pars2theta(pars)

bd_image_pars = ImagePars(shape=(64, 64), pixel_scale=0.11, indexing='ij')
psf = galsim.Gaussian(fwhm=0.15)              # Roman-like PSF for illustration

# build_image_obs(...) is the public replacement for the old configure_psf.
# Passing int_model=model enables the fused k-space PSF path: render and
# convolve in a single FFT pass, with cusp anti-aliasing handled by oversample.
obs = build_image_obs(
    image_pars=bd_image_pars,
    psf=psf,
    oversample=5,                              # project default
    int_model=model,
)
image = model.render_image(theta, obs=obs)

plt.imshow(image, origin='lower', cmap='viridis')
plt.colorbar(label='Flux')
plt.title('Bulge + Disk Composite (PSF-convolved, oversample=5)')
plt.show()
```

**TODO** — once **PR #41** (render-config / tolerance API) lands on `main`,
swap the `build_image_obs` + `render_image` snippet above for the production
rendering entry point and expose its tolerance / oversample knobs explicitly.
The cell below previews that workflow by varying `oversample` directly on the
current API.

### Rendering-accuracy convergence

`oversample` is the current accuracy knob (see `_kspace_render_core` in
`kl_pipe/intensity.py`): it extends the k-grid to `N × Nyquist` before IFFT
and bins back to the coarse grid, suppressing the n=4 bulge cusp's aliasing.
At `oversample=1` the cusp aliases into a single bright pixel + axis-aligned
ringing even *with* PSF; `oversample=5` (the project default) is already
qualitatively converged for Gaussian PSFs (~7e-5 relative error per the PSF
oversampled-rendering convention); higher values vanish into numerical noise.

```{code-cell} python
import jax.numpy as jnp

oversamples = [1, 5, 11, 21]
images = []
for N in oversamples:
    obs_N = build_image_obs(
        image_pars=bd_image_pars,
        psf=psf,
        oversample=N,
        int_model=model,
    )
    images.append(model.render_image(theta, obs=obs_N))

# Use the highest-oversample render as the "true" reference
ref = images[-1]
fig, axes = plt.subplots(1, len(oversamples), figsize=(4 * len(oversamples), 4))
for ax, N, img in zip(axes, oversamples, images):
    im = ax.imshow(img, origin='lower', cmap='viridis')
    ax.set_title(f'oversample={N}')
    plt.colorbar(im, ax=ax, label='Flux')
plt.tight_layout()
plt.show()

# Numerical convergence: peak-pixel deviation from the oversample=21 reference
for N, img in zip(oversamples, images):
    rel_err = float(jnp.max(jnp.abs(img - ref)) / jnp.max(ref))
    print(f"oversample={N:>2d}:  peak={float(jnp.max(img)):8.4f}   "
          f"max |Δ| / peak = {rel_err:.3e}")
```

For this PSF (Gaussian FWHM=0.15", ≈1.4 px) the PSF already band-limits the
bulge cusp before the pixel grid sees it, so the `oversample` knob is mostly
invisible to the eye. Drop the PSF (or make it much narrower than the pixel
scale) and the `oversample=1` panel shows the cross-shaped FFT ringing the
old bare-`_render_kspace` cell produced.

### Shared centroids

By default, each component has its own centroid (`disk_x0`, `bulge_x0`, etc.). Pass `shared_centroids=True` to link them:

```{code-cell} python
model_shared = BulgeDiskModel(shared_centroids=True)
print("Shared:", model_shared.PARAMETER_NAMES)
print(f"Count: {len(model_shared.PARAMETER_NAMES)} (vs {len(model.PARAMETER_NAMES)} independent)")
```

### Turning off bulge shear

`fixed_params` can override shared parameters for specific components. To zero out shear for the bulge while keeping it for the disk:

```{code-cell} python
from kl_pipe.intensity import CompositeIntensityModel, ComponentSpec
from kl_pipe.intensity import InclinedExponentialModel, InclinedSersicModel

model_no_bulge_shear = CompositeIntensityModel(
    components=[
        ComponentSpec(InclinedExponentialModel(), prefix='disk'),
        ComponentSpec(InclinedSersicModel(), prefix='bulge',
                      fixed_params={'n_sersic': 4.0, 'g1': 0.0, 'g2': 0.0}),
    ],
)
# g1/g2 still in PARAMETER_NAMES (disk needs them), but bulge always sees zeros
print("g1 in params:", 'g1' in model_no_bulge_shear.PARAMETER_NAMES)
```

### Generic two-component

Any two `IntensityModel` subclasses can be composed:

```{code-cell} python
two_sersic = CompositeIntensityModel(
    components=[
        ComponentSpec(InclinedSersicModel(), prefix='thin_disk'),
        ComponentSpec(InclinedSersicModel(), prefix='thick_disk'),
    ],
)
print("Free n for both:", [p for p in two_sersic.PARAMETER_NAMES if 'n_sersic' in p])
```

### Three-component example

```{code-cell} python
three_comp = CompositeIntensityModel(
    components=[
        ComponentSpec(InclinedExponentialModel(), prefix='disk'),
        ComponentSpec(InclinedSersicModel(), prefix='bulge',
                      fixed_params={'n_sersic': 4.0}),
        ComponentSpec(InclinedSersicModel(), prefix='bar',
                      fixed_params={'n_sersic': 1.5}),
    ],
)
frac_params = [p for p in three_comp.PARAMETER_NAMES if '_frac' in p]
print("Fraction params:", frac_params)
print("disk_frac = 1 - bulge_frac - bar_frac (derived)")
```

---

## TODOs:

- Add example for MCMC inference
