# Bayesian Inference with MCMC Sampling

This tutorial demonstrates Bayesian parameter inference using the `kl_pipe` sampling module.

> **TL;DR:** emcee for exploration, nautilus for evidence/model comparison, **numpyro for production** (recommended -- handles multi-scale gradients, provides R-hat/ESS), blackjax for simple velocity-only problems (known issues with joint models).

**NOTE:** To read this as a Jupyter Notebook, run:
```bash
jupytext --to ipynb docs/tutorials/sampling.md
```

---

## Design Philosophy

The sampling module follows the same JAX-compatible, functional design as the rest of `kl_pipe`:

1. **InferenceTask**: Bundles model, likelihood, priors, and data into a single object
2. **PriorDict**: Separates sampled parameters (have Prior objects) from fixed parameters (numeric values)
3. **Factory Pattern**: `build_sampler('emcee', task, config)` creates samplers with a unified interface
4. **SamplerResult**: Unified output format across all backends

---

## Quick Overview

```{code-cell} python
import os
CI_MODE = os.environ.get('KL_PIPE_CI', '0') == '1'
if CI_MODE:
    print("CI mode: using reduced MCMC settings for faster execution")

from kl_pipe.sampling import (
    InferenceTask,
    EnsembleSamplerConfig,
    NestedSamplerConfig,
    GradientSamplerConfig,
    NumpyroSamplerConfig,
    build_sampler,
    get_available_samplers,
)

# See what samplers are available
print("Available samplers:", get_available_samplers())
```

---

## Section 1: Velocity-Only Inference with emcee

Let's start with a simple velocity-only inference problem using the emcee ensemble sampler.

### 1.1 Generate Synthetic Data

```{code-cell} python
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from kl_pipe.velocity import CenteredVelocityModel
from kl_pipe.parameters import ImagePars
from kl_pipe.synthetic import SyntheticVelocity

# Define true parameters
true_pars = {
    'v0': 10.0,           # km/s systemic velocity
    'vcirc': 200.0,       # km/s asymptotic velocity
    'vel_rscale': 5.0,    # arcsec turnover radius
    'cosi': 0.6,          # ~53 deg inclination
    'theta_int': 0.785,   # ~45 deg position angle
    'g1': 0.0,
    'g2': 0.0,
}

# Generate synthetic velocity data
image_pars = ImagePars(shape=(32, 32), pixel_scale=0.3, indexing='ij')
synth = SyntheticVelocity(true_pars, model_type='arctan', seed=42)
snr = 20
data_noisy = synth.generate(image_pars, snr=snr, include_poisson=False)
variance = synth.variance

print(f"Data shape: {data_noisy.shape}")
print(f"Variance: {variance:.2f} (km/s)^2")

# Plot the data
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
im0 = axes[0].imshow(synth.data_true.T, origin='lower', cmap='RdBu_r')
axes[0].set_title('True Velocity Field')
plt.colorbar(im0, ax=axes[0], label='km/s')

im1 = axes[1].imshow(data_noisy.T, origin='lower', cmap='RdBu_r')
axes[1].set_title(f'Noisy Data (SNR={snr})')
plt.colorbar(im1, ax=axes[1], label='km/s')
plt.tight_layout()
plt.show()
```

### 1.2 Define Priors

The `PriorDict` class automatically separates sampled parameters (those with `Prior` objects) from fixed parameters (numeric values).

```{code-cell} python
from kl_pipe.priors import Uniform, Gaussian, TruncatedNormal, PriorDict

# Define priors for parameters we want to sample
# Parameters with Prior objects will be sampled
# Parameters with numeric values will be fixed
priors = PriorDict({
    # Sampled parameters
    'v0': Gaussian(10.0, 5.0),          # Weakly constrained around true
    'vcirc': Uniform(100, 300),          # Broad uniform prior
    'vel_rscale': Uniform(1.0, 10.0),    # Reasonable range
    'cosi': TruncatedNormal(0.5, 0.3, 0.1, 0.99),  # Gaussian truncated to valid range
    'theta_int': Uniform(0, np.pi),      # Position angle

    # Fixed parameters (not sampled)
    'g1': 0.0,
    'g2': 0.0,
})

print(f"Sampled parameters: {priors.sampled_names}")
print(f"Fixed parameters: {priors.fixed_names}")
print(f"Number of dimensions: {priors.n_sampled}")
```

### 1.3 Create InferenceTask

The `InferenceTask` bundles everything needed for sampling: model, likelihood function, priors, and data.

```{code-cell} python
from kl_pipe.sampling import InferenceTask

# Create the velocity model
model = CenteredVelocityModel()

# Create the inference task
task = InferenceTask.from_velocity_model(
    model=model,
    priors=priors,
    data_vel=jnp.array(data_noisy),
    variance_vel=variance,
    image_pars=image_pars,
)

print(f"Sampled parameters: {task.sampled_names}")
print(f"Fixed parameters: {task.fixed_params}")
print(f"Number of dimensions: {task.n_params}")

# Test that the log posterior is finite
import jax.random as random
key = random.PRNGKey(42)
theta_test = task.sample_prior(key, 1)[0]
log_prob_fn = task.get_log_posterior_fn()
print(f"Log posterior at prior sample: {log_prob_fn(theta_test):.2f}")
```

### 1.4 Configure and Run emcee

```{code-cell} python
from kl_pipe.sampling import EnsembleSamplerConfig, build_sampler

# Configure the sampler
config = EnsembleSamplerConfig(
    n_walkers=32,
    n_iterations=200 if CI_MODE else 2000,
    burn_in=50 if CI_MODE else 500,
    thin=1,
    seed=42,
    progress=not CI_MODE,
)

# Build and run the sampler
sampler = build_sampler('emcee', task, config)
result = sampler.run()

print(f"Number of samples: {result.n_samples}")
print(f"Acceptance fraction: {result.acceptance_fraction:.1%}")
```

### 1.5 Analyze Results

```{code-cell} python
from kl_pipe.sampling.diagnostics import plot_corner, plot_trace, print_summary

# Print summary statistics
print_summary(result, true_values=true_pars)

# Corner plot
fig = plot_corner(result, true_values=true_pars, sampler_info={'name': 'emcee'})
plt.show()

# Trace plots (useful for convergence diagnosis)
fig = plot_trace(result)
plt.show()
```

---

## Section 2: Nested Sampling with nautilus

Nautilus uses neural networks to efficiently explore the parameter space and provides evidence estimates for model comparison.

### 2.1 Configure nautilus

```{code-cell} python
from kl_pipe.sampling import NestedSamplerConfig

# nautilus configuration
config_nautilus = NestedSamplerConfig(
    n_live=100 if CI_MODE else 500,
    n_networks=4,
    seed=42,
    progress=not CI_MODE,
)

print("Nested sampler configuration:")
print(f"  Live points: {config_nautilus.n_live}")
```

### 2.2 Run nautilus

```{code-cell} python
# Use the same task as before
sampler_nautilus = build_sampler('nautilus', task, config_nautilus)
result_nautilus = sampler_nautilus.run()

print(f"Number of samples: {result_nautilus.n_samples}")
```

### 2.3 Evidence and Results

```{code-cell} python
# Nautilus provides the Bayesian evidence
print_summary(result_nautilus, true_values=true_pars)

if result_nautilus.evidence is not None:
    print(f"\nLog evidence: {result_nautilus.evidence:.2f}")
    # TODO: nautilus doesn't currently expose evidence_error

# Corner plot
fig = plot_corner(result_nautilus, true_values=true_pars, sampler_info={'name': 'nautilus'})
plt.show()
```

The log evidence (Z) is useful for Bayesian model comparison. If you have two models with evidences Z1 and Z2, the Bayes factor is:

$$\text{Bayes factor} = \frac{Z_1}{Z_2} = e^{\log Z_1 - \log Z_2}$$

---

## Section 3: Gradient-Based Sampling with NumPyro (Recommended)

NumPyro provides a robust NUTS implementation with superior mass matrix adaptation and built-in Z-score reparameterization. This is the **recommended gradient-based sampler** for joint velocity+intensity models.

### 3.1 Why NumPyro?

NumPyro is particularly effective when:
- Parameters span multiple scales (e.g., intensity ~10^7, velocity ~10^3)
- You need robust convergence diagnostics (R-hat, ESS)
- The posterior has parameter correlations

The key feature is **Z-score reparameterization**, which automatically normalizes parameter scales so the sampler sees O(1) gradients for all parameters.

### 3.2 Configure NumPyro

```{code-cell} python
from kl_pipe.sampling import NumpyroSamplerConfig, ReparamStrategy

config_numpyro = NumpyroSamplerConfig(
    n_samples=200 if CI_MODE else 1250,
    n_warmup=100 if CI_MODE else 625,
    n_chains=1 if CI_MODE else 4,
    dense_mass=True,
    reparam_strategy='prior',
    target_accept_prob=0.8,
    seed=42,
    progress=not CI_MODE,
)

print("NumPyro configuration:")
print(f"  Samples per chain: {config_numpyro.n_samples}")
print(f"  Chains: {config_numpyro.n_chains}")
print(f"  Reparameterization: {config_numpyro.reparam_strategy}")
```

### 3.3 Run NumPyro NUTS

```{code-cell} python
sampler_numpyro = build_sampler('numpyro', task, config_numpyro)
result_numpyro = sampler_numpyro.run()

print(f"Total samples: {result_numpyro.n_samples}")
print(f"Acceptance rate: {result_numpyro.acceptance_fraction:.1%}")

# Corner plot
fig = plot_corner(result_numpyro, true_values=true_pars, sampler_info={'name': 'numpyro'})
plt.show()
```

### 3.4 Check Convergence Diagnostics

```{code-cell} python
# R-hat should be close to 1.0 (< 1.01 is good)
r_hats = result_numpyro.get_rhat()
print("R-hat values:")
for name, rhat in r_hats.items():
    status = "OK" if rhat < 1.01 else "WARNING"
    print(f"  {name}: {rhat:.4f} {status}")

# Effective sample size
ess = result_numpyro.get_ess()
print("\nEffective sample size:")
for name, n_eff in ess.items():
    print(f"  {name}: {n_eff:.0f}")

# Divergences
n_div = result_numpyro.diagnostics.get('n_divergences', 0)
print(f"\nDivergences: {n_div}")
```

### 3.5 Z-Score Reparameterization Strategies

NumPyro supports three reparameterization strategies:

```{code-cell} python
from kl_pipe.sampling import ReparamStrategy

# Strategy 1: Prior-based (default, fast)
# Uses prior mean/std for scaling
config_prior = NumpyroSamplerConfig(
    reparam_strategy=ReparamStrategy.PRIOR,
    # ...
)

# Strategy 2: Empirical (slower, more robust)
# Runs short warmup to estimate posterior scales
config_empirical = NumpyroSamplerConfig(
    reparam_strategy=ReparamStrategy.EMPIRICAL,
    empirical_warmup_frac=0.1,  # Use 10% of warmup for estimation
    # ...
)

# Strategy 3: None (sample in physical space)
# Only use if you know parameters are well-scaled
config_none = NumpyroSamplerConfig(
    reparam_strategy=ReparamStrategy.NONE,
    # ...
)
```

---

## Section 4: BlackJAX

BlackJAX provides JAX-native HMC/NUTS sampling using `GradientSamplerConfig`. It works for simple velocity-only models but has **known issues with joint velocity+intensity models** where parameter gradients span multiple orders of magnitude. For joint models, use NumPyro instead.

Configuration reference:

```python
from kl_pipe.sampling import GradientSamplerConfig

config = GradientSamplerConfig(
    n_samples=2000,
    n_warmup=500,
    algorithm='nuts',       # or 'hmc'
    target_acceptance=0.8,
    seed=42,
)
sampler = build_sampler('blackjax', task, config)
```

See `tests/test_blackjax.py` for diagnostic patterns and known limitations.

---

## Section 5: Joint Velocity + Intensity Inference with Shear

For kinematic lensing, we jointly fit velocity and intensity maps to constrain the lensing shear.

### 5.1 Generate Joint Data

```{code-cell} python
from kl_pipe.intensity import InclinedExponentialModel
from kl_pipe.model import KLModel
from kl_pipe.synthetic import SyntheticIntensity

# True parameters including shear
true_pars_joint = {
    # Velocity
    'v0': 10.0,
    'vcirc': 200.0,
    'vel_rscale': 5.0,
    # Intensity
    'flux': 1.0,
    'int_rscale': 3.0,
    'int_h_over_r': 0.1,
    'int_x0': 0.0,
    'int_y0': 0.0,
    # Shared geometry
    'cosi': 0.6,
    'theta_int': 0.785,
    'g1': 0.03,    # Non-zero shear!
    'g2': -0.02,
}

# Generate velocity data
image_pars_vel = ImagePars(shape=(32, 32), pixel_scale=0.3, indexing='ij')
vel_pars = {k: v for k, v in true_pars_joint.items() if k in CenteredVelocityModel().PARAMETER_NAMES}
synth_vel = SyntheticVelocity(vel_pars, model_type='arctan', seed=42)
data_vel = synth_vel.generate(image_pars_vel, snr=30, include_poisson=False)
var_vel = synth_vel.variance

# Generate intensity data
image_pars_int = ImagePars(shape=(48, 48), pixel_scale=0.2, indexing='ij')
int_pars = {k: v for k, v in true_pars_joint.items() if k in InclinedExponentialModel().PARAMETER_NAMES}
synth_int = SyntheticIntensity(int_pars, model_type='exponential', seed=43)
data_int = synth_int.generate(image_pars_int, snr=30, include_poisson=False)
var_int = synth_int.variance

# Plot both datasets
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
im0 = axes[0].imshow(data_vel.T, origin='lower', cmap='RdBu_r')
axes[0].set_title('Velocity Map')
plt.colorbar(im0, ax=axes[0], label='km/s')

im1 = axes[1].imshow(data_int.T, origin='lower', cmap='viridis')
axes[1].set_title('Intensity Map')
plt.colorbar(im1, ax=axes[1], label='Flux')
plt.tight_layout()
plt.show()
```

### 5.2 Create Joint Model

The `KLModel` combines velocity and intensity models, with shared geometric parameters.

```{code-cell} python
# Create component models
vel_model = CenteredVelocityModel()
int_model = InclinedExponentialModel()

# Create joint model with shared parameters
joint_model = KLModel(
    velocity_model=vel_model,
    intensity_model=int_model,
    shared_pars={'cosi', 'theta_int', 'g1', 'g2'},  # These appear once in joint theta
)

print(f"Joint model parameters: {joint_model.PARAMETER_NAMES}")
print(f"Velocity parameters: {vel_model.PARAMETER_NAMES}")
print(f"Intensity parameters: {int_model.PARAMETER_NAMES}")
print(f"Shared parameters: {joint_model.shared_pars}")
```

### 5.3 Define Joint Priors

```{code-cell} python
priors_joint = PriorDict({
    # Velocity params
    'v0': Gaussian(10.0, 5.0),
    'vcirc': Uniform(100, 300),
    'vel_rscale': Uniform(1.0, 10.0),

    # Intensity params
    'flux': Uniform(0.1, 5.0),
    'int_rscale': Uniform(0.5, 10.0),
    'int_h_over_r': 0.1,  # Fixed
    'int_x0': 0.0,  # Fixed
    'int_y0': 0.0,  # Fixed

    # Shared geometric params
    'cosi': TruncatedNormal(0.5, 0.3, 0.1, 0.99),
    'theta_int': Uniform(0, np.pi),

    # Shear - sample these to constrain from joint fit!
    'g1': Uniform(-0.1, 0.1),
    'g2': Uniform(-0.1, 0.1),
})

print(f"Sampled parameters ({priors_joint.n_sampled}):")
for name in priors_joint.sampled_names:
    print(f"  {name}")
```

### 5.4 Create Joint InferenceTask

```{code-cell} python
task_joint = InferenceTask.from_joint_model(
    model=joint_model,
    priors=priors_joint,
    data_vel=jnp.array(data_vel),
    data_int=jnp.array(data_int),
    variance_vel=var_vel,
    variance_int=var_int,
    image_pars_vel=image_pars_vel,
    image_pars_int=image_pars_int,
)

print(f"Joint task has {task_joint.n_params} sampled parameters")
```

### 5.5 Run Joint Inference with NumPyro

For joint models, NumPyro is recommended due to its Z-score reparameterization:

```{code-cell} python
# NumPyro handles multi-scale gradients automatically
config_joint = NumpyroSamplerConfig(
    n_samples=200 if CI_MODE else 2500,
    n_warmup=100 if CI_MODE else 1250,
    n_chains=1 if CI_MODE else 4,
    dense_mass=True,
    seed=42,
    progress=not CI_MODE,
)

sampler_joint = build_sampler('numpyro', task_joint, config_joint)
result_joint = sampler_joint.run()

print_summary(result_joint, true_values=true_pars_joint)

# Check convergence
print(f"\nMax R-hat: {max(result_joint.get_rhat().values()):.4f}")
print(f"Divergences: {result_joint.diagnostics.get('n_divergences', 0)}")
```

### 5.6 Examine Shear Constraints

```{code-cell} python
# Corner plot focusing on shear parameters
fig = plot_corner(
    result_joint,
    params=['g1', 'g2', 'cosi', 'vcirc'],
    true_values=true_pars_joint,
    sampler_info={'name': 'numpyro'},
)
plt.show()

# Get shear constraints
summary = result_joint.get_summary()
print("\nShear Constraints:")
print(f"  g1 = {summary['g1']['quantiles'][0.5]:.4f} "
      f"+{summary['g1']['quantiles'][0.84] - summary['g1']['quantiles'][0.5]:.4f} "
      f"-{summary['g1']['quantiles'][0.5] - summary['g1']['quantiles'][0.16]:.4f}")
print(f"  g2 = {summary['g2']['quantiles'][0.5]:.4f} "
      f"+{summary['g2']['quantiles'][0.84] - summary['g2']['quantiles'][0.5]:.4f} "
      f"-{summary['g2']['quantiles'][0.5] - summary['g2']['quantiles'][0.16]:.4f}")
print(f"\nTrue values: g1={true_pars_joint['g1']}, g2={true_pars_joint['g2']}")
```

---

## Section 6: TNG Data Vector Integration

The TNG50 simulation provides realistic galaxy morphologies and kinematics that introduce model mismatch -- the analytic models in `kl_pipe` (arctan rotation curves, exponential disks) can't perfectly describe the complex structure of simulated galaxies. This is a feature, not a bug: it lets us test how well the inference pipeline performs under realistic conditions.

### 6.1 Load TNG Data

```{code-cell} python
try:
    from kl_pipe.tng import TNG50MockData, TNGDataVectorGenerator, TNGRenderConfig
    tng_data = TNG50MockData()
    TNG_AVAILABLE = True
except Exception:
    TNG_AVAILABLE = False
    print("TNG50 data not available. Skipping TNG sections.")
    print("Download with: make download-cyverse-data")

if TNG_AVAILABLE:
    print(f"Number of galaxies: {tng_data.n_galaxies}")
    print(f"Available SubhaloIDs: {tng_data.subhalo_ids[:10]}...")
    galaxy = tng_data.get_galaxy(subhalo_id=8)
```

### 6.2 Generate Data Vectors

```{code-cell} python
if TNG_AVAILABLE:
    # Create render configuration
    image_pars_tng = ImagePars(shape=(32, 32), pixel_scale=0.2, indexing='ij')
    render_config = TNGRenderConfig(
        image_pars=image_pars_tng,
        band='r',                       # r-band photometry
        use_native_orientation=True,    # Use TNG catalog orientation
        target_redshift=0.3,            # Scale to z=0.3 (Roman-like)
        use_cic_gridding=True,          # Cloud-in-Cell interpolation
    )

    # Generate velocity and intensity maps from particle data
    gen = TNGDataVectorGenerator(galaxy)
    velocity_map, var_vel_tng = gen.generate_velocity_map(render_config, snr=30.0, seed=42)
    intensity_map, var_int_tng = gen.generate_intensity_map(render_config, snr=30.0, seed=43)

    print(f"Velocity map shape: {velocity_map.shape}")
    print(f"Intensity map shape: {intensity_map.shape}")
    print(f"Native inclination: {gen.native_inclination_deg:.1f} deg")
    print(f"Native PA: {gen.native_pa_deg:.1f} deg")

    # Plot the TNG data
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    im0 = axes[0].imshow(velocity_map.T, origin='lower', cmap='RdBu_r')
    axes[0].set_title('TNG Velocity Map')
    plt.colorbar(im0, ax=axes[0], label='km/s')

    im1 = axes[1].imshow(intensity_map.T, origin='lower', cmap='viridis')
    axes[1].set_title('TNG Intensity Map')
    plt.colorbar(im1, ax=axes[1], label='Flux')
    plt.tight_layout()
    plt.show()
```

### 6.3 Normalize Intensity

TNG luminosities are in physical units (erg/s). For sampling with `kl_pipe`'s normalized intensity model, we need to estimate the flux normalization.

```{code-cell} python
if TNG_AVAILABLE:
    flux_estimate = float(np.sum(intensity_map))
    intensity_normalized = intensity_map / flux_estimate

    print(f"Total flux: {flux_estimate:.2e}")
    print(f"Normalized peak: {intensity_normalized.max():.4f}")
```

### 6.4 Estimate Initial Parameters

Use the TNG catalog orientation as starting estimates:

```{code-cell} python
if TNG_AVAILABLE:
    initial_pars = {
        'cosi': gen.native_cosi,
        'theta_int': gen.native_pa_rad,
        'g1': 0.0,
        'g2': 0.0,
    }

    print(f"Initial cosi: {initial_pars['cosi']:.3f}")
    print(f"Initial theta_int: {initial_pars['theta_int']:.3f} rad")
```

### 6.5 Build Joint Task and Run

```{code-cell} python
if TNG_AVAILABLE:
    priors_tng = PriorDict({
        # Velocity
        'v0': Gaussian(0.0, 20.0),
        'vcirc': Uniform(50, 400),
        'vel_rscale': Uniform(0.5, 15.0),

        # Intensity
        'flux': Uniform(0.01, 10.0),
        'int_rscale': Uniform(0.2, 10.0),
        'int_h_over_r': 0.1,
        'int_x0': 0.0,
        'int_y0': 0.0,

        # Geometry -- use TNG estimates to inform priors
        'cosi': TruncatedNormal(gen.native_cosi, 0.2, 0.05, 0.99),
        'theta_int': Uniform(0, np.pi),
        'g1': Uniform(-0.1, 0.1),
        'g2': Uniform(-0.1, 0.1),
    })

    vel_model_tng = CenteredVelocityModel()
    int_model_tng = InclinedExponentialModel()
    joint_model_tng = KLModel(
        velocity_model=vel_model_tng,
        intensity_model=int_model_tng,
        shared_pars={'cosi', 'theta_int', 'g1', 'g2'},
    )

    task_tng = InferenceTask.from_joint_model(
        model=joint_model_tng,
        priors=priors_tng,
        data_vel=jnp.array(velocity_map),
        data_int=jnp.array(intensity_normalized),
        variance_vel=var_vel_tng,
        variance_int=var_int_tng / flux_estimate**2,
        image_pars_vel=image_pars_tng,
        image_pars_int=image_pars_tng,
    )

    config_tng = NumpyroSamplerConfig(
        n_samples=200 if CI_MODE else 1250,
        n_warmup=100 if CI_MODE else 625,
        n_chains=1 if CI_MODE else 4,
        dense_mass=True,
        seed=42,
        progress=not CI_MODE,
    )

    sampler_tng = build_sampler('numpyro', task_tng, config_tng)
    result_tng = sampler_tng.run()

    print_summary(result_tng)
    print(f"\nMax R-hat: {max(result_tng.get_rhat().values()):.4f}")
    print(f"Divergences: {result_tng.diagnostics.get('n_divergences', 0)}")
```

### 6.6 Interpreting TNG Results

With TNG data there is no single "true" parameter set -- the analytic model is an approximation. Expect model mismatch: the posterior will be shifted from the TNG catalog values because the arctan rotation curve and exponential disk don't perfectly describe TNG morphology. The key question is whether shear constraints remain unbiased despite this mismatch.

```{code-cell} python
if TNG_AVAILABLE:
    tng_reference = {
        'cosi': gen.native_cosi,
        'theta_int': gen.native_pa_rad,
        'g1': 0.0,
        'g2': 0.0,
    }

    fig = plot_corner(
        result_tng,
        params=['g1', 'g2', 'cosi', 'theta_int'],
        true_values=tng_reference,
        sampler_info={'name': 'numpyro'},
    )
    plt.show()
```

---

## Section 7: Diagnostic Analysis

### 7.1 Trace Plots for Convergence

Trace plots show parameter values over the sampling iterations. Look for:
- **Stationarity**: Chains should fluctuate around a stable mean
- **Mixing**: Chains should explore the full range of the posterior
- **No trends**: Avoid drifting or slow exploration

```{code-cell} python
fig = plot_trace(result_joint)
plt.show()
```

### 7.2 Corner Plots

Corner plots reveal parameter degeneracies and correlations. The `plot_corner` function includes:

- **MAP summary box** (upper right): median +/- std for each parameter, with per-parameter sigma offsets color-coded (green <2, orange 2-3, red >3)
- **Joint N-sigma**: Mahalanobis distance from truth using posterior covariance
- **+/-1 sigma shading** on diagonal histograms (gray bands)
- **True values** (black solid lines/markers) and **MAP values** (red dashed)

```{code-cell} python
fig = plot_corner(result_joint, true_values=true_pars_joint, sampler_info={'name': 'numpyro'})
plt.show()
```

### 7.3 Parameter Recovery Assessment

```{code-cell} python
from kl_pipe.sampling.diagnostics import plot_recovery

fig, recovery_stats = plot_recovery(result_joint, true_pars_joint)
plt.show()

# The joint Nsigma statistic accounts for parameter correlations
print(f"Joint Nsigma: {recovery_stats['joint_nsigma']:.2f}")
print(f"P-value: {recovery_stats['joint_pvalue']:.4f}")
```

### 7.4 Comparing Samplers

Overlay results from different samplers to compare their posteriors using `plot_corner_comparison`. The baseline sampler (default: numpyro) is shown with filled contours; others with dashed contours clipped to the baseline region.

```{code-cell} python
from kl_pipe.sampling.diagnostics import plot_corner_comparison

# Run samplers on the same problem
results_comparison = {}

config_quick = EnsembleSamplerConfig(
    n_walkers=24, n_iterations=100 if CI_MODE else 500, burn_in=20 if CI_MODE else 100, seed=42, progress=False)
results_comparison['emcee'] = build_sampler('emcee', task, config_quick).run()

config_nautilus_quick = NestedSamplerConfig(n_live=50 if CI_MODE else 200, seed=42, progress=False)
results_comparison['nautilus'] = build_sampler('nautilus', task, config_nautilus_quick).run()

config_numpyro_quick = NumpyroSamplerConfig(
    n_samples=100 if CI_MODE else 500, n_warmup=50 if CI_MODE else 200, n_chains=1, seed=42, progress=False)
results_comparison['numpyro'] = build_sampler('numpyro', task, config_numpyro_quick).run()

fig = plot_corner_comparison(results_comparison, true_values=true_pars)
plt.show()
```

---

## Section 8: Running Tests

### Makefile targets

| Target | What it runs |
|--------|-------------|
| `make test-sampling` | Sampling diagnostics tests (excludes nautilus) |
| `make test-sampling-all` | All sampling tests including nautilus (slow) |
| `make test-tng-diagnostics` | TNG sampling end-to-end tests |
| `make test-basic` | Fast tests excluding TNG and slow |
| `make test-all` | Everything |
| `make diagnostics` | Generate HTML report from test output images |

### TestConfig pattern

Tests use a `TestConfig` container that holds output directories, SNR-dependent tolerance tables, and image parameters:

```python
from tests.test_utils import TestConfig, redirect_sampler_output
from pathlib import Path

config = TestConfig(
    output_dir=Path("tests/out"),
    enable_plots=True,
    verbose_terminal=False,
    seed=42,
)

# Redirect sampler output to log file (keeps test output clean)
log_path = config.get_sampler_log_path("my_test", "numpyro")
with redirect_sampler_output(log_path):
    result = sampler.run()
```

---

## Section 9: Choosing a Sampler

### Decision flowchart

1. **Need evidence for model comparison?** --> nautilus
2. **Joint velocity+intensity model?** --> numpyro (handles multi-scale gradients)
3. **Simple velocity-only, want fast results?** --> emcee or numpyro
4. **Multi-modal posterior?** --> emcee (gradient-free, handles multiple modes)
5. **Production run with convergence guarantees?** --> numpyro (R-hat, ESS)

### Comparison table

| Sampler | Gradients | Evidence | R-hat/ESS | Best For |
|---------|-----------|----------|-----------|----------|
| **emcee** | No | No | No | Multi-modal posteriors, easy setup, initial exploration |
| **nautilus** | No | Yes | No | Model comparison, evidence estimation |
| **numpyro** | Yes | No | Yes | Joint models, multi-scale parameters, production runs |
| **blackjax** | Yes | No | No | Simple velocity-only models (known issues with joint) |

---

## Section 10: Common Issues & Troubleshooting

| Problem | Likely Cause | Fix |
|---------|-------------|-----|
| Poor mixing (emcee) | Too few walkers or degenerate posterior | Increase `n_walkers` to >4x n_params |
| Low acceptance (emcee) | Walkers stuck in low-probability region | Check prior bounds; increase burn-in |
| Divergences (numpyro) | Numerical issues in likelihood/prior | Try `reparam_strategy='empirical'`; increase `target_accept_prob` to 0.9 |
| High R-hat (numpyro) | Chains not converged | Increase `n_warmup` and `n_samples`; check for multimodality |
| Zero variance (blackjax) | Gradient collapse in joint models | Use numpyro instead -- blackjax lacks Z-score reparam |
| Low ESS (numpyro) | Strong correlations or poor mass matrix | Use `dense_mass=True`; increase warmup |
| Slow nautilus | Too many live points or complex posterior | Reduce `n_live` for exploration; increase for final runs |
| `NaN` in log-posterior | Parameter outside model domain | Check prior bounds match model requirements |
| Hitting max tree depth (numpyro) | Step size too large or posterior geometry | Increase `max_tree_depth` to 12-15 |

---

## Section 11: Next Steps

- **Architecture reference**: `kl_pipe/sampling/README.md` -- full config reference, design patterns, adding backends
- **Test examples**: `tests/test_sampling.py`, `tests/test_numpyro.py`, `tests/test_sampling_diagnostics.py`
- **TNG integration**: `docs/tutorials/tng50_data.md` -- detailed TNG data pipeline
- **BlackJAX diagnostics**: `tests/test_blackjax.py` -- gradient diagnostic patterns
