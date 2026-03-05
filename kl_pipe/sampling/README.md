# Sampling Module

Architecture and implementation reference for the MCMC sampling infrastructure. For a practical usage guide with runnable examples, see [`docs/tutorials/sampling.md`](../../docs/tutorials/sampling.md).

---

## Module Architecture

### `kl_pipe/sampling/`

| File | Purpose |
|------|---------|
| `__init__.py` | Public API exports; imports all backends to trigger registration |
| `base.py` | `Sampler` ABC and `SamplerResult` dataclass |
| `configs.py` | Config dataclasses per sampler type, YAML loading, `parse_prior_spec()` |
| `task.py` | `InferenceTask` -- bundles model + likelihood + priors + data |
| `factory.py` | `build_sampler()` factory, `_SAMPLER_REGISTRY`, `_register_builtins()` |
| `emcee.py` | `EmceeSampler` -- ensemble MCMC (gradient-free) |
| `nautilus.py` | `NautilusSampler` -- neural nested sampling |
| `blackjax.py` | `BlackJAXSampler` -- JAX-native HMC/NUTS |
| `numpyro.py` | `NumpyroSampler` -- NUTS with Z-score reparam (RECOMMENDED) |
| `ultranest.py` | `UltraNestSampler` -- placeholder, NOT IMPLEMENTED |
| `diagnostics.py` | Trace plots, corner plots, recovery plots, sampler comparison |

### Related modules

| File | Purpose |
|------|---------|
| `kl_pipe/priors.py` | `Prior` ABC, `Uniform`, `Gaussian`, `LogUniform`, `TruncatedNormal`, `PriorDict` |
| `kl_pipe/diagnostics.py` | `compute_joint_nsigma()`, `plot_parameter_recovery()`, `plot_data_comparison_panels()` |

---

## Design Patterns

### Factory + Auto-Registration

`_SAMPLER_REGISTRY` is a module-level dict mapping string names to `Sampler` subclasses. `_register_builtins()` runs at import time and registers all five backends plus aliases:

```
emcee       -> EmceeSampler
nautilus    -> NautilusSampler
blackjax    -> BlackJAXSampler
numpyro     -> NumpyroSampler
ultranest   -> UltraNestSampler
nuts        -> NumpyroSampler  (alias)
hmc         -> NumpyroSampler  (alias)
```

Users interact via `build_sampler(name, task, config)` which does a case-insensitive lookup, creates a default config if none is provided, and returns the sampler instance.

### Prior-Based Parameter Selection

`PriorDict` separates parameters into **sampled** (have a `Prior` object) and **fixed** (numeric values). This single dict drives everything downstream:

```python
priors = PriorDict({
    'vcirc': Uniform(100, 300),  # sampled
    'cosi': Gaussian(0.5, 0.2),  # sampled
    'v0': 10.0,                   # fixed
})
# priors.sampled_names -> ['cosi', 'vcirc']  (sorted)
# priors.fixed_values  -> {'v0': 10.0}
```

Sampled parameter order is always **sorted alphabetically** -- this is the canonical theta ordering used throughout.

### JIT-Compatible Posterior

`InferenceTask._build_full_theta()` maps the sampled-params-only theta array back to the full model parameter space using pre-computed index arrays and `jnp.ndarray.at[].set()`. This is fully JIT-compatible.

`_log_posterior_jittable()` uses `jnp.where(jnp.isfinite(log_prior), ...)` to avoid branching on -inf prior values, which would break JIT tracing.

### Z-Score Reparameterization (NumPyro)

The NumPyro backend samples in a standardized latent space where all parameters are O(1). For each parameter:

```
z ~ Normal(0, 1)           # latent variable
theta = loc + scale * z    # physical parameter
```

The `numpyro.factor("log_posterior", log_post_fn(theta))` call adds the full log-posterior (prior + likelihood) to the model. The latent Normal(0,1) is purely for numerical conditioning -- it is NOT the prior.

Per-prior-type scaling rules (`compute_reparam_scales()`):

| Prior Type | loc | scale |
|------------|-----|-------|
| `Gaussian(mu, sigma)` | `mu` | `sigma` |
| `TruncatedNormal(mu, sigma, ...)` | `mu` | `sigma` |
| `Uniform(low, high)` | `(low+high)/2` | `(high-low)/4` |
| `LogUniform(low, high)` | `exp(log_mid)` | `exp(log_mid) * log_scale` |

---

## Sampler Reference

| Name | Config Class | Gradients | Evidence | Multi-chain | R-hat/ESS | Status |
|------|-------------|-----------|----------|-------------|-----------|--------|
| `emcee` | `EnsembleSamplerConfig` | No | No | No | No | Stable |
| `nautilus` | `NestedSamplerConfig` | No | Yes | No | No | Stable |
| `blackjax` | `GradientSamplerConfig` | Yes | No | No | No | KNOWN ISSUES with joint models |
| **`numpyro`** | **`NumpyroSamplerConfig`** | **Yes** | **No** | **Yes** | **Yes** | **RECOMMENDED** |
| `ultranest` | `NestedSamplerConfig` | No | Yes | No | No | NOT IMPLEMENTED |

Aliases: `nuts` and `hmc` both resolve to `NumpyroSampler`.

---

## Quick Start

### Minimal emcee example

```python
from kl_pipe.velocity import CenteredVelocityModel
from kl_pipe.priors import Uniform, Gaussian, TruncatedNormal, PriorDict
from kl_pipe.sampling import InferenceTask, EnsembleSamplerConfig, build_sampler

priors = PriorDict({
    'vcirc': Uniform(100, 350),
    'cosi': TruncatedNormal(0.5, 0.2, 0.1, 0.99),
    'theta_int': Uniform(0, 3.14159),
    'g1': Gaussian(0, 0.05),
    'g2': Gaussian(0, 0.05),
    'v0': 10.0,
    'vel_rscale': 5.0,
    'vel_x0': 0.0,
    'vel_y0': 0.0,
})

task = InferenceTask.from_velocity_model(
    model=CenteredVelocityModel(),
    priors=priors,
    data_vel=observed_velocity,
    variance_vel=25.0,
    image_pars=image_pars,
)

config = EnsembleSamplerConfig(n_walkers=64, n_iterations=5000, burn_in=1000, seed=42)
result = build_sampler('emcee', task, config).run()

summary = result.get_summary()
print(f"vcirc = {summary['vcirc']['quantiles'][0.5]:.1f} km/s")
```

### NumPyro joint model example

```python
from kl_pipe.model import KLModel
from kl_pipe.sampling import NumpyroSamplerConfig, build_sampler

config = NumpyroSamplerConfig(
    n_samples=2000, n_warmup=1000, n_chains=4,
    dense_mass=True, reparam_strategy='prior', seed=42,
)
result = build_sampler('numpyro', task_joint, config).run()

# Check convergence
print(f"Max R-hat: {max(result.get_rhat().values()):.4f}")
print(f"Divergences: {result.diagnostics['n_divergences']}")
```

---

## Config Reference

### `BaseSamplerConfig`

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `seed` | `Optional[int]` | `None` | Random seed. None = system entropy |
| `progress` | `bool` | `True` | Show progress bar |

### `EnsembleSamplerConfig(BaseSamplerConfig)`

| Field | Type | Default | Validation |
|-------|------|---------|------------|
| `n_walkers` | `int` | `32` | >= 2 |
| `n_iterations` | `int` | `2000` | > burn_in |
| `burn_in` | `int` | `500` | < n_iterations |
| `thin` | `int` | `1` | >= 1 |
| `moves` | `Optional[Any]` | `None` | Custom emcee move proposals |
| `vectorize` | `bool` | `False` | Vectorize log_prob over walkers |

### `NestedSamplerConfig(BaseSamplerConfig)`

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `n_live` | `int` | `500` | Number of live points |
| `n_networks` | `int` | `4` | Neural network ensemble size (nautilus) |
| `verbose` | `bool` | `True` | Detailed progress output |
| `log_dir` | `Optional[str]` | `None` | Checkpoint directory (ultranest) |
| `resume` | `bool` | `False` | Resume from checkpoint (ultranest) |

### `GradientSamplerConfig(BaseSamplerConfig)`

| Field | Type | Default | Validation |
|-------|------|---------|------------|
| `n_samples` | `int` | `2000` | Samples after warmup |
| `n_warmup` | `int` | `500` | Warmup iterations |
| `max_tree_depth` | `int` | `10` | Max NUTS tree depth |
| `target_acceptance` | `float` | `0.8` | In (0, 1) |
| `step_size` | `float` | `0.1` | Initial step size |
| `num_integration_steps` | `int` | `10` | Leapfrog steps (HMC only) |
| `algorithm` | `str` | `'nuts'` | `'nuts'` or `'hmc'` |

### `NumpyroSamplerConfig(BaseSamplerConfig)`

| Field | Type | Default | Validation |
|-------|------|---------|------------|
| `n_samples` | `int` | `2000` | Samples per chain |
| `n_warmup` | `int` | `1000` | Warmup iterations |
| `n_chains` | `int` | `4` | >= 1 |
| `dense_mass` | `bool` | `True` | Dense vs diagonal mass matrix |
| `max_tree_depth` | `int` | `10` | Max NUTS tree depth |
| `target_accept_prob` | `float` | `0.8` | In (0, 1) |
| `reparam_strategy` | `ReparamStrategy` | `PRIOR` | `'none'`, `'prior'`, `'empirical'` |
| `empirical_warmup_frac` | `float` | `0.1` | Fraction of warmup for empirical |
| `chain_method` | `str` | `'sequential'` | `'sequential'`, `'parallel'`, `'vectorized'` |
| `save_warmup` | `bool` | `False` | Save warmup samples |
| `save_mass_matrix` | `bool` | `False` | Save adapted inverse mass matrix |
| `init_strategy` | `str` | `'prior'` | `'prior'`, `'median'`, `'jitter'` |

### `ReparamStrategy` Enum

| Value | Description |
|-------|-------------|
| `NONE` | Sample in physical space (no transform) |
| `PRIOR` | Z-score using prior mean/std (default, fast) |
| `EMPIRICAL` | Estimate scales from short warmup (slower, robust) |

---

## InferenceTask API

### Properties

| Property | Returns | Description |
|----------|---------|-------------|
| `parameter_names` | `Tuple[str, ...]` | Full model parameter names |
| `sampled_names` | `list` | Names of sampled parameters (sorted) |
| `n_params` | `int` | Number of sampled parameters |
| `fixed_params` | `Dict[str, float]` | Fixed parameter values |

### Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `log_likelihood` | `(theta_sampled) -> float` | Log-likelihood for sampled params |
| `log_prior` | `(theta_sampled) -> float` | Log-prior probability |
| `log_posterior` | `(theta_sampled) -> float` | Log-posterior (non-JIT) |
| `get_log_posterior_fn` | `() -> Callable` | JIT-compiled log-posterior |
| `get_log_posterior_and_grad_fn` | `() -> Callable` | JIT-compiled (log_post, grad) |
| `get_bounds` | `() -> list[(low, high)]` | Parameter bounds from priors |
| `sample_prior` | `(rng_key, n_samples) -> jnp.ndarray` | Draw prior samples |

### Factory Methods

```python
InferenceTask.from_velocity_model(model, priors, data_vel, variance_vel, image_pars, meta_pars=None)
InferenceTask.from_intensity_model(model, priors, data_int, variance_int, image_pars, meta_pars=None)
InferenceTask.from_joint_model(model, priors, data_vel, data_int, variance_vel, variance_int,
                               image_pars_vel, image_pars_int, meta_pars=None)
```

---

## SamplerResult Reference

### Core fields (always present)

| Field | Type | Description |
|-------|------|-------------|
| `samples` | `np.ndarray (n_samples, n_params)` | Posterior samples |
| `log_prob` | `np.ndarray (n_samples,)` | Log-posterior per sample |
| `param_names` | `List[str]` | Sampled parameter names |
| `fixed_params` | `Dict[str, float]` | Fixed parameter values |

### Optional fields

| Field | Type | Description |
|-------|------|-------------|
| `evidence` | `Optional[float]` | Log-evidence (nested samplers) |
| `evidence_error` | `Optional[float]` | Evidence uncertainty |
| `chains` | `Optional[np.ndarray]` | Pre-flattening chains (ensemble) |
| `blobs` | `Optional[np.ndarray]` | Per-sample extra data |
| `acceptance_fraction` | `Optional[float]` | Mean acceptance rate |
| `autocorr_time` | `Optional[np.ndarray]` | Autocorrelation times |
| `converged` | `bool` | Backend-specific convergence flag |
| `diagnostics` | `Dict[str, Any]` | Backend-specific diagnostics |
| `metadata` | `Dict[str, Any]` | Timing, backend name, etc. |

### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `get_chain(name)` | `np.ndarray` | Samples for one parameter |
| `get_summary(quantiles=(0.16,0.5,0.84))` | `dict` | Mean, std, quantiles per param |
| `get_rhat(param_name=None)` | `float` or `dict` or `None` | R-hat convergence diagnostic |
| `get_ess(param_name=None)` | `float` or `dict` or `None` | Effective sample size |
| `to_dict(include_samples=True)` | `dict` | Serializable representation |

### Per-backend `diagnostics` dict contents

**emcee:**
```python
{'backend': 'emcee', 'n_walkers': int, 'n_iterations': int}
```

**nautilus:**
```python
{'backend': 'nautilus', 'n_live': int, 'n_eff': int}
# evidence in result.evidence; evidence_error: TODO (not currently exposed)
```

**numpyro:**
```python
{
    'diverging': np.ndarray,           # per-sample divergence flags
    'n_divergences': int,
    'divergence_rate': float,
    'accept_prob': np.ndarray,         # per-sample acceptance probabilities
    'mean_accept_prob': float,
    'num_steps': np.ndarray,           # leapfrog steps per sample
    'mean_tree_depth': float,
    'step_size': float,                # adapted step size
    'r_hat': Dict[str, float],         # per-param R-hat
    'ess': Dict[str, float],           # per-param effective sample size
    'reparam_strategy': str,
    'reparam_scales': Dict[str, (float, float)],
    'inverse_mass_matrix': np.ndarray, # only if save_mass_matrix=True
}
```

**blackjax:**
```python
{'backend': 'blackjax', 'algorithm': str, 'n_warmup': int}
```

---

## Diagnostics Functions

### `kl_pipe/sampling/diagnostics.py`

| Function | Signature | Description |
|----------|-----------|-------------|
| `check_convergence_warnings` | `(result) -> dict` | Detect zero-variance, low acceptance, small n |
| `add_convergence_annotation` | `(fig, warnings, position='bottom')` | Annotate figure with warnings |
| `plot_trace` | `(result, params=None, figsize=(12,3), output_path=None) -> Figure` | Trace plots per parameter |
| `plot_corner` | `(result, params=None, true_values=None, map_values=None, output_path=None, ...) -> Figure` | Corner plot with MAP box, sigma annotations, joint Nsigma |
| `plot_corner_comparison` | `(results, params=None, true_values=None, output_path=None, ...) -> Figure` | Overlaid corner comparing multiple samplers |
| `plot_recovery` | `(result, true_values, output_path=None, ...) -> (Figure, dict)` | Two-panel recovery plot with joint Nsigma |
| `print_summary` | `(result, true_values=None)` | Print tabular summary to stdout |

### `kl_pipe/diagnostics.py`

| Function | Signature | Description |
|----------|-----------|-------------|
| `compute_joint_nsigma` | `(recovered, true_values, covariance=None, samples=None, param_names=None) -> dict` | Mahalanobis distance in posterior covariance |
| `nsigma_to_color` | `(nsigma) -> str` | Green (<2), orange (2-3), red (>3) |
| `plot_data_comparison_panels` | `(data_noisy, data_true, model_eval, test_name, output_dir, ...) -> Path` | 2x3 panel data comparison |
| `plot_combined_data_comparison` | `(vel_data..., int_data..., test_name, output_dir, ...) -> Path` | 4x3 panel joint data comparison |
| `plot_parameter_recovery` | `(true_values, recovered_values, output_dir, test_name, ...) -> dict` | Two-panel recovery plot (for optimizers and samplers) |

---

## Adding a New Sampler

### Step 1: Create `kl_pipe/sampling/mysampler.py`

```python
from kl_pipe.sampling.base import Sampler, SamplerResult
from kl_pipe.sampling.configs import BaseSamplerConfig  # or custom config

class MySampler(Sampler):
    requires_gradients = False
    provides_evidence = False
    config_class = BaseSamplerConfig  # or your custom config class

    def run(self) -> SamplerResult:
        log_prob_fn = self.task.get_log_posterior_fn()
        # ... run your sampler ...
        return SamplerResult(
            samples=samples,
            log_prob=log_probs,
            param_names=self.task.sampled_names,
            fixed_params=self.task.fixed_params,
            diagnostics={'backend': 'mysampler'},
            metadata={'sampler': 'mysampler'},
        )
```

### Step 2: Add a config class (if needed)

In `configs.py`, add a dataclass inheriting from `BaseSamplerConfig` with sampler-specific fields and `__post_init__` validation.

### Step 3: Register in `factory.py`

Add to `_register_builtins()`:

```python
from kl_pipe.sampling.mysampler import MySampler
register_sampler('mysampler', MySampler)
```

### Step 4: Update `__init__.py`

Add the import and include in `__all__`:

```python
from kl_pipe.sampling.mysampler import MySampler
```

### Step 5: Write tests

Create `tests/test_mysampler.py` with at minimum:
- Config validation tests
- Velocity-only parameter recovery
- Joint model parameter recovery (if applicable)
- Convergence diagnostic checks

### Step 6: Add YAML support

In `configs.py` `SamplingYAMLConfig.get_sampler_config()`, add a branch for your sampler name mapping to the config class.

---

## YAML Configuration

### Schema

```yaml
model_type: velocity | intensity | joint
velocity_model: offset | centered
intensity_model: inclined_exp
shared_params: [cosi, theta_int, g1, g2]  # joint only

priors:
  vcirc:
    type: uniform
    low: 100
    high: 300
  cosi:
    type: truncated_normal
    mu: 0.5
    sigma: 0.2
    low: 0.1
    high: 0.99
  v0: 10.0  # fixed

sampler: numpyro
sampler_config:
  n_samples: 2000
  n_warmup: 1000
  n_chains: 4
  dense_mass: true
  reparam_strategy: prior
  seed: 42

data:
  velocity_path: /path/to/velocity.fits
  variance: 25.0
```

### `parse_prior_spec()` supported types

| YAML `type` | Class | Aliases |
|-------------|-------|---------|
| `uniform` | `Uniform` | -- |
| `gaussian` | `Gaussian` | `normal` |
| `loguniform` | `LogUniform` | `log_uniform` |
| `truncated_normal` | `TruncatedNormal` | `truncatednormal` |

Numeric values (int/float) are treated as fixed parameters.

### Loading

```python
from kl_pipe.sampling.configs import SamplingYAMLConfig

config = SamplingYAMLConfig.from_yaml('config.yaml')
priors = config.get_prior_dict()
sampler_config = config.get_sampler_config()
```

---

## Testing Infrastructure

### Test file map

| File | Markers | What it tests |
|------|---------|---------------|
| `tests/test_sampling.py` | -- | Config validation, factory, basic emcee/nautilus runs |
| `tests/test_numpyro.py` | -- | NumPyro velocity + joint recovery, reparam strategies, convergence |
| `tests/test_blackjax.py` | -- | BlackJAX velocity recovery, gradient diagnostics |
| `tests/test_sampling_diagnostics.py` | `slow` (nautilus) | Multi-sampler comparison, corner/trace/recovery plots |
| `tests/test_tng_sampling_diagnostics.py` | `tng_diagnostics` | TNG data vector + sampling end-to-end |
| `tests/test_priors.py` | -- | Prior log_prob, sampling, PriorDict |
| `tests/conftest.py` | -- | Warning suppression (emcee chain length, JAXopt deprecation) |

### Makefile targets

| Target | What it runs |
|--------|-------------|
| `make test-sampling` | `test_sampling_diagnostics.py` (excludes nautilus) |
| `make test-sampling-all` | `test_sampling_diagnostics.py` with `INCLUDE_NAUTILUS=1` |
| `make test-tng-diagnostics` | `-m tng_diagnostics` tests |
| `make test-basic` | All tests except TNG and slow |
| `make test` | All tests except slow and tng_diagnostics |
| `make test-all` | Everything |
| `make diagnostics` | Generate HTML report from test output images |

### Key test utilities (`tests/test_utils.py`)

**`TestConfig`** -- configuration container for parameter recovery tests. Holds output directories, tolerance tables (SNR-dependent relative + absolute), parameter bounds, and image parameters.

```python
config = TestConfig(
    output_dir=Path("tests/out"),
    enable_plots=True,
    verbose_terminal=False,
    seed=42,
)
```

**`redirect_sampler_output(log_path, also_terminal=False)`** -- context manager that captures sampler stdout to a file, optionally teeing to terminal.

```python
with redirect_sampler_output(Path("tests/out/emcee.log")):
    result = sampler.run()
```

---

## Dependencies

- `emcee>=3.1` -- Ensemble sampler
- `nautilus-sampler>=1.0` -- Neural nested sampler
- `blackjax>=1.0` -- JAX-native gradient samplers
- `numpyro>=0.13` -- Probabilistic programming with JAX
- `jax`, `jaxlib` -- Autodiff and JIT compilation
- `corner` -- Corner plots
- `matplotlib` -- All plotting
- `scipy` -- Chi-squared statistics in diagnostics
