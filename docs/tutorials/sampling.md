---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Bayesian Inference with MCMC Sampling

Posterior inference with the `kl_pipe` sampling module, built on the
`SourceModel` + `InferenceTask` + `PriorDict` stack from `quickstart.md`.

> **TL;DR:** emcee for exploration, nautilus for evidence / model comparison,
> **numpyro for production** (gradients, R-hat / ESS, multi-chain), blackjax for
> simple velocity-only problems. For the joint broadband + grism configuration,
> use numpyro with the **Laplace preconditioner** (Section 6).

```{code-cell} python
import os
CI_MODE = os.environ.get('KL_PIPE_CI', '0') == '1'

import jax
jax.config.update('jax_enable_x64', True)
import numpy as np
import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt

from kl_pipe.parameters import ImagePars
from kl_pipe.velocity import CenteredVelocityModel
from kl_pipe.intensity import InclinedExponentialModel
from kl_pipe.source import SourceModel
from kl_pipe.observation import build_velocity_obs, build_image_obs, build_grism_obs
from kl_pipe.noise import add_velocity_noise, add_intensity_noise
from kl_pipe.priors import PriorDict, Uniform, Gaussian, TruncatedNormal
from kl_pipe.sampling import (
    InferenceTask,
    EnsembleSamplerConfig,
    NestedSamplerConfig,
    GradientSamplerConfig,
    NumpyroSamplerConfig,
    build_sampler,
    get_available_samplers,
)
from kl_pipe.sampling.diagnostics import (
    plot_corner, plot_trace, print_summary, plot_recovery,
)

print("Available samplers:", get_available_samplers())
```

## Design philosophy

1. **InferenceTask** bundles source, likelihood, priors, and data; exposes a
   JIT-compiled, differentiable log-posterior.
2. **PriorDict** separates sampled parameters (with `Prior` objects) from fixed
   ones (numeric); sampled names sort alphabetically into the sampling vector.
3. **Factory**: `build_sampler(name, task, config)` returns a unified interface.
4. **SamplerResult** is the common output across all backends.

Parameters use the dotted-key namespace (`cosi`, `vel.vcirc`, `F087.flux`,
`Halpha.dispersion`); see `quickstart.md`.

---

## Section 1: Velocity-only inference with emcee

### Generate synthetic data

We render a velocity field from a velocity-only `SourceModel` and add noise.

```{code-cell} python
vsource = SourceModel(velocity_model=CenteredVelocityModel())
true_vel = {
    'cosi': 0.6,
    'theta_int': 0.785,
    'g1': 0.0,
    'g2': 0.0,
    'vel.v0': 10.0,
    'vel.vcirc': 200.0,
    'vel.rscale': 2.0,
}
image_pars = ImagePars(shape=(32, 32), pixel_scale=0.3, indexing='ij')

v_true = np.asarray(vsource.render_velocity(true_vel, build_velocity_obs(image_pars)))
v_noisy, v_var = add_velocity_noise(v_true, target_snr=20, seed=42)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, img, title in [(axes[0], v_true, 'true velocity'),
                       (axes[1], v_noisy, 'noisy (SNR=20)')]:
    vmax = float(np.max(np.abs(v_true)))
    im = ax.imshow(img, origin='lower', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax.set_title(title); plt.colorbar(im, ax=ax, label='km/s')
plt.tight_layout(); plt.show()
```

### Priors and InferenceTask

```{code-cell} python
priors = PriorDict({
    'cosi': TruncatedNormal(0.5, 0.3, 0.1, 0.99),
    'theta_int': Uniform(0.0, np.pi),
    'g1': 0.0,
    'g2': 0.0,
    'vel.v0': Gaussian(10.0, 5.0),
    'vel.vcirc': Uniform(100.0, 300.0),
    'vel.rscale': Uniform(0.5, 8.0),
})
print(f"sampled: {priors.sampled_names}   fixed: {priors.fixed_names}")

vel_obs = build_velocity_obs(image_pars, data=jnp.asarray(v_noisy), variance=v_var)
task = InferenceTask.from_obs(vsource, priors, velocity_obs=vel_obs)

key = random.PRNGKey(42)
theta0 = task.sample_prior(key, 1)[0]
print(f"log-posterior at a prior draw: {float(task.log_posterior(theta0)):.2f}")
```

### Configure and run emcee

```{code-cell} python
config_emcee = EnsembleSamplerConfig(
    n_walkers=32,
    n_iterations=200 if CI_MODE else 2000,
    burn_in=50 if CI_MODE else 500,
    seed=42,
    progress=not CI_MODE,
)
result_emcee = build_sampler('emcee', task, config_emcee).run()
print(f"samples: {result_emcee.n_samples}   acceptance: {result_emcee.acceptance_fraction:.1%}")
```

```{code-cell} python
print_summary(result_emcee, true_values=true_vel)
fig = plot_corner(result_emcee, true_values=true_vel, sampler_info={'name': 'emcee'})
plt.show()
fig = plot_trace(result_emcee)
plt.show()
```

---

## Section 2: Nested sampling with nautilus

Nautilus uses neural networks to explore the posterior and returns the Bayesian
evidence for model comparison. It reuses the same `task`.

```{code-cell} python
config_nautilus = NestedSamplerConfig(
    n_live=100 if CI_MODE else 500, n_networks=4, seed=42, progress=not CI_MODE,
)
result_nautilus = build_sampler('nautilus', task, config_nautilus).run()
print_summary(result_nautilus, true_values=true_vel)
if result_nautilus.evidence is not None:
    print(f"log evidence: {result_nautilus.evidence:.2f}")
```

The log evidence `Z` gives Bayes factors for model comparison: for models with
evidences `Z1`, `Z2`, the Bayes factor is `exp(log Z1 - log Z2)`.

---

## Section 3: Gradient-based NUTS with NumPyro (recommended)

NumPyro's NUTS uses the gradients JAX provides, adapts a (dense) mass matrix, and
reports R-hat / ESS across chains. Its **Z-score reparameterization** rescales
each parameter by its prior so the sampler sees O(1) gradients, which is what
makes it robust when parameters span very different scales (intensity ~1e2-1e4,
velocity ~1e2, shear ~1e-2).

```{code-cell} python
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
result_numpyro = build_sampler('numpyro', task, config_numpyro).run()
print(f"samples: {result_numpyro.n_samples}   acceptance: {result_numpyro.acceptance_fraction:.1%}")
fig = plot_corner(result_numpyro, true_values=true_vel, sampler_info={'name': 'numpyro'})
plt.show()
```

### Convergence diagnostics

```{code-cell} python
for name, rhat in result_numpyro.get_rhat().items():
    print(f"  R-hat {name:<12}: {rhat:.4f} {'OK' if rhat < 1.01 else 'WARN'}")
print("min ESS:", f"{min(result_numpyro.get_ess().values()):.0f}")
print("divergences:", result_numpyro.diagnostics.get('n_divergences', 0))
```

### Reparameterization strategies

`reparam_strategy` controls the Z-score scaling: `'prior'` (default, uses prior
mean/std), `'empirical'` (a short warmup estimates posterior scales; more robust,
slower), or `'none'` (sample in physical space; only if parameters are already
well-scaled). Set it on `NumpyroSamplerConfig(reparam_strategy=...)`.

---

## Section 4: BlackJAX

BlackJAX provides JAX-native HMC / NUTS via `GradientSamplerConfig`. It works for
simple velocity-only problems but has known issues on joint velocity+intensity
models where gradients span many orders of magnitude (it lacks the Z-score
reparam); use NumPyro there.

```python
config = GradientSamplerConfig(n_samples=2000, n_warmup=500,
                               algorithm='nuts', target_acceptance=0.8, seed=42)
result = build_sampler('blackjax', task, config).run()
```

See `tests/test_blackjax.py` for diagnostics and the known limitations.

---

## Section 5: Joint velocity + intensity with shear

Jointly fitting a velocity map and a broadband image constrains the lensing
shear. The source carries a velocity model and a broadband band sharing the
geometry; `from_obs` takes both a `velocity_obs` and an `image_obs`.

```{code-cell} python
joint_source = SourceModel(
    velocity_model=CenteredVelocityModel(),
    broadband_models={'F087': InclinedExponentialModel()},
)
true_joint = {
    'cosi': 0.6,
    'theta_int': 0.785,
    'g1': 0.03,            # non-zero shear
    'g2': -0.02,
    'vel.v0': 10.0,
    'vel.vcirc': 200.0,
    'vel.rscale': 2.0,
    'F087.flux': 100.0,
    'F087.rscale': 0.6,
    'F087.h_over_r': 0.1,
    'F087.x0': 0.0,
    'F087.y0': 0.0,
}
ip_vel = ImagePars(shape=(32, 32), pixel_scale=0.3, indexing='ij')
ip_img = ImagePars(shape=(48, 48), pixel_scale=0.2, indexing='ij')
f087 = joint_source.broadband_models['F087']

v_t = np.asarray(joint_source.render_velocity(true_joint, build_velocity_obs(ip_vel)))
v_n, v_v = add_velocity_noise(v_t, target_snr=30, seed=42)
img_clean = np.asarray(joint_source.render_broadband(
    true_joint, build_image_obs(ip_img, broadband_key='F087', int_model=f087), 'F087'))
i_n, i_v = add_intensity_noise(img_clean, target_snr=30, seed=43)

obs_vel_j = build_velocity_obs(ip_vel, data=jnp.asarray(v_n), variance=v_v)
obs_img_j = build_image_obs(ip_img, broadband_key='F087', int_model=f087,
                            data=jnp.asarray(i_n), variance=i_v)
```

```{code-cell} python
# Shear is sampled; the vcirc prior is the Tully-Fisher prior (see quickstart
# Example 4) that pins inclination so shape and shear separate.
priors_joint = PriorDict({
    'cosi': TruncatedNormal(0.5, 0.3, 0.1, 0.99),
    'theta_int': Uniform(0.0, np.pi),
    'g1': Uniform(-0.1, 0.1),
    'g2': Uniform(-0.1, 0.1),
    'vel.v0': Gaussian(10.0, 5.0),
    'vel.vcirc': TruncatedNormal(200.0, 37.0, 100.0, 300.0),
    'vel.rscale': Uniform(0.5, 8.0),
    'F087.flux': Uniform(10.0, 500.0),
    'F087.rscale': Uniform(0.1, 2.0),
    'F087.h_over_r': 0.1,
    'F087.x0': 0.0,
    'F087.y0': 0.0,
})
task_joint = InferenceTask.from_obs(
    joint_source, priors_joint, velocity_obs=obs_vel_j, image_obs={'F087': obs_img_j},
)
print(f"joint task: {task_joint.n_params} sampled params, "
      f"intensity oversample={obs_img_j.render_config.oversample}")

config_joint = NumpyroSamplerConfig(
    n_samples=200 if CI_MODE else 2000,
    n_warmup=100 if CI_MODE else 1000,
    n_chains=1 if CI_MODE else 4,
    dense_mass=True, seed=42, progress=not CI_MODE,
)
result_joint = build_sampler('numpyro', task_joint, config_joint).run()
print_summary(result_joint, true_values=true_joint)
print(f"max R-hat: {max(result_joint.get_rhat().values()):.4f}")
```

```{code-cell} python
fig = plot_corner(result_joint, params=['g1', 'g2', 'cosi', 'vel.vcirc'],
                  true_values=true_joint, sampler_info={'name': 'numpyro'})
plt.show()
```

---

## Section 6: Grism and joint photometry + grism (the production configuration)

The Roman kinematic-lensing target is a broadband image fit jointly with a
slitless grism roll. The source adds an emission line; `from_obs` takes
`image_obs` and `grism_obs` dicts. See `grism.md` for the forward model.

```{code-cell} python
from kl_pipe.lines import EmissionLine, LINE_LAMBDAS
from kl_pipe.dispersion import build_grism_pars_for_line
from kl_pipe.render import RenderConfig
import galsim

Z = 1.0
ip = ImagePars(shape=(28, 28), pixel_scale=0.11, indexing='ij')
psf = galsim.Gaussian(fwhm=0.18)              # Gaussian stand-in, Roman-like FWHM
rc = RenderConfig(oversample=3)               # low for tutorial speed

f087 = InclinedExponentialModel()
halpha = InclinedExponentialModel()
src = SourceModel(
    velocity_model=CenteredVelocityModel(),
    broadband_models={'F087': f087},
    emission_lines={'Halpha': EmissionLine(intensity=halpha)},
)
truth = {
    'cosi': 0.6,
    'theta_int': 0.785,
    'g1': 0.02,
    'g2': -0.01,
    'vel.v0': 10.0,
    'vel.vcirc': 200.0,
    'vel.rscale': 0.3,
    'F087.flux': 100.0,
    'F087.rscale': 0.3,
    'F087.h_over_r': 0.1,
    'F087.x0': 0.0,
    'F087.y0': 0.0,
    'Halpha.flux': 100.0,
    'Halpha.rscale': 0.25,
    'Halpha.h_over_r': 0.1,
    'Halpha.x0': 0.0,
    'Halpha.y0': 0.0,
    'Halpha.dispersion': 50.0,
    'z': Z,
}

gp = build_grism_pars_for_line(LINE_LAMBDAS['Halpha'], redshift=Z, image_pars=ip, dispersion=1.1)
img_clean = np.asarray(src.render_broadband(
    truth, build_image_obs(ip, psf=psf, render_config=rc, int_model=f087, broadband_key='F087'), 'F087'))
g_clean = np.asarray(src.render_grism(truth, build_grism_obs(gp, z=Z, psf=psf, render_config=rc)))
i_n, i_v = add_intensity_noise(img_clean, target_snr=100, seed=1)
g_n, g_v = add_intensity_noise(g_clean, target_snr=100, seed=2)

obs_F087 = build_image_obs(ip, psf=psf, render_config=rc, int_model=f087, broadband_key='F087',
                           data=jnp.asarray(i_n), variance=i_v)
obs_grism = build_grism_obs(gp, z=Z, psf=psf, render_config=rc,
                            data=jnp.asarray(g_n), variance=g_v)

priors = PriorDict({
    'cosi': TruncatedNormal(0.6, 0.15, 0.05, 0.99),
    'theta_int': TruncatedNormal(0.785, 0.3, 0.0, np.pi / 2),
    'g1': Uniform(-0.25, 0.25),
    'g2': Uniform(-0.25, 0.25),
    'vel.v0': Gaussian(10.0, 10.0),
    'vel.vcirc': TruncatedNormal(200.0, 37.0, 80.0, 400.0),    # TF prior
    'vel.rscale': TruncatedNormal(0.3, 0.1, 0.05, 1.0),
    'F087.flux': TruncatedNormal(100.0, 20.0, 30.0, 250.0),
    'F087.rscale': TruncatedNormal(0.3, 0.08, 0.05, 1.0),
    'F087.h_over_r': 0.1,
    'F087.x0': TruncatedNormal(0.0, 0.1, -0.5, 0.5),
    'F087.y0': TruncatedNormal(0.0, 0.1, -0.5, 0.5),
    'Halpha.flux': TruncatedNormal(100.0, 20.0, 30.0, 250.0),
    'Halpha.rscale': TruncatedNormal(0.25, 0.08, 0.05, 1.0),
    'Halpha.h_over_r': 0.1,
    'Halpha.x0': TruncatedNormal(0.0, 0.1, -0.5, 0.5),
    'Halpha.y0': TruncatedNormal(0.0, 0.1, -0.5, 0.5),
    'Halpha.dispersion': TruncatedNormal(50.0, 20.0, 5.0, 150.0),
    'z': Z,
})
task_pg = InferenceTask.from_obs(src, priors, image_obs={'F087': obs_F087},
                                 grism_obs={'roll0': obs_grism})
print(f"joint phot+grism task: {task_pg.n_params} sampled params")
```

### The Laplace preconditioner

This joint posterior is correlated and the gradient is dominated by the grism
`build_cube` cost, so NUTS warmup (climbing from an identity mass matrix) is the
bottleneck. The **Laplace preconditioner** starts NUTS at the MAP with a fixed
inverse-Hessian mass matrix, so warmup only tunes the step size. On the flagship
test this is ~5x faster (8.5 vs ~42 min) with equal recovery and better
convergence. Enable it with `precondition='laplace'` and a short warmup:

```{code-cell} python
config_laplace = NumpyroSamplerConfig(
    n_samples=150 if CI_MODE else 500,
    n_warmup=30 if CI_MODE else 50,        # short: the MAP + Laplace metric do the work
    n_chains=1 if CI_MODE else 2,
    precondition='laplace',                # MAP init + fixed Laplace mass matrix
    n_map_starts=2 if CI_MODE else 4,
    chain_method='vectorized',
    seed=42, progress=not CI_MODE,
)
result_pg = build_sampler('numpyro', task_pg, config_laplace).run()
print(f"max R-hat: {max(result_pg.get_rhat().values()):.4f}")
print(f"min ESS:   {min(result_pg.get_ess().values()):.0f}")
```

```{code-cell} python
fig = plot_corner(result_pg, params=['g1', 'g2', 'cosi', 'vel.vcirc', 'Halpha.dispersion'],
                  true_values=truth, sampler_info={'name': 'numpyro+laplace'})
plt.show()
```

The plain dense-mass path (`precondition='none'`, the default) is the stable
anchor; switch to `'laplace'` with a short warmup for the joint phot+grism
configuration. `tests/test_flagship.py` runs the full production version.

---

## Section 7: TNG data-vector integration

TNG50 galaxies provide realistic morphologies and kinematics that the analytic
models cannot perfectly describe, which tests inference under model mismatch.
The cell below is tagged skip-execution (it needs the CyVerse TNG data, absent in
CI); run it locally after `make download-cyverse-data`. The data-vector pipeline
itself is covered in `tng50_data.md`.

```{code-cell} python
:tags: [skip-execution]

from kl_pipe.tng import TNG50MockData, TNGDataVectorGenerator, TNGRenderConfig

galaxy = TNG50MockData().get_galaxy(subhalo_id=8)
gen = TNGDataVectorGenerator(galaxy)
ip_tng = ImagePars(shape=(32, 32), pixel_scale=0.2, indexing='ij')
rcfg = TNGRenderConfig(image_pars=ip_tng, band='r', use_native_orientation=True,
                       target_redshift=0.3, use_cic_gridding=True)
vel_map, v_var = gen.generate_velocity_map(rcfg, snr=30.0, seed=42)
int_map, i_var = gen.generate_intensity_map(rcfg, snr=30.0, seed=43)
flux_est = float(np.sum(int_map))
int_norm = int_map / flux_est

src_tng = SourceModel(velocity_model=CenteredVelocityModel(),
                      broadband_models={'r': InclinedExponentialModel()})
rmodel = src_tng.broadband_models['r']
priors_tng = PriorDict({
    'cosi': TruncatedNormal(gen.native_cosi, 0.2, 0.05, 0.99),
    'theta_int': Uniform(0.0, np.pi),
    'g1': Uniform(-0.1, 0.1),
    'g2': Uniform(-0.1, 0.1),
    'vel.v0': Gaussian(0.0, 20.0),
    'vel.vcirc': Uniform(50.0, 400.0),
    'vel.rscale': Uniform(0.5, 15.0),
    'r.flux': Uniform(0.01, 10.0),
    'r.rscale': Uniform(0.2, 10.0),
    'r.h_over_r': 0.1,
    'r.x0': 0.0,
    'r.y0': 0.0,
})
obs_vel_t = build_velocity_obs(ip_tng, data=jnp.asarray(vel_map), variance=v_var)
obs_int_t = build_image_obs(ip_tng, broadband_key='r', int_model=rmodel,
                            data=jnp.asarray(int_norm), variance=i_var / flux_est**2)
task_tng = InferenceTask.from_obs(src_tng, priors_tng,
                                  velocity_obs=obs_vel_t, image_obs={'r': obs_int_t})
cfg_tng = NumpyroSamplerConfig(n_samples=1250, n_warmup=625, n_chains=4,
                               dense_mass=True, seed=42, progress=True)
result_tng = build_sampler('numpyro', task_tng, cfg_tng).run()
print_summary(result_tng)
print(f"max R-hat: {max(result_tng.get_rhat().values()):.4f}")
```

With TNG data there is no single "true" parameter set; the analytic model is an
approximation, so the posterior is shifted from the catalog orientation by model
mismatch. The science question is whether the shear constraint stays unbiased
despite it. See `tng50_data.md` for the data-vector pipeline.

---

## Section 8: Diagnostics

```{code-cell} python
fig = plot_trace(result_joint)
plt.show()
fig, recovery_stats = plot_recovery(result_joint, true_joint)
plt.show()
print(f"joint Nsigma: {recovery_stats['joint_nsigma']:.2f}")
```

`plot_corner` annotates the median +/- std per parameter, the joint N-sigma
(Mahalanobis distance from truth using the posterior covariance), and overlays
true (black) and MAP (red) values. `plot_corner_comparison` overlays posteriors
from several samplers run on the same task.

---

## Section 9: Choosing a sampler

| Sampler | Gradients | Evidence | R-hat / ESS | Best for |
|---|---|---|---|---|
| emcee | No | No | No | multi-modal posteriors, exploration |
| nautilus | No | Yes | No | model comparison, evidence |
| **numpyro** | **Yes** | No | **Yes** | joint models, multi-scale params, production |
| blackjax | Yes | No | No | simple velocity-only |

Decision shortcuts: need evidence -> nautilus; joint / production -> numpyro
(add the Laplace preconditioner for phot+grism); multi-modal -> emcee.

---

## Section 10: Troubleshooting

| Problem | Likely cause | Fix |
|---|---|---|
| Poor mixing (emcee) | too few walkers / degenerate posterior | `n_walkers` > 4x n_params |
| Divergences (numpyro) | stiff geometry | `reparam_strategy='empirical'`; raise `target_accept_prob` to 0.9 |
| High R-hat | not converged | more warmup / samples; check multimodality |
| Zero variance (blackjax) | gradient collapse on joint models | use numpyro |
| Low ESS | strong correlations | `dense_mass=True`; or the Laplace preconditioner |
| Slow phot+grism warmup | warmup-dominated wallclock | `precondition='laplace'` + short warmup |
| Hitting max tree depth | step size / geometry | raise `max_tree_depth` to 12-15 |
| NaN log-posterior | parameter outside model domain | check prior bounds |

---

## Section 11: Running tests and next steps

| Target | Runs |
|---|---|
| `make test-sampling` | sampling diagnostics (excludes nautilus) |
| `make test-sampling-all` | all sampling tests including nautilus |
| `pytest tests/test_flagship.py` | the full joint phot+grism Laplace run |

- **Config reference**: `kl_pipe/sampling/README.md`.
- **Test examples**: `tests/test_numpyro.py`, `tests/test_sampling.py`.
- **TNG pipeline**: `tng50_data.md`. **Forward model**: `quickstart.md`, `grism.md`.
