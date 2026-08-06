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

# Quickstart Tutorial

A practical introduction to the kinematic lensing pipeline. The pipeline models
rotationally-supported disk galaxies through their line-of-sight velocity field
and their broadband + emission-line surface brightness, then fits that forward
model to data. This tutorial walks the full path: describe a source, render it,
build a likelihood, and run an inference.

The pipeline targets Roman Space Telescope weak lensing, but it is not
Roman-specific: any rotating-galaxy imaging + slitless-spectroscopy dataset
(including archival data) can be modeled with the same objects, given an
appropriate PSF, pixel scale, and dispersion.

**NOTE:** to read this as a Jupyter notebook, convert it locally:
```bash
jupytext --to ipynb .../kl_roman_pipe/docs/tutorials/quickstart.md
```

```{code-cell} python
# kl_pipe.psf and the FFT rendering path require 64-bit precision.
import jax
jax.config.update('jax_enable_x64', True)

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
```

## Design Philosophy

The pipeline is built around three principles:

1. **A single source description.** Everything you can observe (velocity maps,
   broadband images, dispersed grism frames) is rendered from one `SourceModel`
   object that bundles a velocity model, per-band broadband intensity models,
   and per-line emission models, all sharing the same galaxy geometry.

2. **Functional, JAX-compatible core.** Models are immutable and hold no
   per-fit state; parameters flow in as data. This is what lets
   [JAX](https://docs.jax.dev), the array/autodiff library the pipeline is
   built on, differentiate the whole forward model and compile it to fast
   machine code. If you have not used JAX: it mirrors the NumPy API
   (`jax.numpy` in place of `numpy`), adding the rules that arrays are
   immutable and functions must be pure. Those rules buy you `jax.grad` (exact
   gradients) and `jax.jit` (compile once, run fast). In a quickstart fit you
   rarely call JAX directly; the inference objects wrap it for you.

3. **Coordinate-plane abstraction.** A source is evaluated face-on in its
   `disk` plane and transformed back out through four frames
   (`obs -> cen -> source -> gal -> disk`) that successively remove the
   centroid, lensing shear, position angle, and inclination. The models apply
   these transforms; you never write them yourself.

---

## Key Classes

### The source (`SourceModel`)

`SourceModel` is the central object. Any subset of its three slots may be
populated:

- `velocity_model` -- the rotation curve + line-of-sight projection. Required
  for the velocity and grism channels.
- `broadband_models` -- a dict of per-filter intensity models, keyed by your
  filter labels (`'F087'`, `'F184'`, ...).
- `emission_lines` -- a dict of per-line emission models, keyed by line name.
  Rest wavelengths auto-resolve from the `LINE_LAMBDAS` registry for known
  lines (e.g. `'Halpha'`).

```{code-cell} python
from kl_pipe.source import SourceModel
from kl_pipe.velocity import CenteredVelocityModel

# A velocity-only source. Enough for the first five examples.
source = SourceModel(velocity_model=CenteredVelocityModel())
print("velocity params:", source.velocity_model.PARAMETER_NAMES)
```

**Available velocity models:**
- `CenteredVelocityModel` -- arctangent rotation curve, centroid fixed at origin
- `OffsetVelocityModel` -- adds free centroid offsets `x0`, `y0`

**Available intensity models** (used for broadband and emission-line components):
- `InclinedExponentialModel` -- 3D exponential disk (n=1, exact FT); the default
- `InclinedSpergelModel`, `InclinedSersicModel`, `InclinedDeVaucouleursModel`
- `CompositeIntensityModel` / `BulgeDiskModel` -- multi-component (Section 7)

### The parameter convention (dotted keys)

`SourceModel` is driven by a flat dict of named parameters. The key namespace
tells the source which component a parameter belongs to:

| Key pattern | Meaning | Example |
|---|---|---|
| no prefix | shared geometry (all components) | `cosi`, `theta_int`, `g1`, `g2` |
| `vel.<name>` | velocity-model parameter | `vel.vcirc`, `vel.rscale`, `vel.v0` |
| `<band>.<name>` | broadband intensity parameter | `F087.flux`, `F087.rscale` |
| `<line>.<name>` | emission-line parameter | `Halpha.flux`, `Halpha.dispersion` |
| `<line>.cont.<name>` | continuum under a line | `Halpha.cont.flux_per_nm` |

The geometric parameters (`cosi`, `theta_int`, `g1`, `g2`) are shared: there is
one inclination, one position angle, one shear for the whole galaxy, common to
the velocity field and every intensity component. That sharing is the physical
content of kinematic lensing.

Parameter resolution is a two-step fallback: for component `F184`, the renderer
looks up `F184.rscale` first, then the bare `rscale`. This has two useful
consequences:

- **Sharing across components.** To give two broadband bands the same
  morphology but independent flux, supply the morphology as bare top-level keys
  (`rscale`, `h_over_r`) and only the flux as band-prefixed keys
  (`F087.flux`, `F184.flux`). Each band's missing `F<band>.rscale` falls back to
  the shared `rscale`. Used in the Section 9 capstone.
- **Per-component override.** The reverse is also allowed: a component-prefixed
  key overrides the shared one for that component only. For example, giving
  `Halpha.cosi` a value (or its own prior) lets the emission line take a
  different inclination than the broadband. This is physically unusual but
  permitted; the shared geometry is the default, not a constraint.

Units follow the repo conventions: arcsec for coordinates, km/s for velocity,
radians for `theta_int`, `cosi = cos(inclination)` (1 = face-on, 0 = edge-on),
dimensionless shear with `|g| < 1`, and integrated flux (not surface
brightness).

### Image geometry (`ImagePars`)

`ImagePars` defines the pixel grid. It takes a `shape`, an `indexing`
convention, and exactly one of `pixel_scale` (arcsec/pixel) or a `wcs` (an
`astropy.wcs.WCS`):

```{code-cell} python
from kl_pipe.parameters import ImagePars

# 64x64 image at 0.1 arcsec/pixel, numpy 'ij' (row, col) indexing.
image_pars = ImagePars(shape=(64, 64), pixel_scale=0.1, indexing='ij')

print(f"Image: {image_pars.Nx} x {image_pars.Ny} pixels")
print(f"Field of view: {image_pars.Nx * image_pars.pixel_scale:.1f} x "
      f"{image_pars.Ny * image_pars.pixel_scale:.1f} arcsec")
```

Passing a `wcs` instead of a `pixel_scale` is how you carry a real sky
orientation (used in the capstone): the position angle and shear are then
rotated from the celestial frame into the detector frame at render time.

The coordinate grid is always centered so that `(0, 0)` is the geometric center
of the stamp, regardless of whether the grid has an even or odd number of
pixels. The pixel-center coordinates are `arange(N) - (N - 1) / 2`, so an
even-N axis puts pixel centers at half-integers (the center falls on a pixel
edge) and an odd-N axis puts them at integers (the center falls on a pixel):

```{code-cell} python
from kl_pipe.utils import build_map_grid_from_image_pars

for N in (4, 5):
    ip = ImagePars(shape=(N, N), pixel_scale=1.0, indexing='ij')
    X, _ = build_map_grid_from_image_pars(ip, unit='pixel')
    parity = 'even -> half-integer centers' if N % 2 == 0 else 'odd -> integer centers'
    print(f"N={N} ({parity}): x pixel-centers = {np.asarray(X[0])}")
```

A source at `x0 = y0 = 0` therefore always sits at the true stamp center; the
parity only changes whether that center coincides with a pixel or a pixel edge.

### Observations (`ImageObs`, `VelocityObs`, `GrismObs`)

A `SourceModel` carries no instrument state. The PSF, pixel response, FFT render
grid, and (for a fit) the data and variance all live on an *observation*
object. There are three, one per channel, each built by a helper:

| Channel | Class | Builder |
|---|---|---|
| broadband image | `ImageObs` | `build_image_obs` |
| velocity map | `VelocityObs` | `build_velocity_obs` |
| dispersed grism | `GrismObs` | `build_grism_obs` |

You render *through* an obs (`source.render_broadband(pars, obs, band)`), and a
fit bundles obs into an `InferenceTask`. The next examples use them in both
modes: an obs built **without** data is a pure render target (forward model
only); an obs built **with** `data` + `variance` is what a likelihood compares
against.

---

## Example 1: Render and plot a velocity field

We render the line-of-sight velocity map directly from the `SourceModel`, then
add noise at a target SNR. (Velocity is a flux-weighted moment of the spectral
cube, not a photon count, so its noise is Gaussian by construction; Poisson
statistics live at the datacube layer.)

```{code-cell} python
from kl_pipe.observation import build_velocity_obs
from kl_pipe.noise import add_velocity_noise

# True parameters (dotted-key namespace). CenteredVelocityModel has no
# vel.x0 / vel.y0: the centroid is fixed at the origin.
true_pars = {
    'cosi': 0.6,          # ~53 deg inclination
    'theta_int': 0.785,   # ~45 deg position angle
    'g1': 0.0,
    'g2': 0.0,
    'vel.v0': 10.0,       # km/s systemic
    'vel.vcirc': 200.0,   # km/s asymptotic
    'vel.rscale': 1.0,    # arcsec turnover radius (well inside the 6.4" FOV, so
                          # the flat part of the rotation curve is sampled and
                          # vcirc is constrained rather than extrapolated)
}

# An obs built WITHOUT data is just a render target: a grid (+ optional PSF).
# Here there is no PSF, so this renders the noiseless LOS velocity field. We
# attach real data + variance to a separate obs in Example 2.
vel_obs = build_velocity_obs(image_pars)
velocity_true = np.asarray(source.render_velocity(true_pars, vel_obs))

# Add noise at SNR=100; returns the noisy map and a per-pixel variance map.
velocity_noisy, variance_vel = add_velocity_noise(velocity_true, target_snr=100, seed=42)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, img, title in [
    (axes[0], velocity_true, 'True velocity field'),
    (axes[1], velocity_noisy, 'Noisy data (SNR=100)'),
    (axes[2], velocity_noisy - velocity_true, 'Noise realization'),
]:
    vmax = float(np.max(np.abs(velocity_true)))
    im = ax.imshow(img, origin='lower', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel('x (pixels)'); ax.set_ylabel('y (pixels)')
    plt.colorbar(im, ax=ax, label='km/s')
plt.tight_layout()
plt.show()

print(f"Noise std: {float(np.sqrt(np.mean(variance_vel))):.2f} km/s")
```

---

## Example 2: Priors, an InferenceTask, and the likelihood

`InferenceTask` bundles a source, its priors, and the data into one object that
exposes a log-posterior. Priors use a `PriorDict`: entries with a `Prior` object
are *sampled*; entries with a bare number are *fixed*. The sampled names, sorted
alphabetically, define the canonical sampling vector.

```{code-cell} python
from kl_pipe.priors import PriorDict, Uniform, Gaussian
from kl_pipe.sampling import InferenceTask

priors = PriorDict({
    # shared geometry
    'cosi': Uniform(0.05, 0.99),
    'theta_int': Uniform(0.0, np.pi),
    'g1': 0.0,                            # fixed (no lensing in this example)
    'g2': 0.0,
    # velocity
    'vel.v0': Gaussian(10.0, 10.0),
    'vel.vcirc': Uniform(50.0, 400.0),
    'vel.rscale': Uniform(0.2, 5.0),
})

# Rebuild the velocity obs WITH the data + variance from Example 1.
vel_obs_data = build_velocity_obs(image_pars, data=velocity_noisy, variance=variance_vel)
task = InferenceTask.from_obs(source, priors, velocity_obs=vel_obs_data)

print("Sampled parameters:", task.sampled_names)
print("Fixed parameters:  ", task.fixed_params)
```

The task evaluates a JIT-compiled, differentiable log-posterior on the sampled
vector (in `sampled_names` order). The simplest forward-model sanity check is to
sweep one parameter while holding the rest at truth: the curve should peak at
the true value. We add a quadratic interpolation around the grid maximum to
locate the peak more precisely than the grid spacing allows.

```{code-cell} python
theta_true = jnp.array([true_pars[name] for name in task.sampled_names])
i_vcirc = task.sampled_names.index('vel.vcirc')

vcirc_grid = np.linspace(150, 250, 101)
logp = np.array([float(task.log_posterior(theta_true.at[i_vcirc].set(v)))
                 for v in vcirc_grid])

# Quadratic-interpolate the peak from the 3 points around the grid maximum.
k = int(np.argmax(logp))
if 0 < k < len(vcirc_grid) - 1:
    y0, y1, y2 = logp[k - 1], logp[k], logp[k + 1]
    dx = vcirc_grid[1] - vcirc_grid[0]
    offset = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)   # vertex of the parabola
    peak = vcirc_grid[k] + offset * dx
else:
    peak = vcirc_grid[k]

plt.figure(figsize=(9, 5))
plt.plot(vcirc_grid, logp, 'b-', lw=2)
plt.axvline(true_pars['vel.vcirc'], color='r', ls='--', label=f"truth: {true_pars['vel.vcirc']:.0f}")
plt.axvline(peak, color='g', ls=':', label=f"interp. peak: {peak:.1f}")
plt.xlabel('vcirc (km/s)'); plt.ylabel('log-posterior')
plt.title('Posterior slice along vcirc'); plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout(); plt.show()

print(f"interpolated peak vcirc = {peak:.2f} km/s   (truth {true_pars['vel.vcirc']:.0f})")
```

This slicing technique is used throughout the test suite
(`tests/test_likelihood_slices.py`) to validate that likelihoods peak at the
true parameters and that the forward model is implemented correctly.

---

## Example 3: Parameter recovery with an optimizer

For a point estimate, maximize the log-posterior. `kl_pipe.optimization`
provides `multi_start_minimize`, a multi-start wrapper around
`scipy.optimize.minimize` (multi-start guards against the shallow local minima
and boundary attractors that single-start L-BFGS-B falls into). The task
provides the value-and-gradient function (JAX computes the gradient by automatic
differentiation) and the prior bounds.

```{code-cell} python
from kl_pipe.optimization import multi_start_minimize

logp_and_grad = task.get_log_posterior_and_grad_fn()

# scipy minimizes, so the objective returns the NEGATIVE log-posterior and its
# gradient. This is the only adapter you need.
def objective(theta):
    val, grad = logp_and_grad(jnp.asarray(theta))
    return -float(val), -np.asarray(grad)

# A slightly-wrong starting guess, in sampled order
# [cosi, theta_int, vel.rscale, vel.v0, vel.vcirc].
theta_init = np.asarray(theta_true) + np.array([0.05, 0.1, 0.3, 2.0, -20.0])
result = multi_start_minimize(
    objective, x0=theta_init, bounds=task.get_bounds(),
    n_starts=8, jac=True, options={'maxiter': 1000},
)
pars_fit = {name: float(v) for name, v in zip(task.sampled_names, result.x)}
print(f"converged: {result.success}    final log-posterior: {-result.fun:.2f}")
```

Visualize the recovery with `plot_parameter_recovery`, the same two-panel
diagnostic the optimizer-recovery tests use (top: true vs recovered values;
bottom: fractional error per parameter):

```{code-cell} python
import tempfile
from pathlib import Path
from IPython.display import Image
from kl_pipe.diagnostics.imaging import plot_parameter_recovery

res = plot_parameter_recovery(
    true_values={n: true_pars[n] for n in task.sampled_names},
    recovered_values=pars_fit,
    output_dir=Path(tempfile.mkdtemp()),
    test_name='quickstart_optimizer',
)

print(f"{'parameter':<12}{'fit':>10}{'truth':>10}{'rel. error':>12}")
for n in task.sampled_names:
    rel = abs(pars_fit[n] - true_pars[n]) / abs(true_pars[n]) if true_pars[n] else float('nan')
    print(f"{n:<12}{pars_fit[n]:>10.4f}{true_pars[n]:>10.4f}{rel:>11.1%}")

Image(res['output_path'])
```

`cosi`, `vcirc`, and `rscale` are typically the worst-recovered parameters here,
and that is expected, not a bug. A velocity map constrains the line-of-sight
projection `vcirc * sin(i)`, so `vcirc` and `cosi` trade off along a degeneracy
that velocity data alone cannot break. The recovery is much better on the
observable *product* than on the individual parameters:

```{code-cell} python
def vcirc_sini(p):
    return p['vel.vcirc'] * np.sqrt(1.0 - p['cosi'] ** 2)

prod_true, prod_fit = vcirc_sini(true_pars), vcirc_sini(pars_fit)
print(f"vcirc*sin(i):  truth={prod_true:.2f}  fit={prod_fit:.2f}  "
      f"rel err={abs(prod_fit - prod_true) / prod_true:.2%}")
```

This is exactly why the optimizer-recovery tests assert on `vcirc * sin(i)`
rather than on `vcirc` and `cosi` separately. Breaking the degeneracy needs
external information, which is the subject of the next example.

---

## Example 4: Joint fitting, shear, and the Tully-Fisher prior

From an image alone, observed ellipticity is `intrinsic-shape(i) + shear`: a
degeneracy photometric weak lensing cannot break. A velocity field breaks it by
constraining inclination and the kinematic axis independently, so a joint
velocity + imaging fit separates intrinsic shape from shear.

That separation needs the inclination. The velocity field pins it two ways: the
`1/cos(i)` stretch of the 2D field (geometric) and the `vcirc * sin(i)`
amplitude (degenerate with `vcirc`). Toward face-on, or at modest resolution and
SNR, the geometric handle weakens, a `vcirc - cosi` correlation survives, and it
leaks into the shear error.

Tully-Fisher closes this. The TF relation ties a disk's circular velocity to its
luminosity, so the galaxy's photometry predicts `vcirc` independent of
inclination. As a prior on `vcirc` it pins `vcirc`, hence `sin(i)`, hence the
intrinsic shape, hence the shear.

We show this on a joint velocity + broadband fit with free shear at `cosi = 0.8`,
where the velocity geometry alone cannot pin the inclination.

```{code-cell} python
import galsim
from kl_pipe.intensity import InclinedExponentialModel
from kl_pipe.observation import build_image_obs
from kl_pipe.noise import add_intensity_noise
from kl_pipe.render import RenderConfig

# Joint source: velocity + one broadband band.
f087 = InclinedExponentialModel()
joint_source = SourceModel(
    velocity_model=CenteredVelocityModel(),
    broadband_models={'F087': f087},
)

# Mildly face-on, with shear. cosi=0.8 -> i~37 deg, so the 1/cos(i) field
# stretch is weak and inclination is not pinned by the velocity geometry alone.
jt = {
    'cosi': 0.8,
    'theta_int': 0.6,
    'g1': 0.05,
    'g2': -0.03,
    'vel.v0': 10.0,
    'vel.vcirc': 210.0,
    'vel.rscale': 1.0,
    'F087.flux': 100.0,
    'F087.rscale': 1.0,
    'F087.h_over_r': 0.1,
    'F087.x0': 0.0,
    'F087.y0': 0.0,
}
jgrid = ImagePars(shape=(48, 48), pixel_scale=0.1, indexing='ij')
jpsf = galsim.Gaussian(fwhm=0.12)
jrc = RenderConfig(oversample=3)

# Velocity data (no PSF) + broadband data (PSF-convolved).
# Fluxes here are in arbitrary units; to work in physical units (e.g. a
# catalog AB magnitude or a published survey depth), convert with the
# helpers in kl_pipe.photometry (ab_mag_to_ujy, fnu_to_flambda, ...).
vobs_clean = build_velocity_obs(jgrid)
v_true = np.asarray(joint_source.render_velocity(jt, vobs_clean))
v_noisy, v_var = add_velocity_noise(v_true, target_snr=100, seed=10)

iobs_clean = build_image_obs(jgrid, psf=jpsf, render_config=jrc,
                             int_model=f087, broadband_key='F087')
i_true = np.asarray(joint_source.render_broadband(jt, iobs_clean, 'F087'))
i_noisy, i_var = add_intensity_noise(i_true, target_snr=200, seed=11)

vobs = build_velocity_obs(jgrid, data=v_noisy, variance=v_var)
iobs = build_image_obs(jgrid, psf=jpsf, render_config=jrc, int_model=f087,
                       broadband_key='F087', data=jnp.asarray(i_noisy), variance=i_var)
```

The TF prior is a Gaussian on `vcirc` at the TF-predicted value, with a width
set by the relation's intrinsic scatter. (A helper that derives the center from
photometry is planned; here we set it by hand.) We compare a flat `vcirc` prior
against the TF prior, shear free in both. Everything except the `vcirc` prior is
shared, so we build that common block once.

```{code-cell} python
# TF intrinsic scatter at z~1 is ~0.08 dex in log10(vcirc) (Ubler et al. 2017,
# ApJ 842, 121, KMOS3D; relation: Tully & Fisher 1977, A&A 54, 661). The 1-sigma
# width in km/s at vcirc~210 is sigma = vcirc * ln(10) * 0.08 ~ 39 km/s.
sigma_tf = 210.0 * np.log(10) * 0.08

shear_free_priors = {
    'cosi': Uniform(0.05, 0.99),
    'theta_int': Uniform(0.0, np.pi),
    'g1': Uniform(-0.25, 0.25),
    'g2': Uniform(-0.25, 0.25),
    'vel.v0': Gaussian(10.0, 10.0),
    'vel.rscale': Uniform(0.2, 5.0),
    'F087.flux': Uniform(10.0, 1000.0),
    'F087.rscale': Uniform(0.2, 5.0),
    'F087.h_over_r': 0.1,
    'F087.x0': 0.0,
    'F087.y0': 0.0,
}

def fit_with_vcirc_prior(vcirc_prior):
    priors = PriorDict({**shear_free_priors, 'vel.vcirc': vcirc_prior})
    t = InferenceTask.from_obs(
        joint_source, priors, image_obs={'F087': iobs}, velocity_obs=vobs
    )
    grad_fn = t.get_log_posterior_and_grad_fn()
    def neg_logpost(th):
        val, grad = grad_fn(jnp.asarray(th))
        return -float(val), -np.asarray(grad)
    x0 = np.array([jt[n] for n in t.sampled_names])
    r = multi_start_minimize(neg_logpost, x0=x0, bounds=t.get_bounds(),
                             n_starts=6, jac=True)
    return dict(zip(t.sampled_names, (float(v) for v in r.x)))

fits = {
    'no TF':   fit_with_vcirc_prior(Uniform(50.0, 400.0)),
    'with TF': fit_with_vcirc_prior(Gaussian(210.0, sigma_tf)),
}
```

```{code-cell} python
keys = ['g1', 'g2', 'cosi', 'vel.vcirc']
rel = lambda v, t: abs(v - t) / abs(t) if t else float('nan')

header = f"{'':<14}" + ''.join(f"{k:>11}" for k in keys)
print(header)
print(f"{'truth':<14}" + ''.join(f"{jt[k]:>11.3f}" for k in keys))
for label, f in fits.items():
    print(f"{label:<14}" + ''.join(f"{f[k]:>11.3f}" for k in keys))
    print(f"{'  rel. error':<14}" + ''.join(f"{rel(f[k], jt[k]):>10.0%} " for k in keys))
```

```{code-cell} python
shear_err = {lab: np.hypot(f['g1'] - jt['g1'], f['g2'] - jt['g2'])
             for lab, f in fits.items()}
plt.figure(figsize=(5, 4))
plt.bar(list(shear_err), list(shear_err.values()), color=['indianred', 'steelblue'])
plt.ylabel('|g_fit - g_true|'); plt.title('Shear recovery error')
plt.tight_layout(); plt.show()
```

With a flat `vcirc` prior the `vcirc - cosi` correlation leaves the inclination
loose, so the intrinsic shape, and the shear, are poorly recovered. The TF prior
pins `vcirc` and tightens the shear. How much it helps depends on the galaxy and
the data: within the TF scatter the fit here is already prior-dominated, so
tightening the prior further changes little. This is the chain that lets the
joint broadband + grism fit in the next sections measure shear.

---

## Example 5: Sampling a velocity-only posterior with NUTS

The optimizer gives a point estimate; a sampler gives the full posterior with
uncertainties and correlations. NumPyro NUTS is the recommended sampler (it uses
the gradients JAX provides). We fit the same velocity-only problem with the TF
prior, a small enough setup to converge in under a minute.

```{code-cell} python
from kl_pipe.sampling import NumpyroSamplerConfig, build_sampler
from kl_pipe.priors import TruncatedNormal

priors_tf = PriorDict({
    'cosi': Uniform(0.05, 0.99),
    'theta_int': Uniform(0.0, np.pi),
    'g1': 0.0,
    'g2': 0.0,
    'vel.v0': Gaussian(10.0, 10.0),
    # TF prior on vcirc (Ubler+2017 ~0.08 dex -> ~37 km/s at 200 km/s):
    'vel.vcirc': TruncatedNormal(200.0, 37.0, 50.0, 400.0),
    'vel.rscale': Uniform(0.2, 5.0),
})
task_tf = InferenceTask.from_obs(source, priors_tf, velocity_obs=vel_obs_data)

config = NumpyroSamplerConfig(
    n_warmup=400, n_samples=600, n_chains=2,
    chain_method='vectorized', seed=0, progress=False,
)
sampler = build_sampler('numpyro', task_tf, config)
result = sampler.run()

summary = result.get_summary()
rhats = result.get_rhat()
print(f"{'param':<12}{'mean':>10}{'std':>10}{'truth':>10}{'rel.err':>9}{'R-hat':>8}")
for n in task_tf.sampled_names:
    mean = summary[n]['mean']
    rel = abs(mean - true_pars[n]) / abs(true_pars[n]) if true_pars[n] else float('nan')
    print(f"{n:<12}{mean:>10.3f}{summary[n]['std']:>10.3f}"
          f"{true_pars[n]:>10.3f}{rel:>8.1%}{rhats[n]:>8.3f}")
```

```{code-cell} python
from kl_pipe.sampling.diagnostics import plot_corner

fig = plot_corner(result, true_values={n: true_pars[n] for n in task_tf.sampled_names})
plt.show()
```

R-hat near 1.0 indicates the chains converged. With the TF prior, `cosi` and
`vcirc` are now individually constrained (not just their product), and the
corner plot shows the residual `vcirc`-`cosi` correlation the TF prior did not
fully remove.

---

## Example 6: The primary Roman configuration -- joint broadband image + grism emission line

This is the primary configuration the pipeline is built for: a broadband image
that constrains the disk morphology, fit jointly with a slitless grism roll
whose emission line is dispersed into a 2D frame. (It need not be exactly one of
each; the next sections add more bands and rolls.) Rotation shears the dispersed
line antisymmetrically across the kinematic major axis, and fitting both
channels together, with the TF prior, decouples that kinematic axis from the
morphological one and lets the fit measure shear.

The emission line gets its own intensity model, distinct from the broadband
(line-emitting gas and broadband stellar light need not share a scale length),
plus a continuum under the line. All components still share the galaxy geometry.

```{code-cell} python
import galsim
from kl_pipe.intensity import InclinedExponentialModel
from kl_pipe.lines import EmissionLine, LINE_LAMBDAS
from kl_pipe.observation import build_image_obs, build_grism_obs
from kl_pipe.dispersion import build_grism_pars_for_line
from kl_pipe.render import RenderConfig
from kl_pipe.noise import add_intensity_noise

Z = 1.0
roman_pars = ImagePars(shape=(32, 32), pixel_scale=0.11, indexing='ij')  # Roman native
# Gaussian PSF with a Roman-like FWHM. This is a stand-in: the true Roman PSF is
# diffraction-limited with structure (not Gaussian); only the size is Roman-like.
psf = galsim.Gaussian(fwhm=0.18)

# Broadband, line, and continuum each get their own intensity model.
f087_model = InclinedExponentialModel()
halpha_model = InclinedExponentialModel()
halpha_cont = InclinedExponentialModel()
roman_source = SourceModel(
    velocity_model=CenteredVelocityModel(),
    broadband_models={'F087': f087_model},
    emission_lines={'Halpha': EmissionLine(intensity=halpha_model, continuum=halpha_cont)},
)

roman_truth = {
    'cosi': 0.6,
    'theta_int': np.pi / 4,
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
    'Halpha.cont.flux_per_nm': 25.0,
    'Halpha.cont.rscale': 0.25,
    'Halpha.cont.h_over_r': 0.1,
    'Halpha.cont.x0': 0.0,
    'Halpha.cont.y0': 0.0,
    'z': Z,
}
```

We size the FFT render grid with an explicit `RenderConfig`. `oversample=3` is
deliberately low to keep this tutorial fast: it trades rendering fidelity for
speed and is fine here because the priors are loose and we are not chasing
sub-percent accuracy. For production inference with tight priors you would let
`InferenceTask.from_obs` auto-derive the grid from the prior bounds (Section 8),
which guards against the high-spatial-frequency aliasing that a too-low
oversample would otherwise fold into the likelihood.

```{code-cell} python
rc = RenderConfig(oversample=3)

grism_pars = build_grism_pars_for_line(
    LINE_LAMBDAS['Halpha'], redshift=Z, image_pars=roman_pars, dispersion=1.1,
)

# Clean renders (no data) -> add noise -> attach data + variance.
img_obs_clean = build_image_obs(roman_pars, psf=psf, render_config=rc,
                                int_model=f087_model, broadband_key='F087')
grism_obs_clean = build_grism_obs(grism_pars, z=Z, psf=psf, render_config=rc)

F087_true = np.asarray(roman_source.render_broadband(roman_truth, img_obs_clean, 'F087'))
grism_true = np.asarray(roman_source.render_grism(roman_truth, grism_obs_clean))

# add_intensity_noise applies the same matched-filter SNR model to any 2D flux
# image, broadband or dispersed grism.
F087_noisy, var_F087 = add_intensity_noise(F087_true, target_snr=100, seed=1)
grism_noisy, var_grism = add_intensity_noise(grism_true, target_snr=150, seed=2)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].imshow(F087_noisy, origin='lower', cmap='viridis'); axes[0].set_title('F087 broadband')
axes[1].imshow(grism_noisy, origin='lower', cmap='magma');  axes[1].set_title('Halpha grism')
for ax in axes: ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout(); plt.show()
```

Two notes on the noise used throughout this tutorial:

- **The requested SNR is the matched-filter SNR of the whole stamp**, and it
  is realized exactly (`var = ||T||^2 / SNR^2`). For a grism stamp a more
  physically meaningful label normalizes on the emission line alone
  (`kl_pipe.noise.grism_line_noise`, with a line-only render as the
  template): the continuum dominates the dispersed power but carries no
  kinematic signal. The whole-stamp version here keeps the example short.
- **Choosing a pathway.** Declared-SNR noise is the right tool for
  controlled experiments: you pick the depth, it is exact by construction,
  and the flux units stay arbitrary. When the question is instead what
  *Roman* would deliver for a galaxy of a given physical flux, carry
  physical units and derive the noise from the published survey depths (a
  flat depth-anchored background plus the source's own shot noise). That
  pathway, and when to prefer each, is worked out in the Roman reference
  tutorial ("Physical flux units and Roman noise" in
  `roman_reference.md`).

Wire both channels into one joint `InferenceTask`. `from_obs` takes a dict of
broadband obs (keyed by band) and a dict of grism obs (keyed by an arbitrary
roll label). Shear priors are set to +/- 0.25, comfortably wider than any
realistic reduced shear.

```{code-cell} python
roman_priors = PriorDict({
    'cosi': TruncatedNormal(0.6, 0.15, 0.05, 0.99),
    'theta_int': TruncatedNormal(np.pi / 4, 0.3, 0.0, np.pi / 2),
    'g1': Uniform(-0.25, 0.25),
    'g2': Uniform(-0.25, 0.25),
    'vel.v0': Gaussian(10.0, 10.0),
    'vel.vcirc': TruncatedNormal(200.0, 37.0, 80.0, 400.0),   # TF prior (Ubler+2017)
    'vel.rscale': TruncatedNormal(0.3, 0.1, 0.05, 1.0),
    'F087.flux': TruncatedNormal(100.0, 20.0, 30.0, 250.0),
    'F087.rscale': TruncatedNormal(0.3, 0.08, 0.05, 1.0),
    'F087.h_over_r': 0.1,                                     # fixed (low observable sensitivity)
    'F087.x0': TruncatedNormal(0.0, 0.1, -0.5, 0.5),
    'F087.y0': TruncatedNormal(0.0, 0.1, -0.5, 0.5),
    'Halpha.flux': TruncatedNormal(100.0, 20.0, 30.0, 250.0),
    'Halpha.rscale': TruncatedNormal(0.25, 0.08, 0.05, 1.0),
    'Halpha.h_over_r': 0.1,                                   # fixed
    'Halpha.x0': TruncatedNormal(0.0, 0.1, -0.5, 0.5),
    'Halpha.y0': TruncatedNormal(0.0, 0.1, -0.5, 0.5),
    'Halpha.dispersion': TruncatedNormal(50.0, 20.0, 5.0, 150.0),
    'Halpha.cont.flux_per_nm': TruncatedNormal(25.0, 15.0, 0.0, 200.0),
    'Halpha.cont.rscale': roman_truth['Halpha.cont.rscale'],   # fixed to line truth
    'Halpha.cont.h_over_r': roman_truth['Halpha.cont.h_over_r'],
    'Halpha.cont.x0': roman_truth['Halpha.cont.x0'],
    'Halpha.cont.y0': roman_truth['Halpha.cont.y0'],
    'z': Z,                                                   # fixed
})

obs_F087 = build_image_obs(roman_pars, psf=psf, render_config=rc, int_model=f087_model,
                           broadband_key='F087', data=jnp.asarray(F087_noisy), variance=var_F087)
obs_grism = build_grism_obs(grism_pars, z=Z, psf=psf, render_config=rc,
                            data=jnp.asarray(grism_noisy), variance=var_grism)

roman_task = InferenceTask.from_obs(
    roman_source, roman_priors,
    image_obs={'F087': obs_F087},
    grism_obs={'roll0': obs_grism},
)
theta_roman = jnp.array([roman_truth[n] for n in roman_task.sampled_names])
print(f"sampled dimension: {roman_task.n_params}")
print(f"joint log-posterior at truth: {float(roman_task.log_posterior(theta_roman)):.2f}")
```

A finite log-posterior at truth confirms the joint forward model and likelihood
are wired correctly. Sampling this ~17-dimensional posterior takes minutes to
tens of minutes, too long to run inside a tutorial, so the sampler call is left
commented. The two cells below show the two ways to run it.

The straightforward way (dense mass matrix, long warmup):

```{code-cell} python
# config_dense = NumpyroSamplerConfig(
#     n_warmup=300, n_samples=300, n_chains=2, chain_method='vectorized',
#     dense_mass=True, seed=42, progress=True,
# )
# result = build_sampler('numpyro', roman_task, config_dense).run()
# Rough runtime at this 32x32, oversample=3 setup: ~40 min (warmup-dominated).
```

The faster way (Laplace preconditioner): NUTS starts at the MAP with a fixed
inverse-Hessian mass matrix, so warmup only has to tune the step size instead of
climbing from an identity metric. On the flagship test this is ~5x faster (8.5
vs ~42 min) with equal recovery and better convergence.

```{code-cell} python
# config_laplace = NumpyroSamplerConfig(
#     n_warmup=50, n_samples=300, n_chains=2, chain_method='vectorized',
#     precondition='laplace',   # MAP init + fixed Laplace mass matrix
#     seed=42, progress=True,
# )
# result = build_sampler('numpyro', roman_task, config_laplace).run()
# Rough runtime at this setup: ~8-10 min. Recommended for joint phot+grism fits.
```

See the sampling tutorial for the full run, convergence diagnostics, and the
Laplace preconditioner in depth.

---

## Example 7: Multi-component intensity models

Real galaxies have distinct morphological components (disk, bulge, bar).
`CompositeIntensityModel` sums N components in a single k-space pass (one IFFT),
and `BulgeDiskModel` is a convenience subclass for the common disk + bulge case:
an exponential disk (n=1, exact FT) plus a de Vaucouleurs bulge (n=4).

```{code-cell} python
from kl_pipe.intensity import (
    BulgeDiskModel, CompositeIntensityModel, ComponentSpec,
    InclinedExponentialModel, InclinedSersicModel,
)

bd = BulgeDiskModel()
print("Bulge+disk parameters:", bd.PARAMETER_NAMES)
```

Flux is parameterized as `total_flux` + `bulge_frac` (the B/T ratio); the
component fluxes are derived internally as `disk_flux = total_flux * (1 -
bulge_frac)`. A bulge profile has a central cusp that aliases without a PSF, so
multi-component rendering should always go through a PSF-convolved obs:

```{code-cell} python
bd_pars = {
    'cosi': 0.5,
    'theta_int': 0.3,
    'g1': 0.02,
    'g2': -0.01,
    'total_flux': 1e4,
    'bulge_frac': 0.25,
    'disk_rscale': 1.5,
    'disk_h_over_r': 0.1,
    'disk_x0': 0.0,
    'disk_y0': 0.0,
    'bulge_hlr': 0.4,
    'bulge_h_over_hlr': 0.3,
    'bulge_x0': 0.0,
    'bulge_y0': 0.0,
}
bd_image_pars = ImagePars(shape=(64, 64), pixel_scale=0.11, indexing='ij')
bd_psf = galsim.Gaussian(fwhm=0.15)

bd_source = SourceModel(broadband_models={'F087': bd})
obs_bd = build_image_obs(bd_image_pars, psf=bd_psf, render_config=RenderConfig(oversample=5),
                         int_model=bd, broadband_key='F087')
# Supply all params under the F087 namespace via the bare keys above (they
# resolve through the per-component -> shared fallback).
image_bd = np.asarray(bd_source.render_broadband(bd_pars, obs_bd, 'F087'))

plt.figure(figsize=(5, 4))
plt.imshow(image_bd, origin='lower', cmap='viridis'); plt.colorbar(label='flux/pixel')
plt.title('Bulge + disk composite (PSF-convolved)'); plt.show()
```

`CompositeIntensityModel` composes any intensity models via `ComponentSpec`,
each with a `prefix` that namespaces its parameters and optional `fixed_params`
that override shared values for that component only. A few patterns:

```{code-cell} python
# Disk + bulge with shear zeroed for the bulge only (the disk keeps it):
disk_bulge = CompositeIntensityModel(components=[
    ComponentSpec(InclinedExponentialModel(), prefix='disk'),
    ComponentSpec(InclinedSersicModel(), prefix='bulge',
                  fixed_params={'n_sersic': 4.0, 'g1': 0.0, 'g2': 0.0}),
])

# Two-component disk (thin + thick), both Sersic with free index:
two_disk = CompositeIntensityModel(components=[
    ComponentSpec(InclinedSersicModel(), prefix='thin'),
    ComponentSpec(InclinedSersicModel(), prefix='thick'),
])

# Three components (disk + bulge + bar):
three = CompositeIntensityModel(components=[
    ComponentSpec(InclinedExponentialModel(), prefix='disk'),
    ComponentSpec(InclinedSersicModel(), prefix='bulge', fixed_params={'n_sersic': 4.0}),
    ComponentSpec(InclinedSersicModel(), prefix='bar', fixed_params={'n_sersic': 1.5}),
])
for label, m in [('disk+bulge', disk_bulge), ('thin+thick', two_disk), ('3-component', three)]:
    print(f"{label:<12} ({len(m.PARAMETER_NAMES)} params): {m.PARAMETER_NAMES}")
```

`shared_centroids=True` links the components' centroids into one shared
`x0`/`y0` instead of per-component offsets. The intensity-model details, the
full parameter lists, and the rendering-accuracy convergence study are covered
in the intensity-models tutorial.

---

## Example 8: RenderConfig and rendering accuracy

Intensity rendering happens in k-space: the analytic profile FT is multiplied by
the pixel-response FT and the PSF FT, then inverse-transformed in one FFT pass.
`RenderConfig` controls the grid that FFT runs on:

- `oversample` -- how far past the base Nyquist frequency the k-grid extends
  before binning back. Higher oversample suppresses aliasing of high-spatial-
  frequency profile content (sharp cusps, compact or highly-inclined disks).
- `pad_factor`, `folding_threshold`, `maxk_threshold` -- padding and the
  aliasing budget.

Each intensity model exposes `maxk(params)` (the wavenumber where its FT drops
below threshold) and `stepk(params)` (the spacing needed to avoid flux folding).
`maxk` grows as the scale length shrinks and as `cosi` decreases (inclination
compresses the FT along one axis, extending it by `1/cosi`). For inference,
`InferenceTask.from_obs` calls `RenderConfig.for_priors`, which evaluates the
worst case (smallest scale, smallest `cosi`) across the prior bounds and sizes
the grid so even the most demanding sample the chain visits renders without
aliasing. When a PSF is present its FT caps the effective `maxk`, so the
required grid is smaller than the bare profile would imply.

```{code-cell} python
m = InclinedExponentialModel()
# maxk / stepk take a parameter dict (bare names), not a theta array.
demo = {
    'cosi': 0.6,
    'theta_int': 0.0,
    'g1': 0.0,
    'g2': 0.0,
    'flux': 1.0,
    'rscale': 0.3,
    'h_over_r': 0.1,
    'x0': 0.0,
    'y0': 0.0,
}
print(f"maxk  = {float(m.maxk(demo)):.1f} rad/arcsec  (compact / inclined -> larger)")
print(f"stepk = {float(m.stepk(demo)):.1f} rad/arcsec")
```

The trade-off is direct: a narrower `cosi` prior gives a smaller worst-case
`maxk`, a smaller grid, and a faster fit. Forcing a low `oversample` by hand
(as in Example 6) is a speed-vs-fidelity choice that is only safe when the
priors are loose. The relationship between `oversample` and rendering accuracy
is quantified in the test suite:

- `tests/test_psf.py::test_oversample_convergence` -- residual vs GalSim
  ground truth decreases monotonically with `oversample` (N=5 within 1e-4).
- `tests/test_grism_bandwidth.py::test_worst_case_converges_at_predicted_oversample`
  -- the prior-derived `oversample` is the factor at which grism rendering
  converges.
- `tests/test_render.py` and `tests/test_render_config.py` -- how the
  PSF-capped worst-case `maxk` sets `oversample`.

---

## Example 9: A fully-worked Roman mock

A reference-quality setup, less pedagogical than the examples above: two
broadband bands and two grism roll angles for a single emission line, with sky
orientation carried by a WCS, a realistic PSF, a bulge + disk composite shared
across the two broadbands (and reused for the line's continuum), and a separate
exponential disk for the emission line. The sampler call is left commented; the
point is the data + model construction.

```{code-cell} python
from astropy.wcs import WCS

def make_wcs(shape, pixel_scale_arcsec, pa_deg):
    """A simple TAN WCS with a sky position angle, for the celestial->detector rotation.

    Built from cdelt + a pc rotation matrix (not a cd matrix) so the
    pipeline's image_rotation_from_wcs can read the rotation via get_pc().
    """
    w = WCS(naxis=2)
    cd = pixel_scale_arcsec / 3600.0
    rot = np.deg2rad(pa_deg)
    w.wcs.cdelt = [-cd, cd]
    w.wcs.pc = np.array([[np.cos(rot), -np.sin(rot)],
                         [np.sin(rot), np.cos(rot)]])
    w.wcs.crpix = [shape[1] / 2 + 0.5, shape[0] / 2 + 0.5]
    w.wcs.crval = [150.0, 2.0]
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    w.pixel_shape = (shape[1], shape[0])   # (NAXIS1=Ncol, NAXIS2=Nrow)
    return w

shape = (40, 40)
ps = 0.11
# Two broadbands share an image grid (could differ in practice); WCS adds a 20 deg PA.
# With a WCS, indexing may be omitted: it defaults to numpy 'ij' (Nrow, Ncol).
img_pars = ImagePars(shape=shape, wcs=make_wcs(shape, ps, pa_deg=20.0))
roman_psf = galsim.Gaussian(fwhm=0.18)
Zc = 1.0
```

The source: a `BulgeDiskModel` for each broadband and for the continuum (same
model class, so they share morphology through the bare-key fallback), and an
`InclinedExponentialModel` for the emission line itself.

```{code-cell} python
bd_F087, bd_F184 = BulgeDiskModel(), BulgeDiskModel()
bd_cont = BulgeDiskModel()
line_model = InclinedExponentialModel()

cap_source = SourceModel(
    velocity_model=CenteredVelocityModel(),
    broadband_models={'F087': bd_F087, 'F184': bd_F184},
    emission_lines={'Halpha': EmissionLine(intensity=line_model, continuum=bd_cont)},
)

# Morphology as BARE keys -> shared across F087, F184, and the continuum.
# Flux as band/component-prefixed keys -> independent per band.
cap_truth = {
    'cosi': 0.55,
    'theta_int': 0.6,
    'g1': 0.03,
    'g2': -0.02,
    'vel.v0': 5.0,
    'vel.vcirc': 210.0,
    'vel.rscale': 0.35,
    # shared bulge+disk morphology (bare keys, no band prefix):
    'bulge_frac': 0.3,
    'disk_rscale': 0.35,
    'disk_h_over_r': 0.1,
    'disk_x0': 0.0,
    'disk_y0': 0.0,
    'bulge_hlr': 0.18,
    'bulge_h_over_hlr': 0.3,
    'bulge_x0': 0.0,
    'bulge_y0': 0.0,
    # independent broadband flux (band-prefixed):
    'F087.total_flux': 120.0,
    'F184.total_flux': 180.0,
    # emission line: its own exponential disk + flux + dispersion:
    'Halpha.flux': 90.0,
    'Halpha.rscale': 0.28,
    'Halpha.h_over_r': 0.1,
    'Halpha.x0': 0.0,
    'Halpha.y0': 0.0,
    'Halpha.dispersion': 55.0,
    # continuum reuses the shared bulge+disk morphology, with its own flux:
    'Halpha.cont.total_flux': 20.0,
    'z': Zc,
}
```

Build the four observations (two broadbands, two grism rolls) and the joint
task. The two rolls differ in dispersion angle; here we approximate that with
two grism configs whose WCS carry different position angles.

```{code-cell} python
rc_cap = RenderConfig(oversample=3)   # low for tutorial speed; auto-derive in production

# Broadband obs, one per band:
obs_bb = {}
for i, (band, model) in enumerate((('F087', bd_F087), ('F184', bd_F184))):
    clean = np.asarray(cap_source.render_broadband(
        cap_truth, build_image_obs(img_pars, psf=roman_psf, render_config=rc_cap,
                                   int_model=model, broadband_key=band), band))
    # deterministic per-band seeds; hash() is salted per process and would
    # make the noise realization change from run to run
    noisy, var = add_intensity_noise(clean, target_snr=100, seed=20 + i)
    obs_bb[band] = build_image_obs(img_pars, psf=roman_psf, render_config=rc_cap,
                                   int_model=model, broadband_key=band,
                                   data=jnp.asarray(noisy), variance=var)

# Grism obs, two rolls at different sky PAs:
obs_rolls = {}
for j, (roll, pa) in enumerate((('roll0', 20.0), ('roll90', 110.0))):
    gp_pars = ImagePars(shape=shape, wcs=make_wcs(shape, ps, pa_deg=pa))
    gpr = build_grism_pars_for_line(LINE_LAMBDAS['Halpha'], redshift=Zc,
                                    image_pars=gp_pars, dispersion=1.1)
    clean = np.asarray(cap_source.render_grism(
        cap_truth, build_grism_obs(gpr, z=Zc, psf=roman_psf, render_config=rc_cap)))
    noisy, var = add_intensity_noise(clean, target_snr=150, seed=30 + j)
    obs_rolls[roll] = build_grism_obs(gpr, z=Zc, psf=roman_psf, render_config=rc_cap,
                                      data=jnp.asarray(noisy), variance=var)
```

The full data vector: two broadband bands, two grism rolls, and (for context)
the underlying velocity field. The two rolls disperse the Halpha line in
different directions relative to the galaxy, which is what gives the joint fit
its handle on the kinematic axis.

```{code-cell} python
vel_ctx = np.asarray(cap_source.render_velocity(cap_truth, build_velocity_obs(img_pars)))

fig, axes = plt.subplots(2, 3, figsize=(13, 8))
panels = [
    (axes[0, 0], obs_bb['F087'].data, 'F087 broadband', 'viridis'),
    (axes[0, 1], obs_bb['F184'].data, 'F184 broadband', 'viridis'),
    (axes[0, 2], vel_ctx, 'LOS velocity (truth)', 'RdBu_r'),
    (axes[1, 0], obs_rolls['roll0'].data, 'Halpha grism, roll 0 deg', 'magma'),
    (axes[1, 1], obs_rolls['roll90'].data, 'Halpha grism, roll 90 deg', 'magma'),
]
for ax, img, title, cmap in panels:
    arr = np.asarray(img)
    kw = dict(vmin=-np.max(np.abs(arr)), vmax=np.max(np.abs(arr))) if cmap == 'RdBu_r' else {}
    im = ax.imshow(arr, origin='lower', cmap=cmap, **kw)
    ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046)
axes[1, 2].axis('off')
plt.tight_layout(); plt.show()
```

```{code-cell} python
# Priors: sample geometry + velocity + shared morphology + per-band flux + line;
# fix the thickness params and z. Shear at +/- 0.25; TF prior on vcirc.
cap_priors = PriorDict({
    'cosi': TruncatedNormal(0.55, 0.15, 0.05, 0.99),
    'theta_int': TruncatedNormal(0.6, 0.3, 0.0, np.pi),
    'g1': Uniform(-0.25, 0.25),
    'g2': Uniform(-0.25, 0.25),
    'vel.v0': Gaussian(5.0, 10.0),
    'vel.vcirc': TruncatedNormal(210.0, 37.0, 80.0, 400.0),     # TF prior (Ubler+2017)
    'vel.rscale': TruncatedNormal(0.35, 0.1, 0.05, 1.0),
    'bulge_frac': TruncatedNormal(0.3, 0.1, 0.0, 0.9),
    'disk_rscale': TruncatedNormal(0.35, 0.08, 0.05, 1.0),
    'disk_h_over_r': 0.1,
    'disk_x0': 0.0,
    'disk_y0': 0.0,
    'bulge_hlr': TruncatedNormal(0.18, 0.05, 0.02, 0.6),
    'bulge_h_over_hlr': 0.3,
    'bulge_x0': 0.0,
    'bulge_y0': 0.0,
    'F087.total_flux': TruncatedNormal(120.0, 25.0, 30.0, 400.0),
    'F184.total_flux': TruncatedNormal(180.0, 30.0, 30.0, 500.0),
    'Halpha.flux': TruncatedNormal(90.0, 20.0, 20.0, 250.0),
    'Halpha.rscale': TruncatedNormal(0.28, 0.08, 0.05, 1.0),
    'Halpha.h_over_r': 0.1,
    'Halpha.x0': 0.0,
    'Halpha.y0': 0.0,
    'Halpha.dispersion': TruncatedNormal(55.0, 20.0, 5.0, 150.0),
    'Halpha.cont.total_flux': TruncatedNormal(20.0, 12.0, 0.0, 150.0),
    'z': Zc,
})

cap_task = InferenceTask.from_obs(
    cap_source, cap_priors,
    image_obs=obs_bb,
    grism_obs=obs_rolls,
)
theta_cap = jnp.array([cap_truth[n] for n in cap_task.sampled_names])
print(f"sampled dimension: {cap_task.n_params}")

# The bulge+disk morphology is SHARED across F087 and F184: because the prior
# uses bare keys (disk_rscale, bulge_hlr, bulge_frac) rather than band-prefixed
# ones, each appears once in the sampled vector and both bands resolve to it.
# Only the flux is band-specific. (Were morphology not shared, these would
# double to F087.disk_rscale + F184.disk_rscale, etc.)
shared_morph = [n for n in cap_task.sampled_names
                if n in ('bulge_frac', 'disk_rscale', 'bulge_hlr')]
per_band_flux = [n for n in cap_task.sampled_names if n.endswith('total_flux')]
print(f"shared morphology (one copy, both bands): {shared_morph}")
print(f"per-band flux (independent):              {per_band_flux}")
print(f"joint log-posterior at truth: {float(cap_task.log_posterior(theta_cap)):.2f}")
```

An optimizer run on the full data vector finds a point estimate (a sampler would
characterize the full posterior). We start from a mild perturbation of truth and
take a single descent.

```{code-cell} python
grad_fn = cap_task.get_log_posterior_and_grad_fn()
def neg_logpost(th):
    val, grad = grad_fn(jnp.asarray(th))
    return -float(val), -np.asarray(grad)

rng = np.random.default_rng(0)
x0 = theta_cap + 0.05 * np.abs(theta_cap) * rng.standard_normal(cap_task.n_params)
res_cap = multi_start_minimize(
    neg_logpost, x0=np.asarray(x0), bounds=cap_task.get_bounds(),
    n_starts=1, jac=True, options={'maxiter': 150},
)
cap_fit = dict(zip(cap_task.sampled_names, (float(v) for v in res_cap.x)))

res = plot_parameter_recovery(
    true_values={n: cap_truth[n] for n in cap_task.sampled_names},
    recovered_values=cap_fit,
    output_dir=Path(tempfile.mkdtemp()),
    test_name='roman_capstone_optimizer',
)
Image(res['output_path'])
```

The MAP point estimate recovers the well-constrained parameters (fluxes, scale
lengths) but not every parameter robustly: with 16 free parameters and the
shape/shear and `vcirc - cosi` degeneracies, the shear in particular needs the
full posterior to characterize. That is the sampler's job, run via the commented
cell below.

```{code-cell} python
# Run it (commented; minutes to hours at production settings):
# config = NumpyroSamplerConfig(
#     n_warmup=50, n_samples=500, n_chains=4, chain_method='vectorized',
#     precondition='laplace', max_tree_depth=10, seed=42, progress=True,
# )
# result = build_sampler('numpyro', cap_task, config).run()
```

This is the template for a realistic multi-band, multi-roll Roman fit:
geometry shared across every channel, morphology shared across the broadbands
and the continuum, flux free per band, the emission line on its own profile, and
the whole forward model rendered through WCS-aware observations. Swap in real
data + variance arrays and the same task runs unchanged.

---

## Where to go next

- **intensity_models.md** -- the intensity model zoo, multi-component sources,
  and render-grid control in depth.
- **grism.md** -- datacube assembly and grism dispersion mechanics.
- **sampling.md** -- inference in depth: emcee, nautilus, NumPyro NUTS, the
  Laplace preconditioner, and convergence diagnostics.
- **tng50_data.md** -- fitting realistic TNG50 mock galaxies.

Worked references in the test suite: `tests/test_likelihood_slices.py`
(forward-model validation), `tests/test_optimizer_recovery.py` (gradient fits),
and `tests/test_flagship.py` (the full Roman case end-to-end).
