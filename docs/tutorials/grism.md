---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Grism & Datacube Tutorial

A slitless grism disperses each spatial pixel along a single direction on the detector, mapping wavelength to position. For a rotating galaxy observed at an emission line, the receding and approaching sides shift by `±λ_rest * v_LOS / c` along the dispersion direction, so the velocity gradient appears as an asymmetry in the dispersed image between the two halves of the source. This tutorial walks through the forward model from emission lines + velocity field, to 3D datacube, to dispersed image, and finally to a JAX-compatible likelihood for inference.

**NOTE:** Convert to a Jupyter Notebook with:
```bash
jupytext --to ipynb docs/tutorials/grism.md
```

---

## Key Classes

| Class | Purpose |
|---|---|
| `LineSpec`, `EmissionLine` | Per-line wavelength + parameter routing (`kl_pipe.spectral`) |
| `SpectralConfig` | Bundle of lines + LSF mode + spectral oversampling |
| `SpectralModel` | Builds a 3D datacube `(Nrow, Ncol, Nλ)` from velocity + intensity + line list |
| `CubePars` | Wavelength grid for cube assembly |
| `GrismPars` | Instrument config: dispersion direction, magnitude, reference wavelength |
| `GrismObs` | Observation container: PSF + dispersion + data + variance + oversample |
| `KLModel` (with `spectral_model=`) | Unified interface for imaging, datacubes, and grism rendering |

The cube-to-grism flow is: `SpectralModel.build_cube` → per-slice PSF convolution → `disperse_cube` → 2D pixel-response → detector image. `KLModel.render_grism` does this in one call.

---

## Example 1: Velocity + Intensity Maps

Start with the same single-disk forward model used in the quickstart, at Roman's pixel scale.

```{code-cell} python
import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import matplotlib.pyplot as plt

from kl_pipe.parameters import ImagePars
from kl_pipe.velocity import CenteredVelocityModel
from kl_pipe.intensity import InclinedExponentialModel
from kl_pipe.utils import build_map_grid_from_image_pars

image_pars = ImagePars(shape=(32, 32), pixel_scale=0.11, indexing='ij')   # roman scale, 32x32 cutout

vel_model = CenteredVelocityModel()
int_model = InclinedExponentialModel()

vel_pars = {
    'cosi': 0.5, 'theta_int': 0.7, 'g1': 0.0, 'g2': 0.0,
    'v0': 10.0, 'vcirc': 200.0, 'vel_rscale': 0.5,
}
int_pars = {
    'cosi': 0.5, 'theta_int': 0.7, 'g1': 0.0, 'g2': 0.0,
    'flux': 100.0, 'int_rscale': 0.3, 'int_h_over_r': 0.1,
    'int_x0': 0.0, 'int_y0': 0.0,
}

theta_vel = vel_model.pars2theta(vel_pars)
theta_int = int_model.pars2theta(int_pars)

X, Y = build_map_grid_from_image_pars(image_pars)
vmap = vel_model(theta_vel, 'obs', X, Y)
imap = int_model.render_unconvolved(theta_int, image_pars)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
im0 = axes[0].imshow(np.array(imap), origin='lower', cmap='magma')
axes[0].set_title('Intensity (face-on disk)')
plt.colorbar(im0, ax=axes[0], label='Flux')

vmax = float(jnp.max(jnp.abs(vmap - vel_pars['v0'])))
im1 = axes[1].imshow(
    np.array(vmap), origin='lower', cmap='RdBu_r',
    vmin=vel_pars['v0'] - vmax, vmax=vel_pars['v0'] + vmax,
)
axes[1].set_title('LOS velocity')
plt.colorbar(im1, ax=axes[1], label='km/s')
fig.tight_layout()
plt.show()
```

These two maps are the inputs to the spectral pipeline: the intensity sets the emission morphology and the velocity sets the per-pixel Doppler shift.

---

## Example 2: Spectral Configuration

A `SpectralConfig` declares which emission lines to model, how to broaden them, and how finely to sample the wavelength axis. Single-line H-alpha is the simplest case:

```{code-cell} python
from kl_pipe.spectral import (
    SpectralConfig, SpectralModel,
    halpha_line, make_spectral_config,
    roman_grism_R, C_KMS, HALPHA, NII_6583,
)

config = SpectralConfig(
    lines=(halpha_line(),),
    lsf_mode='absorbed',       # see "Known limitations" below
    spectral_oversample=5,
)
sm = SpectralModel(config, int_model, vel_model)
print(f"Spectral parameters: {sm.PARAMETER_NAMES}")
```

The `halpha_line()` factory returns an `EmissionLine(LineSpec)` with `lambda_rest = 656.28 nm` and `param_prefix = 'Ha'`. Each line contributes a `<prefix>_flux` parameter (per-line) and a `<prefix>_cont` parameter (continuum normalization at the line). The shared `vel_dispersion` parameter controls the intrinsic line width in the cube.

**Roman grism resolving power at the observed wavelength**:

```{code-cell} python
z = 1.0
lam_obs = HALPHA.lambda_rest * (1 + z)             # 1312.56 nm at z=1
R = roman_grism_R(lam_obs)                         # ~605
sigma_inst = C_KMS / (2.355 * R)                   # ~213 km/s
print(f"At z={z}: lam_obs = {lam_obs:.1f} nm, R = {R:.0f}, sigma_inst = {sigma_inst:.0f} km/s")
```

`sigma_inst` is the published spectral-resolution element converted to a velocity sigma; see Example 8 ("Known limitations") for how it is currently consumed.

---

## Example 3: Datacube Assembly

`CubePars` defines the wavelength grid; `SpectralModel.build_cube` evaluates the model on that grid:

```{code-cell} python
from kl_pipe.spectral import CubePars

lam_center = HALPHA.lambda_rest * (1 + z)
dlam = lam_center * 2000.0 / C_KMS                 # +/- 2000 km/s window in lambda
cp = CubePars.from_range(image_pars, lam_center - dlam, lam_center + dlam, 1.0)
print(f"CubePars: {cp.n_lambda} bins, delta_lambda = {float(cp.delta_lambda):.2f} nm")

theta_spec = jnp.array([z, 50.0, 100.0, 0.01])     # z, vel_dispersion, Ha_flux, Ha_cont
cube = sm.build_cube(theta_spec, theta_vel, theta_int, cp)
print(f"Datacube shape: {cube.shape}")
```

Inspecting spaxel spectra along the kinematic major axis recovers the Doppler shift:

```{code-cell} python
cr, cc = 16, 16
pixels = {
    'center':       (cr, cc),
    'approaching':  (cr, max(cc - 6, 0)),
    'receding':     (cr, min(cc + 6, 31)),
    'minor axis':   (min(cr + 6, 31), cc),
}

fig, ax = plt.subplots(figsize=(8, 4))
lam = np.array(cp.lambda_grid)
for label, (r, c) in pixels.items():
    ax.plot(lam, np.array(cube[r, c, :]), label=f'{label} ({r},{c})')
ax.axvline(lam_center, color='gray', ls='--', alpha=0.5, label=f'Ha obs ({lam_center:.1f} nm)')
ax.set_xlabel('Wavelength (nm)')
ax.set_ylabel('Flux density')
ax.set_title('Spaxel spectra')
ax.legend(fontsize=8)
fig.tight_layout()
plt.show()
```

Approaching and receding pixels are blue- and red-shifted relative to the systemic-velocity centerline; minor-axis pixels sit on the centerline.

The library provides a multi-panel diagnostic for routine inspection:

```{code-cell} python
from kl_pipe.diagnostics.datacube import plot_datacube_overview

fig = plot_datacube_overview(
    cube, np.array(cp.lambda_grid),
    imap=np.array(imap), vmap=np.array(vmap),
    lam_center=lam_center, v0=vel_pars['v0'],
    title='Datacube overview — Ha at z=1',
)
plt.show()
```

---

## Example 4: Grism Dispersion

`GrismPars` specifies the dispersion direction, plate scale, and reference wavelength. `build_grism_pars_for_line` is a convenience for a single-line target:

```{code-cell} python
from kl_pipe.dispersion import GrismPars, disperse_cube, build_grism_pars_for_line

gp = build_grism_pars_for_line(
    HALPHA.lambda_rest, redshift=z,
    image_pars=image_pars, dispersion=1.1,         # Roman grism: ~1.1 nm/pix
)
print(f"GrismPars: dispersion={gp.dispersion} nm/pix, lambda_ref={gp.lambda_ref:.1f} nm, "
      f"angle={gp.dispersion_angle:.1f} rad")

cp_grism = gp.to_cube_pars(z=z)                    # wavelength grid matched to grism coverage
cube_for_grism = sm.build_cube(theta_spec, theta_vel, theta_int, cp_grism)
grism = disperse_cube(cube_for_grism, gp, cp_grism.lambda_grid)
print(f"Dispersed image shape: {grism.shape}")
```

Compare the dispersed image to the wavelength-stacked (no-dispersion) image. The stacked image is the broadband-equivalent integral; dispersion remaps it along the dispersion axis:

```{code-cell} python
dl = float(cp_grism.lambda_grid[1] - cp_grism.lambda_grid[0])
stacked = np.array(jnp.sum(cube_for_grism, axis=2) * dl)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].imshow(stacked, origin='lower', cmap='magma')
axes[0].set_title('Spectrally stacked (no dispersion)')

im = axes[1].imshow(np.array(grism), origin='lower', cmap='magma')
axes[1].set_title(f'Grism image (angle={gp.dispersion_angle:.1f} rad)')
plt.colorbar(im, ax=axes[1])
fig.tight_layout()
plt.show()
```

---

## Example 5: Rendering with KLModel

`KLModel` with a `spectral_model` exposes a unified interface that takes a single composite parameter vector and dispatches to imaging, datacube, or grism rendering. Subtracting a model rendered at `vcirc = 0` from the full model isolates the contribution of rotation to the dispersed image:

```{code-cell} python
from kl_pipe.model import KLModel

shared_pars = {'cosi', 'theta_int', 'g1', 'g2'}
kl = KLModel(vel_model, int_model, shared_pars=shared_pars, spectral_model=sm)

spec_dict = {'z': z, 'vel_dispersion': 50.0, 'Ha_flux': 100.0, 'Ha_cont': 0.01}
merged = {**vel_pars, **int_pars, **spec_dict}
theta = kl.pars2theta(merged)

grism_rot   = np.array(kl.render_grism(theta, gp))
theta_norot = kl.pars2theta({**merged, 'vcirc': 0.0})
grism_norot = np.array(kl.render_grism(theta_norot, gp))
diff = grism_rot - grism_norot

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
axes[0].imshow(grism_rot,   origin='lower', cmap='magma');   axes[0].set_title('vcirc=200 km/s')
axes[1].imshow(grism_norot, origin='lower', cmap='magma');   axes[1].set_title('vcirc=0 km/s')
vmax = np.max(np.abs(diff))
im = axes[2].imshow(diff, origin='lower', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
axes[2].set_title('vcirc=200 minus vcirc=0')
plt.colorbar(im, ax=axes[2])
fig.tight_layout()
plt.show()
```

The difference image is antisymmetric across the kinematic major axis: pixels on the approaching side lose flux at the systemic wavelength (their flux has been Doppler-shifted into another bin) and pixels on the receding side gain it. This antisymmetry is the observable kinematic lensing exploits — shear breaks the symmetry between the kinematic and morphological axes, and the joint photometry + grism fit decouples them.

The dispersion direction controls how strongly the rotation term appears. With the kinematic major axis aligned to the dispersion axis (`theta_int = 0` and `dispersion_angle = 0`), the rotation contribution is maximally extended along the dispersion direction; orthogonal alignment suppresses it:

```{code-cell} python
from kl_pipe.diagnostics.grism import plot_dispersion_angle_study

pars_study   = {**vel_pars, **int_pars, 'theta_int': 0.0}
merged_study = {**pars_study, **spec_dict}
theta_study  = kl.pars2theta(merged_study)

gp_ref    = build_grism_pars_for_line(HALPHA.lambda_rest, redshift=z, image_pars=image_pars, dispersion=1.1)
cp_study  = gp_ref.to_cube_pars(z=z)
cube_study = kl.render_cube(theta_study, cp_study)
dl = float(cp_study.lambda_grid[1] - cp_study.lambda_grid[0])
broadband = np.array(jnp.sum(cube_study, axis=2) * dl)

def render_at_angle(angle):
    gp_a = GrismPars(
        image_pars=image_pars, dispersion=1.1,
        lambda_ref=HALPHA.lambda_rest * 2.0,
        dispersion_angle=angle,
    )
    return np.array(kl.render_grism(theta_study, gp_a, cube_pars=cp_study))

fig = plot_dispersion_angle_study(render_at_angle, broadband,
                                  title='Dispersion angle study (theta_int=0)')
plt.show()
```

---

## Example 6: Grism Likelihood + InferenceTask

The grism likelihood is a Gaussian chi-squared on the dispersed 2D image, JIT-compiled and gradient-friendly. Wire it into an `InferenceTask` with `from_grism_obs`:

```{code-cell} python
import galsim
from kl_pipe.observation import build_grism_obs
from kl_pipe.priors import PriorDict, Uniform
from kl_pipe.sampling.task import InferenceTask
from kl_pipe.likelihood import create_jitted_likelihood_grism

psf = galsim.Gaussian(fwhm=0.18)                     # roman-like PSF

# render a clean grism image, then add Gaussian noise calibrated to SNR=1000
obs_noiseless = build_grism_obs(gp, z=z, psf=psf)
clean = np.array(kl.render_grism(theta, obs_noiseless))
snr = 1000
variance = float(np.sum(clean**2)) / snr**2
rng = np.random.default_rng(0)
data = clean + rng.normal(0.0, np.sqrt(variance), size=clean.shape)

obs_grism = build_grism_obs(gp, z=z, psf=psf, data=jnp.asarray(data), variance=variance)

# 1) raw likelihood (no priors involved)
log_like_fn = create_jitted_likelihood_grism(kl, obs_grism)
print(f"log L at truth: {float(log_like_fn(theta)):.2f}")
print(f"log L at vcirc=150 (truth=200): {float(log_like_fn(theta.at[kl.PARAMETER_NAMES.index('vcirc')].set(150.0))):.2f}")

# 2) wrap into an InferenceTask with a PriorDict
priors_dict = {name: float(merged[name]) for name in kl.PARAMETER_NAMES}    # all fixed by default
priors_dict['Ha_flux'] = Uniform(20.0, 200.0)                                # let one parameter float
priors_dict['vcirc']   = Uniform(50.0, 400.0)
priors = PriorDict(priors_dict)

task = InferenceTask.from_grism_obs(kl, priors, obs_grism)
print(f"Sampled parameters: {task.sampled_names}")
print(f"Log posterior at the true sampled values: "
      f"{float(task.log_posterior(jnp.array([merged['Ha_flux'], merged['vcirc']]))):.2f}")
```

Gradients work out of the box, so any JAX-aware optimizer or sampler can drive the task:

```{code-cell} python
from scipy.optimize import minimize

grad_fn = jax.jit(jax.grad(log_like_fn))

free_names   = ['Ha_flux', 'vcirc']
free_indices = [kl.PARAMETER_NAMES.index(n) for n in free_names]

theta_init = np.array(theta)
rng = np.random.default_rng(42)
for i in free_indices:                                # perturb the initial guess
    theta_init[i] *= 1 + 0.15 * rng.standard_normal()

def neg_log_like(x):
    full = jnp.asarray(theta_init).at[jnp.asarray(free_indices)].set(jnp.asarray(x))
    return -float(log_like_fn(full))

def neg_grad(x):
    full = jnp.asarray(theta_init).at[jnp.asarray(free_indices)].set(jnp.asarray(x))
    return -np.asarray(grad_fn(full))[free_indices]

result = minimize(
    neg_log_like, x0=np.array([theta_init[i] for i in free_indices]),
    method='L-BFGS-B', jac=neg_grad,
    bounds=[(20.0, 200.0), (50.0, 400.0)],
    options={'maxiter': 500, 'ftol': 1e-8},
)

for name, val in zip(free_names, result.x):
    truth = merged[name]
    print(f"{name:10s}: recovered {float(val):7.2f}  (truth {truth:7.2f}, "
          f"err {100*abs(float(val)-truth)/abs(truth):5.2f}%)")
```

---

## Example 7: Joint Photometry + Grism Likelihood

Joint inference adds a broadband image term. `InferenceTask.from_joint_photometry_grism_obs` constructs the corresponding task; the underlying log-likelihood is the sum of the two independent Gaussian terms.

```{code-cell} python
from kl_pipe.observation import build_image_obs
from kl_pipe.likelihood import create_jitted_likelihood_joint_photometry_grism

# build a synthetic broadband image at the same parameters
obs_int_noiseless = build_image_obs(image_pars, int_model=kl.intensity_model, psf=psf)
clean_int = np.array(kl.intensity_model.render_image(
    kl.get_intensity_pars(theta), obs=obs_int_noiseless,
    render_config=obs_int_noiseless.render_config,
))
variance_int = float(np.sum(clean_int**2)) / 1000**2
data_int = clean_int + np.random.default_rng(1).normal(0.0, np.sqrt(variance_int), size=clean_int.shape)
obs_int = build_image_obs(
    image_pars, int_model=kl.intensity_model, psf=psf,
    data=jnp.asarray(data_int), variance=variance_int,
)

# joint task and joint log-posterior
task_joint = InferenceTask.from_joint_photometry_grism_obs(kl, priors, obs_int, obs_grism)
print(f"Sampled parameters: {task_joint.sampled_names}")

# the raw joint likelihood is also exposed for direct evaluation
log_like_joint = create_jitted_likelihood_joint_photometry_grism(kl, obs_int, obs_grism,
                                                                 render_config_int=obs_int.render_config)
print(f"Joint log L at truth: {float(log_like_joint(theta)):.2f}")
```

The joint task plugs into the sampler infrastructure the same way `from_intensity_obs` / `from_velocity_obs` / `from_joint_obs` tasks do; see `docs/tutorials/sampling.md` for MCMC examples (numpyro is recommended for production joint runs).

---

## Example 8: Multi-Line Spectroscopy (H-alpha + [N II])

The H-alpha + [N II] 6548/6583 triplet shares geometry but allows per-line flux normalizations:

```{code-cell} python
from kl_pipe.spectral import halpha_nii_lines

lines = halpha_nii_lines()
for line in lines:
    print(f"  {line.line_spec.name:12s} lambda_rest = {line.line_spec.lambda_rest:6.2f} nm  "
          f"prefix = '{line.line_spec.param_prefix}'  own_params = {line.own_params}")

config_3 = make_spectral_config()                     # default: Ha + NII triplet
sm_3 = SpectralModel(config_3, int_model, vel_model)
print(f"\nSpectral parameters: {sm_3.PARAMETER_NAMES}")
```

Render a spatially integrated spectrum to inspect the triplet:

```{code-cell} python
lam_min = 654.0 * (1 + z) - 10
lam_max = 659.0 * (1 + z) + 10
cp_3 = CubePars.from_range(image_pars, lam_min, lam_max, 0.3)

theta_vel_static = vel_model.pars2theta({**vel_pars, 'vcirc': 0.0, 'v0': 0.0})
# z, vel_disp, Ha_flux, Ha_cont, NII6548_flux, NII6548_cont, NII6583_flux, NII6583_cont
theta_spec_3 = jnp.array([z, 50.0, 100.0, 0.0, 30.0, 0.0, 90.0, 0.0])

cube_3 = sm_3.build_cube(theta_spec_3, theta_vel_static, theta_int, cp_3)
spectrum = np.array(jnp.sum(cube_3, axis=(0, 1)))

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(np.array(cp_3.lambda_grid), spectrum, 'k-')
for ls in lines:
    ax.axvline(ls.line_spec.lambda_rest * (1 + z), color='gray', ls='--', alpha=0.5)
    ax.text(ls.line_spec.lambda_rest * (1 + z), ax.get_ylim()[1] * 0.9,
            ls.line_spec.name, ha='center', fontsize=8, color='gray')
ax.set_xlabel('Wavelength (nm)')
ax.set_ylabel('Spatially integrated flux density')
ax.set_title('Ha + [N II] triplet at z=1 (no rotation)')
fig.tight_layout()
plt.show()
```

Per-line fluxes are independent parameters, enabling line-ratio measurements (e.g. [N II] 6583 / H-alpha as a metallicity / ionization diagnostic) alongside the kinematic fit.

---

## Diagnostic Functions

`kl_pipe.diagnostics` provides reusable plotting helpers for datacubes and grism images. Available functions:

- `plot_datacube_overview()` — intensity + velocity + stacked flux + wavelength-channel slices
- `plot_grism_overview()` — 3-row composite: input maps, wavelength channels, grism with the rotating - non-rotating difference
- `plot_dispersion_angles()` — grism rendered at the four cardinal dispersion angles
- `plot_dispersion_angle_study()` — deep-dive comparison of grism vs broadband per angle

Example use:

```{code-cell} python
from kl_pipe.diagnostics.grism import plot_grism_overview

cube_diag  = kl.render_cube(theta, cp_grism)
grism_diag = np.array(kl.render_grism(theta, gp, cube_pars=cp_grism))
grism_norot_diag = np.array(kl.render_grism(theta_norot, gp, cube_pars=cp_grism))

fig = plot_grism_overview(
    np.array(cube_diag), grism_diag,
    np.array(cp_grism.lambda_grid), gp,
    imap=np.array(imap), vmap=np.array(vmap),
    grism_norot=grism_norot_diag, v0=vel_pars['v0'],
    title='Grism overview — Ha at z=1',
)
plt.show()
```

---

## Known Limitations

The grism subsystem is in a working but evolving state. Two specific caveats apply to the current rendering chain:

1. **Line broadening absorbs the instrumental sigma.** `SpectralModel.build_cube` applies emission-line broadening as `sigma_eff = sqrt(vel_disp^2 + sigma_inst^2)`, where `sigma_inst` is derived from the quoted grism resolving power R. For slitless geometry the spectral resolution element is the PSF projected along the dispersion axis — so applying both the in-quadrature `sigma_inst` here and a PSF convolution per wavelength slice likely double-counts the same physical effect. In practice, **`vel_dispersion` is poorly identified by the grism likelihood under the current model**. A future refactor will drop the `sigma_inst` absorption entirely; see `docs/plans/phase2_lsf_refactor.md`.

2. **Photometric and emission centroids are shared.** Both broadband image rendering (through `kl_model.intensity_model`) and grism cube assembly (through the emission spatial component of the same `intensity_model`) currently use a single `int_x0` / `int_y0`. If the photometric and grism observations have independent astrometric solutions, the shared centroid is the wrong degree of freedom — they should each carry their own. A planned `SourceModel` refactor decouples per-component intensity models and centroids; see `docs/plans/phase3_sourcemodel_refactor.md`.

---

## Reference Tests

- `tests/test_grism_core.py` — spectral parameter routing, dispersion sign and angle, KLModel integration, JIT + grad.
- `tests/test_datacube.py` — cube assembly, flux conservation, spectral oversampling convergence, JAX vs numpy reference.
- `tests/test_cube_psf.py` — per-slice PSF convolution under oversampling.
- `tests/test_grism_likelihood.py` — grism likelihood unit tests, parameter-slice tests, smoke optimizer recovery.

---

## TODOs

- Add MCMC inference example for grism-only and joint photometry + grism (see `docs/tutorials/sampling.md` for the velocity / intensity / joint analogues).
- Document the planned `SourceModel` API (`docs/plans/phase3_sourcemodel_refactor.md`) once it lands; the API will replace `KLModel` and resolve the centroid-coupling limitation above.
