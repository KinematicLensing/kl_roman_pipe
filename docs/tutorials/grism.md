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

The Roman Space Telescope will observe galaxies through a slitless grism, dispersing each spatial pixel along wavelength. For rotating galaxies, this encodes kinematic information — Doppler-shifted emission lines — directly into 2D grism images. This tutorial walks through the full spectral pipeline: building 3D datacubes from velocity + intensity models, dispersing them into grism images, and visualizing the velocity signature that enables kinematic lensing.

**NOTE:** Convert to Jupyter Notebook with:
```bash
jupytext --to ipynb docs/tutorials/grism.md
```

---

## 1. Setup & Physical Picture

We start with the same velocity and intensity models from the quickstart tutorial, now at Roman's pixel scale (0.11"/pix).

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

# roman pixel scale, 32x32 cutout
image_pars = ImagePars(shape=(32, 32), pixel_scale=0.11, indexing='ij')

vel_model = CenteredVelocityModel()
int_model = InclinedExponentialModel()

# galaxy parameters
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

# render maps
X, Y = build_map_grid_from_image_pars(image_pars)
vmap = vel_model(theta_vel, 'obs', X, Y)
imap = int_model.render_unconvolved(theta_int, image_pars)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
im0 = axes[0].imshow(np.array(imap), origin='lower', cmap='magma')
axes[0].set_title('Intensity map')
plt.colorbar(im0, ax=axes[0], label='Flux')

vmax = float(jnp.max(jnp.abs(vmap - vel_pars['v0'])))
im1 = axes[1].imshow(
    np.array(vmap), origin='lower', cmap='RdBu_r',
    vmin=vel_pars['v0'] - vmax, vmax=vel_pars['v0'] + vmax
)
axes[1].set_title('Velocity map')
plt.colorbar(im1, ax=axes[1], label='km/s')
fig.tight_layout()
plt.show()
```

The intensity map shows an inclined exponential disk; the velocity map shows the classic rotation pattern with approaching (blue) and receding (red) sides. A grism disperses each spatial pixel along wavelength — so rotation creates wavelength-dependent spatial structure in the 2D grism image.

---

## 2. Emission Lines & Spectral Configuration

The spectral model is configured by choosing emission lines and instrumental parameters.

```{code-cell} python
from kl_pipe.spectral import (
    LineSpec, EmissionLine, SpectralConfig, SpectralModel,
    halpha_line, halpha_nii_lines, make_spectral_config,
    roman_grism_R, C_KMS, HALPHA, NII_6583,
)

# single H-alpha line (656.28 nm rest)
ha = halpha_line()
print(f"H-alpha: lambda_rest = {HALPHA.lambda_rest} nm, prefix = '{HALPHA.param_prefix}'")
print(f"  own_params (overrides from broadband): {ha.own_params}")

# at z=1, what is the Roman grism resolving power?
z = 1.0
lam_obs = HALPHA.lambda_rest * (1 + z)  # ~1312.56 nm
R = roman_grism_R(lam_obs)
sigma_inst = C_KMS / (2.355 * R)  # instrumental broadening in km/s
print(f"\nAt z={z}: lambda_obs = {lam_obs:.1f} nm, R = {R:.0f}, sigma_inst = {sigma_inst:.0f} km/s")
```

The `SpectralConfig` bundles lines + LSF mode + spectral oversampling:

```{code-cell} python
config = SpectralConfig(
    lines=(halpha_line(),),
    lsf_mode='absorbed',       # LSF broadening absorbed into line Gaussian
    spectral_oversample=5,     # 5x oversampling of wavelength grid
)

sm = SpectralModel(config, int_model, vel_model)
print(f"SpectralModel parameters: {sm.PARAMETER_NAMES}")
```

The `lsf_mode='absorbed'` means the instrumental line spread function is folded into the emission line width analytically: `sigma_eff = sqrt(vel_dispersion^2 + sigma_inst^2)`. This avoids an expensive 1D convolution along the wavelength axis.

---

## 3. Building a Datacube

A datacube is a 3D array `(Nrow, Ncol, Nlambda)` — an image at each wavelength. `CubePars` defines the wavelength grid.

```{code-cell} python
from kl_pipe.spectral import CubePars

# option 1: explicit wavelength range and spacing (nm)
lam_center = HALPHA.lambda_rest * (1 + z)
dlam = lam_center * 2000.0 / C_KMS  # +/- 2000 km/s window
cp = CubePars.from_range(image_pars, lam_center - dlam, lam_center + dlam, 1.0)
print(f"from_range: {cp.n_lambda} wavelength bins, delta_lambda = {float(cp.delta_lambda):.2f} nm")

# option 2: spacing matched to resolving power R
cp_R = CubePars.from_R(image_pars, lam_center - dlam, lam_center + dlam, R=1000)
print(f"from_R:     {cp_R.n_lambda} wavelength bins, delta_lambda = {float(cp_R.delta_lambda):.2f} nm")
```

Now build the datacube:

```{code-cell} python
# spectral parameters: z, vel_dispersion, Ha_flux, Ha_cont
theta_spec = jnp.array([z, 50.0, 100.0, 0.01])

cube = sm.build_cube(theta_spec, theta_vel, theta_int, cp)
print(f"Datacube shape: {cube.shape}  (Nrow, Ncol, Nlambda)")
```

The library provides a multi-panel diagnostic:

```{code-cell} python
from kl_pipe.diagnostics.datacube import plot_datacube_overview

fig = plot_datacube_overview(
    cube, np.array(cp.lambda_grid),
    imap=np.array(imap), vmap=np.array(vmap),
    lam_center=lam_center, v0=vel_pars['v0'],
    title='Datacube overview — H-alpha at z=1',
)
plt.show()
```

Let's look at spectra at specific spatial pixels to see the Doppler shifts:

```{code-cell} python
cr, cc = 16, 16  # center pixel
pixels = {
    'center': (cr, cc),
    'approaching': (cr, max(cc - 6, 0)),
    'receding': (cr, min(cc + 6, 31)),
    'minor axis': (min(cr + 6, 31), cc),
}

fig, ax = plt.subplots(figsize=(8, 4))
lam = np.array(cp.lambda_grid)
for label, (r, c) in pixels.items():
    ax.plot(lam, np.array(cube[r, c, :]), label=f'{label} ({r},{c})')
ax.axvline(lam_center, color='gray', ls='--', alpha=0.5, label=f'Ha obs ({lam_center:.1f} nm)')
ax.set_xlabel('Wavelength (nm)')
ax.set_ylabel('Flux density')
ax.set_title('Spaxel spectra — Doppler shifts from rotation')
ax.legend(fontsize=8)
fig.tight_layout()
plt.show()
```

Each spatial pixel has a Doppler-shifted emission line. The approaching side (blue-shifted) and receding side (red-shifted) are clearly separated.

---

## 4. Grism Dispersion

The grism maps each wavelength slice to a different spatial offset. `GrismPars` defines the instrument configuration.

```{code-cell} python
from kl_pipe.dispersion import GrismPars, disperse_cube, build_grism_pars_for_line

# convenience factory: centers wavelength grid on Ha at z=1
gp = build_grism_pars_for_line(
    HALPHA.lambda_rest, redshift=z,
    image_pars=image_pars, dispersion=1.1,  # nm/pixel (Roman)
)
print(f"GrismPars: dispersion={gp.dispersion} nm/pix, "
      f"lambda_ref={gp.lambda_ref:.1f} nm, angle={gp.dispersion_angle:.1f} rad")
```

`GrismPars.to_cube_pars()` automatically computes the wavelength grid from the grism parameters:

```{code-cell} python
cp_grism = gp.to_cube_pars(z=1.0)
print(f"Auto-computed: {cp_grism.n_lambda} wavelength bins, "
      f"range [{float(cp_grism.lambda_grid[0]):.1f}, {float(cp_grism.lambda_grid[-1]):.1f}] nm")
```

Now disperse the datacube into a 2D grism image:

```{code-cell} python
cube_for_grism = sm.build_cube(theta_spec, theta_vel, theta_int, cp_grism)
grism = disperse_cube(cube_for_grism, gp, cp_grism.lambda_grid)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
dl = float(cp_grism.lambda_grid[1] - cp_grism.lambda_grid[0])
stacked = np.array(jnp.sum(cube_for_grism, axis=2) * dl)
axes[0].imshow(stacked, origin='lower', cmap='magma')
axes[0].set_title('Spectrally stacked (no dispersion)')

im = axes[1].imshow(np.array(grism), origin='lower', cmap='magma')
axes[1].set_title(f'Grism image (angle={gp.dispersion_angle:.1f} rad)')
plt.colorbar(im, ax=axes[1])
fig.tight_layout()
plt.show()
```

---

## 5. The Velocity Signature

The key insight for kinematic lensing: rotation creates an asymmetric pattern in the grism image. We isolate this by subtracting a non-rotating reference.

```{code-cell} python
from kl_pipe.model import KLModel

shared_pars = {'cosi', 'theta_int', 'g1', 'g2'}
kl = KLModel(vel_model, int_model, shared_pars=shared_pars, spectral_model=sm)

spec_dict = {'z': z, 'vel_dispersion': 50.0, 'Ha_flux': 100.0, 'Ha_cont': 0.01}
merged = {**vel_pars, **int_pars, **spec_dict}
theta = kl.pars2theta(merged)

# rotating galaxy
grism_rot = np.array(kl.render_grism(theta, gp))

# non-rotating reference
merged_norot = {**merged, 'vcirc': 0.0}
theta_norot = kl.pars2theta(merged_norot)
grism_norot = np.array(kl.render_grism(theta_norot, gp))

diff = grism_rot - grism_norot

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
axes[0].imshow(grism_rot, origin='lower', cmap='magma')
axes[0].set_title('Rotating (vcirc=200)')

axes[1].imshow(grism_norot, origin='lower', cmap='magma')
axes[1].set_title('Non-rotating (vcirc=0)')

vmax = np.max(np.abs(diff))
im = axes[2].imshow(diff, origin='lower', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
axes[2].set_title('Velocity signature (difference)')
plt.colorbar(im, ax=axes[2])
fig.tight_layout()
plt.show()
```

The difference shows the antisymmetric velocity signature: approaching side loses flux (blue-shifted away from line center) while the receding side gains flux. This is the signal we use for kinematic lensing.

---

## 6. Dispersion Angle Matters

The relative angle between the kinematic axis (`theta_int`) and the dispersion axis determines how well rotation is resolved.

```{code-cell} python
from kl_pipe.diagnostics.grism import plot_dispersion_angles

def build_grism_fn(angle):
    gp_a = GrismPars(
        image_pars=image_pars, dispersion=1.1,
        lambda_ref=HALPHA.lambda_rest * 2.0,
        dispersion_angle=angle,
    )
    return np.array(kl.render_grism(theta, gp_a))

fig = plot_dispersion_angles(
    build_grism_fn,
    title='Dispersion angle comparison (theta_int=0.7 rad)',
)
plt.show()
```

Maximum velocity information is extracted when the dispersion axis is parallel to the kinematic major axis. The library provides a deeper diagnostic:

```{code-cell} python
from kl_pipe.diagnostics.grism import plot_dispersion_angle_study

# use theta_int=0 so kinematic axis is purely along x
pars_study = {**vel_pars, **int_pars, 'theta_int': 0.0}
merged_study = {**pars_study, **spec_dict}
theta_study = kl.pars2theta(merged_study)

gp_ref = build_grism_pars_for_line(
    HALPHA.lambda_rest, redshift=z,
    image_pars=image_pars, dispersion=1.1,
)
cp_study = gp_ref.to_cube_pars(z=z)
cube_study = kl.render_cube(theta_study, cp_study)
dl = float(cp_study.lambda_grid[1] - cp_study.lambda_grid[0])
broadband = np.array(jnp.sum(cube_study, axis=2) * dl)

def build_grism_study(angle):
    gp_a = GrismPars(
        image_pars=image_pars, dispersion=1.1,
        lambda_ref=HALPHA.lambda_rest * 2.0,
        dispersion_angle=angle,
    )
    return np.array(kl.render_grism(theta_study, gp_a, cube_pars=cp_study))

fig = plot_dispersion_angle_study(
    build_grism_study, broadband,
    title='Dispersion angle study (theta_int=0)',
)
plt.show()
```

When dispersion is along x (0 deg) and the kinematic axis is also along x, the velocity signature is maximally stretched along the dispersion direction.

---

## 7. The KLModel: Unified Interface

`KLModel` with a `spectral_model` provides a single interface for imaging, datacubes, and grism rendering — all from the same parameter vector.

```{code-cell} python
print(f"KLModel parameters ({len(kl.PARAMETER_NAMES)} total):")
for i, name in enumerate(kl.PARAMETER_NAMES):
    print(f"  [{i:2d}] {name}")
```

The `render()` dispatch supports datacube and grism modes. 2D imaging uses the component models directly (see quickstart tutorial).

```{code-cell} python
# datacube (3D)
cube_out = kl.render(theta, 'cube', cp_grism)
print(f"datacube: {cube_out.shape}")

# grism (2D dispersed)
grism_out = kl.render(theta, 'grism', gp)
print(f"grism:    {grism_out.shape}")
```

JIT compilation and automatic differentiation work out of the box:

```{code-cell} python
from functools import partial

# JIT-compiled grism rendering
cp_jit = gp.to_cube_pars(z=z)
render_jit = jax.jit(partial(kl.render_grism, grism_pars=gp, cube_pars=cp_jit))
grism_fast = render_jit(theta)
print(f"JIT grism shape: {grism_fast.shape}")

# gradient of a scalar loss w.r.t. all parameters
def loss(th):
    return jnp.sum(kl.render_grism(th, gp, cube_pars=cp_jit)**2)

grad = jax.grad(loss)(theta)
print(f"Gradient shape: {grad.shape}, all finite: {bool(jnp.isfinite(grad).all())}")
```

The same `KLModel` works for imaging, datacubes, and grism — this is the foundation for joint inference across data types.

---

## 8. Multi-Line Spectroscopy (Ha + [NII])

Real observations capture multiple emission lines. The H-alpha + [NII] triplet is the primary target.

```{code-cell} python
# 3-line complex: Ha + [NII] 6548 + [NII] 6583
lines = halpha_nii_lines()
for line in lines:
    print(f"  {line.line_spec.name}: lambda_rest={line.line_spec.lambda_rest:.2f} nm, "
          f"own_params={line.own_params}")

config_3line = make_spectral_config()  # defaults to Ha + NII
sm_3line = SpectralModel(config_3line, int_model, vel_model)
print(f"\nSpectralModel parameters: {sm_3line.PARAMETER_NAMES}")
```

```{code-cell} python
# wider wavelength range to cover all 3 lines
lam_min = 654.0 * (1 + z) - 10
lam_max = 659.0 * (1 + z) + 10
cp_3line = CubePars.from_range(image_pars, lam_min, lam_max, 0.3)

theta_vel_norot = vel_model.pars2theta({**vel_pars, 'vcirc': 0.0, 'v0': 0.0})
# z, vel_disp, Ha_flux, Ha_cont, NII6548_flux, NII6548_cont, NII6583_flux, NII6583_cont
theta_spec_3 = jnp.array([z, 50.0, 100.0, 0.0, 30.0, 0.0, 90.0, 0.0])

cube_3line = sm_3line.build_cube(theta_spec_3, theta_vel_norot, theta_int, cp_3line)

# spatially integrated spectrum
total_spec = np.array(jnp.sum(cube_3line, axis=(0, 1)))
lam_3 = np.array(cp_3line.lambda_grid)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(lam_3, total_spec, 'k-')
for ls in lines:
    lam_obs = ls.line_spec.lambda_rest * (1 + z)
    ax.axvline(lam_obs, color='gray', ls='--', alpha=0.5)
    ax.text(lam_obs, ax.get_ylim()[1] * 0.9, ls.line_spec.name,
            ha='center', fontsize=8, color='gray')
ax.set_xlabel('Wavelength (nm)')
ax.set_ylabel('Spatially integrated flux density')
ax.set_title('Ha + [NII] triplet at z=1 (no rotation)')
fig.tight_layout()
plt.show()
```

Each line's flux is a free parameter, enabling line ratio measurements alongside kinematics.

---

## 9. Diagnostic Plots (Library Functions)

`kl_pipe.diagnostics` provides reusable plotting functions for datacubes and grism images.

```{code-cell} python
from kl_pipe.diagnostics.grism import plot_grism_overview

kl_3 = KLModel(vel_model, int_model, shared_pars=shared_pars, spectral_model=sm)
gp_diag = build_grism_pars_for_line(
    HALPHA.lambda_rest, redshift=z,
    image_pars=image_pars, dispersion=1.1,
)
cp_diag = gp_diag.to_cube_pars(z=z)

cube_diag = kl_3.render_cube(theta, cp_diag)
grism_diag = np.array(kl_3.render_grism(theta, gp_diag, cube_pars=cp_diag))

merged_norot_diag = {**merged, 'vcirc': 0.0}
theta_norot_diag = kl_3.pars2theta(merged_norot_diag)
grism_norot_diag = np.array(kl_3.render_grism(theta_norot_diag, gp_diag, cube_pars=cp_diag))

fig = plot_grism_overview(
    np.array(cube_diag), grism_diag,
    np.array(cp_diag.lambda_grid), gp_diag,
    imap=np.array(imap), vmap=np.array(vmap),
    grism_norot=grism_norot_diag,
    v0=vel_pars['v0'],
    title='Grism overview — Ha at z=1',
)
plt.show()
```

Available diagnostics:
- `plot_datacube_overview()` — intensity, velocity, stacked flux, wavelength channel slices
- `plot_grism_overview()` — master 3-row panel: maps, channels, grism + velocity signature
- `plot_dispersion_angles()` — grism images at cardinal dispersion angles
- `plot_dispersion_angle_study()` — deep-dive: grism vs broadband at each angle

These are reusable in your own analysis scripts and notebooks.
