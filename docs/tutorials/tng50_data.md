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

# Working with TNG50 Mock Data

This tutorial loads TNG50 galaxies, renders them as 2D intensity and
line-of-sight velocity maps at a Roman-like redshift and pixel scale, and fits
the rendered maps with the pipeline's analytic models.

## Prerequisites

The TNG50 files live in `data/tng50/` and are downloaded from CyVerse with

    make download-cyverse-data

Particle and catalog files:
- `gas_data_analysis.npz`: gas particles (coordinates, velocities, masses, SFR, metals)
- `stellar_data_analysis.npz`: stellar particles (coordinates, velocities, ages, metallicities, luminosities)
- `subhalo_data_analysis.npz`: per-galaxy catalog (orientation, sizes, masses, photometry)

plus the BC03 photometry table (`SDSS_hr_stelib_stellar_photometrics.hdf5`) and
the SDSS filter curves (`*_SDSS.res`) used to recompute dust-attenuated
luminosities at new orientations.

The first cell checks for these files. Set `KLPIPE_TNG_DATA_DIR` to read them
from another directory (CI points it at a synthetic stand-in).

```{code-cell} python
import os
from pathlib import Path
from kl_pipe.tng.loaders import DEFAULT_DATA_DIR

TNG_DATA_DIR = Path(os.environ.get('KLPIPE_TNG_DATA_DIR', DEFAULT_DATA_DIR))
TNG_FILES = [
    'gas_data_analysis.npz',
    'stellar_data_analysis.npz',
    'subhalo_data_analysis.npz',
    'SDSS_hr_stelib_stellar_photometrics.hdf5',
] + [f'{b}_SDSS.res' for b in 'ugriz']
missing = [f for f in TNG_FILES if not (TNG_DATA_DIR / f).exists()]
if missing:
    raise FileNotFoundError(
        f"TNG50 data missing from {TNG_DATA_DIR}: {missing}\n"
        "Download with: make download-cyverse-data"
    )
print(f"TNG data directory: {TNG_DATA_DIR}")
```

```{code-cell} python
import time
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

from kl_pipe.tng import TNG50MockData, TNGDataVectorGenerator, TNGRenderConfig
from kl_pipe.parameters import ImagePars
from kl_pipe.velocity import CenteredVelocityModel
from kl_pipe.intensity import InclinedExponentialModel
from kl_pipe.source import SourceModel
from kl_pipe.observation import build_velocity_obs, build_image_obs
from kl_pipe.priors import PriorDict, Uniform, Gaussian, CircularUniform
from kl_pipe.sampling import InferenceTask
from kl_pipe.sampling.initialization import find_map, prior_starts

t_start = time.time()
```

## 1. Loading the dataset

`TNG50MockData` loads all galaxies from `data_dir`.

```{code-cell} python
tng_data = TNG50MockData(data_dir=TNG_DATA_DIR)
print(tng_data)

print(f"{'SubhaloID':>9} {'inc[deg]':>8} {'log M*':>6} {'R_e[kpc]':>8} {'Vmax':>6}")
for gal in tng_data.subhalo:
    inc = gal['Inclination_star']  # (90, 180) = viewed from below
    print(
        f"{gal['SubhaloID']:>9d} {min(inc, 180 - inc):>8.1f} "
        f"{np.log10(gal['SubhaloStellarMass']):>6.2f} {gal['R_e_fit_r']:>8.2f} "
        f"{gal['SubhaloMaxCircVel']:>6.1f}"
    )
```

Galaxies are accessed by array index (`tng_data[i]`) or by SubhaloID. Each is a
dict with `'gas'`, `'stellar'` and `'subhalo'` entries. We use SubhaloID 561676, a
barred disk with log M* = 10.6 and a well-populated gas disk.

```{code-cell} python
galaxy = tng_data.get_galaxy(subhalo_id=561676)
print(f"stellar particles: {len(galaxy['stellar']['Coordinates']):,}")
print(f"gas particles:     {len(galaxy['gas']['Coordinates']):,}")
```

## 2. The data-vector generator

`TNGDataVectorGenerator` projects particles onto a pixel grid. Intensity comes
from stellar luminosities (dust-attenuated by default); velocity is the
mass-weighted gas line-of-sight velocity, with the systemic velocity removed.

```{code-cell} python
gen = TNGDataVectorGenerator(galaxy, data_dir=TNG_DATA_DIR)
sub = galaxy['subhalo']

print(f"catalog inclination (stellar): {gen.native_inclination_deg:.1f} deg")
print(f"catalog position angle:        {gen.native_pa_deg:.1f} deg")
```

The catalog inclination is morphological; the kinematic inclination from the
stellar angular momentum typically differs by 5-15 deg. For 561676 most gas
mass sits in an extended halo (median particle radius ~90 kpc) that rotates
roughly opposite to the disk; the gas within 20 kpc is aligned with the stars to within
~6 deg. Keep the default `preserve_gas_stellar_offset=True`, which rotates gas
with the stellar rotation matrix. The velocity map averages all gas along each
line of sight, so beyond ~1 arcsec at z = 1 it is dominated by this halo gas,
which would not be seen in H-alpha (Section 6 masks it).

## 3. Rendering at a Roman-like redshift

TNG50 snapshot 99 galaxies are at z = 0, so an angular scale requires
`target_redshift`; without it the renderer raises. We place the galaxy at
z = 1 (H-alpha inside the Roman grism band) on 0.11 arcsec pixels (Roman WFI).

```{code-cell} python
Z_OBS = 1.0
image_pars = ImagePars(shape=(48, 48), pixel_scale=0.11, indexing='ij')

config = TNGRenderConfig(
    image_pars=image_pars,
    band='r',
    use_native_orientation=True,
    target_redshift=Z_OBS,
)
intensity, _ = gen.generate_intensity_map(config)
velocity, _ = gen.generate_velocity_map(config)

sin_i = np.sin(np.radians(gen.native_inclination_deg))
print(f"flux in stamp / catalog dusted r luminosity: "
      f"{intensity.sum() / sub['Dusted_Luminosity_r']:.3f}")
print(f"max |v_LOS| = {np.abs(velocity).max():.1f} km/s; "
      f"Vmax sin(i) = {sub['SubhaloMaxCircVel'] * sin_i:.1f} km/s")
```

Intensity is in the particle luminosity units (erg/s per pixel, rest-frame SDSS
band); velocity is in km/s. The stamp captures nearly all of the light, and
the peak LOS velocity is close to the catalog Vmax projected by sin(i).

```{code-cell} python
def show_maps(intensity, velocity, title, vmax=None):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    im = axes[0].imshow(np.log10(np.clip(intensity, 1e-6 * intensity.max(), None)),
                        origin='lower', cmap='viridis')
    axes[0].set_title('log10 intensity')
    plt.colorbar(im, ax=axes[0])
    vmax = vmax or np.abs(velocity).max()
    im = axes[1].imshow(velocity, origin='lower', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    axes[1].set_title('v_LOS [km/s]')
    plt.colorbar(im, ax=axes[1])
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()

show_maps(intensity, velocity,
          f'561676 native orientation, z={Z_OBS} '
          f'(inc={gen.native_inclination_deg:.0f} deg)')
```

Maps are indexed `[y, x]` (rows = y), so `imshow(..., origin='lower')` shows them with x to the right and y up.

Any other galaxy follows the same path. SubhaloID 490815 is a lower-mass,
unbarred disk whose gas and stars are aligned to within 10 deg at all radii:

```{code-cell} python
gen_b = TNGDataVectorGenerator(tng_data.get_galaxy(subhalo_id=490815),
                               data_dir=TNG_DATA_DIR)
int_b, _ = gen_b.generate_intensity_map(config)
vel_b, _ = gen_b.generate_velocity_map(config)
show_maps(int_b, vel_b, f'490815 native orientation, z={Z_OBS} '
          f'(inc={gen_b.native_inclination_deg:.0f} deg)')
```

## 4. Custom orientation and shear

With `use_native_orientation=False` the generator rotates the particles to a
face-on stellar disk and re-observes it at the requested `cosi`, `theta_int`
(radians, from +x) and shear `(g1, g2)`. With `use_dusted=True` (default) dust
attenuation is recomputed for the new line of sight, which takes a few seconds
per render.

```{code-cell} python
cases = [
    ('face-on', dict(cosi=1.0, theta_int=0.0, g1=0.0, g2=0.0)),
    ('cos i = 0.5', dict(cosi=0.5, theta_int=0.0, g1=0.0, g2=0.0)),
    ('cos i = 0.2', dict(cosi=0.2, theta_int=0.0, g1=0.0, g2=0.0)),
    ('cos i = 0.5, g1 = 0.2', dict(cosi=0.5, theta_int=0.0, g1=0.2, g2=0.0)),
]
fig, axes = plt.subplots(2, len(cases), figsize=(3.2 * len(cases), 6.4))
for j, (label, geo) in enumerate(cases):
    cfg = TNGRenderConfig(image_pars=image_pars, use_native_orientation=False,
                          pars={**geo, 'x0': 0.0, 'y0': 0.0}, target_redshift=Z_OBS)
    I, _ = gen.generate_intensity_map(cfg)
    V, _ = gen.generate_velocity_map(cfg)
    axes[0, j].imshow(np.log10(np.clip(I, 1e-6 * I.max(), None)), origin='lower')
    axes[0, j].set_title(label)
    axes[1, j].imshow(V, origin='lower', cmap='RdBu_r', vmin=-200, vmax=200)
    print(f"{label:>22}: max |v_LOS| = {np.abs(V).max():6.1f} km/s")
for ax in axes.ravel():
    ax.axis('off')
plt.tight_layout()
plt.show()
```

The rotations are 3D, so the disk keeps its vertical structure toward edge-on
and the face-on velocity field is dominated by dispersion rather than rotation.
The shear (g1 > 0) stretches the image along x.

## 5. Redshift, band and star-formation maps

At fixed pixel scale the galaxy shrinks with redshift. Bands are rest-frame
SDSS `u, g, r, i, z`; `generate_sfr_map` grids the gas star-formation rate
(Msun/yr per pixel), a proxy for H-alpha emission.

```{code-cell} python
fig, axes = plt.subplots(1, 6, figsize=(18, 3.2))
for ax, z in zip(axes[:3], [0.5, 1.0, 1.5]):
    I, _ = gen.generate_intensity_map(
        TNGRenderConfig(image_pars=image_pars, target_redshift=z))
    ax.imshow(np.log10(np.clip(I, 1e-6 * I.max(), None)), origin='lower')
    ax.set_title(f'r band, z={z}')
for ax, band in zip(axes[3:5], ['u', 'i']):
    I, _ = gen.generate_intensity_map(
        TNGRenderConfig(image_pars=image_pars, band=band, target_redshift=Z_OBS))
    ax.imshow(np.log10(np.clip(I, 1e-6 * I.max(), None)), origin='lower')
    ax.set_title(f'{band} band, z={Z_OBS}')
sfr = gen.generate_sfr_map(config)
axes[5].imshow(np.log10(np.clip(sfr, 1e-6 * sfr.max(), None)), origin='lower',
               cmap='magma')
axes[5].set_title('SFR (H-alpha proxy)')
for ax in axes:
    ax.axis('off')
plt.tight_layout()
plt.show()

print(f"SFR in stamp: {sfr.sum():.2f} Msun/yr (catalog SubhaloSFR {sub['SubhaloSFR']:.2f})")
```

Other options on `TNGRenderConfig`: `psf` (a GalSim object; velocity is
convolved flux-weighted), `apply_cosmological_dimming` ((1+z)^-4 surface
brightness dimming), `use_cic_gridding` (cloud-in-cell, default; False = nearest
grid point) and `center_on_peak`.

## 6. Fitting the maps with the analytic models

We render 561676 at a known orientation and shear, add noise, and fit the
pipeline's analytic models. TNG galaxies are not thin exponential disks with
arctan rotation curves, so this is a test under model mismatch: there is no
exact truth for `vcirc` or the scale radii, and the geometric parameters are
only expected to land near the input. The section ends with a short
investigation of where the fit breaks down and why.

```{code-cell} python
geo_true = dict(cosi=0.6, theta_int=0.8, g1=0.03, g2=-0.02, x0=0.0, y0=0.0)
cfg_fit = TNGRenderConfig(image_pars=image_pars, use_native_orientation=False,
                          pars=geo_true, target_redshift=Z_OBS)
int_map, int_var = gen.generate_intensity_map(cfg_fit, snr=100, seed=1)
vel_map, vel_var = gen.generate_velocity_map(cfg_fit, snr=100, seed=2)

# unit total flux keeps the flux prior O(1)
flux_norm = float(int_map.sum())
int_map, int_var = int_map / flux_norm, int_var / flux_norm**2

# fit velocity only where H-alpha would be detected (star-forming gas)
sfr_fit = gen.generate_sfr_map(cfg_fit)
vel_mask = sfr_fit > 0.02 * sfr_fit.max()
print(f"velocity pixels used: {vel_mask.sum()} / {vel_mask.size}")
```

### Velocity map

The observation objects carry data, variance and mask; `InferenceTask.from_obs`
bundles them with a `SourceModel` and priors into a differentiable
log-posterior. `find_map` runs multi-start L-BFGS-B from prior draws.

```{code-cell} python
geometry_priors = {
    'cosi': Uniform(0.1, 0.99),
    'theta_int': CircularUniform(),  # full turn: rotation sense is data-determined
    'g1': Uniform(-0.2, 0.2),
    'g2': Uniform(-0.2, 0.2),
}
velocity_priors = {
    'vel.v0': Gaussian(0.0, 20.0),
    'vel.vcirc': Uniform(50.0, 400.0),
    'vel.rscale': Uniform(0.02, 2.0),
}
n_starts = 8

vel_source = SourceModel(velocity_model=CenteredVelocityModel())
obs_vel = build_velocity_obs(image_pars, data=jnp.asarray(vel_map),
                             variance=jnp.asarray(vel_var), mask=jnp.asarray(vel_mask))
vel_priors = PriorDict({**geometry_priors, **velocity_priors})
vel_task = InferenceTask.from_obs(vel_source, vel_priors, velocity_obs=obs_vel)
vel_fit = find_map(vel_task, prior_starts(vel_task, n_starts, seed=0))

def print_map(task, fit):
    print(f"MAP from {n_starts} starts, {fit.n_evals} evaluations, {fit.wall_s:.1f} s")
    for name, val in zip(task.sampled_names, fit.theta_map):
        ref = geo_true.get(name)
        print(f"  {name:>10} = {val:+9.3f} " + (f"(input {ref:+.3f})" if ref is not None else ''))

print_map(vel_task, vel_fit)
```

A velocity map alone constrains `cosi` and the shear only weakly (they trade
off along a ridge with `vcirc`), so fewer starts can settle elsewhere on that
ridge at nearly the same residual; a sampler maps the full ridge.

Parameters use the dotted-key namespace (`vel.vcirc`, `r.rscale`; shared
geometry has no prefix); see `quickstart.md`. For posteriors rather than a MAP,
pass the same task to `build_sampler('numpyro', task, config)` as in
`sampling.md`.

```{code-cell} python
pars_map = vel_priors.theta_to_full_pars(jnp.asarray(vel_fit.theta_map))
model_vel = np.asarray(vel_source.render_velocity(pars_map, obs_vel))
vel_resid = np.where(vel_mask, vel_map - model_vel, np.nan)

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
panels = [
    (np.where(vel_mask, vel_map, np.nan), 'data [km/s]', 250),
    (np.where(vel_mask, model_vel, np.nan), 'MAP model [km/s]', 250),
    (vel_resid, 'residual [km/s]', 100),
]
for ax, (img, title, lim) in zip(axes, panels):
    im = ax.imshow(img, origin='lower', cmap='RdBu_r', vmin=-lim, vmax=lim)
    ax.set_title(title)
    plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.show()
print(f"rms residual in mask: {np.sqrt(np.nanmean(vel_resid**2)):.1f} km/s "
      f"(noise sigma {np.sqrt(vel_var.mean()):.1f} km/s)")
```

### Adding the image

A joint fit adds an r-band `InclinedExponentialModel` sharing the geometry.

```{code-cell} python
joint_source = SourceModel(
    velocity_model=CenteredVelocityModel(),
    broadband_models={'r': InclinedExponentialModel()},
)
obs_int = build_image_obs(image_pars, broadband_key='r',
                          int_model=joint_source.broadband_models['r'],
                          data=jnp.asarray(int_map), variance=jnp.asarray(int_var))
joint_priors = PriorDict({
    **geometry_priors, **velocity_priors,
    'r.flux': Uniform(0.5, 1.5),
    'r.rscale': Uniform(0.05, 1.5),
    'r.h_over_r': 0.1,
    'r.x0': 0.0,
    'r.y0': 0.0,
})
joint_task = InferenceTask.from_obs(joint_source, joint_priors,
                                    velocity_obs=obs_vel, image_obs={'r': obs_int})
joint_fit = find_map(joint_task, prior_starts(joint_task, n_starts, seed=0))
print_map(joint_task, joint_fit)
```

The velocity map recovers the input geometry; the joint fit does not, and the
shear runs to the prior edge. The image has far more constraining pixels than
the masked velocity map, so it dominates the shared geometry.

### Investigating the joint fit

Before blaming the galaxy, rule out the pipeline. Replace the TNG maps with an
analytic galaxy at the same geometry, add noise the same way, and run the same
joint fit. If this does not recover the input, the problem is in the setup, not
the galaxy:

```{code-cell} python
from kl_pipe.noise import add_intensity_noise, add_velocity_noise

ctrl_pars = {**geo_true, 'vel.v0': 0.0, 'vel.vcirc': 200.0, 'vel.rscale': 0.3,
             'r.flux': 1.0, 'r.rscale': 0.3, 'r.h_over_r': 0.1, 'r.x0': 0.0, 'r.y0': 0.0}
clean_int_obs = build_image_obs(image_pars, broadband_key='r',
                                int_model=joint_source.broadband_models['r'])
ctrl_int, ctrl_int_var = add_intensity_noise(
    np.asarray(joint_source.render_broadband(ctrl_pars, clean_int_obs, 'r')),
    target_snr=100, seed=1)
ctrl_vel, ctrl_vel_var = add_velocity_noise(
    np.asarray(joint_source.render_velocity(ctrl_pars, build_velocity_obs(image_pars))),
    target_snr=100, seed=2)

ctrl_task = InferenceTask.from_obs(
    joint_source, joint_priors,
    velocity_obs=build_velocity_obs(image_pars, data=jnp.asarray(ctrl_vel),
                                    variance=jnp.asarray(ctrl_vel_var),
                                    mask=jnp.asarray(vel_mask)),
    image_obs={'r': build_image_obs(image_pars, broadband_key='r',
                                    int_model=joint_source.broadband_models['r'],
                                    data=jnp.asarray(ctrl_int),
                                    variance=jnp.asarray(ctrl_int_var))},
)
print_map(ctrl_task, find_map(ctrl_task, prior_starts(ctrl_task, n_starts, seed=0)))
```

The analytic galaxy lands close to the input, with no parameter at a prior
edge, so the fitting code and the shared geometry are fine. That leaves the galaxy itself. A single inclined disk is round when
seen face-on, so any elongation of the face-on light must be absorbed by
inclination or shear. Render 561676 face-on and unsheared and measure its axis
ratio from second moments:

```{code-cell} python
from kl_pipe.utils import build_map_grid_from_image_pars

X, Y = [np.asarray(a) for a in build_map_grid_from_image_pars(image_pars)]
R = np.hypot(X, Y)

def axis_ratio(img, rmax):
    """Second-moment axis ratio b/a within radius rmax [arcsec]."""
    w = np.clip(img, 0, None) * (R < rmax)
    w = w / w.sum()
    mx, my = (w * X).sum(), (w * Y).sum()
    qxx = (w * (X - mx) ** 2).sum()
    qyy = (w * (Y - my) ** 2).sum()
    qxy = (w * (X - mx) * (Y - my)).sum()
    ev = np.linalg.eigvalsh([[qxx, qxy], [qxy, qyy]])
    return np.sqrt(ev[0] / ev[1])

faceon = dict(cosi=1.0, theta_int=0.0, g1=0.0, g2=0.0, x0=0.0, y0=0.0)
cfg_faceon = TNGRenderConfig(image_pars=image_pars, use_native_orientation=False,
                             pars=faceon, target_redshift=Z_OBS)
int_faceon, _ = gen.generate_intensity_map(cfg_faceon)

plt.imshow(np.log10(np.clip(int_faceon, 1e-6 * int_faceon.max(), None)), origin='lower')
plt.title('561676 face-on, no shear')
plt.show()
for rmax in (0.6, 1.2):
    print(f"face-on axis ratio within {rmax}\": {axis_ratio(int_faceon, rmax):.2f} "
          "(1.0 for a round disk)")
```

The face-on light is clearly elongated: the galaxy has a bar and spiral
structure that no axisymmetric disk model contains. The joint fit explains that
shape the only way it can, through inclination and shear. This is intrinsic
shape noise appearing in a realistic mock, and it is what TNG galaxies are for:
measuring how non-disk structure biases the shear, and testing models or
analysis choices that reduce it.

## Summary

| Class | Purpose |
|---|---|
| `TNG50MockData` | Load all galaxies; access by index or SubhaloID |
| `TNGDataVectorGenerator` | Project one galaxy's particles to intensity, velocity and SFR maps |
| `TNGRenderConfig` | Grid, band, orientation, redshift, PSF and gridding options |

Key `TNGRenderConfig` fields:
- `target_redshift`: required; sets the angular scale
- `use_native_orientation`: simulation orientation (True) or `pars` (False)
- `pars`: `cosi`, `theta_int`, `g1`, `g2`, `x0`, `y0` for a custom orientation
- `preserve_gas_stellar_offset`: rotate gas with the stellar matrix (default True)
- `band`, `use_dusted`: rest-frame SDSS band and dust attenuation

See `kl_pipe/tng/README.md` for details and `tests/test_tng_data_vectors.py` for
the validation tests.

```{code-cell} python
print(f"tutorial wall time: {time.time() - t_start:.0f} s")
```
