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

# Grism and Datacube Forward Modeling

A slitless grism disperses each spatial pixel along one direction on the
detector, mapping wavelength to position. For a rotating galaxy observed at an
emission line, the receding and approaching sides shift by `+/- lambda_rest *
v_LOS / c` along the dispersion direction, so the velocity gradient appears as an
asymmetry between the two halves of the dispersed image. That asymmetry is the
kinematic signal the pipeline exploits.

This tutorial covers the forward model only: emission lines + velocity field ->
3D datacube -> dispersed 2D image. Fitting a grism observation to data is in
`sampling.md`; the model objects here assume the `quickstart.md` basics.

```{code-cell} python
import jax
jax.config.update('jax_enable_x64', True)

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import galsim

from kl_pipe.parameters import ImagePars
from kl_pipe.velocity import CenteredVelocityModel
from kl_pipe.intensity import InclinedExponentialModel
from kl_pipe.lines import EmissionLine, LINE_LAMBDAS
from kl_pipe.spectral import CubePars
from kl_pipe.dispersion import GrismPars, build_grism_pars_for_line, disperse_cube
from kl_pipe.source import SourceModel
from kl_pipe.observation import build_velocity_obs, build_image_obs, build_grism_obs
```

| Object | Purpose |
|---|---|
| `EmissionLine` | Per-line spatial profile (+ optional continuum / dispersion sharing) |
| `LINE_LAMBDAS` | Rest-wavelength registry (nm) for named lines |
| `CubePars` | Spatial grid + wavelength array for cube assembly |
| `GrismPars` | Dispersion direction, plate scale, reference wavelength |
| `GrismObs` | Observation container: PSF + dispersion + (data + variance) |
| `SourceModel.build_cube` / `.render_grism` | Cube assembly and dispersion |

The cube-to-grism flow is `build_cube` -> per-slice PSF convolution ->
`disperse_cube` -> pixel response -> detector image. `SourceModel.render_grism`
does all of it in one call.

---

## Example 1: The inputs (velocity + emission morphology)

The two inputs to the spectral pipeline are the line-of-sight velocity field and
the emission-line spatial profile. We build a `SourceModel` with a velocity model
and an H-alpha emission line, all sharing the galaxy geometry, and render the
maps. (The line's spatial profile is an intensity model; we render it through a
broadband channel here just to view it.)

```{code-cell} python
Z = 1.0
image_pars = ImagePars(shape=(32, 32), pixel_scale=0.11, indexing='ij')  # Roman scale

vm = CenteredVelocityModel()
line_int = InclinedExponentialModel()

# Full dotted-key parameter dict reused throughout this tutorial.
pars = {
    'cosi': 0.5,
    'theta_int': 0.7,
    'g1': 0.0,
    'g2': 0.0,
    'vel.v0': 10.0,
    'vel.vcirc': 200.0,
    'vel.rscale': 0.5,
    'Halpha.flux': 100.0,
    'Halpha.rscale': 0.3,
    'Halpha.h_over_r': 0.1,
    'Halpha.x0': 0.0,
    'Halpha.y0': 0.0,
    'Halpha.dispersion': 50.0,   # intrinsic line velocity dispersion (km/s)
    'z': Z,
}

# Velocity map (no PSF) and the H-alpha spatial profile (viewed as a broadband).
src_maps = SourceModel(velocity_model=vm, broadband_models={'Halpha': line_int})
vmap = np.asarray(src_maps.render_velocity(pars, build_velocity_obs(image_pars)))
imap = np.asarray(src_maps.render_broadband(
    pars, build_image_obs(image_pars, broadband_key='Halpha', int_model=line_int), 'Halpha'))

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
im0 = axes[0].imshow(imap, origin='lower', cmap='magma')
axes[0].set_title('H-alpha intensity'); plt.colorbar(im0, ax=axes[0], label='flux/pixel')
vmax = float(np.max(np.abs(vmap - pars['vel.v0'])))
im1 = axes[1].imshow(vmap, origin='lower', cmap='RdBu_r',
                     vmin=pars['vel.v0'] - vmax, vmax=pars['vel.v0'] + vmax)
axes[1].set_title('LOS velocity'); plt.colorbar(im1, ax=axes[1], label='km/s')
plt.tight_layout(); plt.show()
```

The intensity sets the emission morphology; the velocity sets the per-pixel
Doppler shift.

---

## Example 2: The emission-line source

The grism source pairs the velocity model with one or more `EmissionLine`
components. An `EmissionLine` carries a spatial intensity profile and, optionally,
a stellar continuum and a shared dispersion. Rest wavelengths auto-resolve from
`LINE_LAMBDAS` by the dict key:

```{code-cell} python
source = SourceModel(
    velocity_model=vm,
    emission_lines={'Halpha': EmissionLine(intensity=line_int)},
)
print(f"Halpha rest wavelength: {LINE_LAMBDAS['Halpha']} nm")
print(f"observed at z={Z}: {LINE_LAMBDAS['Halpha'] * (1 + Z):.1f} nm")
```

The per-line parameters live under the line's namespace (`Halpha.flux`,
`Halpha.rscale`, ..., `Halpha.dispersion`); the intrinsic line width in the cube
is `Halpha.dispersion` only. The slitless instrumental LSF is produced
downstream by the PSF-per-slice + dispersion geometry, not a separate term;
`tests/test_spectral_resolution.py` checks the resulting resolution against the Roman spec
`R = 461 * lambda_um`.

---

## Example 3: Datacube assembly

`CubePars` defines the wavelength grid; `SourceModel.build_cube` evaluates the
model `C(x, y, lambda)` on it. The cube needs the top-level `z` and the per-line
`dispersion` (both already in `pars`):

```{code-cell} python
lam_center = LINE_LAMBDAS['Halpha'] * (1 + Z)
C_KMS = 299792.458
dlam = lam_center * 2000.0 / C_KMS                 # +/- 2000 km/s window
cube_pars = CubePars.from_range(image_pars, lam_center - dlam, lam_center + dlam, 1.0)
print(f"CubePars: {cube_pars.n_lambda} wavelength bins")

cube = source.build_cube(pars, cube_pars)
print(f"Datacube shape: {cube.shape}")
```

Spaxel spectra along the kinematic major axis recover the Doppler shift:
approaching pixels are blueshifted, receding pixels redshifted, minor-axis
pixels sit on the systemic centerline.

```{code-cell} python
lam = np.asarray(cube_pars.lambda_grid)
cube_np = np.asarray(cube)
cr, cc = 16, 16
spaxels = {'approaching': (cr, cc - 6), 'center': (cr, cc),
           'receding': (cr, cc + 6), 'minor axis': (cr + 6, cc)}

fig, ax = plt.subplots(figsize=(8, 4))
for label, (r, c) in spaxels.items():
    ax.plot(lam, cube_np[r, c, :], label=f'{label} ({r},{c})')
ax.axvline(lam_center, color='gray', ls='--', alpha=0.5, label=f'Ha systemic ({lam_center:.1f} nm)')
ax.set_xlabel('wavelength (nm)'); ax.set_ylabel('flux density')
ax.set_title('Spaxel spectra'); ax.legend(fontsize=8)
plt.tight_layout(); plt.show()
```

A multi-panel datacube diagnostic is available for routine inspection:

```{code-cell} python
from kl_pipe.diagnostics.datacube import plot_datacube_overview

fig = plot_datacube_overview(cube_np, lam, imap=imap, vmap=vmap,
                             lam_center=lam_center, v0=pars['vel.v0'],
                             title='Datacube overview -- Ha at z=1')
plt.show()
```

---

## Example 4: Grism dispersion

`GrismPars` specifies the dispersion direction, plate scale, and reference
wavelength; `build_grism_pars_for_line` is the single-line convenience.
`SourceModel.render_grism` builds the cube, convolves per slice with the PSF,
disperses, and applies the pixel response:

```{code-cell} python
gp = build_grism_pars_for_line(
    LINE_LAMBDAS['Halpha'], redshift=Z, image_pars=image_pars, dispersion=1.1,  # ~1.1 nm/pix
)
psf = galsim.Gaussian(fwhm=0.18)   # Gaussian stand-in; Roman-like FWHM, not the true PSF
grism_obs = build_grism_obs(gp, z=Z, psf=psf)
grism = np.asarray(source.render_grism(pars, grism_obs))
print(f"dispersed image shape: {grism.shape}")
```

Compare the dispersed image to the spectrally-stacked (no-dispersion) cube. The
stacked image is the broadband-equivalent integral; dispersion remaps it along
the dispersion axis:

```{code-cell} python
dl = float(lam[1] - lam[0])
stacked = np.sum(cube_np, axis=2) * dl

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].imshow(stacked, origin='lower', cmap='magma'); axes[0].set_title('stacked (no dispersion)')
im = axes[1].imshow(grism, origin='lower', cmap='magma'); axes[1].set_title('dispersed grism')
plt.colorbar(im, ax=axes[1]); plt.tight_layout(); plt.show()
```

---

## Example 5: The rotation signature

Rendering the grism at `vcirc = 200` and `vcirc = 0` and differencing isolates
the contribution of rotation to the dispersed image:

```{code-cell} python
grism_rot = np.asarray(source.render_grism(pars, grism_obs))
grism_norot = np.asarray(source.render_grism({**pars, 'vel.vcirc': 0.0}, grism_obs))
diff = grism_rot - grism_norot

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
axes[0].imshow(grism_rot, origin='lower', cmap='magma'); axes[0].set_title('vcirc=200')
axes[1].imshow(grism_norot, origin='lower', cmap='magma'); axes[1].set_title('vcirc=0')
vmax = float(np.max(np.abs(diff)))
im = axes[2].imshow(diff, origin='lower', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
axes[2].set_title('rotation signature (difference)')
plt.colorbar(im, ax=axes[2]); plt.tight_layout(); plt.show()
```

The difference is antisymmetric across the kinematic major axis: approaching-side
flux is Doppler-shifted into other wavelength bins and receding-side flux shifts
in, so one side loses and the other gains. This antisymmetry is what kinematic
lensing exploits. Shear breaks the symmetry between the kinematic and
morphological axes, and the joint photometry + grism fit decouples them (see
`sampling.md`).

The dispersion direction sets how strongly the rotation term appears: aligning
the kinematic major axis with the dispersion axis maximizes it, orthogonal
alignment suppresses it. The `plot_dispersion_angle_study` diagnostic sweeps this:

```{code-cell} python
from kl_pipe.diagnostics.grism import plot_dispersion_angle_study

pars_aligned = {**pars, 'theta_int': 0.0}
cube_aligned = np.asarray(source.build_cube(pars_aligned, cube_pars))
broadband = np.sum(cube_aligned, axis=2) * dl

def render_at_angle(angle):
    gp_a = GrismPars(image_pars=image_pars, dispersion=1.1,
                     lambda_ref=gp.lambda_ref, dispersion_angle_detector=angle)
    obs_a = build_grism_obs(gp_a, z=Z, psf=psf)
    return np.asarray(source.render_grism(pars_aligned, obs_a))

fig = plot_dispersion_angle_study(render_at_angle, broadband,
                                  title='Dispersion angle study (theta_int=0)')
plt.show()
```

---

## Example 6: Multi-line spectroscopy (H-alpha + [N II])

The H-alpha + [N II] doublet shares geometry and (optionally) the spatial profile
and dispersion, while keeping independent per-line fluxes. `intensity_key` points
a line at another line's spatial profile; `dispersion_key` shares its dispersion:

```{code-cell} python
multi_source = SourceModel(
    velocity_model=vm,
    emission_lines={
        'Halpha': EmissionLine(intensity=InclinedExponentialModel()),
        'NII6584': EmissionLine(intensity_key='Halpha', dispersion_key='Halpha'),
    },
)

multi_pars = {
    **{k: v for k, v in pars.items() if not k.startswith('Halpha')},
    'Halpha.flux': 100.0,
    'Halpha.rscale': 0.3,
    'Halpha.h_over_r': 0.1,
    'Halpha.x0': 0.0,
    'Halpha.y0': 0.0,
    'Halpha.dispersion': 50.0,
    'NII6584.flux': 35.0,        # NII shares Halpha's spatial profile + dispersion
}

lam_min = 654.0 * (1 + Z) - 8
lam_max = 659.0 * (1 + Z) + 8
cp_multi = CubePars.from_range(image_pars, lam_min, lam_max, 0.3)
cube_multi = np.asarray(multi_source.build_cube({**multi_pars, 'vel.vcirc': 0.0}, cp_multi))
spectrum = np.sum(cube_multi, axis=(0, 1))

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(np.asarray(cp_multi.lambda_grid), spectrum, 'k-')
for name in ('Halpha', 'NII6584'):
    lc = LINE_LAMBDAS[name] * (1 + Z)
    ax.axvline(lc, color='gray', ls='--', alpha=0.5)
    ax.text(lc, spectrum.max() * 0.9, name, ha='center', fontsize=8, color='gray')
ax.set_xlabel('wavelength (nm)'); ax.set_ylabel('integrated flux density')
ax.set_title('Ha + [N II] at z=1 (no rotation)')
plt.tight_layout(); plt.show()
```

Independent per-line fluxes enable line-ratio diagnostics ([N II]/H-alpha for
metallicity / ionization) alongside the kinematic fit.

**Validity limit: one line complex per fit.** The grism pathway applies a
single, wavelength-independent PSF to every spectral slice (and, for the
analytic dispersal path, to every line). This is accurate for a tight line
complex like H-alpha + [N II] (~5 nm apart; the instrumental PSF varies by
well under a percent across it), but it is a real bias for widely separated
lines — over the ~170 nm between H-alpha and H-beta the Roman PSF size
changes at the several-percent level. Until per-line PSFs land (issue #51),
fit widely separated lines as separate per-complex posterior runs and
combine the chains afterwards; when multiplying the posteriors, divide out
the shared-parameter prior N-1 times so it is not double-counted.

---

## Diagnostics and reference tests

`kl_pipe.diagnostics` provides array-based plotting helpers:
`plot_datacube_overview`, `plot_grism_overview`, `plot_dispersion_angles`, and
`plot_dispersion_angle_study`. They take rendered arrays, so they work with any
forward-model output.

```{code-cell} python
from kl_pipe.diagnostics.grism import plot_grism_overview

fig = plot_grism_overview(
    cube_np, grism, lam, gp, imap=imap, vmap=vmap,
    grism_norot=grism_norot, v0=pars['vel.v0'],
    title='Grism overview -- Ha at z=1',
)
plt.show()
```

Reference tests:
- `tests/test_grism_core.py` -- spectral routing, dispersion sign / angle, JIT + grad.
- `tests/test_datacube.py` -- cube assembly, flux conservation, spectral-oversample convergence.
- `tests/test_cube_psf.py` -- per-slice PSF convolution under oversampling.
- `tests/test_grism_bandwidth.py` -- grism render-grid sizing vs accuracy.

---

## Where to go next

- **sampling.md** -- grism and joint photometry + grism inference (the fit).
- **quickstart.md** -- the primary Roman configuration end to end.
- **intensity_models.md** -- the emission-line spatial profiles in depth.
