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

# Intensity Models

The surface-brightness side of the pipeline in depth: the single-component model
zoo, multi-component (bulge + disk and beyond) composites, and the `RenderConfig`
machinery that sizes the FFT grid for a target accuracy. This assumes the
`quickstart.md` basics (SourceModel, dotted-key parameters, observations).

```{code-cell} python
import jax
jax.config.update('jax_enable_x64', True)

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import galsim

from kl_pipe.parameters import ImagePars
from kl_pipe.render import RenderConfig
from kl_pipe.source import SourceModel
from kl_pipe.observation import build_image_obs
from kl_pipe.intensity import (
    InclinedExponentialModel,
    InclinedSpergelModel,
    InclinedSersicModel,
    InclinedDeVaucouleursModel,
    BulgeDiskModel,
    CompositeIntensityModel,
    ComponentSpec,
    build_intensity_model,
)
```

All intensity models are stateless: parameters are passed in, never stored.
Every model shares the four geometric parameters (`cosi`, `theta_int`, `g1`,
`g2`) and adds its own profile parameters. Analytic evaluation is in surface
brightness; `render_image` (via `SourceModel.render_broadband`) returns
flux/pixel.

A small helper renders any single intensity model through the standard k-space
path with an optional PSF:

```{code-cell} python
def render(model, pars, image_pars, psf=None, oversample=5):
    """Render one intensity model to a flux/pixel image (standalone, non-inference)."""
    src = SourceModel(broadband_models={'m': model})
    obs = build_image_obs(image_pars, psf=psf, broadband_key='m', int_model=model,
                          render_config=RenderConfig(oversample=oversample))
    return np.asarray(src.render_broadband(pars, obs, 'm'))

image_pars = ImagePars(shape=(64, 64), pixel_scale=0.05, indexing='ij')
psf = galsim.Gaussian(fwhm=0.10)
```

---

## The single-component zoo

| Model | factory name(s) | profile parameters | notes |
|---|---|---|---|
| `InclinedExponentialModel` | `inclined_exp`, `default` | `flux`, `rscale`, `h_over_r` | n=1 disk, exact analytic FT; the default |
| `InclinedSpergelModel` | `inclined_spergel`, `spergel` | `flux`, `rscale`, `h_over_r`, `nu` | Spergel profile; `nu` sets the cuspiness |
| `InclinedSersicModel` | `inclined_sersic`, `sersic` | `flux`, `hlr`, `h_over_hlr`, `n_sersic` | Miller & Pasha emulator; free Sersic index |
| `InclinedDeVaucouleursModel` | `de_vaucouleurs` | `flux`, `rscale`, `h_over_r` | fixed n=4 bulge profile |

(plus the shared `x0`, `y0` centroid). The `build_intensity_model(name)` factory
is case-insensitive and raises `ValueError` on unknown names:

```{code-cell} python
print(build_intensity_model('spergel').name)
print(build_intensity_model('SERSIC').name)
print(InclinedSersicModel().PARAMETER_NAMES)
```

A side-by-side render at a common inclination. All four are PSF-convolved: the
de Vaucouleurs and high-`n` Sersic profiles have a steep central cusp that
aliases without a PSF (and is unphysical anyway, since every real observation is
band-limited by the instrument).

```{code-cell} python
geom = {'cosi': 0.5, 'theta_int': 0.6, 'g1': 0.0, 'g2': 0.0, 'x0': 0.0, 'y0': 0.0}
models_pars = [
    ('exponential', InclinedExponentialModel(),
     {**geom, 'flux': 100.0, 'rscale': 0.3, 'h_over_r': 0.1}),
    ('spergel (nu=0.5)', InclinedSpergelModel(),
     {**geom, 'flux': 100.0, 'rscale': 0.3, 'h_over_r': 0.1, 'nu': 0.5}),
    ('sersic (n=2)', InclinedSersicModel(),
     {**geom, 'flux': 100.0, 'hlr': 0.3, 'h_over_hlr': 0.1, 'n_sersic': 2.0}),
    ('de Vaucouleurs', InclinedDeVaucouleursModel(),
     {**geom, 'flux': 100.0, 'rscale': 0.15, 'h_over_r': 0.3}),
]

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, (name, model, pars) in zip(axes, models_pars):
    img = render(model, pars, image_pars, psf=psf)
    im = ax.imshow(np.log10(img + img.max() * 1e-4), origin='lower', cmap='magma')
    ax.set_title(name); ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout(); plt.show()
```

**Choosing a model.** Use `InclinedExponentialModel` for disks (it is exact and
fast). Use `InclinedSersicModel` when you need a free Sersic index. Use
`InclinedDeVaucouleursModel` for a classical n=4 bulge. `InclinedSpergelModel`
is accurate for disk-like `nu`; a highly cuspy inclined Spergel (`nu` toward the
de Vaucouleurs regime) is numerically fragile, so prefer the de Vaucouleurs or
Sersic models for bulges. See `tests/test_intensity_sersic.py` and the Spergel
validation tests for the accuracy envelopes.

---

## Multi-component composites

`CompositeIntensityModel` sums N components in a single k-space pass (one IFFT).
Each component is a `ComponentSpec(model, prefix=..., fixed_params=...)`: the
`prefix` namespaces that component's parameters, and `fixed_params` overrides
shared values for that component only.

```{code-cell} python
composite = CompositeIntensityModel(components=[
    ComponentSpec(InclinedExponentialModel(), prefix='disk'),
    ComponentSpec(InclinedSersicModel(), prefix='bulge',
                  fixed_params={'n_sersic': 4.0}),
])
print(composite.PARAMETER_NAMES)
```

Flux is parameterized as a single `total_flux` plus per-component fractions
(`bulge_frac`, `bar_frac`, ...); the last component's fraction is derived so they
sum to one. `BulgeDiskModel` is the convenience subclass for the common
exponential-disk + de Vaucouleurs-bulge case:

```{code-cell} python
bd = BulgeDiskModel()
print(bd.PARAMETER_NAMES)
```

Render a bulge + disk and decompose it by rendering each component alone (build a
one-component SourceModel for each, reusing the shared parameter values):

```{code-cell} python
bd_pars = {
    'cosi': 0.45,
    'theta_int': 0.4,
    'g1': 0.0,
    'g2': 0.0,
    'total_flux': 1e4,
    'bulge_frac': 0.4,
    'disk_rscale': 0.5,
    'disk_h_over_r': 0.1,
    'disk_x0': 0.0,
    'disk_y0': 0.0,
    'bulge_hlr': 0.15,
    'bulge_h_over_hlr': 0.3,
    'bulge_x0': 0.0,
    'bulge_y0': 0.0,
}
ip = ImagePars(shape=(80, 80), pixel_scale=0.05, indexing='ij')
psf_bd = galsim.Gaussian(fwhm=0.12)

total = render(bd, bd_pars, ip, psf=psf_bd)

# Disk alone and bulge alone: set the other component's flux fraction to isolate.
disk_only = render(bd, {**bd_pars, 'bulge_frac': 0.0}, ip, psf=psf_bd)
bulge_only = render(bd, {**bd_pars, 'bulge_frac': 1.0}, ip, psf=psf_bd)

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, img, title in [(axes[0], disk_only, 'disk (B/T=0)'),
                       (axes[1], bulge_only, 'bulge (B/T=1)'),
                       (axes[2], total, 'total (B/T=0.4)')]:
    im = ax.imshow(np.log10(img + img.max() * 1e-4), origin='lower', cmap='viridis')
    ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout(); plt.show()
```

A few composite patterns:

```{code-cell} python
# Shared centroid for all components (one x0/y0 instead of per-component):
bd_shared = BulgeDiskModel(shared_centroids=True)
print("shared-centroid params:", len(bd_shared.PARAMETER_NAMES),
      "vs independent:", len(BulgeDiskModel().PARAMETER_NAMES))

# Zero the bulge shear while the disk keeps it (per-component fixed override):
no_bulge_shear = CompositeIntensityModel(components=[
    ComponentSpec(InclinedExponentialModel(), prefix='disk'),
    ComponentSpec(InclinedSersicModel(), prefix='bulge',
                  fixed_params={'n_sersic': 4.0, 'g1': 0.0, 'g2': 0.0}),
])

# Three components (disk + bulge + bar):
three = CompositeIntensityModel(components=[
    ComponentSpec(InclinedExponentialModel(), prefix='disk'),
    ComponentSpec(InclinedSersicModel(), prefix='bulge', fixed_params={'n_sersic': 4.0}),
    ComponentSpec(InclinedSersicModel(), prefix='bar', fixed_params={'n_sersic': 1.5}),
])
print("three-component fractions:",
      [p for p in three.PARAMETER_NAMES if p.endswith('_frac')])
```

To share morphology across several bands while keeping flux independent, supply
the morphology as bare top-level keys and only the flux as band-prefixed keys;
see the `quickstart.md` capstone.

---

## RenderConfig and rendering accuracy

Intensity rendering is a k-space FFT: the analytic profile FT is multiplied by
the pixel-response FT and the PSF FT, then inverse-transformed in one pass.
`RenderConfig` controls the grid:

```{code-cell} python
rc = RenderConfig()
print(f"oversample={rc.oversample}  pad_factor={rc.pad_factor}  "
      f"folding_threshold={rc.folding_threshold}  maxk_threshold={rc.maxk_threshold}")
```

- `oversample` extends the k-grid past the base Nyquist before binning back,
  suppressing aliasing of high-spatial-frequency profile content.
- `pad_factor` zero-pads the real-space grid before the FFT (reduces wrap-around).
- `folding_threshold` / `maxk_threshold` set the aliasing budget used when sizing
  the grid from the profile bandwidth.

Each model reports the wavenumber where its FT falls below threshold (`maxk`) and
the spacing needed to avoid flux folding (`stepk`). `maxk` grows as the scale
length shrinks and as `cosi` decreases (inclination compresses the FT along one
axis, extending it by `1/cos(i)`):

```{code-cell} python
m = InclinedExponentialModel()
for cosi in (0.9, 0.5, 0.2):
    p = {'cosi': cosi, 'theta_int': 0.0, 'g1': 0.0, 'g2': 0.0,
         'flux': 1.0, 'rscale': 0.3, 'h_over_r': 0.1, 'x0': 0.0, 'y0': 0.0}
    print(f"cosi={cosi}:  maxk={float(m.maxk(p)):6.1f}   stepk={float(m.stepk(p)):5.2f} rad/arcsec")
```

For inference you do not size the grid by hand. `RenderConfig.for_priors` (and
the channel helpers `build_image_render_config` / `build_grism_render_config`)
evaluate the worst case (smallest scale, smallest `cosi`) across the prior bounds
and pick the smallest grid that renders even the most demanding sample without
aliasing. A present PSF caps the effective `maxk`, so the required grid is smaller
than the bare profile would imply. `InferenceTask.from_obs` calls these
automatically; the trade-off is direct: a narrower `cosi` prior gives a smaller
worst-case grid and a faster fit.

### Convergence with oversample

The accuracy knob is `oversample`. A bulge cusp aliases at low oversample even
with a PSF; raising it converges the render. We sweep it and measure the
peak-pixel deviation from a high-oversample reference:

```{code-cell} python
oversamples = [1, 3, 5, 9, 15]
images = [render(bd, bd_pars, ip, psf=psf_bd, oversample=N) for N in oversamples]
ref = images[-1]

fig, axes = plt.subplots(1, len(oversamples), figsize=(4 * len(oversamples), 4))
for ax, N, img in zip(axes, oversamples, images):
    ax.imshow(np.log10(img + img.max() * 1e-4), origin='lower', cmap='viridis')
    ax.set_title(f'oversample={N}'); ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout(); plt.show()

for N, img in zip(oversamples, images):
    rel = float(np.max(np.abs(img - ref)) / np.max(ref))
    print(f"oversample={N:>2d}:  max |delta| / peak = {rel:.2e}")
```

The relationship between `oversample` and accuracy is quantified in the test
suite, which a tutorial render only illustrates:

- `tests/test_psf.py::test_oversample_convergence` -- residual vs GalSim ground
  truth, monotonically decreasing (N=5 within 1e-4 for Gaussian PSFs).
- `tests/test_render.py`, `tests/test_render_config.py` -- how the PSF-capped
  worst-case `maxk` sets the derived oversample.
- `tests/test_grism_bandwidth.py::test_worst_case_converges_at_predicted_oversample`
  -- the prior-derived oversample is the factor at which grism rendering
  converges.

See `docs/units_and_conventions.md` for the full render-method unit contract and
`docs/oversampling_convergence.md` for the spectral-oversample default.

---

## Where to go next

- **quickstart.md** -- the basics and the primary Roman configuration.
- **grism.md** -- emission-line datacubes and grism dispersion.
- **sampling.md** -- fitting these models to data.
