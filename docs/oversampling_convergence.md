# Oversampling convergence — spatial + spectral

This document records the convergence behavior of the two oversampling
knobs in the rendering pipeline. Both control how a continuous integral
is discretized; both have a default that trades accuracy for cost.

> **2026-07-04 update:** spectral bin integration now defaults to the exact
> `spectral_method='erf'` path (analytic Gaussian bin integrals via the error
> function; no spectral discretization error, `spectral_oversample` ignored).
> Everything below about `spectral_oversample` applies only to the retained
> `spectral_method='oversample'` comparison path, which additionally
> **under-resolves narrow lines** (σ_λ below one fine sub-bin, i.e.
> dispersion ≲ 17 km/s at osf=15: ~5% voxel value error measured at
> 10 km/s — see `tests/test_spectral_methods.py`).

| Knob | Where | Default | Convergence test | Production-bias anchor |
|---|---|---:|---|---|
| **Spatial `oversample`** | `RenderConfig.oversample` | auto-derived per priors (typically 1-9) | `tests/test_psf.py::test_oversample_convergence` | (covered by convergence test) |
| **Spectral `spectral_oversample`** | `RenderConfig.spectral_oversample` | **15** | `tests/test_datacube.py::test_spectral_oversample_convergence` | `experiments/sweverett/spectral_osf_convergence/` |

`obs.oversample` and `obs.spectral_oversample` are `@property` that read
from `obs.render_config`. Override by passing `render_config=...` to
`build_image_obs` / `build_grism_obs`, or call `.with_render_config(...)`.

## Spatial oversample (`RenderConfig.oversample`)

- **What it controls:** k-grid Nyquist extension during k-space rendering of intensity
  profiles. Extends effective Nyquist to `oversample × π / pixel_scale`.
- **Auto-derivation:** computed from `model.maxk(params)` / `model.stepk(params)` per
  priors, so the chosen value depends on the steepest profile in the prior support.
- **Convergence:** `tests/test_psf.py::test_oversample_convergence` (Exponential ×
  Gaussian PSF on 150×200, hlr=3.0):

  | N | max relative residual vs GalSim |
  |---:|---:|
  | 1 | varies (large) |
  | 3 | smaller |
  | 5 | <1e-4 (assertion) |
  | 9 | smaller |

  Test asserts strict monotone decrease + `N=5 < 1e-4`.
- **Cost:** cubic in N for FFT grid size, but moderated by `pad_factor`.
- **When to override:** very steep source profiles (high Sersic n, edge-on inclined
  disks at high cosi) may push effective maxk above auto-derivation; use
  `kl_pipe.render.build_image_render_config(source, priors, image_pars, psf=psf)`
  with tighter prior bounds.

## Spectral oversample (`RenderConfig.spectral_oversample`)

- **What it controls:** wavelength sub-bin count for cube assembly in
  `SourceModel.build_cube` (and downstream `render_grism`). Each detector wavelength
  pixel's flux integral is approximated by averaging `N` sub-samples in λ.
- **Default = 15** (raised from 5 on 2026-06-07; see "Why the default is 15" below).
- **Cube-level convergence** (`tests/test_datacube.py::test_spectral_oversample_convergence`,
  Halpha at z=1, natural σ=50 km/s, 1.1 nm/px):

  | osf | max relative cube error vs osf=25 |
  |---:|---:|
  | 1 | ~79% (broken) |
  | 3 | 1.7e-2 |
  | 5 | 6.0e-3 |
  | 7 | 3.1e-3 |
  | 9 | 1.8e-3 |
  | **15** | **4.7e-4** |
  | 25 | 0 (reference) |

  Test asserts strict monotone decrease + `osf=15 < 1e-3`.
- **Parameter-level convergence** (`experiments/sweverett/spectral_osf_convergence/`,
  B.4b geometry, SNR=10000, conditional 1D MLE per param):

  | Parameter | osf=5 \|bias\|/σ_Fisher | osf=9 | **osf=15 (default)** | osf=25 (ref) |
  |---|---:|---:|---:|---:|
  | vel.v0 | 14.9 | 4.0 | **1.0** | 0.00 |
  | vel.vcirc | 15.9 | 4.5 | **1.2** | 0.00 |
  | Halpha.dispersion | 24.9 | 7.0 | **1.9** | 0.02 |
  | cosi | 5.9 | 2.0 | **0.9** | 0.53 |
  | theta_int | 9.4 | 2.8 | **1.0** | 0.30 |
  | vel.rscale | 18.7 | 5.3 | **1.4** | 0.02 |
  | g1 | 9.5 | 2.8 | **1.0** | 0.30 |
  | g2 | 5.0 | 0.9 | **0.2** | 0.66 |

  (osf=25 row is the self-consistency check; non-zero values for cosi/g2
  reflect parabolic-fit numerical precision floor when σ_Fisher is tiny.)

- **Cost:** linear in osf. osf=15 is ~3× slower per likelihood evaluation than
  osf=5, ~5/3× slower than osf=9.

### Why the default is 15

Before 2026-06-07 the default was osf=5. Parameter-level convergence study found
that **osf=5 produces parameter bias 5-25× σ_Fisher at SNR=10000** for all
parameters in the joint vel+phot+grism fit. The bias is deterministic
(noise-free synth at osf=25 vs fit at osf=5), so it does not average down with
multi-galaxy stacking — a real systematic.

osf=15 was selected as the production default:

- It brings bias to ~1× σ_Fisher at SNR=10000 (5-25× improvement on every parameter).
- At more typical per-galaxy Roman SNR (≤500), σ_Fisher is ≥20× larger, so the
  residual bias is ≤0.05σ and indistinguishable from noise.
- It costs ~3× per evaluation vs osf=5 — acceptable for typical NUTS / nautilus
  runtimes.

Tightening to 0.1× σ_Fisher at SNR=10000 would require osf ≫ 25 (≥5× cost), an
ROI that does not justify the wallclock for typical use. Users running at
near-anchor SNR or doing ensemble cosmology should consider further bumping
via `RenderConfig(spectral_oversample=N)`.

### When to override

- **Convergence / regression tests:** explicit `RenderConfig(spectral_oversample=N)`
  is required to test convergence behavior or compare against a known reference.
- **High-SNR anchor fits:** set `spectral_oversample=25` (or higher) for fits where
  systematic bias must be below noise.
- **Quick exploratory fits:** osf=5 cuts wallclock 3× at the cost of measurable
  parameter bias — acceptable for first-look quicklook fits.

## Construction patterns

### Using the default everywhere

```python
obs = build_grism_obs(grism_pars, z=z, psf=psf, data=data, variance=var)
# obs.spectral_oversample == 15 (production default)
task = InferenceTask.from_obs(source, priors, grism_obs={'roll0': obs})
# likelihood internally uses obs.spectral_oversample
```

### Overriding the default

```python
from kl_pipe.render import RenderConfig
from dataclasses import replace

# Method 1: pass at obs construction
rc = RenderConfig(spectral_oversample=25)
obs = build_grism_obs(grism_pars, z=z, psf=psf, data=data, variance=var,
                       render_config=rc)

# Method 2: preserve auto-derived spatial sizing, override only spectral
from kl_pipe.render import build_grism_render_config
base_rc = build_grism_render_config(source, priors, grism_pars, psf=psf)
rc = replace(base_rc, spectral_oversample=25)
obs = build_grism_obs(grism_pars, z=z, psf=psf, data=data, variance=var,
                       render_config=rc)
```

### Synth-to-fit consistency

Synthetic data generated by `source.render_grism(pars, obs_clean, ...)` should
use the same `spectral_oversample` as the likelihood that fits it. With the
plumbing through `RenderConfig`, the canonical pattern is to **share the same
RenderConfig (or at least the same spectral_oversample) between the synth-side
obs and the data-side obs.** `InferenceTask.from_obs` raises if multi-roll
grism obs have disagreeing values.
