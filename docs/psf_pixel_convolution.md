# Pixel Integration & PSF Convolution

Reference document for the k-space rendering pipeline's pixel integration
and PSF convolution architecture (post-PR #41).

## Architecture

The pipeline renders intensity models in Fourier space (k-space). Three
factors are multiplied before the inverse FFT:

```
I_measured = IFFT( profile_FT × pixel_FT × PSF_FT )
```

- **profile_FT** — analytic Fourier transform of the surface brightness
  profile (`InclinedExponential`, `InclinedSpergel`, `InclinedDeVaucouleurs`,
  `InclinedSersic` via Miller-Pasha emulator)
- **pixel_FT** — `PixelResponse.ft(KX, KY)`. Default `BoxPixel(pixel_scale)`
  has FT = `sinc(kx · w / 2π) × sinc(ky · w / 2π)`.
- **PSF_FT** — bare PSF FT from GalSim's `drawKImage` (no internal pixel
  convolution; pixel integration is handled by `pixel_FT` above).

This is mathematically exact pixel integration at O(1) cost.

## Rendering Pathway Selection

Decision tree at `_kspace_render_image`/`render_image` entry points
(`kl_pipe/intensity.py`):

```
obs given?
├── obs.kspace_psf_fft set?
│   ├── YES → fused k-space path (production):
│   │        IFFT( profile_FT × pixel_FT × drawKImage_PSF_FT )
│   │        with wrap when oversample > 1
│   └── NO, obs.psf_data set? → real-space PSF fallback:
│          point-sample profile at fine grid + drawImage PSF + bin
│   └── NO PSF at all → no-PSF k-space path:
│          IFFT( profile_FT × pixel_FT ) with wrap when oversample > 1
└── no obs (legacy / one-off) → adaptive theta-driven:
    auto-compute RenderConfig from theta, render via _render_kspace
```

For inference, only the fused k-space path is used (PSF FFT precomputed
into `obs.kspace_psf_fft` by `build_image_obs(int_model=...)`).

## Wrap path (the only multi-grid path)

When `oversample > 1`, `_kspace_render_core` engages the wrap path:

1. Build extended k-grid (size = oversample × base_padded_grid). Nyquist
   extends from `π/pixel_scale` to `oversample · π/pixel_scale`.
2. Evaluate `profile_FT × pixel_FT × PSF_FT` at TRUE frequencies on the
   extended grid.
3. Fold the extended k-grid onto the base grid by modular addition
   (`_wrap_kspace`). This sums the (unavoidable) aliases at each base k-bin
   correctly, capturing high-k profile content that would otherwise be
   evaluated on already-aliased base k-bins.
4. IFFT at base resolution → coarse image.

The wrap is the FT of the point-sampled signal — it does not depend on
whether `pixel_FT` is included. With `pixel_FT` the result is pixel-
integrated; without it, the result is converged point-sampled.

`oversample == 1` skips the wrap (no extension): a single base-grid IFFT.
This is aliased at high k but cheap; useful for tests.

## `pixel_response` semantics

| `pixel_response` | Output |
|---|---|
| `BoxPixel(pixel_scale)` (default from `build_image_obs`) | pixel-integrated image |
| `None` (explicit opt-out) | true point-sampled image |
| Future subclasses (e.g., `RomanPixel` with IPC) | as defined by subclass |

`pixel_response=None` is genuinely point-sampling — there is no implicit
fine-bin-averaging fallback. Use it only when comparing against an external
point-sample reference (e.g., GalSim `drawImage(method='no_pixel')`,
`__call__` surface brightness × pixel_area).

## RenderConfig: single source of truth

`RenderConfig` (`kl_pipe/render.py`) is the canonical sizing recipe:

```python
RenderConfig(oversample, pad_factor, maxk_threshold, folding_threshold, ...)
```

`obs.render_config` is the canonical attribute on `ImageObs`/`VelocityObs`.
`obs.oversample` is a `@property` that reads from it. Drift between obs
and runtime rc is structurally impossible.

### Construction paths

| Use case | rc source | Adaptive |
|---|---|---|
| **JAX inference (production)** | `RenderConfig.for_priors(model, priors, ps, pixel_response=...)` passed at obs build time | No (frozen) |
| **One-off render with PSF, max accuracy** | `RenderConfig.for_model(model, params, ps, pixel_response=...)` passed at obs build time | Yes (per build) |
| **One-off render with PSF, casual** | `RenderConfig()` defaults (built by `build_image_obs` if no rc passed) | No |
| **One-off render no PSF (tutorial)** | Auto-computed inside `render_image(theta, image_pars=ip)` via `_auto_render_config` | Yes (per call) |

The four sources reduce to two implementations: `RenderConfig.for_model`
(theta-adaptive) and `RenderConfig.for_priors` (worst-case over prior
bounds). The legacy no-obs path delegates to `for_model`; production paths
use `for_priors`. No parallel sizing logic.

### Drift prevention

`InferenceTask.from_intensity_obs(model, priors, obs)` does NOT recompute
rc from priors. It reads `obs.render_config` directly and validates
priors fit within the obs's pre-built grid:

- Priors imply tighter rc → use obs's rc as-is (slightly oversized but
  correct).
- Priors imply WIDER rc than obs was built for → raise `ValueError`
  with rebuild instructions:
  ```
  rc = RenderConfig.for_priors(model, priors, pixel_scale, pixel_response=...)
  obs = build_image_obs(image_pars, ..., render_config=rc)
  ```

This is the structural fix to Issue #42 (obs/rc oversample drift).

## Effective maxk: unified product scan

`render.compute_effective_maxk(model, params, pixel_response, psf, threshold)`
scans wavenumbers and returns the largest k where

```
|profile_FT(k) × pixel_FT(k) × PSF_FT(k)| > threshold
```

Whichever factors are passed go into the product. The product is correct
for grid sizing: at the min of any two individual maxks, both factors are
≤ threshold so the product is ≤ threshold². The product crosses threshold
well *before* either single factor — `min(individual_maxks)` is too
conservative (oversizes the grid).

PSF FT is evaluated at `(kx=0, ky=k)` via `psf.kValue(galsim.PositionD(0, k))`
— assumes radially symmetric PSF (true for common GalSim PSFs: Gaussian,
Moffat, Airy without anisotropy). `PixelResponse.ft_radial(k)` provides the
radial slice for any subclass; no `BoxPixel`-only restriction.

## Velocity models (separate convention)

Velocity rendering uses spatial oversampling + binning, NOT k-space sinc.
The flux-weighted projection
`v_obs = Conv(I·v, PSF) / Conv(I, PSF)` is a ratio; sinc can't apply to a
ratio (ratio of integrals ≠ integral of ratio). `obs.pixel_response` is
ignored by the velocity rendering path.

## JAX constraints

`obs.kspace_psf_fft` and `obs.fine_X/fine_Y` are precomputed at
`build_image_obs` time. They are JAX arrays packed into the `ImageObs`
pytree.

Why not lazy/per-call? `precompute_psf_kspace_fft` calls GalSim's
`drawKImage` + `np.fft.fft2` — neither is JAX-traceable. The PSF FFT
must exist as a fixed-shape `jnp.ndarray` before any JIT trace begins.

This is why `obs.render_config` is fixed at construction. To use a
different rc, rebuild the obs.

## Flux/pixel convention

All `render_image` outputs are **flux per pixel** (matching GalSim
`drawImage`'s default). `__call__` and `evaluate_in_disk_plane` return
**surface brightness** (flux per arcsec²); multiply by `pixel_scale**2`
for flux/pixel comparison. See base commit `63a30d5` for the migration.

## Accuracy Characterization

### Sinc vs GalSim `method='auto'`

| Profile | Face-on | Inclined (cosi=0.7) | Notes |
|---------|---------|---------------------|-------|
| Exponential (n=1) | < 1e-3 | < 1e-3 | Tight tolerance |
| Spergel ν=0.5 | < 1e-3 | < 1e-3 | Equivalent to exponential |
| Spergel ν=0.0 | ~2% | — | High-k aliasing from cusp |
| Spergel ν=-0.6 | ~5% | — | Divergent cusp, large aliasing |
| Sersic n=4 | TBD | TBD | Benchmark in progress |

For cuspy profiles (ν < 0), the sinc attenuates but doesn't eliminate
high-k aliasing. GalSim uses real-space integration for these profiles,
which handles the cusp differently. The 5% residual for ν=-0.6 is a
known limitation of any DFT-based approach at the native pixel Nyquist.

### maxk_threshold trade-off

`maxk_threshold` controls how much power above Nyquist is captured via
the k-space wrap operation. Lower threshold → larger k-grid → more
accurate but slower.

| `maxk_threshold` | Oversample (exp, r=0.3", cosi=1.0, 0.11"/px) | Accuracy vs GalSim | Use case |
|------------------|-----------------------------------------------|---------------------|----------|
| `1e-3` (default) | 1 (no wrap) | ~0.5% | Production inference |
| `5e-4` | 3 (wrap active) | <0.05% | Validation, diagnostics |
| `1e-4` | 3 | <0.01% | Ultra-tight validation |

For inclined profiles (cosi < 1), the wrap activates at coarser
thresholds because the compressed FT axis extends further:
- cosi=0.5, `maxk_threshold=1e-3`: oversample=3 already
- cosi=0.3, `maxk_threshold=1e-3`: oversample=5

Trade-off: narrower cosi prior AND/OR tighter `maxk_threshold` →
larger grid → slower. For Roman WL at typical SNR 20-50, the default
`1e-3` aliasing budget is well below noise.

### Residual aliasing (without wrap)

When `oversample == 1`, the residual aliasing floor is set by the
profile FT amplitude at Nyquist. For exponential at Roman pixel scale
(0.11"/px, rscale=0.3"): `A(k_N) ≈ 1e-3` of DC, residual after sinc
≈ 0.5%.

## Historical context

Pre-PR #41 the k-space pipeline applied pixel integration via spatial
oversampling + binning (O(N²) cost) and a separate "bin path" with
half-pixel phase compensation. That path was an *approximation* and
double-integrated when both `pixel_response` and oversample > 1 were
active in the fused PSF case (Bug A — caused ~16.5% systematic in
`int_h_over_r` recovery under flux/pixel convention).

PR #41 unified all multi-grid rendering under the wrap path and made
`obs.render_config` the single source of truth (Bug B / Issue #42 fix).
The `oversample` knob is retained as a manual override / convergence-
testing parameter.
