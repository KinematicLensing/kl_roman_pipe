# Pixel Integration & PSF Convolution

Reference document for the k-space rendering pipeline's pixel integration
and PSF convolution architecture.

## Architecture

The pipeline renders intensity models in Fourier space (k-space), where
three effects are applied as multiplicative factors before the IFFT:

```
I_measured = IFFT( profile_FT × pixel_FT × PSF_FT )
```

- **Profile FT**: analytic Fourier transform of the galaxy surface
  brightness (exponential, Spergel, Sersic emulator)
- **Pixel FT**: `PixelResponse` object — default `BoxPixel` has FT =
  sinc(kx·w/(2π)) × sinc(ky·w/(2π)) where w = pixel_scale
- **PSF FT**: bare PSF FT from `drawKImage` (no pixel convolution)

This is mathematically exact pixel integration at O(1) cost, replacing
the previous O(N²=25×) spatial oversampling approximation.

## PixelResponse

Defined in `kl_pipe/pixel.py`. `PixelResponse` is an ABC with:
- `ft(KX, KY)`: FT of pixel response, normalized to 1 at DC
- `maxk(threshold)`: wavenumber where |FT| drops below threshold

`BoxPixel(pixel_scale)` implements the square top-hat pixel. Lives on
`ImageObs`; `build_image_obs` creates one by default. Pass
`pixel_response=None` to disable (testing, point-sampled comparisons).

## Rendering Paths

### Fused k-space path (preferred)
Profile FT × pixel sinc × bare PSF FT → single IFFT → image.
Used when `obs.kspace_psf_fft` is set (k-space intensity models with PSF).

### Real-space PSF fallback
PSF kernel from `drawImage` (already includes pixel integration).
Source rendered point-sampled (no pixel_response). Pixel integration
via spatial oversampling + binning.

### No-PSF path
Profile FT × pixel sinc → IFFT → image.
Pixel integration without PSF convolution.

### Velocity models
Pixel integration via spatial oversampling (fine grid → bin).
Sinc can't apply to flux-weighted ratio `Conv(I*v, PSF) / Conv(I, PSF)`.
`PixelResponse` on obs is ignored by velocity rendering.

## Adaptive Grid Sizing via RenderConfig

`RenderConfig` in `kl_pipe/render.py` controls k-space grid sizing.

Each intensity model provides:
- `maxk(params, threshold)`: where bare profile FT drops below threshold
  (depends on scale length AND cosi — inclination extends FT by 1/cosi)
- `stepk(params, folding_threshold)`: minimum k-spacing for flux containment
- `_ft_envelope(k, params)`: forward FT evaluation for combined product scan

The effective maxk is computed from the combined product
`profile_FT(k) × pixel_sinc(k)`, not `min(individual maxks)`. The sinc
attenuation at each k significantly reduces the needed grid size for
inclined profiles.

Two factory methods:
- `RenderConfig.for_model(model, params, pixel_scale, ...)` — from specific theta
- `RenderConfig.for_priors(model, priors, pixel_scale, ...)` — worst-case from priors

`InferenceTask` computes `RenderConfig.for_priors` at construction and
freezes it into the JIT'd likelihood closure.

Trade-off: narrower cosi prior → smaller maxk → smaller grid → faster.
`maxk_threshold` (default 1e-3) controls aliasing budget.

## Historical: Spatial Oversampling

The previous approach (superseded for k-space intensity models):
1. Render source at N× finer resolution
2. Convolve with PSF at fine scale
3. Bin N×N fine pixels to coarse pixels (mean for intensity, sum/sum for velocity)

This approximates pixel integration with O(N²) cost. At N=5 (default),
accuracy was ~7e-5 for Gaussian PSFs, ~5e-4 for sharp PSFs.

The `oversample` parameter is retained for:
- Velocity models (spatial oversampling is the only option)
- Benchmarking (compare oversample vs sinc)
- Convergence testing (oversample N→∞ should converge to sinc)

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

When the wrap is not active (oversample=1), the residual aliasing
floor is set by the profile's FT amplitude at Nyquist:

For exponential at Roman pixel scale (0.11"/px, rscale=0.3"):
`A(k_N) ≈ 1e-3 of DC`, residual after sinc ≈ 0.5%.
