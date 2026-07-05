# Units and rendering conventions

Single source of truth for physical units and the input/output contracts of
the render methods in `kl_pipe`. CLAUDE.md's Physical Units table is the
short reference; this document expands on it and adds the cube / dispersion /
grism pipeline rules.

## Physical units (extends CLAUDE.md table)

| Quantity | Unit | Convention |
|----------|------|-----------|
| Spatial coordinates | arcsec | From `ImagePars.pixel_scale` |
| Wavelength | nm | `CubePars.lambda_grid`, `GrismPars.lambda_ref`, `EmissionLine.lambda_rest` |
| Velocities | km/s | Line-of-sight or circular; see `_C_KMS` in `kl_pipe.source` |
| Inclination | `cosi = cos(i)` | 0 = edge-on, 1 = face-on |
| Shear | dimensionless | `g1`, `g2`; reduced shear, |g| < 1 |
| Flux | integrated | `I0 = flux / (2 * pi * r_scale^2)` for canonical normalization |
| Surface brightness (SB) | flux / arcsec² | Per-point intensity from `__call__`, `evaluate_in_disk_plane` |
| Wavenumber (k) | rad/arcsec | `maxk`, `stepk`, k-space grids |

## Render method contract

The contract is: **every render method that returns an observable (a 2D
detector-pixel image) returns flux per coarse pixel.** Intermediate or
continuous representations stay in SB.

| Method | Returns | Notes |
|---|---|---|
| `IntensityModel.__call__(theta, plane, x, y)` | SB (per arcsec²) | Continuous-coord SB evaluation at the (x, y) samples. Multiply by `pixel_scale**2` to convert to flux per (uniform) pixel. |
| `IntensityModel.evaluate_in_disk_plane(theta, X, Y)` | SB (per arcsec²) | Same as `__call__` semantics; called per-plane. |
| `IntensityModel.render_image(theta, obs=obs)` | flux per coarse pixel | k-space FFT chain (profile × pixel_response × PSF), exact pixel integration. Returns observable units. |
| `IntensityModel._render_kspace(...)` | flux per coarse pixel | Internal k-space path; same convention as `render_image`. |
| `VelocityModel.__call__(theta, plane, x, y)` | km/s at (x, y) | LOS velocity (or speed when `return_speed=True`). |
| `VelocityModel.render_image(theta, obs=obs)` | km/s per coarse pixel | Intensive; flux-weighted PSF convolution preserves km/s units. |
| `SourceModel.render_broadband(pars, obs, band_key)` | flux per coarse pixel | Dispatches to `IntensityModel.render_image`; same convention. |
| `SourceModel.render_velocity(pars, obs)` | km/s per coarse pixel | Dispatches to `VelocityModel.render_image`. |
| `SourceModel.build_cube(pars, cube_pars, ...)` | SB per arcsec² per nm | Intermediate; cube voxel = line term `I_line(x, y) × G(λ; x, y)` (SB × 1/nm) plus continuum term `I_cont(x, y)` (already SB/nm — see below). |
| `kl_pipe.dispersion.disperse_cube(cube, grism_pars, lambda_grid, oversample)` | SB per arcsec² | Sums `cube × throughput × dlam` over wavelength; output is wavelength-integrated SB. |
| `kl_pipe.grism._apply_post_dispersion_pixel_response(...)` | flux per coarse pixel | Coarse-pixel sinc in k-space (the pixel integration), samples the box-averaged SB at coarse pixel centers, multiplies by `coarse_ps²` to convert SB → flux per coarse pixel. |
| `SourceModel.render_grism(pars, obs, ...)` | flux per coarse pixel | Final dispersed grism observable. |
| `KLModel.render_grism(theta, obs)` | flux per coarse pixel | Legacy path; same convention. |

## Emission line flux vs continuum flux density (cube assembly)

The cube voxel is a surface-brightness spectral density, `[flux / arcsec² / nm]`
(SB/nm). `build_cube` sums a line term and a continuum term, both in SB/nm. They
carry different front-end units because a line flux and a continuum density are
physically different quantities: a line has a finite integrated flux, a flat
continuum has none (it diverges over infinite λ) and is instead specified per unit
wavelength.

Emission line: `<line>.flux` is an integrated flux `[flux]`. Two normalizations
turn it into an SB/nm voxel:

```
<line>.flux  [flux]
   ÷ spatial profile   (∫∫ I dx dy = flux  ⇒  I carries 1/arcsec²)
I_line(x,y)  [flux/arcsec²] = SB
   × G(λ)               (∫ G dλ = 1        ⇒  G carries 1/nm)
I_line·G(λ)  [flux/arcsec²/nm] = SB/nm
```

`G(λ) = (1/(σ√2π))·exp(...)` is the normalized line-spread kernel. Integrating the
voxel back over λ (`disperse_cube` does this via `× dlam` then summing) and over
space recovers `<line>.flux`. The stored voxel is the **bin average** of `G` over
each coarse channel. With the default `spectral_method='erf'` this bin average is
computed exactly (Gaussian CDF differenced at the channel edges), so `Σ_bins × dλ
= 1` holds for any line width covered by the wavelength window. The retained
`spectral_method='oversample'` path approximates it by midpoint sub-sampling
(`spectral_oversample` sub-bins/channel) and requires the fine grid to resolve
`σ_λ`; it under-resolves narrow lines (dispersion ≲ 17 km/s at osf=15). This is
an accuracy concern, separate from units.

Continuum: `<line>.cont.flux_per_nm` is a spectral density `[flux/nm]`. The
intensity profile is linear in its amplitude, so feeding a `[flux/nm]` amplitude
yields `[flux/nm/arcsec²]` = SB/nm directly, with no extra per-nm factor (an added
`/nm` would double-count). `ContinuumModel` (in `model.py`) wraps a raw
`IntensityModel` and relabels its `flux` parameter to `flux_per_nm` so the density
is explicit in the dotted-key namespace. `EmissionLine` auto-wraps a raw continuum
model; passing the integrated name `<line>.cont.flux` raises. Floats carry no units
at runtime, so the name and that guard are what enforce the distinction; profile
linearity propagates it.

## SB ↔ flux conversion shorthand

Given an SB array sampled on a fine grid at pixel scale `fine_ps`, the
equivalent flux per coarse pixel (coarse_ps = N × fine_ps) is:

```
flux_per_coarse_pixel  =  mean(fine_SB over N×N fine cells) × coarse_ps²
                       =  sum(fine_SB over N×N fine cells) × fine_ps²
                       =  sum(fine_SB) × (coarse_ps / N)²
```

These are equivalent. When `N == 1`, `mean == sum == identity`, and the
conversion reduces to `SB × coarse_ps²`.

This shorthand applies to RAW (point-sampled) SB fields only. A field that
has already been pixel-integrated (e.g. multiplied by the coarse BoxPixel
sinc in k-space) must be READ OUT by sampling at coarse pixel centers, not
mean-binned — averaging an already box-averaged field is a second box
convolution (see "Pixel response" below).

## Pixel response

For broadband imaging, the BoxPixel sinc multiplies the profile FT in
k-space (`IntensityModel._render_kspace`); pixel integration is exact and
the output is already flux per pixel.

For grism, the BoxPixel sinc applies post-dispersion on the dispersed 2D
image (`_apply_post_dispersion_pixel_response`); the source-plane cube
cells are not detector pixels. The sinc (coarse-pixel side, on the fine
k-grid) IS the coarse-pixel integration: after the IFFT each fine cell
holds the coarse-box-averaged SB centered on that cell. The readout
SAMPLES that field at each coarse pixel center (the center fine cell of
each block; oversample must be odd) and multiplies by `coarse_ps²` to get
flux per coarse pixel. Averaging the block instead would apply a second,
unintended coarse-box convolution.

The pixel response is a coarse-detector property — the BoxPixel side length
is `coarse_ps`, not `fine_ps`. This matters when oversample > 1 (the sinc
on the fine k-grid uses the coarse pixel scale).

## Likelihood interface

The likelihood `(data - model)² / variance` requires `data` and `model`
in the same units. The observed grism / image data is in detector flux
(counts × calibration constant ≈ flux per coarse pixel). All `render_*`
methods that feed the likelihood return flux per coarse pixel, by
contract.

## Forward-looking notes

- **IFU `render_cube` (future).** When IFU data products land, the
  observed-cube renderer (let's tentatively call it `SourceModel.render_cube`)
  must return flux per (coarse spatial pixel × wavelength bin) — i.e.
  `SB × coarse_ps² × dlam`. `build_cube` stays in SB-per-nm as the
  intermediate; the observable wrapper applies the unit conversion.
- **Multi-resolution detectors.** When per-pixel detector responses vary
  (e.g. saturation, gain maps), the conversion still goes "SB → flux per
  pixel" via `coarse_ps²` before the per-pixel response is applied. The
  `PixelResponse` abstraction supports this; only `BoxPixel` is implemented
  today, but the `SB × coarse_ps²` step is the unit-conversion contract
  regardless of the response shape.

## See also

- `CLAUDE.md` — Physical Units table (short reference).
- `kl_pipe/render.py` — `RenderConfig`, `for_priors`, `for_grism_priors`.
- `docs/notes/grism_cube_bandwidth.tex` (gitignored) — bandwidth derivation
  for grism cube fine-grid sizing.
