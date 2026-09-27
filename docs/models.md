# Intensity & Velocity Model Reference

## Intensity Models

| Factory name | Class | Shape param | FT type | Description |
|---|---|---|---|---|
| `inclined_exp` (default) | `InclinedExponentialModel` | n=1 fixed | `(1+k²)^{-3/2}` | Exponential disk + sech² vertical |
| `inclined_spergel`, `spergel` | `InclinedSpergelModel` | `nu` (continuous) | `(1+k²)^{-(1+nu)}` | Spergel profile + sech² vertical |
| `de_vaucouleurs` | `InclinedDeVaucouleursModel` | nu=-0.6 fixed | `(1+k²)^{-0.4}` | Spergel at best-fit nu for Sersic n~4 |
| `inclined_sersic`, `sersic` | `InclinedSersicModel` | `n_sersic` in [0.5, 6] | emulator (Miller & Pasha 2025) | Sersic + sech² vertical; finite at R=0 at all n |
| `bulge_disk` | `BulgeDiskModel` | bulge `n_sersic` (fixed 4 by default) | sum of component FTs | Exponential disk + Sersic bulge |

All single-component models share the same 3D structure: radial profile (model-specific) times sech²(z/h_z) vertical profile, integrated along the line of sight. `InclinedSersicModel` is parameterized by `hlr` and `h_over_hlr` instead of `rscale` and `h_over_r`. For n=1 prefer `InclinedExponentialModel` (exact FT).

### Composite models

`CompositeIntensityModel(components=[ComponentSpec(model, prefix, fixed_params), ...], shared_pars=...)` sums N component FTs before one IFFT. Shared params (default `cosi, theta_int, g1, g2`) stay bare; other component params become `<prefix>_<name>` (e.g. `disk_rscale`). Flux is `total_flux` plus `<prefix>_frac` for components 1..N-1; component 0 gets the remainder. `amplitude_param = 'total_flux'`.

`BulgeDiskModel(shared_centroids=False, shear_bulge=True, bulge_nsersic=4.0)` is the disk (`InclinedExponentialModel`, prefix `disk`) + bulge (`InclinedSersicModel`, prefix `bulge`) composite. `bulge_nsersic=None` samples `bulge_n_sersic`; `shear_bulge=False` fixes the bulge shear to zero; `shared_centroids=True` shares `x0, y0`.

### Evaluation paths

- **`render_image`** (k-space FFT): exact analytic FT, fully differentiable. Use for gradient-based inference.
- **`__call__`** (LOS Gauss-Legendre quadrature): real-space evaluation via scipy K_nu callback. NOT auto-differentiable. For nu < 0, diverges at LOS through R=0.
- **`evaluate_in_disk_plane`**: face-on radial profile only. For velocity flux weighting.

### Spergel profile

The Spergel profile (Spergel 2010) generalizes the exponential via:

    I(r) = I_0 * (r/c)^nu * K_nu(r/c)

where c = `rscale` (scale length) and K_nu is the modified Bessel function of the second kind. The analytic FT `(1+k²)^{-(1+nu)}` makes k-space rendering a one-exponent change from the exponential.

| nu | Sersic n (approx) | Profile character |
|---|---|---|
| -0.6 | ~4.0 | de Vaucouleurs (concentrated, cusp at r=0) |
| 0.0 | ~1.6 | log-divergence at r=0 |
| 0.5 | 1.0 | exponential (exact equivalence) |
| 1.0 | ~0.7 | smoother than exponential |
| 2.0 | ~0.5 | very smooth, extended |

For nu < 0, the profile diverges at r=0 as r^{2nu}. The k-space rendering handles this correctly (FT is finite everywhere). The `__call__` LOS quadrature does not — see known limitations below.

### nu ↔ Sersic n mapping

Convenience functions for converting between Spergel nu and Sersic n:

```python
from kl_pipe.intensity import sersic_to_spergel, spergel_to_sersic

nu = sersic_to_spergel(4.0)   # -> ~-0.6
n = spergel_to_sersic(0.5)    # -> 1.0 (exact)
```

These use pre-computed lookup tables from minimizing the integrated radial profile difference. Exact at n=1/nu=0.5. Roundtrip error < 1% for n in [0.3, 6.2].

### Known limitations

- **nu < 0 central divergence**: the volume density diverges as R^{2nu} at R=0. The `__call__` LOS integral through R=0 diverges. Use `render_image` (k-space) for all inference.
- **nu < 0.5 GalSim comparison**: `galsim.Spergel` uses real-space evaluation (`is_analytic_x=True`), our code uses k-space IFFT. These differ near the cusp due to band-limiting. Face-on GalSim regression uses PSF + `method='auto'` for nu < 0.5.
- **Spergel ≠ Sersic**: the mapping is approximate for n ≠ 1. The Spergel profile has different wing behavior from Sersic at the same effective radius.

## Velocity Models

| Factory name | Class | Params | Description |
|---|---|---|---|
| `centered` | `CenteredVelocityModel` | 7 | No centroid params (centered at origin) |
| `offset` (default) | `OffsetVelocityModel` | 9 | Own centroid (`x0`, `y0`) |

Both use the arctan rotation curve: `v_circ(r) = (2/pi) * vcirc * arctan(r / rscale)`.

### Parameter names

| Model | PARAMETER_NAMES |
|---|---|
| CenteredVelocity | `cosi, theta_int, g1, g2, v0, vcirc, rscale` |
| OffsetVelocity | `cosi, theta_int, g1, g2, v0, vcirc, rscale, x0, y0` |
| InclinedExponential | `cosi, theta_int, g1, g2, flux, rscale, h_over_r, x0, y0` |
| InclinedSpergel | `cosi, theta_int, g1, g2, flux, rscale, h_over_r, nu, x0, y0` |
| InclinedDeVaucouleurs | `cosi, theta_int, g1, g2, flux, rscale, h_over_r, x0, y0` |
| InclinedSersic | `cosi, theta_int, g1, g2, flux, hlr, h_over_hlr, n_sersic, x0, y0` |
| BulgeDisk (defaults) | `cosi, theta_int, g1, g2, total_flux, bulge_frac, disk_rscale, disk_h_over_r, disk_x0, disk_y0, bulge_hlr, bulge_h_over_hlr, bulge_x0, bulge_y0` |

### Naming conventions

Class `PARAMETER_NAMES` tuples use bare names (no `vel_`/`int_` prefix). The
namespace is carried by the dotted `SourceModel` keys the priors/theta use
(`vel.rscale`, `F087.rscale`, `Halpha.x0`); shared geometric params stay
top-level (no dot).

| Pattern | Meaning |
|---|---|
| No prefix / top-level | Shared geometric: `cosi`, `theta_int`, `g1`, `g2` |
| `vel.<param>` | Velocity component key: `vel.rscale`, `vel.x0`, `vel.y0` |
| `<band>.<param>` / `<line>.<param>` | Intensity component key: `F087.rscale`, `F087.h_over_r`, `Halpha.x0` |
| `flux` | Total integrated flux (intensity) |
| `nu` | Spergel index (InclinedSpergelModel only) |

### Physical units

| Quantity | Unit |
|---|---|
| Coordinates | arcsec |
| Velocities | km/s |
| Position angle | radians, from +x, [0, 2pi) |
| Inclination | `cosi = cos(i)`: 0=edge-on, 1=face-on |
| Shear | dimensionless g1, g2; \|g\| < 1 |
| Flux | integrated (not surface brightness) |
| Scale height | `h_over_r * rscale` (arcsec) |

## SourceModel and observations

`SourceModel` (`kl_pipe/source.py`) is the object inference works with. It holds any subset of `velocity_model` (one `VelocityModel`), `broadband_models` (`{band: IntensityModel}`), and `emission_lines` (`{line: EmissionLine}`, `kl_pipe/lines.py`). Band and line keys must be disjoint. Model classes see flat theta arrays; only `SourceModel` knows the dotted-key namespace.

### Parameter routing

| Component | Key lookup (first match wins) |
|---|---|
| Velocity model | `vel.<name>`, then `<name>` |
| Broadband model for band `B` | `B.<name>`, then `<name>` |
| Line intensity for line `L` | `L.<name>`, then `<name>` |
| Line continuum for line `L` | `L.cont.<name>`, then `<name>`; amplitude is `L.cont.flux_per_nm` (flux/arcsec²/nm) |
| Line dispersion | `L.dispersion` only (km/s) |
| Redshift | `z` (top level) |

Sharing between lines (`EmissionLine` fields):

| Field | Effect |
|---|---|
| `intensity_key='M'` | Use line `M`'s intensity model and shape params (`M.<name>`); amplitude stays `L.<amplitude_param>` |
| `continuum_key='M'` | Same for the continuum: shape from `M.cont.<name>`, amplitude `L.cont.flux_per_nm` |
| `dispersion_key='M'` | Read `M.dispersion`; chained references raise |

Observations bind to components by key: an `ImageObs` renders `broadband_models[broadband_key]` (the `image_obs` dict key must match), a `VelocityObs` flux-weights its PSF convolution with the intensity of `emission_lines[flux_weight_key]` (`None` allowed only without a PSF), and a `GrismObs` renders every emission line. Celestial-frame `theta_int`, `(g1, g2)` and `(x0, y0)` are rotated into each obs's detector frame from its WCS.

### Observation types (`kl_pipe/observation.py`)

| Type | Builder | Key fields |
|---|---|---|
| `ImageObs` | `build_image_obs(image_pars, *, psf, data, variance, mask, pixel_response, render_config, broadband_key, ...)` | `image_pars`, `render_config`, `psf_data`, `kspace_psf_fft`, `pixel_response`, `broadband_key`, `flux_unit` |
| `VelocityObs` | `build_velocity_obs(image_pars, *, psf, data, variance, mask, render_config, flux_weight_key, ...)` | `ImageObs` fields + `flux_weight_key` |
| `GrismObs` | `build_grism_obs(grism_pars, z, *, psf, data, variance, mask, render_config, velocity_window_kms, ...)` | `grism_pars`, `cube_pars`, `psf_data`, `render_config`, `pixel_response_fft` |

When `render_config` is omitted, `InferenceTask.from_obs` rebuilds the obs with a prior-sized config (`build_image_render_config` / `build_grism_render_config` in `render.py`).

### Render methods

| Method | Output |
|---|---|
| `render_broadband(pars, obs, band_key)` | Image, flux/pixel |
| `render_velocity(pars, obs)` | LOS velocity map, km/s |
| `render_grism(pars, obs)` | Dispersed 2D grism image, flux/pixel |
| `render_grism_group(pars, obs_group)` | Dict of grism images from one shared cube |
| `build_cube(pars, cube_pars)` | Intrinsic (x, y, λ) cube, no PSF |

`pars` is the full dotted-key dict (sampled + fixed).

### Inference

```python
task = InferenceTask.from_obs(
    source, priors,
    image_obs={'F129': obs_f129},
    grism_obs={'roll0': obs_g0},
    velocity_obs=None,
)
```

The likelihood is the sum of Gaussian log-likelihoods over all channels given (`create_jitted_likelihood_from_obs` in `likelihood.py`). At least one obs is required; grism obs require `velocity_model` and `emission_lines`.

There is no instrumental line-spread function parameter: line width in the cube is the intrinsic `L.dispersion`, and slitless spectral blurring comes from the spatial PSF and dispersion.

## Adding an observation type

Files that must change today:

1. `observation.py`: frozen dataclass, pytree flatten/unflatten + `jax.tree_util.register_pytree_node`, `build_<type>_obs` builder.
2. `source.py`: `SourceModel.render_<type>`.
3. `likelihood.py`: `_log_likelihood_<type>_source`, a branch in `_log_likelihood_total_source`, a kwarg on `create_jitted_likelihood_from_obs`.
4. `sampling/task.py`: kwarg on `InferenceTask.from_obs`, source-to-obs validation, `isinstance` branch in `_check_source_priors_fit_obs`.
5. `render.py`: `build_<type>_render_config` if the renderer uses a k-space grid.
6. `synthetic.py`: independent generator (not using `model.py` / `source.py`).
7. Tests: likelihood slices, optimizer recovery, noise calibration (`tests/test_likelihood_slices.py`, `tests/test_optimizer_recovery.py`, `tests/test_noise_calibration.py`).
