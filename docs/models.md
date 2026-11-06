# Intensity & Velocity Model Reference

## Intensity Models

| Factory name | Class | Shape param | FT type | Description |
|---|---|---|---|---|
| `inclined_exp` (default) | `InclinedExponentialModel` | n=1 fixed | `(1+k²)^{-3/2}` | Exponential disk + sech² vertical |
| `inclined_spergel`, `spergel` | `InclinedSpergelModel` | `nu` (continuous) | `(1+k²)^{-(1+nu)}` | Spergel profile + sech² vertical |
| `de_vaucouleurs` | `InclinedDeVaucouleursModel` | nu=-0.6 fixed | `(1+k²)^{-0.4}` | Spergel at best-fit nu for Sersic n~4 |

All intensity models share the same 3D structure: radial profile (model-specific) times sech²(z/h_z) vertical profile, integrated along the line of sight.

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
| `centered` (default) | `CenteredVelocityModel` | 7 | Shared centroid with intensity |
| `offset` | `OffsetVelocityModel` | 9 | Independent centroid (`x0`, `y0`) |

Both use the arctan rotation curve: `v_circ(r) = (2/pi) * vcirc * arctan(r / rscale)`.

### Parameter names

| Model | PARAMETER_NAMES |
|---|---|
| CenteredVelocity | `cosi, theta_int, g1, g2, v0, vcirc, rscale` |
| OffsetVelocity | `cosi, theta_int, g1, g2, v0, vcirc, rscale, x0, y0` |
| InclinedExponential | `cosi, theta_int, g1, g2, flux, rscale, h_over_r, x0, y0` |
| InclinedSpergel | `cosi, theta_int, g1, g2, flux, rscale, h_over_r, nu, x0, y0` |
| InclinedDeVaucouleurs | `cosi, theta_int, g1, g2, flux, rscale, h_over_r, x0, y0` |

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
