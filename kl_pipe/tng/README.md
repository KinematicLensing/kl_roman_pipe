# TNG50 Mock Data Module

This module provides tools for working with TNG50 IllustrisTNG mock galaxy observations, converting 3D particle data into 2D pixelized intensity and velocity maps suitable for kinematic lensing analysis.

## Overview

The TNG50 simulation provides realistic galaxy particle data (gas and stars) with known 3D positions, velocities, masses, and luminosities. This module handles:

1. **Data Loading**: Load pre-processed TNG50 data from CyVerse repository
2. **Coordinate Transformation**: Convert from TNG comoving coordinates to observed frame (arcsec)
3. **Orientation Control**: Render galaxies at native or custom inclinations/PAs
4. **Gridding**: Project 3D particle data onto 2D pixel grids using **CIC** (Cloud-in-Cell) or **NGP** (Nearest-Grid-Point)
5. **Noise Addition**: Add realistic Gaussian noise to mock observations

## Key Components

### Data Loaders (`loaders.py`)

```python
from kl_pipe.tng import TNG50MockData

# Load all TNG50 galaxies
tng_data = TNG50MockData()

# Access individual galaxies
galaxy = tng_data[0]  # First galaxy
galaxy = tng_data.get_galaxy(subhalo_id=8)  # By SubhaloID

# Galaxy data structure:
# - galaxy['gas']: Gas particle data (Coordinates, Velocities, Masses)
# - galaxy['stellar']: Stellar particle data (Coordinates, Velocities, Luminosities_*)
# - galaxy['subhalo']: Metadata (Inclination_star, Position_Angle_star, DistanceMpc, etc.)
```

**Available galaxies:** 5 TNG50 galaxies with SubhaloIDs: 8, 17, 19, 20, 29

### Data Vector Generation (`data_vectors.py`)

#### TNGRenderConfig

Configuration object specifying how to render a galaxy:

```python
from kl_pipe.tng import TNGRenderConfig
from kl_pipe.parameters import ImagePars

image_pars = ImagePars(shape=(128, 128), pixel_scale=0.1, indexing='ij')

config = TNGRenderConfig(
    image_pars=image_pars,
    band='r',                      # Photometric band: 'g', 'r', 'i', 'u', 'z'
    use_dusted=True,               # Use dust-attenuated luminosities
    center_on_peak=True,           # Center on luminosity peak (vs subhalo center)
    use_native_orientation=True,   # Use TNG's native inc/PA (or custom if False)
    pars=None,                     # Custom orientation params (if use_native_orientation=False)
    use_cic_gridding=True,         # Cloud-in-Cell (True) vs Nearest-Grid-Point (False)
    target_redshift=0.7,           # Scale to z=0.7 for Roman-like sub-arcsec obs (TNG native z~0.01)
    preserve_gas_stellar_offset=True,  # Keep physical gas-stellar misalignment (default)
)
```

#### TNGDataVectorGenerator

Main class for generating 2D maps:

```python
from kl_pipe.tng import TNGDataVectorGenerator

gen = TNGDataVectorGenerator(galaxy)

# Generate intensity map (uses stellar particles)
intensity, variance = gen.generate_intensity_map(config, snr=50, seed=42)

# Generate velocity map (uses gas particles)
velocity, variance = gen.generate_velocity_map(config, snr=50, seed=42)
```

**Key features:**
- **Intensity**: Uses stellar particle luminosities (band-dependent, dust-attenuated or raw)
- **Velocity**: Uses gas particle masses and velocities (mass-weighted)
- **Native orientation**: Render at TNG's intrinsic inclination/PA
- **Custom orientation**: Transform to any desired geometry via `pars` dict

## Coordinate Systems

### TNG Native Frame
- **Units**: Comoving kpc/h (must convert to physical)
- **Origin**: Subhalo position
- **Inclination**: 0-180° (0°=face-on from above, 90°=edge-on, 180°=face-on from below)
- **Position Angle**: 0-360° (East of North convention)
- **Redshift**: z~0.011 (Distance ~50 Mpc)
- **Angular Size**: ~21 arcmin (~1300 arcsec) - **too large for sub-arcsec observations!**

### Observer Frame
- **Units**: arcsec
- **Conversion**: Uses DistanceMpc and cosmology (h=0.6774, 206.265 arcsec/rad)
- **Centered**: On luminosity peak or subhalo center
- **Redshift Scaling**: Use `target_redshift` parameter to scale angular size
  - TNG native (z~0.01): Galaxy spans ~1300" (~21 arcmin)
  - Roman-like (z=0.5-1.0): Galaxy spans ~15-25" (sub-arcsec pixels see structure)
  - **Recommended**: `target_redshift=0.7` for 0.1"/pixel observations

### Transform Pipeline
When `use_native_orientation=False`, we apply:
1. **Undo native**: obs→disk (deproject TNG's native orientation to face-on)
2. **Apply new**: disk→obs (project to desired orientation from `pars`)

Uses `kl_pipe.transformation` functions following: obs→cen→source→gal→disk

## Inclination >90° Handling

**Important:** TNG inclinations >90° indicate viewing from below the disk. For numerical stability in coordinate deprojection (which divides by cos(i)), we convert to equivalent <90° view:
- `inc' = 180° - inc`
- `PA' = PA + 180°`
- Vertical velocities (v_z) are negated to preserve physics

This maintains full 0-180° parameter space coverage while avoiding coordinate explosion when cos(i) < 0.

## Gridding Algorithms

### Cloud-in-Cell (CIC)
- Distributes each particle to 4 nearest grid points with bilinear weights
- Produces smoother maps than NGP
- ~0.5s per galaxy
- **Recommended** for most use cases

### Nearest-Grid-Point (NGP)
- Assigns each particle to single nearest grid point
- Faster but produces noisier maps
- Useful for debugging or when exact particle counts matter

## Example Usage

### Basic: Native Orientation

```python
from kl_pipe.tng import TNG50MockData, TNGDataVectorGenerator, TNGRenderConfig
from kl_pipe.parameters import ImagePars

# Load data
tng_data = TNG50MockData()
galaxy = tng_data[0]

# Setup
gen = TNGDataVectorGenerator(galaxy)
image_pars = ImagePars(shape=(64, 64), pixel_scale=0.1, indexing='ij')
config = TNGRenderConfig(
    image_pars=image_pars, 
    use_native_orientation=True,
    target_redshift=0.7  # Scale to Roman-like distance
)

# Generate maps
intensity, int_var = gen.generate_intensity_map(config, snr=50)
velocity, vel_var = gen.generate_velocity_map(config, snr=50)

print(f"Galaxy at inc={gen.native_inclination_deg:.1f}°, PA={gen.native_pa_deg:.1f}°")
```

### Advanced: Custom Orientation

```python
import numpy as np

# Define custom geometry
pars = {
    'cosi': np.cos(np.radians(45.0)),  # 45° inclination
    'theta_int': np.radians(30.0),     # 30° position angle
    'x0': 0.0,                         # No centroid offset
    'y0': 0.0,
    'g1': 0.05,
    'g2': 0.02,
}

config = TNGRenderConfig(
    image_pars=image_pars,
    use_native_orientation=False,
    pars=pars,
    use_cic_gridding=True,
    target_redshift=0.7,  # Scale to Roman-like distance
)

intensity, _ = gen.generate_intensity_map(config, snr=None)  # No noise
velocity, _ = gen.generate_velocity_map(config, snr=None)
```

## Testing

Run TNG tests with:
```bash
pytest tests/test_tng_*.py -v
```

Key test files:
- `test_tng_loaders.py`: Data loading and access
- `test_tng_data_vectors.py`: Rendering, transforms, gridding, diagnostic plots (40 tests)
- `test_tng_likelihood.py`: Model fitting with TNG truth data

Diagnostic plots are saved to `tests/out/tng_diagnostics/` including:
- Inclination and PA sweep comparisons (with/without gas-stellar offset preservation)
- Resolution and SNR grids
- Symmetry-breaking demonstration plots

## Data Provenance

TNG50 mock data downloaded from CyVerse:
- Path: `data/tng50/`
- Files: `gas_data_analysis.npz`, `stellar_data_analysis.npz`, `subhalo_data_analysis.npz`
- See `data/cyverse/README.md` for download instructions

## Particle Types

- **Stellar (PartType4)**: 95k-624k particles per galaxy
  - Used for: Intensity maps
  - Properties: Coordinates, Velocities, Dusted/Raw_Luminosity_{g,r,i,u,z}
  
- **Gas (PartType0)**: 84k-265k particles per galaxy  
  - Used for: Velocity maps
  - Properties: Coordinates, Velocities, Masses

## Orientation Handling

### Angular Momentum-Based Rotation

TNG galaxies come with intrinsic 3D orientations. We use **angular momentum vectors** to properly transform particles to a face-on reference frame before applying user-specified orientations:

- **Stellar particles**: Use `SubhaloSpin` (total angular momentum from TNG catalog)
- **Gas particles**: Compute angular momentum from particle positions and velocities: $\vec{L} = \sum_i m_i (\vec{r}_i \times \vec{v}_i)$

The rotation matrix aligns the angular momentum vector with +Z using the **Rodrigues formula**:
```
R = I + sin(θ)K + (1-cos(θ))K²
```
where K is the skew-symmetric cross-product matrix of the rotation axis.

### Gas-Stellar Misalignment

**Important:** Gas and stellar disks in TNG galaxies are often misaligned by **30-40°**! This is a real physical effect from different formation histories.

The `preserve_gas_stellar_offset` parameter controls how this is handled:

| Setting | Behavior | Use Case |
|---------|----------|----------|
| `True` (default) | Gas uses **stellar** rotation matrix, preserving intrinsic offset | Realistic simulations |
| `False` | Gas uses its **own** rotation matrix, forcing perfect alignment | Synthetic tests |

When `preserve_gas_stellar_offset=True`:
- User's `(cosi, theta_int)` defines **stellar disk** orientation
- Gas disk is tilted by its natural offset relative to stellar
- Intensity and velocity maps show realistic misalignment

### Native Orientation (`use_native_orientation=True`)
- **Stellar and gas particles are simply projected along z-axis** (no transformations)
- Physical stellar-gas offset from TNG simulation is fully preserved
- This is the recommended mode for analyzing realistic TNG galaxies

### Custom Orientation (`use_native_orientation=False`)
1. **Undo native orientation**: Rotate particles to face-on using angular momentum
2. **Apply new orientation**: Project to user's `(cosi, theta_int, g1, g2)` geometry

The `pars` dict specifies stellar disk geometry:
- `cosi`: cos(inclination) for stellar disk (1=face-on, 0=edge-on)
- `theta_int`: position angle for stellar disk (radians)
- `x0`, `y0`: centroid offsets
- `g1`, `g2`: lensing shear (validated: |g| < 1 required)

### LOS Velocity Projection

Line-of-sight velocity follows the formalism from [Xu et al. 2022 (arXiv:2201.00739)](https://arxiv.org/abs/2201.00739):

```
v_LOS = v_y * sin(i) + v_z * cos(i)
```

where (v_x, v_y, v_z) are velocity components in the face-on disk frame after PA rotation. This accounts for:
- In-plane rotational motion (v_y term, dominant for rotating disks)
- Vertical motion from disk thickness (v_z term, captures dispersion)

## Known Limitations

1. **Empty velocity pixels**: Pixels with no gas particles are set to 0, not NaN. Cannot distinguish "no data" from "zero velocity"
2. **No PSF convolution**: Point-spread function effects not yet implemented
3. **Gaussian noise only**: Poisson noise is available via `include_poisson=True` but adds complexity for intensity map visualization (negative fluctuations)
4. **TNG sample size**: Only 5 galaxies currently available (SubhaloIDs: 8, 17, 19, 20, 29)

## TODOs

- [ ] Add PSF convolution
- [ ] Integrate Poisson noise properly
- [ ] Add mask support to likelihoods
- [ ] Expand to more TNG50 galaxies
- [ ] Add NaN handling for empty velocity pixels

## References

- **TNG50 Simulation**: [Nelson et al. 2019](https://ui.adsabs.harvard.edu/abs/2019ComAC...6....2N), [Pillepich et al. 2019](https://ui.adsabs.harvard.edu/abs/2019MNRAS.490.3196P)
- **Velocity Transforms**: [Xu et al. 2022, arXiv:2201.00739](https://arxiv.org/abs/2201.00739)
- **Cloud-in-Cell**: Hockney & Eastwood 1988, "Computer Simulation Using Particles"
