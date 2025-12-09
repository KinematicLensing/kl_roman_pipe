---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.0
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# Working with TNG50 Mock Data

This tutorial demonstrates how to load TNG50 mock galaxy data and render it as 2D intensity and velocity maps for kinematic lensing analysis.

## Prerequisites

Before running this tutorial, download the TNG50 mock data from CyVerse:

```bash
make download-cyverse-data
```

This downloads three data files (~340 MB total) to `data/tng50/`:
- `gas_data_analysis.npz` - Gas particle data (coordinates, velocities, masses)
- `stellar_data_analysis.npz` - Stellar particle data (coordinates, velocities, luminosities)
- `subhalo_data_analysis.npz` - Galaxy metadata (inclination, PA, distance, angular momentum)

## 1. Loading TNG50 Data

```python
import numpy as np
import matplotlib.pyplot as plt
from kl_pipe.tng import TNG50MockData, TNGDataVectorGenerator, TNGRenderConfig
from kl_pipe.parameters import ImagePars
```

### Load all galaxies

```python
# Load all TNG50 mock data
tng_data = TNG50MockData()

print(f"Number of galaxies: {len(tng_data)}")
print(f"Available SubhaloIDs: {tng_data.subhalo_ids}")
```

### Access individual galaxies

```python
# Access by index
galaxy = tng_data[0]

# Or by SubhaloID
galaxy = tng_data.get_galaxy(subhalo_id=8)

# Galaxy contains three components
print(f"Keys: {list(galaxy.keys())}")
print(f"Stellar particles: {len(galaxy['stellar']['Coordinates']):,}")
print(f"Gas particles: {len(galaxy['gas']['Coordinates']):,}")
```

## 2. Creating a Data Vector Generator

The `TNGDataVectorGenerator` handles coordinate transformations and map rendering:

```python
gen = TNGDataVectorGenerator(galaxy)

print(f"Galaxy distance: {gen.distance_mpc:.1f} Mpc")
print(f"Native inclination: {gen.native_inclination_deg:.1f}°")
print(f"Native PA: {gen.native_pa_deg:.1f}°")
print(f"Gas-stellar angular momentum offset: {gen._gas_stellar_L_angle_deg:.1f}°")
```

## 3. Rendering at Native Orientation

The simplest approach renders the galaxy as it appears in the TNG simulation:

```python
# Define image parameters
# TNG galaxies at z~0.01 are HUGE (~20 arcmin), so we use target_redshift to scale them
image_pars = ImagePars(shape=(64, 64), pixel_scale=0.1, indexing='ij')

# Configure rendering
config = TNGRenderConfig(
    image_pars=image_pars,
    band='r',                      # Photometric band for intensity
    use_native_orientation=True,   # Use TNG's native orientation
    target_redshift=0.7,           # Scale to z=0.7 (Roman-like distance)
)

# Generate maps
intensity, int_var = gen.generate_intensity_map(config, snr=50, seed=42)
velocity, vel_var = gen.generate_velocity_map(config, snr=50, seed=42)

print(f"Intensity map shape: {intensity.shape}")
print(f"Velocity range: {velocity.min():.1f} to {velocity.max():.1f} km/s")
```

### Visualize the maps

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Intensity (log scale)
ax = axes[0]
int_log = np.log10(np.clip(intensity, 1e-10, None))
im = ax.imshow(int_log, origin='lower', cmap='viridis')
ax.set_title(f'Intensity (log scale)\ninc={gen.native_inclination_deg:.1f}°, PA={gen.native_pa_deg:.1f}°')
plt.colorbar(im, ax=ax, label='log₁₀(Flux)')

# Velocity (diverging colormap centered on zero)
ax = axes[1]
vmax = np.nanmax(np.abs(velocity))
im = ax.imshow(velocity, origin='lower', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
ax.set_title('Line-of-Sight Velocity')
plt.colorbar(im, ax=ax, label='v_LOS [km/s]')

plt.tight_layout()
plt.show()
```

## 4. Custom Orientation

You can render at any orientation by specifying geometric parameters:

```python
# Define custom orientation
pars = {
    'cosi': np.cos(np.radians(60)),  # 60° inclination
    'theta_int': np.radians(45),      # 45° position angle
    'x0': 0.0, 'y0': 0.0,             # No centroid offset
    'g1': 0.0, 'g2': 0.0,             # No shear
}

config_custom = TNGRenderConfig(
    image_pars=image_pars,
    use_native_orientation=False,
    pars=pars,
    target_redshift=0.7,
    preserve_gas_stellar_offset=True,  # Keep realistic gas-stellar misalignment
)

intensity_custom, _ = gen.generate_intensity_map(config_custom, snr=None)
velocity_custom, _ = gen.generate_velocity_map(config_custom, snr=None)
```

### Gas-Stellar Offset Preservation

TNG galaxies have real physical misalignment between gas and stellar disks (typically 30-40°). The `preserve_gas_stellar_offset` parameter controls this:

```python
# Compare with and without offset preservation
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for idx, preserve_offset in enumerate([True, False]):
    config = TNGRenderConfig(
        image_pars=image_pars,
        use_native_orientation=False,
        pars={'cosi': 0.5, 'theta_int': 0.0, 'x0': 0, 'y0': 0, 'g1': 0, 'g2': 0},
        target_redshift=0.7,
        preserve_gas_stellar_offset=preserve_offset,
    )
    velocity, _ = gen.generate_velocity_map(config, snr=None)
    
    vmax = np.nanmax(np.abs(velocity))
    axes[idx].imshow(velocity, origin='lower', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    title = "Offset Preserved\n(Realistic)" if preserve_offset else "Aligned\n(Synthetic)"
    axes[idx].set_title(title)

plt.suptitle(f'Gas-Stellar L offset: {gen._gas_stellar_L_angle_deg:.1f}°')
plt.tight_layout()
plt.show()
```

## 5. Inclination Sweep

Demonstrate how the galaxy appearance changes with viewing angle:

```python
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
cosi_vals = [1.0, 0.8, 0.6, 0.4, 0.2]  # Face-on to nearly edge-on

for idx, cosi in enumerate(cosi_vals):
    inc_deg = np.degrees(np.arccos(cosi))
    
    pars = {'cosi': cosi, 'theta_int': 0.0, 'x0': 0, 'y0': 0, 'g1': 0, 'g2': 0}
    config = TNGRenderConfig(
        image_pars=image_pars,
        use_native_orientation=False,
        pars=pars,
        target_redshift=0.7,
    )
    
    intensity, _ = gen.generate_intensity_map(config, snr=None)
    velocity, _ = gen.generate_velocity_map(config, snr=None)
    
    # Intensity
    int_log = np.log10(np.clip(intensity, 1e-10, None))
    axes[0, idx].imshow(int_log, origin='lower', cmap='viridis')
    axes[0, idx].set_title(f'inc={inc_deg:.0f}°')
    axes[0, idx].axis('off')
    
    # Velocity
    vmax = 150  # Fixed scale for comparison
    axes[1, idx].imshow(velocity, origin='lower', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    axes[1, idx].axis('off')

axes[0, 0].set_ylabel('Intensity', fontsize=12)
axes[1, 0].set_ylabel('Velocity', fontsize=12)
plt.suptitle('Inclination Sweep (Face-on → Edge-on)')
plt.tight_layout()
plt.show()
```

## 6. Adding Lensing Shear

Apply weak lensing shear to the galaxy:

```python
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
shear_configs = [
    {'g1': 0.0, 'g2': 0.0, 'label': 'No shear'},
    {'g1': 0.1, 'g2': 0.0, 'label': 'g₁=0.1'},
    {'g1': 0.0, 'g2': 0.1, 'label': 'g₂=0.1'},
]

for idx, shear_cfg in enumerate(shear_configs):
    pars = {
        'cosi': 0.7, 'theta_int': 0.0, 'x0': 0, 'y0': 0,
        'g1': shear_cfg['g1'], 'g2': shear_cfg['g2']
    }
    config = TNGRenderConfig(
        image_pars=image_pars,
        use_native_orientation=False,
        pars=pars,
        target_redshift=0.7,
    )
    
    intensity, _ = gen.generate_intensity_map(config, snr=None)
    int_log = np.log10(np.clip(intensity, 1e-10, None))
    
    axes[idx].imshow(int_log, origin='lower', cmap='viridis')
    axes[idx].set_title(shear_cfg['label'])
    axes[idx].axis('off')

plt.suptitle('Effect of Weak Lensing Shear')
plt.tight_layout()
plt.show()
```

## 7. Redshift Scaling

TNG galaxies are at z~0.01 (~50 Mpc), spanning ~20 arcminutes on sky. Use `target_redshift` to scale to Roman-like observations:

```python
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
redshifts = [None, 0.5, 1.0]  # None = native z~0.01

for idx, z in enumerate(redshifts):
    # Adjust image size based on expected angular size
    if z is None:
        # Native: galaxy is ~1200 arcsec, use large pixels
        img_pars = ImagePars(shape=(64, 64), pixel_scale=20.0, indexing='ij')
        z_label = "z~0.01 (native)"
    else:
        # Scaled: galaxy fits in ~10 arcsec
        img_pars = ImagePars(shape=(64, 64), pixel_scale=0.15, indexing='ij')
        z_label = f"z={z}"
    
    config = TNGRenderConfig(
        image_pars=img_pars,
        use_native_orientation=True,
        target_redshift=z,
    )
    
    intensity, _ = gen.generate_intensity_map(config, snr=None)
    int_log = np.log10(np.clip(intensity, 1e-10, None))
    
    axes[idx].imshow(int_log, origin='lower', cmap='viridis')
    axes[idx].set_title(z_label)
    axes[idx].axis('off')

plt.suptitle('Redshift Scaling')
plt.tight_layout()
plt.show()
```

## 8. Noise and SNR

Control the signal-to-noise ratio of generated maps:

```python
fig, axes = plt.subplots(2, 4, figsize=(14, 7))
snr_vals = [None, 100, 50, 20]

config = TNGRenderConfig(
    image_pars=image_pars,
    use_native_orientation=True,
    target_redshift=0.7,
)

for idx, snr in enumerate(snr_vals):
    intensity, _ = gen.generate_intensity_map(config, snr=snr, seed=42)
    velocity, _ = gen.generate_velocity_map(config, snr=snr, seed=42)
    
    # Intensity
    int_log = np.log10(np.clip(intensity, 1e-10, None))
    axes[0, idx].imshow(int_log, origin='lower', cmap='viridis')
    snr_label = "No noise" if snr is None else f"SNR={snr}"
    axes[0, idx].set_title(snr_label)
    axes[0, idx].axis('off')
    
    # Velocity
    vmax = 150
    axes[1, idx].imshow(velocity, origin='lower', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    axes[1, idx].axis('off')

axes[0, 0].set_ylabel('Intensity', fontsize=12)
axes[1, 0].set_ylabel('Velocity', fontsize=12)
plt.suptitle('Effect of SNR on Maps')
plt.tight_layout()
plt.show()
```

## 9. Photometric Bands

Render intensity in different photometric bands:

```python
fig, axes = plt.subplots(1, 5, figsize=(15, 3))
bands = ['u', 'g', 'r', 'i', 'z']

for idx, band in enumerate(bands):
    config = TNGRenderConfig(
        image_pars=image_pars,
        band=band,
        use_native_orientation=True,
        target_redshift=0.7,
    )
    
    intensity, _ = gen.generate_intensity_map(config, snr=None)
    int_log = np.log10(np.clip(intensity, 1e-10, None))
    
    axes[idx].imshow(int_log, origin='lower', cmap='viridis')
    axes[idx].set_title(f'{band}-band')
    axes[idx].axis('off')

plt.suptitle('Photometric Bands (Dust-attenuated)')
plt.tight_layout()
plt.show()
```

## Summary

The `kl_pipe.tng` module provides:

| Class/Function | Purpose |
|---------------|---------|
| `TNG50MockData` | Load all TNG50 galaxy data |
| `TNGDataVectorGenerator` | Transform particles to 2D maps |
| `TNGRenderConfig` | Configure rendering parameters |

Key parameters in `TNGRenderConfig`:
- `use_native_orientation`: Use TNG's intrinsic orientation (True) or custom (False)
- `pars`: Custom orientation dict with `cosi`, `theta_int`, `g1`, `g2`, `x0`, `y0`
- `target_redshift`: Scale angular size to this redshift (default: native z~0.01)
- `preserve_gas_stellar_offset`: Keep physical gas-stellar misalignment (default: True)
- `band`: Photometric band ('u', 'g', 'r', 'i', 'z')
- `use_cic_gridding`: Cloud-in-Cell smoothing (default: True)

## Next Steps

1. See `kl_pipe/tng/README.md` for detailed documentation
2. Use TNG maps with the kinematic lensing likelihood in `kl_pipe.likelihood`
3. Run diagnostic tests: `pytest tests/test_tng_data_vectors.py -v`
4. View diagnostic plots in `tests/out/tng_diagnostics/`
