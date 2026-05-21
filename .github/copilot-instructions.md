# KL Roman Pipeline - AI Coding Agent Instructions

## Project Overview

`kl_pipe` is a **JAX-based kinematic lensing pipeline** for analyzing rotating galaxy observations from Roman Space Telescope. It models velocity fields and intensity maps through sophisticated coordinate transformations, enabling gradient-based parameter inference.

**Branch naming:** When creating branches for PRs, use `cc/<feature-name>` format.

## Core Architecture

### Multi-Plane Coordinate System (Critical)
Models transform coordinates through **5 reference frames** (`kl_pipe/transformation.py`):
- `obs` → `cen` (remove centroid offset)
- `cen` → `source` (remove lensing shear)
- `source` → `gal` (rotate by position angle)
- `gal` → `disk` (deproject inclination)

All model evaluation happens in the `disk` plane (face-on), then values project back through coordinate transformations. **Never evaluate models directly in `obs` coordinates** - use the `Model.__call__(theta, plane, X, Y)` interface which handles transformations automatically.

### Model Structure
Three base classes in `kl_pipe/model.py`:
- **`Model`**: Abstract base requiring `PARAMETER_NAMES` class attribute and `__call__` implementation
- **`VelocityModel`**: Projects 3D circular velocity to line-of-sight using `v_los = sqrt(1-cos²i) * cos(φ) * v_circ`
- **`IntensityModel`**: 3D sech² vertical profile with exponential radial disk; k-space FFT rendering with analytic Fourier transform; LOS Gauss-Legendre quadrature for `__call__`
- **`KLModel`**: Combines velocity + intensity with shared geometric parameters (e.g., `cosi`, `theta_int`, `g1`, `g2`)

### PSF Infrastructure
PSF convolution is handled in `kl_pipe/psf.py`:
- **`PSFData`**: JAX pytree dataclass holding precomputed PSF FFT, padded shape, oversample factor
- **`configure_psf()`**: Accepts `ImagePars` to set up PSF convolution with N-times oversampling (default N=5)
- **Oversampled rendering**: Source evaluated on N× finer grid, convolved at fine resolution, then binned to native pixels to suppress aliasing
- **`convolve_fft()`**: FFT multiply, bins fine→coarse; `convolve_flux_weighted()` for velocity maps

### Sampling Module
`kl_pipe/sampling/` provides a unified interface to multiple MCMC backends:
- **`InferenceTask`**: Bundles model + likelihood + priors + data
- **`PriorDict`**: Separates sampled (Prior objects) from fixed (numeric) parameters
- **`build_sampler(name, task, config)`**: Factory for emcee, nautilus, numpyro, blackjax
- **Recommended**: NumPyro NUTS with Z-score reparameterization for production runs

### Parameter Management
- **`theta`**: JAX array of model parameters in fixed order (immutable, pure functions)
- **`pars`**: Python dict for human-readable parameter names
- Use `model.pars2theta(pars)` and `model.theta2pars(theta)` for conversions
- `PARAMETER_NAMES` tuple defines the canonical ordering (class attribute)

## JAX-Specific Patterns (Enforce Strictly)

1. **Always use `jax.numpy` instead of `numpy`** in model/likelihood code:
   ```python
   import jax.numpy as jnp  # YES
   import numpy as np       # Only for test/synthetic data generation
   ```

2. **JIT compilation via partial application** (`kl_pipe/likelihood.py`):
   ```python
   from functools import partial
   log_like = jax.jit(partial(_log_likelihood_velocity_only,
                               X_vel=X, Y_vel=Y, variance_vel=var,
                               vel_model=model))
   # Now log_like(theta) is fast and differentiable
   ```

3. **Never use conditionals on traced values** - JAX requires static control flow. Use `jnp.where()` for value selection.

4. **Models are stateless** - all parameters passed as `theta`, never stored as instance attributes.

## Development Workflow

### Environment Setup
```bash
# Installation (uses conda-lock for reproducibility)
make install          # Creates 'klpipe' conda env + pip installs package
# Runs: conda-lock install -> pip install -e .

# Testing
make test             # Run pytest suite
make test-fast        # Stop on first failure (-x flag)
make test-coverage    # Generate coverage report

# Formatting
make format           # Auto-format with Black
make check-format     # Verify formatting without changes
```

**Installation details:**
- Uses `conda-lock.yml` for exact reproducibility across platforms
- Package installed in editable mode (`pip install -e .`) for development
- Environment name: `klpipe` (defined in `environment.yaml`)

### Running Tests
- Tests live in `tests/` with `test_*.py` naming
- Use pytest fixtures heavily (see `test_config` pattern in test files)
- **Key test files:**
  - `test_likelihood_slices.py`: Validate likelihoods peak at true parameters via brute-force slicing
  - `test_optimizer_recovery.py`: Fast gradient-based parameter recovery tests
  - `test_velocity.py`, `test_intensity.py`: Unit tests for model components
  - `test_psf.py`, `test_psf_tng.py`: PSF convolution and TNG+PSF tests
  - `test_numpyro.py`, `test_blackjax.py`, `test_sampling_diagnostics.py`: MCMC tests

### Typical Test Pattern
```python
@pytest.fixture(scope="module")
def test_config():
    out_dir = get_test_dir() / "out" / "my_test"
    config = TestConfig(out_dir, include_poisson_noise=False)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    return config

def test_something(test_config):
    # Generate synthetic data
    synth = SyntheticVelocity(true_params, model_type='arctan', seed=42)
    data = synth.generate(test_config.image_pars_velocity, snr=50)
    # ... test logic
```

## Key Files & Patterns

### Synthetic Data Generation (`kl_pipe/synthetic.py`)
- Use `SyntheticVelocity`, `SyntheticIntensity`, `SyntheticKLObservation` classes
- **Always set `seed`** for reproducibility in tests
- Backend functions (e.g., `generate_arctan_velocity_2d`) use `numpy` (not JAX) for data generation

### Likelihood Construction (`kl_pipe/likelihood.py`)
Prefer helper functions over manual construction:
```python
# Velocity-only
log_like = create_jitted_likelihood_velocity(
    vel_model, image_pars, variance, data_vel
)

# Joint velocity + intensity
log_like = create_jitted_likelihood_joint(
    kl_model, image_pars_vel, image_pars_int,
    variance_vel, variance_int, data_vel, data_int
)
```
These return `log_like(theta)` functions ready for scipy.optimize or JAX gradients.

### ImagePars Convention (`kl_pipe/parameters.py`)
```python
from kl_pipe.parameters import ImagePars

# Matrix indexing (default for numpy arrays)
image_pars = ImagePars(shape=(64, 32), pixel_scale=0.1, indexing='ij')
# shape is (Ny, Nx), but image_pars.Nx=32, image_pars.Ny=64

# Cartesian indexing (if preferred)
image_pars = ImagePars(shape=(32, 64), pixel_scale=0.1, indexing='xy')
# shape is (Nx, Ny)
```
**Always check `indexing` attribute** when working with shapes.

## Common Gotchas

1. **Coordinate grid units**: Use `build_map_grid_from_image_pars(image_pars, unit='arcsec', centered=True)` for model evaluation. Default is pixels.

2. **Velocity vs Speed**: `VelocityModel.__call__()` has `return_speed=False` kwarg:
   - `False` (default): Returns line-of-sight velocity (includes projection + v0)
   - `True`: Returns circular speed (scalar, no LOS projection)

3. **Flux conservation in intensity models**: `InclinedExponentialModel` uses `flux` parameter (integrated flux), not central surface brightness. Projection applies `I_obs = I_disk / cos(i)`.

4. **Shared parameters in `KLModel`**: Pass `shared_pars={'cosi', 'theta_int', 'g1', 'g2'}` to avoid duplicating geometric parameters in joint models.

5. **Test output directory**: Tests write to `tests/out/<test_name>/`. This directory is gitignored.

6. **ImagePars API**: Recent migration (Nov 2024) changed `synthetic.generate()` signature:
   - **Old:** `generate(X, Y, snr, ...)`
   - **New:** `generate(image_pars, snr, ...)`
   - Coordinate grids now accessible as `synth.X`, `synth.Y` after generation

7. **JAX meshgrid indexing**: Always specify `indexing='ij'` to match numpy convention. JAX defaults to 'xy'.

8. **JIT compilation timing**: First call is slow (compilation), subsequent calls are microseconds. Don't JIT inside loops.

## Documentation

- **Primary tutorial**: `docs/tutorials/quickstart.md` (Jupytext-compatible markdown with code cells)
- **Sampling tutorial**: `docs/tutorials/sampling.md` (MCMC inference with all backends)
- **TNG tutorial**: `docs/tutorials/tng50_data.md` (TNG50 mock data pipeline)
- **Convert to notebook**: `jupytext --to ipynb docs/tutorials/quickstart.md`
- Examples cover: synthetic data, likelihood construction, optimization, joint modeling, MCMC sampling

## Dependencies of Note

- **JAX**: Core computational backend (auto-diff, JIT)
- **GalSim**: Galaxy image simulation (used in `synthetic.py` for Sersic profiles, alternative backend)
- **NumPyro**: Recommended MCMC sampler (NUTS with Z-score reparameterization)
- **pytest-xdist**: Parallel test execution
- **scipy**: Optimization and special functions

## Conventions & Style

### Parameter Naming
- **Velocity parameters:** Prefix with `vel_` (e.g., `vel_rscale`, `vel_x0`)
- **Intensity parameters:** Prefix with `int_` (e.g., `int_rscale`, `int_x0`)
- **Shared geometric parameters:** No prefix (e.g., `cosi`, `theta_int`, `g1`, `g2`)

### Coordinate System
- **Units:** Coordinates in arcsec, velocities in km/s
- **Position angle:** Radians, measured from +x axis (Cartesian convention)
- **Inclination:** Use `cosi = cos(i)` where i=0° is face-on, i=90° is edge-on
- **Indexing:** Default to 'ij' (numpy convention) for ImagePars

## When Adding New Models

1. Subclass `VelocityModel` or `IntensityModel` in respective files
2. Define `PARAMETER_NAMES` class attribute (tuple of strings)
3. Implement required abstract methods (`evaluate_circular_velocity` or `evaluate_in_disk_plane`)
4. Add property `name` returning a unique string identifier
5. Write unit tests in `tests/test_<model_type>.py`
6. Add integration test in `test_likelihood_slices.py` to verify parameter recovery

## TNG50 Mock Data Module (`kl_pipe/tng/`)

The TNG module provides realistic galaxy mock data from the IllustrisTNG simulation for testing kinematic lensing methods.

### Key Concepts

**5 Available Galaxies**: SubhaloIDs 8, 17, 19, 20, 29 (download via `make download-cyverse-data`)

**Coordinate System Pipeline**: Same 5-plane system as main pipeline, but starting from TNG simulation frame:
1. **TNG native frame** → 3D rotation to **disk plane** (face-on)
2. **disk plane** → Apply user's orientation → **obs plane** (observed)

**Critical: 3D Transformations Not 2D Projections!**
- Uses proper 3D rotation matrices (Rodrigues formula) to transform particle distributions
- Preserves realistic galaxy thickness and velocity dispersion at all viewing angles
- Essential for accurate edge-on views where disk scale height is observable
- Implementation: `_undo_native_orientation()` (line 511-550) and `_apply_new_orientation()` (line 555-683)

### Gas-Stellar Angular Momentum Offset

Gas and stellar components have different angular momentum directions (typically misaligned by 30-40°). This is real physics from different formation histories.

**`preserve_gas_stellar_offset` parameter** (default=True):
- **True** (physically realistic): Gas uses **stellar rotation matrix**, preserving intrinsic ~30-40° offset
  - User's `(cosi, theta_int)` defines **stellar disk** orientation
  - Gas disk appears tilted by its natural offset relative to stellar
  - **Use this for realistic TNG simulations**
- **False** (synthetic test mode): Gas uses **its own rotation matrix**, forcing perfect alignment
  - Both components forced to exact same orientation
  - **Use this only for controlled tests**

**Diagnostic attributes** (accessible after initialization):
```python
gen = TNGDataVectorGenerator(galaxy)
gen._gas_stellar_L_angle_deg       # Angle between L vectors (30-40° typical)
gen._kinematic_inc_stellar_deg     # Inc from stellar angular momentum
gen._kinematic_inc_gas_deg         # Inc from gas angular momentum
gen._L_stellar, gen._L_gas         # Unit vectors in TNG frame
gen._R_to_disk_stellar, gen._R_to_disk_gas  # 3D rotation matrices
```

### Core Classes

**`TNG50MockData`**: Loader for all 5 galaxies
```python
from kl_pipe.tng import TNG50MockData
tng_data = TNG50MockData()
galaxy = tng_data.get_galaxy(subhalo_id=8)  # or tng_data[0]
# Returns dict with 'stellar', 'gas', 'subhalo' keys
```

**`TNGRenderConfig`**: Configuration dataclass
```python
from kl_pipe.tng import TNGRenderConfig
config = TNGRenderConfig(
    image_pars=ImagePars(...),
    band='r',                           # Photometric band (g,r,i,u,z)
    use_native_orientation=True,        # Use TNG's intrinsic inc/PA
    pars=None,                          # Or dict with cosi, theta_int, g1, g2, x0, y0
    target_redshift=0.7,                # Scale to Roman-like distance (TNG native z~0.01)
    preserve_gas_stellar_offset=True,   # Keep physical gas-stellar misalignment
    use_cic_gridding=True,              # Cloud-in-Cell (vs NGP)
)
```

**`TNGDataVectorGenerator`**: Main rendering class
```python
from kl_pipe.tng import TNGDataVectorGenerator
gen = TNGDataVectorGenerator(galaxy)

# Generate maps
intensity, variance = gen.generate_intensity_map(config, snr=50, seed=42)  # Stellar particles
velocity, variance = gen.generate_velocity_map(config, snr=50, seed=42)    # Gas particles
sfr_map = gen.generate_sfr_map(config, snr=None)                           # Gas SFR data
```

### Transformation Implementation Details

**Undoing TNG native orientation** (`_undo_native_orientation()`):
1. Compute angular momentum vectors from particle data:
   - Stellar: Luminosity-weighted L = Σ w_i * (r_i × v_i)
   - Gas: Mass-weighted L = Σ m_i * (r_i × v_i)
2. Build Rodrigues rotation matrix: R = I + sin(θ)K + (1-cos(θ))K² to align L with +Z
3. Apply 3D rotation to coordinates and velocities: coords_disk = (R @ coords.T).T

**Applying new orientation** (`_apply_new_orientation()`):
1. **3D inclination rotation** (line 608-631):
   ```python
   # Rotation matrix around x-axis by angle = arccos(cosi)
   R_incl = [[1,    0,         0     ],
             [0, cos(θ), -sin(θ)],
             [0, sin(θ),  cos(θ)]]
   coords_inclined = (R_incl @ coords_disk.T).T  # Full 3D rotation!
   ```
2. **Project to sky plane** and apply PA rotation (2D rotation in observer's plane)
3. **Apply weak lensing shear** (g1, g2 transformation)
4. **Add position offsets** (x0, y0)

**LOS velocity**: After 3D rotation, `vel_los = velocities_inclined[:, 2]`
- Equivalent to formula `v_LOS = v_y*sin(i) + v_z*cos(i)` but computed via rotation matrix
- Naturally includes both in-plane rotation (v_y) and vertical dispersion (v_z)

### Testing Patterns

**TNG tests** (`tests/test_tng_*.py`):
- Mark with `@pytest.mark.tng50` (requires data download)
- Use `@pytest.mark.tng_diagnostics` for plot-heavy diagnostic tests
- Run diagnostics: `pytest tests/test_tng_data_vectors.py::TestDiagnosticPlots -v -m tng_diagnostics`

**Diagnostic outputs** (`tests/out/tng_diagnostics/`):
- 11 plot types validating transformations, gridding, gas-stellar offset, etc.
- 2 CSV file types: `vertical_extent_*.csv` (disk thickness vs inclination), `inclination_sweep_summary_*.csv` (observables vs orientation)

**Key validations**:
1. 3D structure preserved: Vertical extent analysis shows realistic disk thickness at all inclinations
2. Gas-stellar offset: Inclination sweeps with/without offset preservation show ~30-40° difference
3. Transform correctness: Face-on and edge-on views match expectations
4. Flux conservation: CIC and NGP produce similar total flux (within 20%)

### Common TNG Gotchas

1. **Redshift scaling required**: TNG galaxies at native z~0.01 span ~1300" (~21 arcmin). Always use `target_redshift=0.5-1.0` for Roman-like sub-arcsec observations.

2. **Sparse gas pixels**: Velocity maps may have zeros where no gas particles fall. Not NaN - cannot distinguish "no data" from "zero velocity".

3. **Separate rotation matrices**: Stellar and gas use different R_to_disk matrices unless `preserve_gas_stellar_offset=False`. This is intentional and physically correct!

4. **Inclination >90° handling**: TNG convention uses 0-180°. Code automatically flips >90° to equivalent <90° view (inc'=180-inc, PA'=PA+180) to avoid negative cos(i).

5. **Transform pipeline vs transformation.py**: TNG implements the same math as `transformation.py` but doesn't call those functions directly - it operates on particles, not coordinate grids.

6. **Kinematic vs morphological inclination**: TNG catalog inclination (from moment of inertia) can differ from kinematic inclination (from angular momentum) by 5-15°. Both are valid; they measure different things.

### When Working with TNG Data

1. **Always check gas-stellar offset**: Use `gen._gas_stellar_L_angle_deg` to understand if galaxy has significant misalignment
2. **Use diagnostic attributes**: Access `_kinematic_inc_*_deg` to compare with catalog values
3. **Visualize 3D structure**: Run `experiments/sweverett/tng/offset_exploration.ipynb` to see particle distributions and rotation matrices
4. **Default to preserved offset**: Unless testing synthetic scenarios, keep `preserve_gas_stellar_offset=True`
5. **Document orientation choice**: When reporting results, state whether gas-stellar offset was preserved or forced to align

### TNG Documentation

- **Module README**: `kl_pipe/tng/README.md` - comprehensive API and physics documentation
- **Tutorial**: `docs/tutorials/tng50_data.md` - hands-on examples with plots
- **Test README**: `tests/README.md` - diagnostic test documentation and CSV format descriptions
- **Exploration notebook**: `experiments/sweverett/tng/offset_exploration.ipynb` - validate angular momentum computation
