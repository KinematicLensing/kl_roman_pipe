# KL Roman Pipeline

General-purpose kinematic lensing analysis toolkit, designed to serve as the foundation for Roman Space Telescope weak lensing measurements of rotating galaxies.

This library provides modular tools for modeling galaxy velocity fields and surface brightness profiles, with JAX-based implementations optimized for gradient-based parameter inference.

## Quick Start

```bash
# Prerequisites (one-time setup)
conda install -n base conda-lock  # If not already installed

# Install
make install
```

If a `klpipe` environment already exists, `make install` asks before replacing it (without a terminal, set `KLPIPE_REINSTALL=1`).

**Note for HPC Users:** If you do not have write access to your base environment, install conda-lock into a custom environment (e.g., `mybase`) and run: `BASE_ENV=mybase make install`

**First test run (no download required):**
```bash
make test-basic  # Skips TNG50 and other data-dependent tests
```

**Full fast suite (downloads ~2.6 GB of TNG50 data on first run):**
```bash
make test
```

Then work through the [tutorials](#tutorials).

**Scope:** single-source fitting (locally or on HPC) is the supported path. The ensemble/GPU/HPC campaign tooling (`kl_pipe/ensemble/`, `configs/ensembles/`, `experiments/`) is internal and not yet supported for general use.

## Repository Structure

```
kl_pipe/              # Main pipeline package
├── source.py         # SourceModel: composes velocity + broadband + emission-line components
├── model.py          # Model base classes (Model, VelocityModel, IntensityModel)
├── velocity.py       # Velocity field models (CenteredVelocityModel, OffsetVelocityModel)
├── intensity.py      # Surface brightness models (exponential, Spergel, Sersic, bulge+disk)
├── lines.py          # EmissionLine + LINE_LAMBDAS registry
├── observation.py    # Observation types (ImageObs, VelocityObs, GrismObs) + factories
├── likelihood.py     # JAX log-likelihoods (create_jitted_likelihood_from_obs)
├── coordinates.py    # WCS-derived celestial-to-detector rotation, shear rotation
├── transformation.py # Multi-plane coordinate transformations
├── parameters.py     # ImagePars (image grid, pixel scale, WCS)
├── priors.py         # Prior distributions (Uniform, Gaussian, TruncatedNormal, etc.) + PriorDict
├── psf.py            # PSF convolution (PSFData, oversampled rendering, FFT pipeline)
├── pixel.py          # PixelResponse (BoxPixel sinc k-space pixel integration)
├── render.py         # RenderConfig: k-space grid sizing + render defaults
├── spectral.py       # Datacube grid (CubePars)
├── dispersion.py     # Grism dispersion (GrismPars, 3D->2D projection)
├── grism.py          # Post-dispersion pixel response
├── synthetic.py      # Independent synthetic data generation
├── noise.py          # SNR-based noise utilities
├── photometry.py     # AB mag / flux-density unit conversions
├── optimization.py   # Gradient-based recovery (multi_start_minimize)
├── constants.py      # Physical constants (C_KMS, etc.)
├── utils.py          # Grid builders, path helpers
├── plotting.py       # Velocity/intensity map visualization
├── surveys/          # Published survey parameters (roman.py: HLWAS depths, line limits)
├── diagnostics/      # Diagnostic plotting subpackage (imaging, datacube, grism)
├── sampling/         # MCMC sampling infrastructure
│   ├── base.py       # Sampler ABC, SamplerResult
│   ├── configs.py    # Config dataclasses per sampler type
│   ├── task.py       # InferenceTask.from_obs: source + likelihood + priors + observations
│   ├── factory.py    # build_sampler() registry
│   ├── initialization.py # Optimizer starts, MAP finder, Laplace metric
│   ├── transforms.py # Bounded-to-unconstrained parameter transforms
│   ├── emcee.py      # Ensemble MCMC (gradient-free)
│   ├── nautilus.py   # Neural nested sampling (evidence)
│   ├── blackjax.py   # JAX-native HMC/NUTS
│   ├── numpyro.py    # NUTS, Laplace-preconditioned unconstrained coords (recommended)
│   └── diagnostics.py # Trace, corner, recovery plots
├── ensemble/         # Internal multi-galaxy campaign tooling (not yet supported)
└── tng/              # TNG50 mock data utilities
    ├── loaders.py    # TNG50MockData: gas, stellar, and subhalo data
    ├── data_vectors.py # 3D particle-to-2D map rendering
    └── tng_dust.py   # Dust attenuation kernels

tests/                # Unit tests (pytest)
docs/
├── tutorials/        # Tutorials (markdown sources, converted to notebooks)
├── models.md         # Model, SourceModel, and observation reference
└── README.md         # Index of reference and internal docs
data/
├── cyverse/          # CyVerse data configuration
└── tng50/            # Downloaded TNG50 mock data (gitignored)
```

## Installation

**Prerequisites:** [conda](https://github.com/conda-forge/miniforge) and `conda-lock` in your base environment

```bash
conda install -n base conda-lock  # If not already installed
make install                       # Creates 'klpipe' environment
```

This installs the package in editable mode with all dependencies via `conda-lock.yml`.

## Makefile Targets

### Testing
- `make test-basic` - Fast tests with no data download (start here)
- `make test` - Fast generic-pipeline tests (downloads TNG50 data if needed, ~2.6 GB; excludes slow tests and the Roman ensemble tier)
- `make test-roman-ensemble` - Run the Roman ensemble-campaign tests (ensemble machinery, catalog adapters, prior provenance, Roman PSF, shear calibration)
- `make test-tng` - Run only TNG50-specific tests
- `make test-sampling` - Run MCMC sampling tests (excludes nautilus)
- `make test-fast` - Same marker filter as `make test`, plus `-x` (stop at first failure)
- `make test-coverage` - Generate coverage report
- `make test-tutorials` - Execute all tutorials end-to-end (CI mode)

See [`tests/README.md`](tests/README.md) for markers and all test targets.

### Data Management
- `make download-cyverse-data` - Download TNG50 mock data from CyVerse
- `make clean-cyverse-data` - Remove downloaded data files

### Documentation
- `make tutorials` - Convert markdown tutorials to Jupyter notebooks
- `make test-tutorials` - Convert and execute tutorials (CI smoke test)

### Code Quality
- `make format` - Auto-format code with Black
- `make check-format` - Verify formatting without changes

## Working with TNG50 Data

The pipeline includes utilities for working with TNG50 mock observations (17 galaxies, ~2.6 GB):

```python
from kl_pipe.tng import TNG50MockData

# Load all mock data
mock_data = TNG50MockData()
gas = mock_data.gas
stellar = mock_data.stellar
subhalo = mock_data.subhalo
```

**Data download:** The data downloads automatically when you run `make test` or `make download-cyverse-data`. On first download, you'll be prompted to set up CyVerse authentication (credentials stored securely in `~/.netrc`).

See [`docs/tutorials/tng50_data.md`](docs/tutorials/tng50_data.md) for details.

## Tutorials

Tutorials live in [`docs/tutorials/`](docs/tutorials/). Suggested order:

1. **quickstart.md** - Describe a source, render it, build a likelihood, run an inference
2. **intensity_models.md** - Intensity profiles, bulge + disk composites, and RenderConfig grid sizing
3. **grism.md** - Grism datacube and dispersion forward modeling
4. **sampling.md** - Bayesian inference with MCMC (numpyro recommended)

As needed:
- **roman_reference.md** - Worked template for a full-complexity Roman mock + fit (two bands, two grism rolls, Roman PSF, bulge + disk)
- **tng50_data.md** - Working with TNG50 mock observations

Convert to Jupyter notebooks:
```bash
make tutorials
```

Then open the `.ipynb` files in Jupyter Lab or VS Code.

## Key Features

- **JAX-based:** Automatic differentiation and JIT compilation for fast gradient-based optimization
- **Multi-plane coordinate system:** Proper handling of lensing transformations (5 reference frames)
- **3D intensity model:** Inclined exponential with sech^2 vertical profile (matches GalSim `InclinedExponential`)
- **PSF convolution:** Oversampled FFT pipeline with configurable oversample factor (default N=5)
- **MCMC sampling:** Multiple backends (emcee, nautilus, numpyro, blackjax) with unified interface
- **Modular models:** Easy to extend with new velocity and intensity models
- **Pure functions:** Stateless models for reproducibility
- **Synthetic data generation:** Built-in tools for testing and validation
- **TNG50 integration:** Work with realistic mock observations from IllustrisTNG

## Development

```bash
# Run tests during development
make test-basic             # No downloads
make test-fast              # Same filter as make test, stop at first failure

# Format code before committing
make format

# Check test coverage
make test-coverage
```

See [`CLAUDE.md`](CLAUDE.md) for architecture, coding conventions, and testing rules (written for AI agents, useful for anyone).

## Citation

One day!
