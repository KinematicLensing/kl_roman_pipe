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

**Option 1: Run all tests (recommended, requires ~340 MB download)**
```bash
make test  # Downloads TNG50 data automatically on first run
```

**Option 2: Run only basic tests (no download required)**
```bash
make test-basic  # Skips TNG50 tests
```

## Repository Structure

```
kl_pipe/              # Main pipeline package
├── model.py          # Model base classes (Model, VelocityModel, IntensityModel, KLModel)
├── velocity.py       # Velocity field models (e.g., CenteredVelocityModel)
├── intensity.py      # Surface brightness models (e.g., InclinedExponentialModel)
├── likelihood.py     # Likelihood construction and optimization
├── transformation.py # Multi-plane coordinate transformations
├── parameters.py     # Parameter and coordinate handling (ImagePars, Pars, etc.)
├── priors.py         # Prior distributions (Uniform, Gaussian, TruncatedNormal, etc.)
├── psf.py            # PSF convolution (PSFData, oversampled rendering, FFT pipeline)
├── synthetic.py      # Synthetic data generation
├── noise.py          # SNR-based noise utilities
├── utils.py          # Grid builders, path helpers
├── plotting.py       # Velocity/intensity map visualization
├── diagnostics.py    # Parameter recovery plots, joint Nsigma analysis
├── sampling/         # MCMC sampling infrastructure
│   ├── base.py       # Sampler ABC, SamplerResult
│   ├── configs.py    # Config dataclasses per sampler type
│   ├── task.py       # InferenceTask: model+likelihood+priors+data
│   ├── factory.py    # build_sampler() registry
│   ├── emcee.py      # Ensemble MCMC (gradient-free)
│   ├── nautilus.py   # Neural nested sampling (evidence)
│   ├── blackjax.py   # JAX-native HMC/NUTS
│   ├── numpyro.py    # NUTS w/ Z-score reparam (recommended)
│   └── diagnostics.py # Trace, corner, recovery plots
└── tng/              # TNG50 mock data utilities
    ├── loaders.py    # Load gas, stellar, and subhalo data
    └── data_vectors.py # 3D particle-to-2D map rendering

tests/                # Unit tests (pytest)
docs/
├── tutorials/        # Interactive Jupyter tutorials
│   ├── quickstart.md
│   ├── sampling.md
│   └── tng50_data.md
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
- `make test` - Run all tests (downloads TNG50 data if needed, ~340 MB)
- `make test-basic` - Run only basic tests (no download required)
- `make test-tng` - Run only TNG50-specific tests
- `make test-sampling` - Run MCMC sampling tests (excludes nautilus)
- `make test-fast` - Stop on first failure
- `make test-coverage` - Generate coverage report
- `make test-tutorials` - Execute all tutorials end-to-end (CI mode)

**To run tests without downloading data:**
```bash
conda run -n klpipe pytest tests/ -v -m "not tng50"
# Or use: make test-basic
```

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

The pipeline includes utilities for working with TNG50 mock observations (~340 MB):

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

Interactive tutorials are available in [`docs/tutorials/`](docs/tutorials/):
- **quickstart.md** - Pipeline basics: models, likelihoods, optimization
- **sampling.md** - Bayesian inference with MCMC sampling (emcee, nautilus, numpyro)
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
make test-fast              # Stop on first failure

# Format code before committing
make format

# Check test coverage
make test-coverage
```

See [`.github/copilot-instructions.md`](.github/copilot-instructions.md) for detailed development guidelines and architecture notes.

## Citation

One day!
