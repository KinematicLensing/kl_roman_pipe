# KL Roman Pipeline

General-purpose kinematic lensing analysis toolkit, designed to serve as the foundation for Roman Space Telescope weak lensing measurements of rotating galaxies.

This library provides modular tools for modeling galaxy velocity fields and surface brightness profiles, with JAX-based implementations optimized for gradient-based parameter inference.

**Current development branch:** `se/basic_models`

## Quick Start

```bash
# 0. Prerequisites (one-time setup)
conda install -n base conda-lock  # If not already installed

# 1. Install (requires conda)
make install

# 2. Run basic tests
make test

# 3. Download TNG50 mock data (optional, for advanced tests)
make download-cyverse-data

# 4. Run TNG50 tests
make test-tng50
```

## Repository Structure

```
kl_pipe/              # Main pipeline package
├── model.py          # Model base classes (Model, VelocityModel, IntensityModel)
├── velocity.py       # Velocity field models (e.g., ArctanVelocityModel)
├── intensity.py      # Surface brightness models (e.g., InclinedExponentialModel)
├── likelihood.py     # Likelihood construction and optimization
├── transformation.py # Multi-plane coordinate transformations
├── synthetic.py      # Synthetic data generation
├── parameters.py     # Parameter and coordinate handling
└── tng/              # TNG50 mock data utilities
    └── loaders.py    # Load gas, stellar, and subhalo data

tests/                # Unit tests (pytest)
docs/
├── tutorials/        # Interactive Jupyter tutorials
│   ├── quickstart.md
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
- `make test` - Run all basic tests
- `make test-tng50` - Run tests requiring TNG50 data (downloads if needed)
- `make test-all` - Run everything (basic + TNG50)
- `make test-fast` - Stop on first failure
- `make test-coverage` - Generate coverage report

### Data Management
- `make download-cyverse-data` - Download TNG50 mock data from CyVerse
- `make clean-cyverse-data` - Remove downloaded data files

### Documentation
- `make tutorials` - Convert markdown tutorials to Jupyter notebooks

### Code Quality
- `make format` - Auto-format code with Black
- `make check-format` - Verify formatting without changes

## Working with TNG50 Data

The pipeline includes utilities for working with TNG50 mock observations:

```python
from kl_pipe.tng import TNG50MockData

# Load all mock data
mock_data = TNG50MockData()
gas = mock_data.gas
stellar = mock_data.stellar
subhalo = mock_data.subhalo
```

**First-time setup:** Run `make download-cyverse-data` and follow the prompts to set up CyVerse authentication (stored securely in `~/.netrc`).

See [`docs/tutorials/tng50_data.md`](docs/tutorials/tng50_data.md) for details.

## Tutorials

Interactive tutorials are available in [`docs/tutorials/`](docs/tutorials/):
- **quickstart.md** - Pipeline basics: models, likelihoods, optimization
- **tng50_data.md** - Working with TNG50 mock observations

Convert to Jupyter notebooks:
```bash
make tutorials
```

Then open the `.ipynb` files in Jupyter Lab or VS Code.

## Key Features

- **JAX-based:** Automatic differentiation and JIT compilation for fast gradient-based optimization
- **Multi-plane coordinate system:** Proper handling of lensing transformations (5 reference frames)
- **Modular models:** Easy to extend with new velocity and intensity models
- **Pure functions:** Stateless models for reproducibility
- **Synthetic data generation:** Built-in tools for testing and validation
- **TNG50 integration:** Work with realistic mock observations

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

*Citation information will be added when available.*
