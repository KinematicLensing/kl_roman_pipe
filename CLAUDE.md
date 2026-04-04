# KL Roman Pipeline — AI Agent Instructions

## Science Requirements & Goals

JAX-based kinematic lensing pipeline for Roman Space Telescope weak lensing of rotating galaxies. Joint velocity-intensity fitting breaks the intrinsic-shape/lensing degeneracy, achieving ~10x lower shape noise than photometric WL. Models velocity fields and intensity maps through 5-plane coordinate transformations for gradient-based Bayesian inference. Designed for millions of galaxy fits via fast samplers.

## Scientific Principles

- **Hypotheses before implementation**: know what you expect before writing code
- **Test every feature**; never weaken tests to pass
- **Document negative results** — a well-documented dead end saves future effort
- **Reproducibility is non-negotiable** — every result traceable to code version, environment, data state, and configuration
- Before adding a new model or profile, write a GalSim/analytic comparison test defining expected accuracy before writing the implementation

## Architecture

### Module Map

```
kl_pipe/
├── model.py           # Model ABCs: Model, VelocityModel, IntensityModel, KLModel
├── velocity.py        # CenteredVelocityModel, OffsetVelocityModel, factories
├── intensity.py       # Inclined{Exponential,Spergel,DeVaucouleurs,Sersic}Model, factories
├── likelihood.py      # JAX log-likelihoods, JIT helper constructors
├── transformation.py  # 5-plane coordinate transforms (obs→cen→source→gal→disk)
├── parameters.py      # Pars, SampledPars, MetaPars, MCMCPars, ImagePars
├── priors.py          # Prior ABC, Uniform, Gaussian, LogUniform, TruncatedNormal, PriorDict
├── observation.py     # Observation types: ImageObs, VelocityObs, GrismObs + factory functions
├── psf.py             # PSF convolution: FFT pipeline, PSFData, oversampled rendering
├── spectral.py        # Datacube assembly (x,y,λ): CubePars, SpectralModel, emission lines
├── dispersion.py      # Grism dispersion: GrismPars, 3D→2D spectral projection
├── synthetic.py       # Independent synthetic data generators (NOT using model.py)
├── noise.py           # SNR-based noise: add_intensity_noise, add_velocity_noise
├── utils.py           # Grid builders, path getters
├── plotting.py        # Velocity/intensity map visualization
├── diagnostics/       # Diagnostic plotting subpackage
│   ├── imaging.py     # Parameter recovery plots, joint Nsigma, data comparison panels
│   ├── datacube.py    # Datacube diagnostic plots
│   └── grism.py       # Grism diagnostic plots
├── sampling/          # MCMC infrastructure (see sampling/README.md)
│   ├── base.py        # Sampler ABC, SamplerResult dataclass
│   ├── configs.py     # Config dataclasses per sampler type, YAML loader
│   ├── task.py        # InferenceTask: model+likelihood+priors+data bundle
│   ├── factory.py     # build_sampler() registry pattern
│   ├── emcee.py       # Ensemble MCMC (gradient-free)
│   ├── nautilus.py    # Neural nested sampling (provides evidence)
│   ├── blackjax.py    # JAX-native HMC/NUTS (known issues w/ joint models)
│   ├── numpyro.py     # NUTS w/ Z-score reparam — RECOMMENDED for production
│   ├── ultranest.py   # Placeholder — NOT IMPLEMENTED
│   └── diagnostics.py # Trace, corner, recovery, convergence plots
└── tng/               # TNG50 mock data (see tng/README.md)
    ├── loaders.py     # Load gas/stellar/subhalo data from CyVerse
    └── data_vectors.py # 3D particle→2D map rendering, Rodrigues rotations
```

### 5-Plane Coordinate System (Critical)

All model evaluation happens in the `disk` plane (face-on). Transformations in `transformation.py`:

```
obs → cen    (remove centroid x0, y0)
cen → source (remove lensing shear g1, g2)
source → gal (rotate by position angle theta_int)
gal → disk   (deproject by inclination cosi)
```

**Never evaluate models directly in obs coordinates.** Use `Model.__call__(theta, plane, X, Y)` which handles transforms automatically.

### Model Hierarchy

```
Model (ABC)
├── VelocityModel → evaluate_circular_velocity() → LOS projection
│   ├── CenteredVelocityModel
│   └── OffsetVelocityModel
├── IntensityModel → evaluate_in_disk_plane() + render_image() (k-space FFT)
│   ├── InclinedExponentialModel      (n=1 analytic, default)
│   ├── InclinedSpergelModel          (Spergel profile, adds nu param)
│   ├── InclinedDeVaucouleursModel    (fixed nu=-0.6, Sersic n=4)
│   ├── InclinedSersicModel           (Miller & Pasha emulator, adds n_sersic + int_hlr)
│   └── CompositeIntensityModel       (stub — future disk+bulge)
└── KLModel → combines velocity + intensity with shared geometric params
```

All models are **stateless pure functions**: parameters passed as `theta` array, never stored as instance attributes. `PARAMETER_NAMES` class tuple defines canonical ordering; enforced at subclass creation via `__init_subclass__`. Models carry no PSF or instrument state — that lives in observation objects (`ImageObs`, `VelocityObs`, `GrismObs`) from `observation.py`.

### Factory Pattern

Three registries with case-insensitive `build_*()` functions:
- `build_velocity_model(name)` — 'centered', 'offset', 'default'
- `build_intensity_model(name)` — 'default', 'inclined_exp', 'inclined_spergel', 'spergel', 'de_vaucouleurs', 'inclined_sersic', 'sersic'
- `build_sampler(name, task, config)` — 'emcee', 'nautilus', 'blackjax', 'numpyro', 'ultranest'; aliases 'nuts' and 'hmc' → numpyro

All raise `ValueError` on unknown names.

---

## Code Style & Formatting

- **Black**: `--skip-string-normalization`, default line-length 88
- **Docstrings**: NumPy style, capitalized sentences
- **Inline comments**: lowercase single-sentence (e.g., `# merge catalogs from tiles`)
- **Type hints** whenever possible; use `TYPE_CHECKING` for circular imports
- **Import order**: stdlib → numpy/jax/scipy → local (`from kl_pipe.X import Y`)

### Naming Conventions

| Pattern | Meaning | Examples |
|---------|---------|---------|
| `theta` | JAX array of params in `PARAMETER_NAMES` order | `theta`, `theta_vel` |
| `pars` | Dict of named parameters | `true_pars`, `meta_pars` |
| `PARAMETER_NAMES` | Class tuple defining canonical ordering | — |
| `vel_*` | Velocity model parameter | `vel_rscale`, `vel_x0` |
| `int_*` | Intensity model parameter | `int_rscale`, `int_x0` |
| No prefix | Shared geometric parameter | `cosi`, `theta_int`, `g1`, `g2` |
| `X, Y` | 2D coordinate grids | From `build_map_grid_from_image_pars()` |

### Physical Units

| Quantity | Unit | Convention |
|----------|------|-----------|
| Coordinates | arcsec | From `ImagePars.pixel_scale` |
| Velocities | km/s | LOS or circular |
| Position angle | radians | From +x (Cartesian), [0, 2pi) |
| Inclination | `cosi = cos(i)` | 0=edge-on, 1=face-on |
| Shear | dimensionless | `g1`, `g2`; \|g\| < 1 |
| Flux | integrated (not surface brightness) | `I0 = flux / (2*pi*r_scale^2)` |

**Always perform dimensional sanity checks** on numerical quantities before finalizing code.

---

## JAX Patterns (Enforce Strictly)

1. **`jax.numpy` in all model/likelihood code:**
   ```python
   import jax.numpy as jnp  # YES — model/likelihood code
   import numpy as np       # ONLY for test/synthetic data generation
   ```

2. **JIT via partial application** (freeze static args, trace only theta):
   ```python
   log_like = jax.jit(partial(_log_likelihood_velocity_only,
                               X_vel=X, Y_vel=Y, variance_vel=var,
                               vel_model=model))
   ```

3. **No Python conditionals on traced values** — use `jnp.where()`:
   ```python
   # WRONG: if plane == 'disk': return I_disk
   return jnp.where(condition, I_disk, I_disk / cosi)
   ```

4. **Array mutation** via `.at[].set()` (not in-place assignment)

5. **Meshgrid indexing**: always `indexing='ij'` in JAX (JAX defaults to 'xy')

6. **PRNG**: functional `jax.random.PRNGKey()` + key splitting, no global state

7. **Prior log_prob must be JIT-compatible**: return `-jnp.inf` for out-of-bounds via `jnp.where()`, never `raise`

---

## Defensive Coding Practices

**No silent fallbacks. Fail fast and explicitly.**

- **Always raise explicit errors** with clear messages for unexpected conditions
- **NEVER return plausible-looking placeholder values** that could silently corrupt scientific results
- **NEVER silently return NaN/None** — these propagate through calculations and hide bugs
- **Validate inputs in `__post_init__`** and constructors (types, bounds, shapes)
- **`NotImplementedError`** for abstract methods and unfinished stubs
- **`KeyError`/`ValueError`** for invalid parameter names or model lookups
- **Exception**: inside JIT-traced code, use `jnp.where(..., value, -jnp.inf)` instead of raising (JAX requirement)

```python
# GOOD: loud error
if self.high <= self.low:
    raise ValueError(f"high ({self.high}) must be > low ({self.low})")

# GOOD: JIT-safe prior boundary
return jnp.where(in_bounds, -log_width, -jnp.inf)

# BAD: silent fallback
return 0.0  # could silently corrupt a likelihood calculation
```

---

## Testing Philosophy & Hierarchy

### Test Tiers (strict → loose)

1. **Likelihood Slicing** (`test_likelihood_slices.py`) — brute-force grid search per parameter. Validates forward model correctness. Strictest tolerances (0.1–5% by SNR). **Failures here = model bug.**

2. **Optimizer Recovery** (`test_optimizer_recovery.py`) — gradient-based `scipy.optimize`. 10–20x looser tolerances. Excludes degenerate params (`cosi`, `g1`, `g2`) from pass/fail; checks observable product `vcirc*sin(i)` instead.

3. **Sampling Diagnostics** (`test_sampling_diagnostics.py`, `test_numpyro.py`, `test_blackjax.py`) — full MCMC with corner plots, traces, convergence checks (R-hat < 1.01, ESS > 400).

4. **TNG Diagnostics** (`test_tng_data_vectors.py`, `test_tng_sampling_diagnostics.py`) — 3D rotation validation, gridding, gas-stellar offset, flux conservation. 11 plot types + CSV quantitative diagnostics.

5. **Grism/Datacube** (`test_datacube.py`, `test_grism_core.py`, `test_cube_psf.py`) — datacube assembly, grism dispersion, spectral PSF convolution.

6. **Grism Validation** (`test_grism_validation.py`) — cross-code comparison against geko reference implementation. Requires `make setup-validation-env` + `make render-validation-*`.

### Pytest Markers

| Marker | Meaning | Run with |
|--------|---------|----------|
| `tng50` | Requires TNG50 data from CyVerse | `pytest -m tng50` |
| `tng_diagnostics` | Slow TNG diagnostic plots | `pytest -m tng_diagnostics` |
| `slow` | Significant runtime | `pytest -m slow` |
| `grism_validation` | Cross-code grism validation (requires reference data) | `make test-grism-validation` |

### Key Test Patterns

- **`TestConfig`** in `tests/test_utils.py` — central tolerance configuration. SNR-dependent relative + absolute tolerances with parameter-specific scaling. **Never hardcode tolerances.**
- **Module-scope fixtures** for expensive setup (grids, synthetic data)
- **SNR parametrization**: `@pytest.mark.parametrize("snr", [1000, 50, 10])`
- **Dual pass criteria**: parameter passes if EITHER relative OR absolute error within tolerance
- **Degenerate products**: test `vcirc*sin(i)` instead of `vcirc` and `cosi` independently
- **Diagnostic plots** saved to `tests/out/<test_name>/` (gitignored)
- **Output redirection**: `redirect_sampler_output()` captures sampler stdout to log files

### Tolerance Rules

- **Never loosen likelihood slice tolerances** — they validate model correctness
- **Optimizer tolerances ~10-20x looser** than likelihood slices
- **Parameter-specific scaling must reflect physics**, not just make tests pass
- **Document and loudly flag** any tolerance changes in PRs

### Adding Tests for New Models

1. Unit tests in `test_velocity.py`, `test_intensity.py`, or a dedicated `test_intensity_<name>.py`
2. Likelihood slice tests in `test_likelihood_slices.py`
3. Optimizer recovery tests in `test_optimizer_recovery.py`
4. Update `TestConfig` with model-specific tolerances if needed
5. Add parameter to `absolute_tolerance_floor` if it can be near zero

---

## Sampling Infrastructure

### Recommended Sampler: NumPyro

Use `numpyro` for production. Handles multi-scale gradients (intensity ~10^4x larger than velocity) via Z-score reparameterization with dense mass matrix adaptation.

| Sampler | Gradients | Evidence | Multi-chain | Status |
|---------|-----------|----------|-------------|--------|
| emcee | No | No | No | Stable — exploration |
| nautilus | No | Yes | No | Stable — model comparison |
| **numpyro** | **Yes** | **No** | **Yes** | **RECOMMENDED** |
| blackjax | Yes | No | No | Known issues w/ joint models |
| ultranest | No | Yes | No | NOT IMPLEMENTED |

### InferenceTask

Bundles model + likelihood + priors + data. Preferred factory methods (obs-based):
- `InferenceTask.from_velocity_obs(model, priors, obs)`
- `InferenceTask.from_intensity_obs(model, priors, obs)`
- `InferenceTask.from_joint_obs(model, priors, obs_vel, obs_int)`

Legacy wrappers (construct obs internally, backward-compatible):
- `InferenceTask.from_velocity_model(model, priors, data_vel, variance_vel, image_pars, ...)`
- `InferenceTask.from_intensity_model(model, priors, data_int, variance_int, image_pars, ...)`
- `InferenceTask.from_joint_model(model, priors, data_vel, data_int, variance_vel, variance_int, image_pars_vel, image_pars_int, ...)`

### PriorDict

Separates sampled (Prior objects) vs fixed (numeric) params. Sampled names sorted alphabetically = canonical theta ordering.

```python
priors = PriorDict({
    'vcirc': Uniform(100, 300),     # sampled
    'cosi': TruncatedNormal(...),   # sampled
    'v0': 10.0,                      # fixed
})
```

---

## TNG50 Module

### Key Concepts

- **5 galaxies**: SubhaloIDs 8, 17, 19, 20, 29 (download via `make download-cyverse-data`)
- **3D rotations** via Rodrigues formula (NOT 2D projections) — preserves disk thickness
- **Gas-stellar offset**: ~30-40 deg misalignment is real physics. `preserve_gas_stellar_offset=True` (default) keeps it; `False` forces alignment (synthetic tests only)
- **Redshift scaling**: TNG native z~0.01 spans ~1300". Always use `target_redshift=0.5-1.0` for Roman-like sub-arcsec observations
- **Empty velocity pixels**: set to 0, not NaN — cannot distinguish no-data from zero-velocity

### Gotchas

1. Separate rotation matrices for stellar vs gas (intentional)
2. Kinematic inc (from L) differs from morphological inc (from inertia tensor) by 5-15 deg
3. Inclination > 90 deg auto-flipped to equivalent < 90 deg view
4. CIC gridding recommended over NGP (smoother, flux-conserving)

---

## Workflow

### Makefile Targets

```bash
make install              # create klpipe conda env
make format               # black formatting
make check-format         # verify without modifying
make tutorials            # convert tutorial md → ipynb
make test-tutorials       # convert + execute all tutorials

# Testing
make test                 # fast tests (excludes slow, tng_diagnostics, grism_validation)
make test-basic           # no TNG data required
make test-extended        # excludes tng_diagnostics + grism_validation only
make test-all             # full suite including TNG diagnostics
make test-fast            # fast tests, stop on first failure (-x)
make test-verbose         # verbose with stdout (-v -s)
make test-coverage        # coverage report (html + terminal)
make test-sampling        # MCMC tests (excl. nautilus)
make test-sampling-all    # all MCMC tests (incl. nautilus)
make test-tng             # TNG50 tests only
make test-tng-unit        # TNG50 unit tests (excl. slow diagnostics)
make test-tng-diagnostics # slow TNG diagnostic plots
make clean-test           # remove tests/out/ and .coverage

# Diagnostics
make diagnostics          # HTML report of test diagnostic images
make show-diagnostics     # open most recent existing report
make diagnostics-pdf      # generate timestamped PDF report
make prune-diagnostics    # keep only latest file per day per extension

# Grism cross-code validation
make setup-validation-env     # clone + install geko validation env
make verify-geko-conventions  # verify geko parameter conventions
make render-validation-kl-pipe  # render 28 test combos via kl_pipe
make render-validation-geko     # render 28 test combos via geko
make test-grism-validation      # compare kl_pipe vs geko outputs
```

### Rules for AI Agents

1. **Do not run `make format` or `make test`** — the user will run these manually. Only suggest which tests to run for validation; do not execute them.
2. **Branch naming**: `se/` prefix for user branches, `cc/` prefix for AI-created branches.
3. **Read before editing**: never propose changes to code you haven't read.
4. **Minimal changes**: don't add docstrings, comments, or type annotations to code you didn't change.
5. **JAX first**: prefer `jax.numpy` over `numpy` in model/likelihood code.
6. **Physical sanity checks**: verify units and dimensional consistency for any physics code.
7. **No AI attribution**: never include `Co-Authored-By`, "Generated with Claude Code", or similar AI-generated footers in commit messages, PR descriptions, or comments. All commits must show the user as sole author.
8. **Plan verification**: when a completed plan modifies test files, run `/test-integrity` before the final commit.

### PR Review Focus Categories

When creating PRs, organize the "Review focus" section into these tiers:

- **API** — public interfaces, model APIs, data structures reviewers should scrutinize
- **Internals** — implementation plumbing; review welcome but lower priority
- **Tests** — test infrastructure, tolerances, fixtures; FYI unless tolerance changes

Optionally flag individual items as **⚠ Critical** for correctness-sensitive code (physics, likelihood, coordinate transforms). Example:

```
## Review focus

**API:**
- `kl_pipe/psf.py` — PSF convolution API & FFT pipeline
- ⚠ `kl_pipe/transformation.py` — axis convention changes (critical)

**Internals:**
- `kl_pipe/sampling/numpyro.py` — NUTS backend

**Tests:**
- `tests/test_utils.py` — TestConfig tolerance system
```

### Environment

- **Conda env**: `klpipe` (from `conda-lock.yml`)
- **Package**: installed editable (`pip install -e .`)
- **Key deps**: JAX, NumPyro, BlackJAX, emcee, nautilus, GalSim, scipy, astropy, corner, ArviZ

---

## Adding New Models

1. Subclass `VelocityModel` or `IntensityModel`
2. Define `PARAMETER_NAMES` class tuple
3. Implement `evaluate_circular_velocity()` or `evaluate_in_disk_plane()`
4. Add `name` property returning unique string
5. Register in factory dict in `velocity.py` or `intensity.py`
6. Write unit tests + likelihood slice tests + optimizer recovery tests
7. Update `TestConfig` with parameter-specific tolerances if needed

## Adding New Samplers

1. Subclass `Sampler` in `kl_pipe/sampling/mysampler.py`
2. Set `requires_gradients`, `provides_evidence`, `config_class`
3. Implement `run() -> SamplerResult`
4. Add config dataclass in `configs.py` (if needed)
5. Register in `factory.py` `_register_builtins()`
6. Export in `sampling/__init__.py`
7. Write tests: config validation, velocity-only recovery, joint recovery, convergence

---

## Writing Style

All docs, markdown, READMEs, commit messages, and code comments:
- Concise, scientific tone
- No LLM-speak ("I'd be happy to...", "Great question!", "Certainly!")
- No emojis

---

## Never Do

- Never reduce test tolerances/scope to make tests pass without explicit human approval
- Never swallow exceptions or hide error details
- Never commit .env, credentials, or secrets
- Never add emojis to code, docs, or commits
- Never skip type hints on function signatures

---

## References

- @CONSTITUTION.md — core scientific values; consult when making scope/rigor tradeoffs
- @PARALLAX.md — scientific workflow rules; consult for hypothesis protocol, experiment manifests, agent handoff format

---

## Verification

After writing: `cat CLAUDE.md` to confirm no formatting issues. All module paths verified against actual codebase structure.
