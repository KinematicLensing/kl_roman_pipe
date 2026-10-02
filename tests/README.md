# Test Suite Documentation

This directory contains comprehensive tests for the `kl_pipe` kinematic lensing pipeline. The test suite validates both forward models (synthetic data generation) and inference workflows (parameter recovery).

## Test Organization

Files are grouped by what they cover. Module docstrings give details.

### Models and rendering
- `test_velocity.py`, `test_intensity.py`: velocity and exponential intensity models, transforms, flux conservation
- `test_intensity_spergel.py`, `test_intensity_sersic.py`, `test_composite_intensity.py`: Spergel / de Vaucouleurs, Sersic emulator, `CompositeIntensityModel` / `BulgeDiskModel`
- `test_cusp_guard.py`: Spergel cusp prior validation at construction
- `test_los_quadrature.py`: tanh-substituted line-of-sight quadrature vs the windowed reference
- `test_coordinates.py`: WCS-derived rotation and shear rotation
- `test_render.py`, `test_render_config.py`: `RenderConfig` grid sizing and effective maxk
- `test_psf.py`, `test_pixel_integration.py`, `test_pixel_readout.py`: PSF convolution, k-space pixel response, post-dispersion pixel readout
- `test_galsim_conventions.py`, `test_galsim_reference.py`: GalSim convention checks and GalSim-chromatic grism reference gate
- `test_jax.py`: JIT and gradient compatibility
- `test_precision.py`, `test_precision_robustness.py`, `test_cpu_devices.py`: `KLPIPE_FP32` precision control, float32 robustness, host-device count

### SourceModel, observations, grism
- `test_source.py`, `test_source_render.py`: `SourceModel` / `EmissionLine` construction and render methods
- `test_observation.py`, `test_obs_image_rotation.py`: `ImageObs` / `VelocityObs` / `GrismObs` builders
- `test_datacube.py`, `test_cube_psf.py`, `test_spectral_methods.py`, `test_spectral_resolution.py`: datacube assembly, cube PSF, spectral bin integration, grism resolution
- `test_grism_core.py`, `test_analytic_dispersal.py`, `test_grism_bandwidth.py`, `test_grism_psf_mode.py`, `test_grism_shared_cube.py`: dispersion, analytic dispersal, grism grid sizing, `psf_mode`, shared cube across rolls
- `test_grism_validation.py`: cross-code comparison against geko (requires reference data)
- `test_synth_source_parity.py`: `synthetic.py` vs `SourceModel` rendering parity

### Likelihood, inference, recovery
- `test_from_obs.py`, `test_grism_likelihood.py`: `InferenceTask.from_obs` and the grism likelihood
- `test_noise.py`, `test_noise_calibration.py`: noise generation and chi-square calibration at truth
- `test_priors.py`: prior distributions and `PriorDict`
- `test_likelihood_slices.py`: likelihood slicing (three-gate design, see below)
- `test_optimizer_recovery.py`, `test_optimization.py`: optimizer recovery (k-sigma design) and `multi_start_minimize`
- `test_flagship.py`: joint Roman-like broadband + grism inference
- `test_posterior_slices.py`: posterior-slice diagnostics

### Sampling
- `test_sampling.py`: sampler configs, factory, `InferenceTask`
- `test_sampling_transforms.py`, `test_initialization.py`: unconstrained-coordinate bijections, fit-initialization toolkit
- `test_sampling_diagnostics.py`: emcee + nautilus runs with diagnostic plots
- `test_numpyro.py`, `test_blackjax.py`: NUTS / HMC recovery and convergence

### Photometry and surveys
- `test_photometry.py`, `test_surveys_roman.py`: unit conversions and published Roman HLWAS parameters
- `test_lines.py`: vacuum rest-wavelength registry and air <-> vacuum conversion

### TNG50 (require data in `data/tng50/`, see `data/cyverse/README.md`)
- `test_tng_loaders.py`, `test_tng_mock_data.py`: data loading and structure
- `test_tng_data_vectors.py`: rendering, 3D rotations, gridding, diagnostic plots
- `test_tng_synthetic_fixture.py`: synthetic TNG files (`fixtures/tng_synthetic.py`) load and render; runs without TNG data
- `test_tng_likelihood.py`, `test_psf_tng.py`, `test_tng_sampling_diagnostics.py`: fitting, PSF, and sampling on TNG mocks

### Ensemble tooling (internal; mostly marked `roman_ensemble`)
- `test_ensemble*.py`, `test_population*.py`, `test_calibration.py`, `test_prior_provenance.py`, `test_roman_psf.py`, `test_bench.py`, `test_dashboard.py`: ensemble specs, catalog populations, shear calibration, prior provenance, Roman PSF option, benchmark records, run dashboard

### Support
- `test_utils.py`: shared utilities, `TestConfig`, plotting helpers
- `test_validation_utils.py`: `scripts/validation/utils.py`
- `conftest.py`: shared fixtures
- `fixtures/tng_synthetic.py`: writes synthetic TNG50-format files; `make test-tutorials` uses it when `data/tng50/` is absent

### Pytest markers

Defined in `pyproject.toml`.

| Marker | Meaning |
|--------|---------|
| `tng50` | Requires TNG50 data (`data/tng50/`) |
| `tng_diagnostics` | Slow TNG diagnostic plots |
| `slow` | Significant runtime |
| `diagnostic_plots` | Produces diagnostic figures (also `slow`) |
| `grism_validation` | Cross-code grism validation (requires geko reference data) |
| `galsim_reference` | GalSim-chromatic reference render (self-contained) |
| `cosmohub` | Requires downloaded CosmoHub / Q1 catalog data |
| `roman_ensemble` | Ensemble tooling tier; excluded from `make test`, run via `make test-roman-ensemble` |

### TNG diagnostic outputs

`make test-tng-diagnostics` writes plots and CSV summaries to
`tests/out/tng_diagnostics/`: native high-resolution renders, CIC vs NGP
gridding, inclination and position-angle sweeps (with and without the
gas-stellar offset), resolution and SNR grids, and vertical-extent
measurements versus inclination. These check that 3D rotations preserve
disk thickness, that observables vary physically with viewing angle, that
CIC gridding conserves flux, and that the gas-stellar offset is preserved
or removed as requested.

---

## Test Philosophy

Every pass/fail bound in the recovery tiers is derived from a
measurement plus a stated rule, never hand-tuned. The design separates
three questions that a single noisy-recovery check used to conflate:

1. **Is the forward model correct?** Answered deterministically, on
   noise-free data (slice gates 1 and 2). No random draw can pass or
   fail these; a failure is a real behavior change.
2. **Is the noise bookkeeping correct?** Answered directly by chi-square
   calibration at truth (`test_noise_calibration.py`), not inferred from
   recovery scatter.
3. **Does estimation under noise work?** Answered by the optimizer tier,
   where the bound is the physical noise floor times a multiplier set by
   a declared suite-wide false-alarm budget.

The old scheme ran one noisy recovery check per parameter against
hand-tuned x%/y% tolerances. Several of those sat below the physical
noise floor, so the pinned seed did the passing, not the code. Now:
green gates 1+2 = the model is correct at the stated accuracy; green
k-sigma = the fit landed within what the data can deliver; red =
something real changed. Find the cause, don't retune the number.

## Test Regimes: Likelihood Slicing vs Optimization

The test suite includes two complementary approaches to validate parameter recovery:

### 1. Likelihood Slicing (`test_likelihood_slices.py`) — three-gate design
**Method:** Brute-force grid search along each parameter dimension on NOISE-FREE data

**Purpose:**
- Validates that forward models are implemented correctly
- Ensures likelihoods peak at true parameter values
- Guards the information content (constraining power) of the likelihood
- Most rigorous validation of model correctness

**The three gates (all deterministic — no random draw can pass or fail them):**
1. **Accuracy**: noise-free slice recovery within tight per-parameter budgets
   (frozen `_GATES` tables; budget rule = 10x measured error, floored at
   1e-5 of the scan half-range, rounded up to 1 significant figure)
2. **Information**: each slice's curvature-implied 1-sigma error bar within
   ±20% of a frozen reference (catches both information loss and fake
   precision from discretization artifacts)
3. **Noise plumbing**: `test_noise_calibration.py` checks chi²-per-point at
   truth through the full likelihood for each obs type

Regenerate frozen tables with `KLPIPE_TEST_MEASURE=1` after a deliberate
change; record the reason next to the values.

**When to inspect:**
- Implementing new models
- Debugging parameter recovery issues
- Verifying forward model correctness

### 2. Gradient-Based Optimization (`test_optimizer_recovery.py`) — k-sigma design
**Method:** scipy.optimize with JAX automatic differentiation on a pinned noisy draw

**Purpose:**
- Validates gradient implementations
- Tests realistic estimation under noise (MCMC-like workflows, but faster)
- Complementary to likelihood slicing

**Pass criterion:** every sampled parameter must satisfy
`|recovered - truth| < bias + k * sigma`, where
- `sigma` is the parameter's marginal noise floor, measured per scene from
  the likelihood curvature at truth on the model's own noise-free render,
  with each uniform prior box folded in as a Gaussian of equal variance
  (frozen `_OPT_SIGMAS` tables; regenerate with `KLPIPE_TEST_MEASURE=1`)
- `k` is a single suite-wide multiplier derived from a declared 1%
  false-alarm budget over all bounded checks in the module
  (`suite_false_alarm_k`)
- `bias` is a frozen allowance used only where an independent rendering
  backend (GalSim) disagrees with the model by more than the noise floor
  (currently only the bulge+disk test)

Degenerate directions (e.g. cosi/g1/g2/vcirc/rscale in velocity-only
scenes — the kinematic lensing degeneracy) get honest prior-scale floors
(effectively unchecked) and the observable product `vcirc*sin(i)` is
checked instead. Exactly-flat directions (e.g. Halpha.flux under
normalized flux weighting) are pinned by in-test flatness asserts.

**When to inspect:**
- Validating gradient computations
- Testing optimization workflows
- Performance benchmarking

---

## Tolerance Configuration

The strict tiers no longer use hand-tuned tolerance tables: converted
likelihood-slice tests carry per-test `{budgets, sigma_refs}` tables and
optimizer tests carry per-test `{snr_ref, sigmas[, biases]}` tables, each
frozen with a provenance comment and regenerated via
`KLPIPE_TEST_MEASURE=1`. Every hard-coded bound must state what was
measured, when, on what scene, and the rule that turned the measurement
into the bound.

`TestConfig` in `tests/test_utils.py` still holds:
- `likelihood_slice_tolerance_*` + `likelihood_slice_param_scaling` +
  `psf_tolerance_multiplier`: used only by the legacy slice pattern (the
  skipped Spergel de Vaucouleurs test) and the slice plot annotations
- `absolute_tolerance_floor`: legacy dual-criterion support
- image/grid geometry shared by the recovery tests

The optimizer tolerance dicts and `optimizer_param_scaling` were retired
2026-07-08 in favor of the k-sigma scheme (`get_tolerance` now raises on
`test_type='optimizer'`).

### Provenance and Regeneration

Every frozen number in the recovery tiers is generated by the test
modules themselves in measure mode — there is no external script to
lose. Each table's header comment records the freeze date, the freeze
commit, and the exact command that regenerates it:

```bash
# likelihood-slice tier (_GATES budgets + sigma references)
KLPIPE_TEST_MEASURE=1 pytest tests/test_likelihood_slices.py -s

# optimizer tier (_OPT_SIGMAS floors + bulge_disk biases); the -k
# selection runs each test only at its reference SNR, where the
# measurement is exact
KLPIPE_TEST_MEASURE=1 pytest tests/test_optimizer_recovery.py \
    -k "10000 or not (1000 or 500)" -s
```

Measure mode skips the fit and prints ready-to-freeze table entries:
sigma floors from the likelihood curvature at truth, and (bulge_disk
only) bias entries from a noiseless multi-start fit of the GalSim
render. Paste the printed entries over the frozen ones and update the
provenance comment.

**What k is and how to read it.** A correct pipeline fitting noisy data
recovers each parameter with an error that scatters at its marginal
noise floor sigma. `|error| < k*sigma` therefore fails a correct
pipeline with probability 2*Phi(-k) per check. We declare a 1% budget
for the whole module spuriously failing, split it over the
`_N_BOUNDED_CHECKS = 184` bounded checks (union bound), and solve for
k: `k = Phi^-1(1 - 0.01/184/2) = 4.04` (`suite_false_alarm_k` in
`test_utils.py` carries the derivation). Consequences: k is not a
tunable — if the check count changes, recount `_N_BOUNDED_CHECKS` and k
moves on its own (it only grows logarithmically: 149 checks -> 3.99,
200 -> 4.06); a failure at k=4 means the error is 4x what the data's
own noise can explain, which pure bad luck produces less than once per
hundred full suite runs.

### Pass Criteria

- Converted likelihood-slice tests: absolute budgets on noiseless recovery
  plus the ±20% error-bar band. Deterministic.
- Optimizer tests: `|error| < bias + k * sigma` per parameter, plus the
  `vcirc*sin(i)` product checks (SNR-dependent tolerances in
  `check_degenerate_product_recovery`).
- Legacy pattern (unconverted tests only): passes if EITHER relative OR
  absolute tolerance is met.

---

## How to Interpret Test Failures

### Likelihood Slice Gate Failure
**Serious issue** — the gates are noiseless and deterministic, so a
failure is a real behavior change:
- Gate 1 (budget): forward-model bias or renderer disagreement grew
- Gate 2 (error-bar band): information content changed — blurred
  (constraining power lost) or sharpened beyond the physics (a
  discretization artifact faking precision)

**Action:** Debug the model — never adjust budgets or references without
root-causing and recording the reason

### Optimizer Test Failure
Could indicate:
- A real gradient or model regression (error far above the k-sigma bound)
- A local optimum (switch to multi-start via
  `kl_pipe.optimization.multi_start_minimize` — pattern in
  `test_optimize_bulge_disk` — rather than widening the bound)
- A renderer-vs-model bias that grew past its frozen allowance

**Action:** Diagnose the cause first (identifiability, local minima,
renderer bias — in that order). Never re-widen a bound blindly; every
change to a frozen table needs a stated measurement and rule.

---

## Modifying Bounds

- **Never loosen likelihood slice budgets or error-bar references** to
  make a test pass — they validate model correctness
- Regenerate a frozen table only after a deliberate, understood change:
  run the affected module with `KLPIPE_TEST_MEASURE=1`, paste the printed
  entry, and record the reason in the provenance comment
- The optimizer `k` derives from the declared false-alarm budget and the
  bounded-check count (`_N_BOUNDED_CHECKS`); recount when adding or
  removing tests or parameters instead of tuning `k`
- Bias allowances are 2x a measured, deterministic noiseless-recovery
  offset (rounded up to 1 significant figure), never free parameters
- **Justify any bound change loudly in your PR**


## Running Tests

### Run full test suite
```bash
make test-basic            # Fast tests, no data downloads needed (start here)
make test                  # Fast tests (excludes slow, TNG diagnostics, grism validation, cosmohub, roman_ensemble)
make test-fast             # Same marker filter as `make test`, plus -x (stop at first failure)
make test-extended         # Everything except TNG diagnostics, grism validation, cosmohub
make test-roman-ensemble   # Ensemble tooling tier
make test-coverage         # With coverage report
```

### Run specific test files
```bash
pytest tests/test_velocity.py -v
pytest tests/test_likelihood_slices.py -v
pytest tests/test_optimizer_recovery.py -v
```

### Run specific SNR levels (optimizer tests only; slice tests are noiseless)
```bash
pytest tests/test_optimizer_recovery.py -k "10000" -v
```

### Run tests in parallel (faster)
```bash
pytest tests/ -n auto  # Uses pytest-xdist
```

---

## Test Output

### Diagnostic Plots
Tests generate diagnostic plots in `tests/out/<test_name>/`:
- Velocity/intensity maps (true, noisy, model)
- Residuals and chi-squared distributions
- Likelihood slices along parameter dimensions
- Parameter recovery statistics

**Note:** This directory is gitignored

### Parameter Recovery Statistics
Tests report for each parameter:
- **Recovered value** vs **true value**
- **Relative error** (percentage)
- **Absolute error** (in parameter units)
- **Pass/fail** status against the bound (k-sigma for optimizer tests,
  budgets/references for slice gates)

Example failure output:
```
Failed: Optimizer: Offset velocity failed for SNR=500:
vel.v0: rel 12.00% (tol 4.5%), abs 1.20 (tol 0.45) - recovered 11.20, true 10.00
```

---

## Adding New Tests

### For New Models
1. Add unit tests in existing files (e.g. `test_velocity.py`) or make your own
2. Add likelihood slice tests in `test_likelihood_slices.py` (three-gate
   pattern: register in `_GATES`, measure with `KLPIPE_TEST_MEASURE=1`)
3. Add optimizer tests in `test_optimizer_recovery.py` (k-sigma pattern:
   add an `_OPT_SIGMAS` entry, measure with `KLPIPE_TEST_MEASURE=1`, and
   recount `_N_BOUNDED_CHECKS`)

### For New Parameters
1. Re-measure the affected frozen tables (`KLPIPE_TEST_MEASURE=1`)
2. Document the provenance of every new frozen value

### Best Practices
- Use `@pytest.fixture(scope="module")` for expensive setup (coordinate grids, etc.)
- Use `TestConfig` for all configuration - avoid hardcoded values
- Generate diagnostic plots for complex tests
- Use descriptive test names: `test_<feature>_<scenario>`

---

## Debugging Failed Tests

### Step 1: Identify Test Type
- **Unit test failure:** Check model implementation
- **Likelihood slice failure:** Model bug or coordinate transform error
- **Optimizer failure:** Check if marginal (< 2× tolerance) or systematic

### Step 2: Check Diagnostic Plots
Look in `tests/out/<test_name>/` for:
- Do model predictions look reasonable?
- Are residuals randomly distributed or systematic?
- Does likelihood slice peak at true value?

### Step 3: Run Specific Failed Test
```bash
pytest tests/test_likelihood_slices.py::test_recover_centered_velocity_base[1000] -v -s
```
The `-s` flag shows print statements and allows debugger access

### Step 4: Adjust or Fix
- **Model bug:** Fix implementation, don't adjust tolerance
- **Marginal failure:** Consider loosening tolerance with justification
- **Systematic failure:** Investigate parameter degeneracies or add bounds

---

## Test Development Workflow

1. **Start with likelihood slice tests** - establish ground truth
2. **Add optimizer tests** with looser tolerances
3. **Run full suite** before committing
4. **Document** any tolerance adjustments with scientific reasoning
5. **Inspect plots** for any marginal failures to verify correctness

---

## Future Test Ideas

See [FUTURE_TESTS.md](FUTURE_TESTS.md) for an evolving list of planned tests and
test gaps, organized by science impact and implementation difficulty.

---

## Questions?

For issues or questions about the test suite:
1. Check this README for tolerance configuration
2. Look at `test_utils.py` for implementation details
3. Examine diagnostic plots in `tests/out/`
4. Review existing tests for examples of patterns
