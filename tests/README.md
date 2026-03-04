# Test Suite Documentation

This directory contains comprehensive tests for the `kl_pipe` kinematic lensing pipeline. The test suite validates both forward models (synthetic data generation) and inference workflows (parameter recovery).

## Test Organization

### Unit Tests
- **`test_velocity.py`**: Velocity model evaluation, coordinate transformations, parameter conversions
- **`test_intensity.py`**: Intensity model evaluation, flux conservation, inclination effects
- **`test_utils.py`**: Shared test utilities, tolerance configuration, plotting helpers
- **`test_jax.py`**: JAX-specific functionality (JIT compilation, gradients)

### Integration Tests (Parameter Recovery)
- **`test_likelihood_slices.py`**: Brute-force likelihood slicing for parameter recovery
- **`test_optimizer_recovery.py`**: Gradient-based optimization for parameter recovery

---

## Test Regimes: Likelihood Slicing vs Optimization

The test suite includes two complementary approaches to validate parameter recovery:

### 1. Likelihood Slicing (`test_likelihood_slices.py`)
**Method:** Brute-force grid search along each parameter dimension

**Purpose:**
- Validates that forward models are implemented correctly
- Ensures likelihoods peak at true parameter values
- Tests the full likelihood surface (not just gradient descent paths)
- Most rigorous validation of model correctness

**Characteristics:**
- Slow but comprehensive
- Strict tolerances
- Tests one parameter at a time while holding others fixed (at true values)
- Independent of optimization algorithms

**When to inspect:**
- Implementing new models
- Debugging parameter recovery issues
- Verifying forward model correctness

### 2. Gradient-Based Optimization (`test_optimizer_recovery.py`)
**Method:** scipy.optimize with JAX automatic differentiation

**Purpose:**
- Validates gradient implementations
- Tests realistic inference scenarios (MCMC-like workflows, but faster)
- Faster feedback during development
- Complementary to likelihood slicing

**Characteristics:**
- Fast-ish (at least compared to likelihood slicing)
- Looser tolerances (10-20× vs likelihood slices) - accounts for local optima and parameter degeneracies
- Tests full gradient-based inference pipeline
- Subject to optimizer convergence issues
- Excludes degenerate parameters (e.g. cosi, g1, g2) from pass/fail checks
- Tests observable products (e.g. vcirc*sini) instead of degenerate individual parameters

**When to inspect:**
- Validating gradient computations
- Testing optimization workflows
- Performance benchmarking

---

## Tolerance Configuration

Tolerances are configured in `test_utils.py` within the `TestConfig` class. They control how precisely parameters must be recovered to pass tests.

### Where Tolerances Are Set

All tolerance settings are in **`tests/test_utils.py`** in the `TestConfig.__init__()` method:

```python
class TestConfig:
    def __init__(self, ...):
        # LIKELIHOOD SLICE TOLERANCES (lines ~57-68)
        self.likelihood_slice_tolerance_velocity = {
            1000: 0.001,  # 0.1%
            50: 0.01,     # 1%
            10: 0.05,     # 5%
        }
        
        # OPTIMIZER TOLERANCES (lines ~74-85) - 10-20x looser
        self.optimizer_tolerance_velocity = {
            1000: 0.02,   # 2% (20× looser than likelihood slices)
            50: 0.05,     # 5%
            10: 0.20,     # 20% (very noisy data)
        }
        self.optimizer_tolerance_intensity = {
            1000: 0.025,  # 2.5% (slightly higher due to intensity-specific degeneracies)
            50: 0.05,     # 5%
            10: 0.20,     # 20%
        }
        
        # PARAMETER-SPECIFIC SCALING (lines ~92-119)
        self.likelihood_slice_param_scaling = {
            'g1': {1000: 1.0, 50: 1.5, ...},  # Shear harder to constrain
            'v0': {1000: 1.0, 10: 1.5, ...},  # Small values harder
        }
        
        self.optimizer_param_scaling = {
            'g1': {1000: 2.0, 50: 3.0, ...},  # Much looser for optimizer
            'g2': {1000: 2.0, 50: 3.0, ...},
            'vel_x0': {1000: 1.5, 50: 2.0, ...},
        }
        
        # ABSOLUTE TOLERANCE FLOORS (lines ~122-129)
        self.absolute_tolerance_floor = {
            'g1': 0.002,  # If |true| < 0.002, use absolute not relative
            'g2': 0.002,
        }
```

### Tolerance Structure

Each test type has:
1. **Base tolerance** (varies by SNR): The default relative error threshold
2. **Parameter-specific scaling**: Multipliers for hard-to-constrain parameters
3. **Absolute floor**: Minimum absolute error tolerance (for parameters near zero)

**Final tolerance = base_tolerance × param_scaling**

### SNR-Dependent Tolerances

| SNR  | Likelihood Slice | Optimizer (Velocity) | Optimizer (Intensity) | Use Case |
|------|-----------------|----------------------|----------------------|----------|
| 1000 | 0.1%            | 2.0%                 | 2.5%                 | High-quality data, tight constraints |
| 500  | 0.25%           | 2.5%                 | 3.0%                 | Good data quality |
| 100  | 0.5%            | 3.0%                 | 3.5%                 | Moderate quality |
| 50   | 1.0%            | 5.0%                 | 5.0%                 | Realistic observational data |
| 10   | 5.0%            | 20.0%                | 20.0%                | Low SNR, very weak constraints |

**Note:** Optimizer tolerances are 10-20× looser than likelihood slices to account for local optima, parameter degeneracies (cosi/g1/g2), and gradient noise.

### Parameter-Specific Tolerances

Some parameters are inherently harder to constrain and get looser tolerances:

#### Degenerate Parameters (`cosi`, `g1`, `g2`)
- Geometrically degenerate - multiple combinations fit data similarly
- **EXCLUDED from pass/fail checks** (reported but don't cause test failure)
- Tests `vcirc*sini` product (3-10% tolerance) instead of individual params
- Line-of-sight velocity depends on `vcirc*sin²`, not `vcirc` and `cosi` independently

#### Shear (`g1`, `g2`)
- **Why harder:** ~4% of main velocity signal in most tests
- **Optimizer scaling:** Excluded (see above)
- **Likelihood scaling:** 1.5× at SNR=50, standard otherwise

#### Systemic Velocity (`v0`)
- **Why harder:** Small absolute value, can get washed out by noise
- **Optimizer scaling:** 1.5-2.5× looser
- **Likelihood scaling:** 1.5× at SNR=10

#### Centroid Offsets (`vel_x0`, `vel_y0`, `int_x0`, `int_y0`)
- **Why harder:** Shallow likelihood surfaces, weak constraints from data edges
- **Optimizer scaling:** 1.5-2.5× looser
- **Absolute floor:** 0.1 arcsec

### Tolerance Pass Criteria

A parameter passes if **EITHER** criterion is met:
- **Relative error** < relative_tolerance, **OR**
- **Absolute error** < absolute_tolerance

This dual-criterion approach handles:
- Large parameters: relative error dominates
- Small parameters (near zero): absolute error dominates

---

## How to Interpret Test Failures

### Likelihood Slice Test Failure
**Serious issue** - likely indicates:
- Bug in forward model implementation
- Incorrect coordinate transformations
- Numerical precision problems

**Action:** Debug the model - don't adjust tolerances unless scientifically justified

### Optimizer Test Failure
**Less serious** - could indicate:
- Local optima (optimizer stuck away from true values)
- Parameter degeneracies (multiple solutions fit data similarly - expected for cosi/g1/g2)
- Tight tolerances relative to SNR

**Note:** Degenerate parameters (cosi, g1, g2) are excluded from causing test failures. They're reported with `[EXCLUDED]` tag if outside tolerance, but tests still pass. The physically meaningful product `vcirc*sini` is checked instead.

**Action:** 
1. Check if error is small (< 2× tolerance) - may just need looser thresholds (but check with team first)
2. Check if specific parameters consistently fail - may need parameter-specific scaling
3. Check if failure only at low SNR - expected behavior
4. For excluded params showing large errors - this is informational, not a failure

### Marginal Failures
If a parameter recovers to within ~1.5× tolerance:
- **Likelihood slices:** Re-run with higher resolution or check for bugs
- **Optimizer:** Likely acceptable - consider loosening tolerance or adding bounds

---

## Modifying Tolerances

### When to Loosen Tolerances

**Acceptable reasons:**
- Optimizer consistently fails by small margins (< ~1.5× current tolerance)
- Parameter has known degeneracies or weak constraints
- Adding new model with different parameter sensitivities

**Bad reasons:**
- Making tests pass without understanding failures
- Likelihood slice tests failing (fix the model instead)
- Ignoring systematic errors

**Always** point this change out *loudly* in your PR!

### How to Modify

**For a specific parameter at specific SNR:**
```python
# In TestConfig.__init__() in test_utils.py
self.optimizer_param_scaling = {
    'my_param': {
        1000: 2.0,  # 2× looser than base
        50: 3.0,    # 3× looser
        10: 4.0,    # 4× looser
    }
}
```

**For all parameters at specific SNR:**
```python
# In TestConfig.__init__()
self.optimizer_tolerance_velocity = {
    1000: 0.01,  # Change from 0.005 to 0.01 (1% instead of 0.5%)
}
```

**For absolute error floor:**
```python
# In TestConfig.__init__()
self.absolute_tolerance_floor = {
    'my_param': 0.05,  # Minimum absolute error threshold
}
```

### Design Principles

1. **Never loosen likelihood slice tolerances** - they validate model correctness
2. **Optimizer tolerances should be ~2-5× looser** than likelihood slices
3. **Parameter-specific scaling should reflect physics** - not just to make tests pass
4. **Justify** any tolerance changes to PR reviewers
5. **Use absolute floors for parameters that can be near zero**

---

## Running Tests

### Run full test suite
```bash
make test                  # All tests with verbose output
make test-fast             # Stop on first failure
make test-coverage         # With coverage report
```

### Run specific test files
```bash
pytest tests/test_velocity.py -v
pytest tests/test_likelihood_slices.py -v
pytest tests/test_optimizer_recovery.py -v
```

### Run specific SNR levels
```bash
pytest tests/test_likelihood_slices.py -k "snr1000" -v
pytest tests/test_optimizer_recovery.py -k "snr50" -v
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
- **Pass/fail** status with tolerance thresholds
- **[EXCLUDED]** tag for degenerate parameters (optimizer tests only)

Example output:
```
️  Optimizer: Centered velocity (base) - Excluded parameters outside tolerance:
  g1: rel 1.03% (tol 30.0%), abs 0.0103 (tol 0.0025) - recovered -0.0103, true 0.0000 [EXCLUDED]
  g2: rel 4.85% (tol 30.0%), abs 0.0485 (tol 0.0025) - recovered 0.0485, true 0.0000 [EXCLUDED]

Failed: Optimizer: Offset velocity failed for SNR=50:
vcirc: rel 1.00% (tol 5.0%), abs 2.00 (tol 10.0) - recovered 202.00, true 200.00
```

Note: Excluded parameters show warnings but don't cause test failure.

---

## Adding New Tests

### For New Models
1. Add unit tests in existing files (e.g. `test_velocity.py`) or make your own
2. Add likelihood slice tests in `test_likelihood_slices.py`
3. Add optimizer tests in `test_optimizer_recovery.py`
4. Update `TestConfig` with model-specific tolerances if needed

### For New Parameters
1. Add parameter to tolerance scaling dicts if weakly constrained
2. Add absolute tolerance floor if parameter can be near zero
3. Document physical reason for any special tolerance treatment

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

## Questions?

For issues or questions about the test suite:
1. Check this README for tolerance configuration
2. Look at `test_utils.py` for implementation details
3. Examine diagnostic plots in `tests/out/`
4. Review existing tests for examples of patterns
