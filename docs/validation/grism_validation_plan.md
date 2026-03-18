# Grism Cross-Code Validation Plan

## Motivation

Current datacube/grism tests compare JAX `SpectralModel.build_cube()` against a numpy
re-implementation in `synthetic.py` that encodes identical algorithms. A shared bug
passes both. We need a genuine, independent validation via cross-code comparison: render the
same galaxy through multiple independent codes and compare pixel-level outputs.

The idea is to (a) define a sufficient set of model, instrument, and observational parameters to cover a sufficiently wide set of realistic parameter combinations that we care about for KL and (b) cross-compare `kl_pipe` grism datavector outputs vs each of the alternative codes to see what our relative difference is at a pixel level over a wide range of inputs. This is relative rather than absolute calibration, but gets us closer.


## Other validation happening or TODO *not* in this doc:
- Unit tests (see `tests/test_grism_core.py`)
- Analytical or physical tests (some done, some TODO)
    - Zero-velocity datacube factorization
    - Flux conservation
    - Gaussian source profile through grism
    - Face-on velocity insensitivity
        - `grism(cosi=1, vcirc=200) == grism(cosi=1, vcirc=0)`
    - Grism symmetry tests
        - `grism(theta_int=π, disp_angle=0) = grism(theta_int=0, disp_angle=π)`
    - Predict mean doppler-shifted wavelength per channel
    - etc.

The analytic/physical tests should be added to the unit tests as we go.

Per-code sanity checks (flux conservation, face-on insensitivity, symmetry, etc.) can
also be run independently when cross-code disagreement is found, to help diagnose which
code is more likely correct.

## Codes to Compare

| Code | Owner | Physics | Expected agreement |
|------|-------|---------|--------------------|
| **kl_pipe** | Spencer | Arctan velocity, 3D inclined exponential, Gaussian emission, linear dispersion, shear. Instrument / datavector agnostic.| Reference |
| **geko** | "Spencer" | Same parametric galaxy model, no shear, independent implementation. Optimized for JWST. | ~1%? (primary) |
| **kl-tools** | Jiachuan | Antecedent of `kl_pipe`. Similar model implementations w/o JAX.| <0.5% (primary) |
| **grizli** | Fabian | Realistic Roman data features; what else? | ~5-10%? (secondary) |

### TLDR for collaborators

If you're writing a rendering script for kl-tools or grizli:

1. Look at `scripts/validation/test_params.yaml` for the 35 test configurations
2. Use the utility functions in `scripts/validation/utils.py` to load parameters:
   `get_test_params(test_name)` gives you a flat dict of source + observation params
3. Copy `scripts/validation/render_kl_pipe.py` as a template for your script
4. For each test, render a noiseless grism image + datacube and save as
   `{test_name}.npz` with keys: `cube`, `grism`, `vmap`, `imap`, `lambda_grid`
5. Upload your npz files to CyVerse (directory TBD)

See the Parameter Mapping section for your code's parameter conversions.

## Code Comparison Workflow

1. Define conceptual test configurations we are interested in (~35 output grism datavectors).
2. Define sufficient parameter values and/or sweeps for those tests and package them in a YAML file in `scripts/validation/test_params.yaml`.
3. Define tolerance & acceptance criteria for each test & code (some may have stricter requirements than others).
4. Write a separate script for each code (that is committed in `scripts/validation/`) that loads in the parameters for each test using provided util methods and renders the corresponding datavector (without noise). Scripts are written by their assigned owner.
5. For each test, save outputs as `{test_name}.npz` (one file per test) with keys: `cube`, `grism`, `vmap`, `imap`, `lambda_grid`.
6. Output `.npz` files are then uploaded to CyVerse in a TBD directory.
7. A new `makefile` target will be provided to automatically download these `.npz` files and saved to `tests/data/validation/`.
8. A new `tests/grism_validation.py` file will be created that loads in the 4 `.npz` files and cross-compares them along with visual diagnostics and a summary report using the defined tolerances & success criteria.

### Test Naming Scheme

We will define different conceptual test groups, each with its own tag. This allows test names to be human-readable: `{group}_{description}`. Every test name, group name,
and tag is defined in `scripts/validation/test_params.yaml` and reproduced in the test file's
docstring so collaborators can grep for any ID and find its definition.

NOTE: Sometimes `description` itself is another group tag, but not always. Depends on what the test is trying to accomplish.

### Rectangular Grism Images

My preferred approach is to have an image size is say **32x48** (Nrow != Ncol). This catches any row/col or Nrow/Ncol
transposition bugs that a square grid would hide. However, this may induce subtle implementation differences in any FFT-space rendering that complicates pixel comparisons that we will have to look out for.

### Code-Specific Rendering Parameters

Each code has their own set of internally-used (and inconsistent) parameters independent of model (e.g. `vcirc`) or data (e.g. PSF `fwhm`) parameters. For example, `kl_pipe` allows you to separately set the spatial & spectral oversampling rates, in addition to the velocity window around a modeled emission line. There is a special place in the `test_params.yaml` file where code-specific parameters can be set:

```yaml
rendering:
  kl_pipe:
    # spectral_oversample: number of sub-bins per wavelength pixel used
    spectral_oversample: 5

    # spatial_oversample: number of sub-pixels per pixel used when
    # evaluating the 2D intensity profile via k-space FFT.
    spatial_oversample: 5

    # velocity_window_kms: half-width of the wavelength window around
    # each emission line, in km/s. Determines how far the spectral grid
    # extends beyond the line center
    velocity_window_kms: 3000.0
```

---

## Test Matrix

### Tag definitions

Every test is tagged with the physics and/or data rendering axis it tests:

| Tag | Meaning |
|-----|---------|
| `static` | `vcirc=0`, datacube factorizes as I(x,y) x S(lambda) |
| `rotating` | `vcirc>0`, Doppler shifts break separability |
| `sweep` | Systematic variation of one parameter at a time |
| `psf` | Includes PSF convolution |

### Test group 1: Static Galaxy (non-rotating baselines)

Datacube factorizes as `I(x,y) x S(lambda)` -- analytically verifiable independent of
any code.

The base case parameters are as follows:
```yaml
base_params:
  # source properties
  z: 1.0                    # redshift; places Ha at ~1313 nm (Roman grism range)
  vcirc: 0.0                # asymptotic circular velocity, km/s; 0 = non-rotating
  vel_rscale: 0.5           # velocity turnover radius, arcsec
  v0: 0.0                   # systemic velocity, km/s
  flux: 100.0               # total integrated emission line flux (arbitrary units)
  int_rscale: 0.3           # intensity scale radius, arcsec
  int_h_over_r: 0.1         # disk scale height / scale radius (3D thickness)
  n_sersic: 1.0             # Sersic index (1.0 = exponential disk)
  cosi: 0.5                 # cos(inclination); 1=face-on, 0=edge-on; 0.5 ~ 60 deg
  theta_int: 0.0            # intrinsic position angle, radians from +x axis
  g1: 0.0                   # reduced shear component 1 (fixed at 0 for all tests;
  g2: 0.0                   #  geko has no shear; shear validation is kl_pipe-internal)
  vel_dispersion: 50.0      # intrinsic velocity dispersion of gas, km/s
  continuum: 0.0            # per-line continuum level (flat spectral density)
```

| Name | Tags | Description | Varies from base |
|------|------|-------------|------------------|
| `static_base` | `static` | Non-rotating, aligned, no PSF | -- |
| `static_psf` | `static, psf` | + Gaussian PSF (FWHM=0.15") | PSF |
| `static_compact` | `static` | int_rscale=0.15" (half base) | Source size |
| `static_extended` | `static` | int_rscale=0.6" (2x base) | Source size |

### Test group 2: Rotating Galaxy (kinematic signal)

Rotation maps velocity field to per-pixel Doppler shifts to spatially-varying line
centers.

| Name | Tags | Description | Varies from base |
|------|------|-------------|------------------|
| `rotating_base` | `rotating` | vcirc=200, aligned | vcirc |
| `rotating_edgeon` | `rotating` | vcirc=200, cosi=0.2 | vcirc, cosi |
| `rotating_edgeon_ortho_pa` | `rotating` | vcirc=200, cosi=0.2, PA perp to disp | vcirc, cosi, theta_int |
| `rotating_psf` | `rotating, psf` | vcirc=200 + PSF (FWHM=0.15") | vcirc, PSF |
| `rotating_compact` | `rotating` | vcirc=200, int_rscale=0.15" | vcirc, source size |
| `rotating_extended` | `rotating` | vcirc=200, int_rscale=0.6" | vcirc, source size |

### Test group 3: Inclination Sweep

Inclination changes velocity projection `cosi` and disk ellipticity.

Fixed: `vcirc=200`, `theta_int=0`, `disp_angle=0`. Vary `cosi` in uniform steps of 0.2.

| Name | Tags | cosi | Inclination |
|------|------|------|-------------|
| `incl_sweep_cosi01` | `rotating, sweep` | 0.1 | 84 deg (nearly edge-on) |
| `incl_sweep_cosi03` | `rotating, sweep` | 0.3 | 73 deg |
| `incl_sweep_cosi05` | `rotating, sweep` | 0.5 | 60 deg |
| `incl_sweep_cosi07` | `rotating, sweep` | 0.7 | 46 deg |
| `incl_sweep_cosi09` | `rotating, sweep` | 0.9 | 26 deg (nearly face-on) |

### Test group 4: Position Angle Sweep

PA relative to dispersion direction determines how much kinematic signal appears in the
grism. At PA=0 (aligned), kinematics stretched; at PA=pi (anti-aligned), compressed. 

Fixed: vcirc=200, cosi=0.5, disp_angle=0. Vary theta_int.

| Name | Tags | theta_int | Notes |
|------|------|-----------|-------|
| `pa_sweep_0` | `rotating, sweep` | 0 | Aligned (kinematic stretched) |
| `pa_sweep_30` | `rotating, sweep` | pi/6 | |
| `pa_sweep_45` | `rotating, sweep` | pi/4 | |
| `pa_sweep_90` | `rotating, sweep` | pi/2 | Orthogonal |
| `pa_sweep_120` | `rotating, sweep` | 2pi/3 | |
| `pa_sweep_150` | `rotating, sweep` | 5pi/6 | |
| `pa_sweep_180` | `rotating, sweep` | pi | Anti-aligned (kinematics compressed) |

### Test group 5: Dispersion Angle Sweep (0 to pi)

Rotating the grism direction tests the cos/sin decomposition in the dispersion code.
Full 0 to pi range catches sign errors in the shift components. Similar conceptually to PA sweeps but tests different parts of the code.

Fixed: vcirc=200, cosi=0.5, theta_int=0. Vary dispersion_angle.

| Name | Tags | disp_angle | Notes |
|------|------|------------|-------|
| `dispangle_sweep_0` | `rotating, sweep` | 0 | Aligned (kinematics stretched) |
| `dispangle_sweep_30` | `rotating, sweep` | pi/6 | |
| `dispangle_sweep_45` | `rotating, sweep` | pi/4 | |
| `dispangle_sweep_90` | `rotating, sweep` | pi/2 | Orthogonal |
| `dispangle_sweep_120` | `rotating, sweep` | 2pi/3 | |
| `dispangle_sweep_150` | `rotating, sweep` | 5pi/6 | |
| `dispangle_sweep_180` | `rotating, sweep` | pi | Anti-aligned (kinematics compressed) |

### Test group 6: Spectral Properties

| Name | Tags | Description | Varies from base |
|------|------|-------------|------------------|
| `narrow_lines` | `static` | vel_disp=30 km/s, narrower lines | vel_disp |
| `broad_lines` | `static` | vel_disp=100 km/s, broader lines | vel_disp |

### Test group 7: Redshift Sweep

Redshift changes observed wavelength, which changes the line width in nm
(lambda_obs * vel_dispersion / c) and therefore the line width in pixels. Also shifts
the wavelength grid. No instrumental R(lambda) is applied by any code.

Fixed: vcirc=200, cosi=0.5, theta_int=0, disp_angle=0.

| Name | Tags | z | Notes |
|------|------|---|-------|
| `redshift_sweep_05` | `rotating, sweep` | 0.5 | Ha at ~984 nm |
| `redshift_sweep_08` | `rotating, sweep` | 0.8 | Ha at ~1181 nm |
| `redshift_sweep_10` | `rotating, sweep` | 1.0 | Ha at ~1313 nm (base) |
| `redshift_sweep_15` | `rotating, sweep` | 1.5 | Ha at ~1641 nm |

**Total: 4 + 6 + 5 + 7 + 7 + 2 + 4 = 35 tests**

---

## Parameter Mapping

*Below is all under construction, still verifying some of the parameter mappings. But a place to start*

Owners shoudl fill out their section as they work so we can udnerstand how to map parameters between kl_pipe and the relevant codes.

### kl_pipe to geko

| kl_pipe | geko | Conversion | Verified? |
|---------|------|------------|-----------|
| `cosi` | `i` (deg) | `i = degrees(arccos(cosi))` | Yes |
| `theta_int` (rad, from +x) | `PAmorph` (deg) | `PAmorph = degrees(theta_int)` | **NO** |
| `vcirc` (km/s) | `Va` (km/s) | Direct | Yes |
| `vel_rscale` (arcsec) | `rt` (arcsec) | Direct | Yes |
| `int_rscale` (arcsec) | `re` | **NO**: `r_hl = 1.678 * r_scale` for n=1? | |
| `flux` (integrated) | `Ie` (SB at re) | `Ie = flux / (2*pi*re^2)` for n=1? | **NO** |
| `v0` (km/s) | `v0` (km/s) | Direct | Yes |
| `vel_dispersion` (km/s) | `sigma0` (km/s) | Direct | Yes |
| `int_h_over_r` | `q0` | **NO**: different geometric meaning | |
| `g1, g2` | N/A | geko has no shear; fix to 0 in kl_pipe | N/A |
| `theta_int` (shared PA) | `PAmorph`, `PAkin` | Set both equal in geko | -- |

### kl_pipe to kl-tools (Jiachuan)

kl-tools is the antecedent of kl_pipe, sharing the same core 5-plane coordinate
system.

- Uses `sini` instead of `cosi`; `sini = sqrt(1 - cosi^2)`
- Datacube + grism only available on `tng-grism` branch
- Expected (or hope): <0.5% agreement with kl_pipe 

### grizli (Fabian)

- Need significant input here from Fabian++
- Expected: looser agreement, but hopefully <10% to start

---

## Acceptance Criteria

NOTE: These likely will change as we get a better understanding of the relative performance of each code. The numbers below remain *aspirational* for now.

Two tolerance tiers reflecting different levels of expected agreement:

### Primary tier (geko, kl-tools)

Same underlying physics (arctan velocity, exponential disk, Gaussian emission).

| Metric | Threshold | Description |
|--------|-----------|-------------|
| `total_flux_rtol` | 0.5% | \|sum(A)-sum(B)\| / sum(A) |
| `peak_pixel_rtol` | 1% | \|max(A)-max(B)\| / max(A) |
| `max_pixel_residual` | 1% | max(\|A-B\|) / max(A), peak-normalized |
| `rms_residual` | 1% | rms(A-B) / rms(A) |
| `spatial_centroid_pix` | 0.25 pix | \|centroid(A)-centroid(B)\| in pixels |
| `spectral_centroid_nm` | 0.05 nm | per-spaxel first-moment agreement (datacube only) |

### Secondary tier (grizli)

Fundamentally different approach, may need looser tolerances.

| Metric | Threshold |
|--------|-----------|
| `total_flux_rtol` | 5% |
| `peak_pixel_rtol` | 10% |
| `max_pixel_residual` | 15% |
| `rms_residual` | 10% |
| `spatial_centroid_pix` | 1.0 pix |
| `spectral_centroid_nm` | 0.5 nm |

NOTE: These are initial "estimates" (i.e. made-up). All will need be refined after the first comparison run.

---

## Infrastructure

### Files

| File | Description |
|------|-------------|
| `scripts/validation/test_params.yaml` | Single source of truth: base params, 35 tests, tolerances |
| `scripts/validation/utils.py` | Config reader, per-code param mappers, comparison metrics |
| `scripts/validation/render_kl_pipe.py` | Renders all tests via KLModel, outputs .npz |
| `scripts/validation/render_geko.py` | Renders all tests via geko |
| `scripts/validation/render_kl_tools.py` | Renders all tests via kl-tools |
| `scripts/validation/render_grizli.py` | Renders all tests via grizli |
| `tests/test_grism_validation.py` | Cross-code comparison test suite, ~N_tests x 2 tests |
| `tests/test_validation_utils.py` | Unit tests for utils.py (runs under default make test) |
| `docs/validation/grism_validation_plan.md` | This document |

### New Makefile targets

```bash
make render-validation-kl-pipe    # python scripts/validation/render_kl_pipe.py
make render-validation-geko       # python scripts/validation/render_geko.py
make download-validation-data     # CyVerse -> tests/data/validation/
make test-grism-validation              # pytest -m grism_validation
```

### Test suite structure

[UNDER CONSTRUCTION]

Tests are organized by test group, with parametrized tests per class:

```
TestStaticGalaxy         # 4 tests x 2 (grism + datacube)
TestRotatingGalaxy       # 6 tests x 2
TestInclinationSweep     # 5 tests x 2
TestPASweep              # 7 tests x 2
TestDispAngleSweep       # 7 tests x 2
TestSpectralProperties   # 2 tests x 2
TestRedshiftSweep        # 4 tests x 2
TestDiagnosticPlots      # 35 tests (side-by-side comparison plots)
```

Marker: `@pytest.mark.grism_validation`. Excluded from default `make test` and
`make test-extended` runs since they require external reference data.

Each cross-code test:
1. Loads kl_pipe + reference .npz via `load_reference_data()`
2. Computes metrics via `compare_images()` / `compare_datacubes()`
3. Asserts against `load_tolerances()`
4. Skips gracefully if reference data is missing

Diagnostic plots saved to `tests/out/grism-validation/`.

## Open Questions

We can assemble our major open questions here for the whole process.

### Geko Parameter mapping

Before the first real comparison run, we need to verify several parameter conversions
between kl_pipe and geko. Getting any of these wrong produces results that look like
code disagreement but are actually input mismatch:

- **PA convention**: is geko's `PAmorph=0` along +x or +y? CW or CCW?
- **`re`**: half-light radius or scale radius?
- **`q0` vs `int_h_over_r`**: intrinsic axis ratio vs height/radius — different geometry
- **Flux normalization**: geko `Ie` = SB at re, or integrated flux?
- **Rotation sign**: does geko's `Va` follow the same sign convention?

Resolution for all of these: inspect geko source code, then render an asymmetric test
case (e.g. `pa_sweep_45`) in both codes and visually compare before running the full
matrix.

### kl-tools Parameter mapping

`kl-tools` mapping is expected to be straightforward given the lineage, but need to
confirm things like the shear convention hasn't diverged since the branch.

### Dispersion model differences

- `geko` uses 4th-order polynomial dispersion (JWST calibration); ours is linear. How much does this matter?
- `grizli`: I don't know enough about the implementation details to predict what level of
  reproducibility we can expect. Need significant input from Fabian.

### Out of scope

- **Shear**: Excluded from cross-code comparison (`geko` has no shear). Shear validation
  is kl_pipe-internal only.
- **Non-zero v0**: All codes should subtract systemic velocity themselves or never apply one.
