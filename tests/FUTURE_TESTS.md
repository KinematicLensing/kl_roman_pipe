# Future Test Ideas

Evolving list of planned tests and test gaps, organized by science impact and
implementation difficulty. Each entry includes science motivation, implementation
approach, and expected assertions.

---

## A. Shear + PSF Interaction (Highest Priority)

**Science motivation:** Core weak lensing measurement. Lensing shears the source
*before* PSF convolution; the PSF circularizes it. Current regression tests only
use round (g1=g2=0) sources --- they validate convolution math but not the
shear-PSF coupling that drives our science.

**What to test:**
- Generate sheared exponential via GalSim: `Exponential(hlr=3).shear(g1=0.05, g2=0.03)`
- Convolve with PSF via both GalSim native and JAX pipeline
- Residuals should match unsheared regression level (~5e-3 default, ~5e-4 rigorous)
- Key check: *difference* between sheared and unsheared residual maps --- if
  PSF x shear coupling introduces systematic bias beyond baseline, that's a problem

**Implementation approach:**
- Parametrize over shear values: `(0.05, 0)`, `(0, 0.05)`, `(0.05, 0.03)`
- GalSim path: `Convolve(sheared_source, psf).drawImage()`
- Pipeline path: `precompute_psf_fft` + `convolve_fft` on sheared source, OR
  full `model.render_image()` with nonzero g1/g2 + PSF configured
- Use `oversample_image_pars` (150x200) for boundary safety

**Practical notes:**
- GalSim applies shear analytically in Fourier domain; our pipeline applies via
  coordinate transforms in `transformation.py`. Verifying agreement is the point.
- Testing at `model.render_image` level is more end-to-end (includes oversampled
  rendering on sheared fine grid).

**Assertions:**
- `max|residual|/peak < 5e-3` (same as unsheared)
- Shear doesn't introduce *additional* systematic error beyond baseline floor

**Difficulty:** Medium

---

## B. Flux-Weighted Velocity PSF with Realistic Rotation

**Science motivation:** `convolve_flux_weighted` is the most scientifically
critical convolution --- determines how PSF smoothing biases recovered rotation
curves. Current tests only check: (1) constant velocity stays constant,
(2) qualitative shift. Neither validates accuracy for a *rotating disk* under PSF.

**What to test:**
- Exponential intensity + arctan rotation curve (`CenteredVelocityModel`)
- Compare two independent paths:
  - Ground truth: high-res (1000x1000, 0.03"/pix) velocity x intensity, convolve, divide
  - Pipeline: `convolve_flux_weighted()` at standard resolution with oversampling
- OR: convergence test --- N=5 vs N=15 oversampling (simpler, still valuable)

**Implementation approach:**
- Render at standard resolution via pipeline; render ground truth at ~10x, convolve
  with numpy, downsample
- Alternatively: compare `model.render_image()` for velocity at N=5 vs N=9 vs N=15

**Practical notes:**
- No GalSim ground truth for velocity maps (GalSim doesn't know about velocities)
- Flux-weighted PSF is `sum(v*I*PSF) / sum(I*PSF)` --- errors partially cancel,
  but near galaxy center where gradient is steepest, PSF mixes approaching and
  receding sides

**Assertions:**
- Velocity residual < 1% of velocity range (SNR=inf)
- Convergence: N=5 within 0.5% of N=15

**Difficulty:** Medium-high

---

## C. Off-Center / Edge-of-Stamp Galaxies

**Science motivation:** Real galaxies aren't centered in stamps. Centroid offsets
shift source toward edges where FFT wrapping can corrupt results. All PSF tests
currently center at (0,0).

**What to test:**
- Render exponential at offsets: (0,0), (2",0), (0,3"), (2",3")
- Convolve with PSF, compare against GalSim (`Exponential.shift(dx, dy)`)
- Check residuals don't grow unacceptably with offset

**Implementation approach:**
- Use `oversample_image_pars` (150x200, 0.3"/pix = 45"x60" FOV)
- GalSim: `Exponential(hlr=3).shift(dx, dy)` convolved with PSF
- Pipeline: `model.render_image()` with nonzero x0, y0 + PSF configured

**Practical notes:**
- FFT padding should handle moderate offsets. Danger is source extending past
  padded boundary.
- Roman stamps typically 4-6x half-light radius; offsets of ~1-2 pixels typical.
- Existing `test_wrap_around_artifact` tests the extreme (corner source) case;
  this covers the realistic intermediate regime.

**Assertions:**
- `max|residual|/peak` doesn't increase by >2x vs centered for offsets up to 5"
- Absolute residual still within 5e-3

**Difficulty:** Low-medium

---

## D. Pixel Scale Regime (Roman 0.11"/pix)

**Science motivation:** All PSF tests use 0.3"/pix. Roman grism/imaging pixel
scale is 0.11"/pix. PSF FWHM/pixel_scale ratio determines sampling --- at
0.3"/pix a 0.625" PSF spans ~2 pixels (barely Nyquist); at 0.11"/pix it spans
~5.7 pixels (well-sampled). Oversampling strategy may behave differently.

**What to test:**
- Run standard GalSim regression at 0.11"/pix
- Check oversample convergence: does N=5 still achieve 5e-4?
- Finer scale should *help* --- this is a confirmation test

**Implementation approach:**
- `roman_image_pars = ImagePars(shape=(250, 300), pixel_scale=0.11, indexing='ij')`
- Run `test_oversample_convergence` logic with Gaussian PSF
- One GalSim regression case at Roman pixel scale

**Practical notes:**
- At 0.11"/pix, 250x300 stamp = 27.5"x33" --- need larger stamp or smaller hlr
  for hlr=3" sources. Fine grid at N=5 = 1250x1500, still manageable.
- Really a sanity check that pipeline generalizes across pixel scales.

**Assertions:**
- Oversample convergence still monotonic (N=1 > N=3 > N=5)
- N=5 residual < 1e-4 (should be *better* than 0.3"/pix)

**Difficulty:** Low

---

## E. Joint Model PSF Path Consistency

**Science motivation:** `configure_joint_psf()` configures velocity+intensity PSFs
in one call, setting `velocity_model._psf_flux_model = intensity_model`. No test
verifies this joint path matches separately configured models.

**What to test:**
- Configure PSF via `configure_joint_psf()` on KLModel
- Separately configure on velocity_model and intensity_model
- Render both maps from both configurations, verify match

**Implementation approach:**
- KLModel with CenteredVelocityModel + InclinedExponentialModel
- Path A: `kl_model.configure_joint_psf(psf_vel=psf, psf_int=psf, ...)`
- Path B: `vel_model.configure_velocity_psf(psf, ..., flux_model=int_model, ...)`
  and `int_model.configure_psf(psf, ...)`
- Compare `PSFData` objects and render outputs

**Practical notes:**
- Joint config sets `_psf_flux_theta` to None (overridden at likelihood eval time);
  separate config uses fixed `flux_theta`. Comparison needs to account for this.
- Simplest: verify PSFData objects identical + `_psf_flux_model is intensity_model`.

**Assertions:**
- `PSFData` objects identical between joint and separate configuration
- `velocity_model._psf_flux_model is intensity_model` after joint config

**Difficulty:** Low

---

## F. PSF Model Mismatch / Robustness

**Science motivation:** Real PSFs estimated from stars have errors (wrong FWHM,
ellipticity, wings). How sensitive is parameter recovery to PSF model error? This
is a systematic error budget question.

**What to test:**
- Generate data with PSF_true (Gaussian FWHM=0.65")
- Fit with PSF_wrong (FWHM=0.60" or 0.70" --- 8% error)
- Measure bias as function of PSF error magnitude
- Key params: g1, g2, int_rscale, cosi

**Implementation approach:**
- Reuse `test_optimizer_recovery` infrastructure
- Generate data with one PSF, configure model with different PSF
- Parametrize over PSF error: 0%, 2%, 5%, 10% FWHM error
- Separate file: `test_psf_systematics.py`, mark `@pytest.mark.slow`

**Practical notes:**
- More diagnostic than unit test --- no simple pass/fail threshold.
- Results inform systematic error budget for Roman pipeline.
- Start simple: Gaussian FWHM mismatch. Later: ellipticity, profile shape.

**Assertions:**
- Informational --- diagnostic plots showing bias vs PSF error
- Rough expectation: shear bias ~ O(delta_FWHM / FWHM)

**Difficulty:** Medium

---

## G. TNG + PSF Integration

**Science motivation:** TNG50 has realistic clumpy morphologies, not smooth
exponentials. PSF convolution may behave differently --- sharper features alias
more, flux-weighted velocity PSF more sensitive to spatial structure.

**What to test:**
- Render TNG50 velocity+intensity, convolve with PSF
- Check: flux conservation, velocity range compression
- Compare PSF effect on TNG vs smooth model with matched params

**Implementation approach:**
- Requires TNG50 data (`@pytest.mark.tng50`)
- Use `data_vectors.py` to render maps, apply PSF via pipeline
- Extend existing `test_psf_tng.py`

**Practical notes:**
- Main question: does oversampling strategy (designed for smooth exponentials)
  work for clumpy TNG morphologies? Probably yes, worth confirming.
- Lower priority than A-C.

**Assertions:**
- Flux conservation < 1e-6
- Velocity range decreases monotonically with PSF FWHM
- No NaN/Inf in output

**Difficulty:** Low-medium
