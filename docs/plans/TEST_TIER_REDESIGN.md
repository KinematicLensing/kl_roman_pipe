# Strict test tier redesign (three-gate scheme)

Status: vision ADOPTED by user 2026-07-08; phase 1 IMPLEMENTED same day
(decisions in section 5). Evidence base:
experiments/sweverett/analytic_audit/ (probes 1-6) and
docs/sessions/2026-07-08_analytic_dispersal_audit.md.

## 1. Problem being solved

The likelihood-slice tier asserts noisy-data recovery against hand-tuned
tolerances. For several parameters the tolerance sits below the physical
noise floor, so the pinned seed does the passing, not the code (measured:
the joint phot+grism test fails most re-rolled seeds on every dispersal
pathway, via vel.vcirc / vel.rscale / Halpha.dispersion). The tolerance
table (per-SNR bases, per-parameter multipliers, a PSF factor applied
twice by accident) is not derivable from anything. One noisy check is
conflating three questions: model correctness, constraint strength,
noise handling.

## 2. Adopted design

Three deterministic gates per slice test; no random draw can pass or
fail any of them.

- Gate 1 (accuracy): slices on NOISE-FREE data; assert |recovered -
  truth| < tight per-parameter budget. Self-rendered channels measure
  estimator/model bias; GalSim-backed channels measure renderer-vs-
  reference accuracy. Measured headroom vs today's tolerances: 650x to
  1e11 across all 17 parameters of the joint test.
- Gate 2 (information): from the same noiseless slice, read the
  likelihood peak curvature; sigma = 1/sqrt(curvature) is the error bar
  the data can deliver (validated against 500-draw Monte Carlo on all
  17 parameters, agreement within a few percent). Freeze today's sigma
  per parameter as an in-test reference (GalSim-floor pattern); assert
  future values stay within a band. Catches information-destroying or
  information-faking changes (the n=25 grid faked sigma(dispersion)
  0.78 km/s vs the true 4.8 -- a 6.1x artifact this gate fails loudly).
- Gate 3 (noise plumbing): one dedicated unit test per obs type --
  known noise draw, assert chi2 at truth matches expectation. Variance
  bookkeeping is tested directly, not via recovery scatter.

Removed from the tier: noisy-draw recovery (estimation belongs to the
optimizer/sampler tiers) and the SNR parametrization (a noiseless
argmax is provably invariant to the assumed noise level; verified
bit-level; halves tier runtime).

Noisy bounds elsewhere (optimizer tier, any future noisy check):
|error| < k * sigma with sigma measured and k a single suite-wide
constant derived from a stated false-alarm policy. This retires the
hard-coded 3.3-sigma bounds from 1fc9316 and the ~3.3-sigma v0
carve-out convention.

TestConfig consequences: slice-tier SNR tables / param scalings / PSF
multipliers become dead for converted tests, replaced by per-test
{budget, sigma reference} tables with provenance comments. The
psf_tolerance_multiplier double-application bug in get_tolerance
(absolute tolerance gets 1.5 twice = 2.25x) is fixed where multipliers
survive; resulting breaks handled case-by-case (user pre-approved).

Style constraint (user): plain-language comments; no experiment
war-stories or Fisher jargon in code; numbers carry a one-line
provenance pointer only.

## 3. Migration order

1. Joint phot+grism likelihood-slice test (the contested one): convert
   to gates 1+2 on the analytic default arm; shared helpers in
   test_utils (noiseless recovery assert + curvature reference assert).
   Slice-arm fate: pending decision (section 5).
2. Gate 3 chi2-calibration unit tests per obs type.
3. Remaining likelihood-slice tests via the same helpers.
4. Optimizer tier (phase 2): fix double-psf bug, recalibrate bounds as
   k * sigma with the derived k.
5. Sampler tier: unchanged (already posterior-width based).
GPU/Vista: on hold until this lands (user).

## 4. Worked example (Halpha.dispersion, joint test)

Today: noisy draw vs effective 1.69 km/s tolerance over a 4.8 km/s
floor -> ~1 in 3 seeds pass. New: Gate 1 bias budget ~0.05 km/s
(measured 0.00015); Gate 2 reference 4.8 km/s +/- band (n=25 artifact
reads 0.78 -> loud fail). Deterministic.

## 5. Decisions (user, 2026-07-08) and phase-1 implementation record

1. Gate 1 budget rule: measured x10, floored at 1e-5 of the scan
   half-range (numerical-drift guard), rounded up to 1 significant
   figure. DECIDED + implemented (_JOINT_GRISM_BUDGETS); inspect
   individual budgets as they trip.
2. Gate 2 band: +/-20%. DECIDED + implemented
   (_JOINT_GRISM_SIGMA_REFS, assert_slice_curvature_references).
3. k derivation: DEFERRED to the optimizer-tier phase (user wants the
   concept fully understood first; no k is needed anywhere in phase 1 --
   the converted tier is noiseless and the chi2-calibration window is a
   plain 5-sigma-of-chi2 statistical spread).
4. Slice-arm coverage: DROPPED from the recovery test (user), conditional
   on production-settings equivalence coverage, which was added:
   test_analytic_dispersal.py::test_production_slice_rule_matches_analytic
   (slice_width_kms=40, os=5 vs analytic; frozen at 7e-4 of peak /
   5e-5 flux, ~3x measured). Slice machinery unit tests remain pinned
   to 'slice'.
5. Sequencing: single PR (user). Phase-1 contents:
   - get_tolerance double-psf-multiplier bug FIXED (absolute tolerance
     was 2.25x base instead of 1.5x).
   - compute_parameter_bounds zero-truth bounded params (shear at 0)
     produced a ZERO-WIDTH scan -- those slices were vacuous; now scan
     fraction * physical half-span.
   - Joint phot+grism slice test converted to gates 1+2 (noiseless,
     no SNR parametrization, vel.v0 sampled again, analytic pathway).
   - Gate 3 added: tests/test_noise_calibration.py (image / grism /
     velocity obs chi2-per-point at truth through the full likelihood).
   - 'dispersion' likelihood_slice_param_scaling entry retired (its 3.0
     was calibrated on the n=25 quantization artifact; the converted
     test was its only consumer).
   - Fallout fixed with cause-level diagnoses (not tolerance widening):
     optimizer joint vel+phot+line test's Halpha.flux is exactly
     unidentifiable (normalized flux weighting cancels the amplitude)
     -> excluded as unidentifiable.

## 5b. Remaining phases

- Phase 3 (user folded into this PR, 2026-07-08): all remaining
  likelihood-slice recovery tests converted to the three gates via a
  shared registry (_GATES in test_likelihood_slices.py) and
  assert_noiseless_gates; per-test tables measured with
  KLPIPE_TEST_MEASURE=1 and frozen. SNR parametrizations dropped
  (nominal snr kept as a signature default for variance normalization).
  The skipped Spergel de Vauc test stays on the legacy pattern until its
  renderer limitation is fixed. The bulge-disk test's g1/g2 exclusions
  removed (the zero-width-scan fix made those slices real).
- Phase 2 (optimizer tier): recalibrate bounds as k * sigma with
  marginal floors; k decision parked with the user (suite-budget
  derivation proposed and explained: declare a suite false-alarm
  budget, e.g. 1% over ~50 bounded checks -> k ~= 3.7).
- Sampler tier: unchanged.

## 6. Evidence pointers

- probes 1-3: derivation exact; dense-slice information == analytic;
  n=25 fake curvature 6.1x; 40-seed pass-rate measurements.
- probe 6: noiseless headroom table (all 17 params); curvature-sigma vs
  Monte Carlo validation; suite-budget k table (17 checks -> 3.44,
  50 -> 3.72, 200 -> 4.06).
- Bench (loaded-machine caveat): analytic 36 ms/grad on the test scene
  vs 805 ms for slice n=151; compile 3.2 s vs 12.6 s.
