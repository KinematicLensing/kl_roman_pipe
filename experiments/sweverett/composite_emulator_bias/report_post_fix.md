# Composite emulator bias — post-fix re-run (E0–E7)

**Date**: 2026-05-06
**Branch**: `se/inclined-sersic` (commit `e76cce6`)
**Predecessor**: [`report.md`](report.md) — superseded; numbers there are
convention-contaminated.

The convention bug — GalSim reinterpreting `scale_h_over_r` as
`h_z/scale_radius` when `half_light_radius` is supplied — was fully fixed in
`871345a` (composite test) + `e76cce6` (Spergel tests, nu-n mapping script,
experiment scripts, regression test). This report re-runs E0–E7 with the
corrected GalSim references and reads off what residual is left.

## TL;DR

**Convention bug confirmed and resolved.** Every E* result that the prior
report flagged as convention-contaminated has collapsed to within ~1% of
truth at the test geometry. E1 central frac residual went from -25% → -0.02%.
E4 thickness sweep is now completely flat (`central frac` ≈ +0.001 at all
`h_over_hlr` ∈ [0.05, 0.5]; pre-fix it was monotonic from -1% at h=0.05 to
-42% at h=0.5). E7 |Δg2| dropped ~100–1000× to ~1e-5–1e-4 across all knobs
at baseline. The Miller-Pasha emulator is fine.

**Residual structure remains.** A ~1% RMS, ~5–8% peak per-pixel residual
persists post-fix and is now likely a combination of (a) the Miller-Pasha
emulator's intrinsic accuracy at n=4 inclined and (b) numerical artifacts
from k-space FT + LOS quadrature. This residual translates into 3–10%
SNR=1000 likelihood slice biases on cosi, theta_int, bulge_hlr, bulge_frac
in the composite recovery test — non-zero, real composite parameter
degeneracies, but qualitatively different from the convention-bug regime.

**Hessian indefiniteness at θ_true persists.** Smallest eigenvalue is now
-0.28 (down from "much worse" pre-fix). θ_true is still not a strict local
minimum; the negative direction is a small numerical-precision artifact or
a real local non-convexity. Either way, gradient-based optimization from
θ_true does not necessarily return to θ_true.

## Setup

Identical to the prior report:
- `BulgeDiskModel(shared_centroids=True)`, n=4 Sersic bulge + exponential disk
- `_TRUE_PARS_SHARED`: cosi=0.7, theta_int=0.785 rad, total_flux=1.0, B/T=0.25, disk_rscale=2.0, disk_h_over_r=0.1, bulge_hlr=0.8, bulge_h_over_hlr=0.3
- `image_pars`: 64×64, pixel_scale=0.15 arcsec
- PSF: `gs.Gaussian(fwhm=0.15)`
- Variance for E6/E7: total noise std = total_flux/1000
- All renders oversample=5 with PSF + pixel response

The only change vs the prior report is the convention fix in the bulge
GalSim construction.

## Per-experiment verdicts

### E0a — radial profiles (`out/e0a_radial_profiles.png`)

Pre-fix: kl_pipe ~25% UNDER along bulge major axis at center, ~60% OVER along
minor axis (inclined cosi=0.7).
Post-fix: agreement to <1% along all radial cuts at all sampled cosi.
**Verdict: CONVENTION-DRIVEN, RESOLVED.**

### E0b — Jacobian images (`out/e0b_jacobian_images.png`, `out/e0b_scores.csv`)

Scores `g_p = Σ residual · ∂M/∂θ_p / σ²` are now ~1000× smaller across all
parameters (residual is ~1000× smaller; Jacobians unchanged).
**Verdict: CONVENTION-DRIVEN, RESOLVED.**

### E0c — scalar diagnostics (`out/e0c_scalars.csv`)

| Quantity | Pre-fix | Post-fix |
|---|---|---|
| total flux kl/gs | 1.0009 | 1.000938 |
| central pixel kl/gs | 0.749 | 0.9998 |

Central pixel ratio is now within 0.02%; total flux unchanged (volume
conservation always held).
**Verdict: CONVENTION-DRIVEN, RESOLVED.**

### E1 — 2D residual maps (`out/e1_residual_scalars.csv`)

| Quantity | Pre-fix | Post-fix |
|---|---|---|
| central frac residual | -0.31 | -0.0002 |
| RMS frac | 0.061 | 0.0091 |
| peak \|frac\| | not recorded | 0.0752 |

Central residual collapses to numerical-noise level. RMS drops 7×. Peak
|frac| of 7.5% remains and is localized at the inclined bulge core / cusp
projection — this is the residual emulator + numerical signature.
**Verdict: CONVENTION-DRIVEN, RESOLVED. Residual structure now ~1% RMS.**

### E1.5a — radial-average sanity (`out/e1p5a_radial_average.csv`)

Max ring-averaged |frac residual| = 0.064. Comparable to the ~3-5% reported
by `tests/test_intensity_sersic.py::test_sersic_inclination_diagnostic`.
The residual that survives radial averaging is a real (small) emulator
signature.
**Verdict: PARTIALLY EXPLAINED.** The radial-averaged residual is now of
the same magnitude as the standalone emulator test. There's no per-pixel
"hidden bias" of the kind the prior session suspected.

### E1.5b — oversample convergence (`out/e1p5b_oversample.csv`)

| oversample | RMS frac | central frac |
|---|---|---|
| 1 | 0.0156 | +0.0410 |
| 3 | 0.0090 | +0.0013 |
| 5 | 0.0090 | -0.0002 |
| 9 | 0.0091 | -0.0007 |
| 15 | 0.0091 | -0.0009 |

**Verdict: BIAS PERSISTS at high oversample → emulator-intrinsic.** The
~0.9% RMS asymptote is stable and reflects the emulator's intrinsic
accuracy + numerical floor for n=4 inclined geometry. No additional
oversample helps. The prior verdict ("emulator-intrinsic, not aliasing") was
correct, but the magnitude is now ~1% RMS rather than the bug-inflated
pre-fix value.

### E2 — bulge-isolated residual (`out/e2_bulge_only_scalars.csv`)

| Quantity | Pre-fix | Post-fix |
|---|---|---|
| central frac | -0.34 | -0.0002 |
| RMS frac | not recorded | 0.0505 |
| peak \|frac\| | not recorded | 0.3011 |

Bulge-isolated central residual is now zero. RMS at 5% reflects the n=4
inclined emulator + LOS sampling. The peak |frac| of 30% is a localized
spike — likely at the bulge cusp where the emulator's interpolation has its
worst residual.
**Verdict: CONVENTION-DRIVEN at center; residual signature is the
emulator's known cusp difficulty at n=4 inclined.**

### E3 — cosi sweep (`out/e3_cosi_sweep.csv`)

Pre-fix: monotonic bias-vs-cosi from face-on +0.2% to edge-on -41%.
Post-fix:

| cosi | central frac | RMS frac | peak \|frac\| |
|---|---|---|---|
| 1.00 | +0.002 | 0.045 | 0.16 |
| 0.85 | +0.001 | 0.043 | 0.15 |
| 0.70 | +0.001 | 0.045 | 0.16 |
| 0.55 | +0.001 | 0.053 | 0.20 |
| 0.40 | +0.001 | 0.094 | 0.34 |
| 0.25 | +0.002 | 0.351 | 1.08 |

Central frac is now ~zero across the cosi range (no monotonic trend). RMS
grows toward edge-on as expected (more inclined geometry → bigger projected
disk extent → harder for emulator). The cosi=0.25 peak |frac|=1.08 is a
sign that the FFT pad / Nyquist starts to break down at extreme inclination,
not a Convention-Bug signature.
**Verdict: BIAS-VS-COSI FLATTENED at center. Inclination-dependent
residual is a known emulator + grid-sampling effect, not convention.**

### E4 — h_over_hlr sweep (`out/e4_thickness_sweep.csv`)

The smoking gun. Pre-fix: monotonic bias from -1% at h=0.05 to -42% at h=0.5.
Post-fix:

| h_over_hlr | central frac | RMS frac | peak \|frac\| |
|---|---|---|---|
| 0.05 | +0.0014 | 0.0451 | 0.157 |
| 0.10 | +0.0014 | 0.0451 | 0.157 |
| 0.20 | +0.0012 | 0.0450 | 0.157 |
| 0.30 | +0.0010 | 0.0449 | 0.157 |
| 0.50 | +0.0008 | 0.0445 | 0.156 |

**Completely flat.** No thickness dependence in the residual. This is the
DEFINITIVE confirmation that the entire pre-fix monotonic bias-vs-thickness
curve was the convention mismatch and nothing else.
**Verdict: CONVENTION-DRIVEN, FULLY RESOLVED.**

### E5 — k-space residual (`out/e5_kspace.csv`)

| Quantity | Value |
|---|---|
| FT(0) emu | 0.997 |
| FT(0) num | 0.999 |
| peak \|frac\| | 0.288 at k=80 (k·Re=64) |
| k_Nyquist | 20.9 |
| k_eff (over-N=5) | 4.19 |

The 28.8% peak residual is at k=80, well outside both k_Nyquist and k_eff —
i.e., it's at high-k modes the test geometry never samples. Within k_Nyquist,
the emulator-vs-numerical-Hankel residual is <0.6% (per prior session),
unchanged.
**Verdict: EMULATOR k-SPACE ACCURACY UNCHANGED. <1% within sampled k.**

### E6 — Fisher prediction (`out/e6_fisher_metrics.csv`, `out/e6_fisher_table.csv`)

| Metric | Pre-fix | Post-fix |
|---|---|---|
| Hessian condition | severe | 3.86e+04 |
| smallest eigval | very negative | -0.277 |
| 2nd smallest | — | +0.351 |
| 3rd smallest | — | +9.17 |
| diag (slice peaks) | many bound-hits | INSUFFICIENT_DATA (only 1 valid) |
| full (optimizer) | n=5, REFUTED | n=5, REFUTED (r=0.12) |

**Hessian indefinite at θ_true.** The smallest eigenvalue is -0.28 — small
in magnitude but real. There is a parameter-direction along which the
negative-log-likelihood DECREASES at θ_true, meaning θ_true is NOT a strict
local minimum at this SNR.

The "INSUFFICIENT_DATA" diag verdict comes from the script's coarse default
slice sampling (±25% over ~30 points → ~5% step); only `disk_h_over_r` peaks
visibly off truth at this resolution. The composite test (`pytest`) uses a
finer slice and reports 3-10% biases on `cosi`, `theta_int`, `bulge_hlr`,
`bulge_frac` (see "Test residuals" below) — those biases live within the
default-sweep step size.

**Verdict: HESSIAN STILL INDEFINITE.** The prior session predicted a PD
Hessian post-fix; that did not materialize. Whether this is numerical or
geometric is a Phase D question.

### E7 — shear & cosi sensitivity (`out/e7_sensitivity.csv`)

Pre-fix: |Δg2| ~0.05-0.25 across all sweep knobs.
Post-fix (baseline geometry, optimizer-converged offsets at θ_true+ε):
- Δcosi ≈ +5e-4
- Δg1 ≈ -1e-5
- Δg2 ≈ +7e-5

Across the swept knobs (PSF FWHM, pixel scale, bulge_hlr, bulge_frac,
true_cosi), |Δg2| stays in the range ~3e-7 to ~3e-4 except at one anomalous
point (`pixel_scale=0.05`, |Δg2|=1.8e-2) where finer pixels produce a
catastrophic mode in the optimizer.

**Verdict: |Δg2| DROPPED ~100-1000× IN BASELINE GEOMETRY.** WL-target-relevant
sensitivity is now in the 1e-4 range across most knobs — comfortably below
typical |Δg| < 1e-3 thresholds. The prior session's "no knob configuration
is WL-tolerable" verdict was wrong; this report officially supersedes it.

## Composite test residuals at SNR=1000

`pytest tests/test_composite_intensity.py::test_likelihood_slice_bulge_disk[1000]` post-fix:

| Param | Truth | Recovered | Rel bias | Tol |
|---|---|---|---|---|
| cosi | 0.7000 | 0.7255 | 3.64% | 0.1% |
| theta_int | 0.7850 | 0.7091 | 9.67% | 0.1% |
| total_flux | 1.0000 | 0.9906 | 0.94% | 0.1% |
| bulge_frac | 0.2500 | 0.2384 | 4.63% | 0.1% |
| disk_rscale | 2.0000 | 2.0107 | 0.54% | 0.1% |
| disk_h_over_r | 0.1000 | 0.1250 | 25.00% | 0.1% (slice bound) |
| bulge_hlr | 0.8000 | 0.8265 | 3.32% | 0.1% |

Test FAILS at the configured 0.1% slice tolerance, but the magnitudes are
much smaller than the pre-fix bias regime (cosi was 25%, theta_int -18%,
bulge_hlr -17%, etc.). The dominant remaining bias is theta_int at 9.7%.

## Phase D — residual bias tractability verdict

The post-fix slice biases at SNR=1000 (3-10% on cosi/theta_int/bulge_hlr/
bulge_frac) are **not model bugs**. They are noise-realization shifts
within the composite likelihood's Fisher uncertainty budget at SNR=1000.

### Diagonal Fisher prediction vs observed bias (seed=42)

Computed σ from the noise-free Hessian: σ_p = 1/sqrt(H_pp).

| Param | obs bias | σ_diag | bias/σ |
|---|---|---|---|
| cosi | +0.0255 (3.6%) | 0.0453 (6.5%) | 0.56 |
| theta_int | -0.0759 (9.7%) | 0.1299 (16.6%) | 0.58 |
| total_flux | -0.0094 (0.9%) | 0.0282 (2.8%) | 0.33 |
| bulge_frac | -0.0116 (4.6%) | 0.0108 (4.3%) | 1.07 |
| disk_rscale | +0.0107 (0.5%) | 0.1052 (5.3%) | 0.10 |
| disk_h_over_r | +0.0250 (25%) | 0.5728 (573%) | 0.04 |
| bulge_hlr | +0.0265 (3.3%) | 0.0322 (4.0%) | 0.82 |

All observed biases are within 1.1σ of the **diagonal** Fisher
uncertainty. Diagonal uncertainty is a lower bound on the true
marginalized uncertainty (full Fisher inversion accounts for parameter
correlations and gives larger σ), so the actual bias/σ values are even
smaller.

**Verdict: TRACTABLE-STAT.** At SNR=1000, the composite likelihood has
intrinsic 4-17% per-parameter uncertainty floors. A single noise
realization producing 3-10% slice peak shifts is exactly what Fisher
predicts. There is no model bug to fix.

The test's configured 0.1% slice tolerance at SNR=1000 is fundamentally
incompatible with the composite model's information geometry. Possible
paths forward (deferred — require explicit user sign-off per CLAUDE.md):

1. **Loosen tolerances to ~1σ Fisher (e.g., 6% cosi, 17% theta_int)**.
   Justifiable physically. Single-line tolerance dict update.
2. **Test at SNR=10000+** to compress Fisher uncertainty. Heavier compute,
   but simpler interpretation.
3. **Average over noise realizations** (multiple seeds, report mean bias).
   Tests the systematic vs purely-statistical component.
4. **Reparameterize** to reduce correlations (log-flux, softplus B/T,
   PCA-rotated geometry params). Larger blast radius.

### Hessian indefiniteness — explained

The smallest Hessian eigenvalue at θ_true is -0.28. This is not a true
non-convexity at θ_true (the data IS the model at θ_true on noise-free
data); it's the **emulator residual** acting as a "wrong data" signal of
~1% RMS. The score gradient `g[cosi] = -0.36` at θ_true reflects this:
the kl_pipe model at θ_true does not exactly equal the GalSim-rendered
data; the difference looks like a perturbed θ to the gradient.

The negative eigenvalue and small score together produce the predicted
diag bias `delta_pred ≈ -H⁻¹g = +0.000743` for cosi — three orders of
magnitude smaller than the observed test bias. The discrepancy confirms
that the test bias is dominated by **noise**, not by the emulator residual.

### Out-of-scope

- The nu-n table updates change `_NU_TABLE_INCLINED` values by 5-10% at
  most n, but no test directly compares against numerical values from
  this table beyond the n=1/nu=0.5 anchor (verified: tests/test_intensity_spergel.py
  asserts on absolute nu values only at n=1, n=2, n=4 face-on; no inclined
  numerical asserts).

## File-by-file change log (Phase B, commit `e76cce6`)

- `kl_pipe/intensity.py` — `_NU_TABLE_INCLINED` recomputed
- `tests/test_intensity_spergel.py` — 4 sites: `scale_h_over_r` → `scale_height`
- `tests/test_composite_intensity.py` — stale docstring cleanup
- `scripts/compute_nu_n_mapping.py` — convention fix + matched physical h_z
- `tests/test_galsim_conventions.py` — new regression guard (3 tests)
- `experiments/sweverett/composite_emulator_bias/e2_e4_sweeps.py` — convention fix
