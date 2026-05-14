# Composite emulator bias diagnosis — report

> ⚠ **SUPERSEDED** — see [`report_post_fix.md`](report_post_fix.md) for the
> post-fix re-run (commit `e76cce6`). All numbers below were computed at
> `026f94a`, before the convention fix landed in `871345a` + `e76cce6`. They
> are kept for the historical record. The qualitative diagnosis flagged at
> the top of this document (convention mismatch, NOT emulator bias) is
> confirmed; quantitative numbers should be read from the post-fix report.

**Date**: 2026-05-06
**Branch**: `se/inclined-sersic` (commit `026f94a` + experiment scripts)
**Goal**: build a quantitative bridge from the per-pixel residual map to the observed parameter biases (slice peaks + optimizer recovery), via Fisher / linearized analysis. Then sweep experimental knobs to predict shear and cosi bias for weak-lensing science.

---

## ⚠ CRITICAL UPDATE — root cause is a parameter-convention mismatch, not an emulator bias

A post-experiment convention check (`check_h_convention.py`) reveals that the "Miller-Pasha n=4 inclined emulator bias" is almost entirely a parameter-convention mismatch between kl_pipe and the GalSim ground truth used in the test:

- **kl_pipe** treats `int_h_over_hlr` as `h_z / half_light_radius` → for the test setup (`hlr=0.8`, `int_h_over_hlr=0.3`), `h_z_klpipe = 0.24` arcsec.
- **GalSim** (per its own docs): when `half_light_radius` and `scale_h_over_r` are both supplied, `scale_h_over_r` is interpreted as `h_z / scale_radius`, where `scale_radius = R_e / b_n^n`. For n=4, `b_4^n ≈ 3463`, so `h_z_galsim ≈ 6.93e-5` arcsec — essentially zero.
- That's a **~3500× physical thickness mismatch.**

When kl_pipe is rendered at `int_h_over_hlr = 8.7e-5` (matching GalSim's actual physical h_z), the central frac residual collapses from -25% to **+0.14%**. Equivalently, when GalSim is given `scale_height=0.24` directly (matching kl_pipe's physical h_z), the central frac residual is **+0.10%**.

Both cases show the **emulator + 3D inclined rendering pipeline is accurate** when the comparison is at matched physical thickness. The diagnostic is correct ("kl differs from gs by 25-31%"), but the attribution was wrong.

| Case | central kl/gs | central frac | RMS frac |
|---|---|---|---|
| A) test default (mismatched physical h_z) | 0.749 | -25.1% | 6.1% |
| B) kl matched to gs (h_over_hlr=8.7e-5) | 1.001 | +0.14% | 4.5% |
| C) gs matched to kl (scale_height=0.24 arcsec) | 1.001 | +0.10% | 4.5% |

### Implications for E0-E7 below

- **E0a face-on agreement**: still correct (face-on is independent of h_z; sini=0 nullifies the thickness factor)
- **E1, E2 residuals at test geometry**: real, but caused by convention mismatch
- **E1.5b oversample sweep**: "emulator-intrinsic, not aliasing" verdict still correct, but "intrinsic to the kl_pipe convention used", not "intrinsic to the emulator math"
- **E3 cosi sweep**: bias-vs-cosi monotonic; explained by sini gating the thickness mismatch (not by inclination of a correct profile)
- **E4 thickness sweep**: bias-vs-`int_h_over_hlr` is the smoking gun — kl_pipe's effective h_z grows linearly while GalSim's stays at scale_radius-scale; they diverge as h grows
- **E5 k-space residual**: <0.6% within k_Nyquist — confirms the radial Hankel emulator is fine
- **E6 Fisher analysis**: Hessian indefiniteness at θ_true and the shape-but-not-magnitude diag verdict are *real signatures* of the convention mismatch, not of model failure. After convention fix, expect θ_true to become a local minimum and Fisher to match.
- **E7 shear & cosi sensitivity**: |Δg2| ~0.05-0.25 across all knobs is **also driven by the convention mismatch**. After convention fix, the WL-target analysis must be redone — the "no knob configuration is WL-tolerable" conclusion likely does NOT hold.

### Recommended fix path

Option 1 (test-side): change `tests/test_composite_intensity.py:_generate_galsim_composite` to pass GalSim `scale_height=int_h_over_hlr * hlr` (physical h_z in arcsec) instead of `scale_h_over_r=int_h_over_hlr`. This makes the test compare physically equivalent disks. Composite recovery tests likely pass after this, and the strict-xfail decorators can lift.

Option 2 (kl_pipe-side): rename `int_h_over_hlr` to `int_h_over_rscale` and divide by `b_n^n` internally to match GalSim's convention. Larger blast radius (production code change), affects all callers.

Option 1 is the minimal-blast-radius fix. Recommended.

### Where else this convention may bite

- `InclinedExponentialModel` (n=1, b_1^1 ≈ 1.68): smaller mismatch (~1.7×) but still systematic. Existing `test_intensity.py` tests use the same convention pattern — likely have a smaller version of this bias hidden in their tolerances.
- `InclinedSpergelModel`, `InclinedDeVaucouleursModel` (n=4 effective): same mismatch as the n=4 Sersic.
- Any other code path that round-trips `int_h_over_hlr` against GalSim ground truth.

The fix should be propagated everywhere this comparison happens.

---

## Setup

Geometry (from `tests/test_composite_intensity.py`, except E7 which overrides shear):
- `BulgeDiskModel(shared_centroids=True)` — exponential disk + n=4 Sersic bulge
- `pars`: cosi=0.7, theta_int=0.785 rad, total_flux=1.0, B/T=0.25, disk_rscale=2.0, disk_h_over_r=0.1, bulge_hlr=0.8, bulge_h_over_hlr=0.3
- `image_pars`: 64×64 grid, pixel_scale=0.15 arcsec
- PSF: `gs.Gaussian(fwhm=0.15)`
- Variance: spatially uniform, total noise std = total_flux/1000 (matches `add_intensity_noise(include_poisson=False, target_snr=1000)`)
- E7 only: g1=0.02, g2=0.0 to exercise the cosi-shear degeneracy
- All renders use oversample=5 (default) with PSF + pixel-response on

## Findings

### E0a — radial intensity profiles (`out/e0a_radial_profiles.png`)

- **Face-on (cosi=1.0)**: kl_pipe and GalSim agree to <1% across the full radial range. The emulator's *radial profile* is essentially correct face-on.
- **Inclined cosi=0.7, major-axis cut**: kl_pipe is ~25% UNDER GalSim at center, monotonically converging to truth at large r.
- **Inclined cosi=0.7, minor-axis cut**: kl_pipe is ~60% OVER GalSim at center (sharp positive spike), then converges.

Interpretation: the emulator's *projected* shape is anisotropic-wrong. Along the minor axis (compressed direction), kl_pipe over-predicts the central intensity; along the major axis (uncompressed), it under-predicts. The face-on agreement rules out a face-on radial profile error — the bias is in the inclined LOS integration.

### E0b — per-parameter Jacobian images (`out/e0b_jacobian_images.png`, `out/e0b_scores.csv`)

Visual Fisher: `g_p = Σ residual · ∂M/∂θ_p / σ²` is the spatial overlap of the residual map with each parameter's Jacobian image. The Jacobians show:

- `cosi`, `theta_int`: dipole/quadrupole patterns sensitive to inclined geometry
- `bulge_hlr`, `bulge_frac`, `total_flux`: monopole-ish patterns sensitive to bulge size/normalization

Score values at θ_true (from `e0b_scores.csv`):

| param | g_p |
|---|---|
| cosi | -1.51e+01 |
| theta_int | -1.82e-03 |
| bulge_hlr | +1.57e+02 |
| bulge_frac | -2.88e+02 |
| total_flux | -7.54e+01 |

The bulge_frac score is largest in magnitude — the residual aligns most strongly with the bulge_frac Jacobian. Confirms that the residual structure is dominated by the bulge component.

### E0c — flux + central pixel scalars

| quantity | kl/gs ratio |
|---|---|
| total_flux | 1.000915 |
| central_pixel | 0.6859 |

Total flux agrees to 0.1% (volume integral conserved). Central pixel kl/gs = 0.69 → emulator under-predicts the peak by 31%. Confirms the handoff's "same total flux, lower peak" claim.

### E1 — 2D residual at test geometry (`out/e1_residual_maps.png`)

- Central frac residual: -31.4%
- RMS frac (over all pixels): 2.4%
- Peak |frac|: 52.5% (at the central pixel; clipped to ±0.5 in the plot)

The 2D residual has a mostly monopole-ish structure at the center (under-predict) with quadrupole-flavored sidelobes along the projected major/minor axes. The 1D cuts (panels E, F) show the spatial structure clearly.

### E1.5a — radial-average sanity check

Max ring-averaged |frac residual|: **0.195 (19.5%)**. Higher than the ~3-5% reported by `test_sersic_inclination_diagnostic` because (a) our test geometry has bulge_hlr=0.8 (smaller, more concentrated than that test's hlr=2.0) and (b) the inclined sech² thickness h/hlr=0.3 is larger than the diagnostic baseline. The order-of-magnitude is consistent — our setup is methodologically sound, the larger number reflects the different geometry, not a bug.

### E1.5b — oversample convergence (`out/e1b_oversample_convergence.png`)

| oversample | RMS frac | central frac |
|---|---|---|
| 1 | 0.02623 | -0.2858 |
| 3 | 0.02353 | -0.3131 |
| 5 | 0.02359 | -0.3141 |
| 9 | 0.02361 | -0.3145 |
| 15 | 0.02362 | -0.3146 |

**Verdict**: bias persists at high oversample → emulator-intrinsic, NOT aliasing. RMS converges by oversample=3 and is flat through oversample=15. Going from 1→3 closes a small ~10% portion of the residual; everything beyond is the emulator itself. Rules out "just oversample more" as a fix and confirms the prior session's attribution.

### E2 — bulge-isolated residual (`out/e2_bulge_only_residual.png`)

With bulge_frac=1.0 (disk component zeroed):

| quantity | value |
|---|---|
| central frac | -0.340 |
| RMS frac | 0.065 |
| peak |frac| | 0.761 |

Bulge-only RMS is 2.7x the composite (0.065 vs 0.024) because the disk dilutes the average. Central residual goes from -31% (composite) to -34% (bulge-only) — disk contribution is small at center but non-zero. Confirms bulge dominates the residual.

### E3 — cosi sweep, bulge-only (`out/e3_cosi_sweep.png`, `out/e3_cosi_sweep.csv`)

| cosi | central frac | RMS frac | peak |frac| |
|---|---|---|---|
| 1.00 (face-on) | +0.002 | 0.045 | 0.159 |
| 0.85 | -0.159 | 0.048 | 0.284 |
| 0.70 (test) | -0.251 | 0.061 | 0.614 |
| 0.55 | -0.316 | 0.088 | 1.035 |
| 0.40 | -0.366 | 0.154 | 1.719 |
| 0.25 (near edge-on) | -0.408 | 0.441 | 3.36 |

**Bias is monotonic with inclination.** Face-on is essentially correct; near-edge-on the central pixel is under-predicted by 41% with peak frac residuals exceeding 3 (i.e., kl_pipe is >3x off GalSim at some pixels). The bias scales primarily with how much the disk gets compressed, consistent with E0a's anisotropic-projection finding.

### E4 — h_over_hlr sweep, bulge-only (`out/e4_thickness_sweep.png`, `out/e4_thickness_sweep.csv`)

| h_over_hlr | central frac | RMS frac | peak |frac| |
|---|---|---|---|
| 0.05 | -0.010 | 0.045 | 0.158 |
| 0.10 | -0.041 | 0.046 | 0.158 |
| 0.20 | -0.143 | 0.050 | 0.317 |
| 0.30 (test) | -0.251 | 0.061 | 0.614 |
| 0.50 | -0.421 | 0.105 | 1.214 |

**Bias scales strongly with thickness.** Thin (h/hlr=0.05): -1% central; thick (h/hlr=0.5): -42% central. **The bias is in the LOS integration**, not the radial FT alone — confirmed because varying thickness while holding the radial profile fixed changes the bias dramatically.

### E5 — k-space residual emulator vs numerical (`out/e5_kspace_residual.png`)

For n=4 Sersic FT compared to GalSim's numerical Hankel:

- |frac residual| within k_Nyquist ≈ 21 rad/arcsec: **<0.6%**
- |frac residual| at k = 4*k_Nyquist ≈ 84 rad/arcsec: **29%**
- Peak |frac residual| at k*R_e ≈ 64 (high-k tail)

The emulator is accurate within the sampled-grid Nyquist, but increasingly biased at higher k. With oversample=5, the rendering integrates k up to k_eff*5 ≈ 105 rad/arcsec, so the high-k under-shoot DOES contribute to the rendered image's central pixel.

### E6 — Fisher / linearized bias prediction (`out/e6_fisher_table.png`, `out/e6_fisher_table.csv`)

Computed g and H at θ_true on noise-free data with SNR=1000 variance.

**Hessian condition number**: 1.46e+03 (well-conditioned for direct solve).

**Smallest 3 Hessian eigenvalues**: -398.7, -136.1, +6.85.

**THE HESSIAN IS INDEFINITE AT θ_TRUE** (two negative eigenvalues). This means θ_true is a *saddle point* of the noise-free NLL, not a local minimum. The model bias is so severe that θ_true does not even sit at a stable point of the inferred-likelihood landscape — moving in some directions strictly improves the fit.

#### Diagonal Fisher (vs slice peaks)

n=4 valid params (theta_int, total_flux, bulge_frac, disk_rscale, bulge_hlr; theta_int dropped by zero-shift filter):

| Param | truth | slice peak | Δp_obs | Δp_diag (pred) |
|---|---|---|---|---|
| total_flux | 1.000 | 1.060 | +0.060 | +0.060 (1.0×) |
| bulge_frac | 0.250 | 0.284 | +0.034 | +0.034 (1.0×) |
| disk_rscale | 2.000 | 1.955 | -0.045 | -0.046 (1.0×) |
| bulge_hlr | 0.800 | 0.650 | -0.150 | -0.290 (1.93×) |

- Pearson r = **0.979** ✓ (≥ 0.95)
- Slope = **1.716** ✗ (target [0.8, 1.25])
- RMS rel err = **0.465** ✗ (target ≤ 0.20)

**Verdict: PARTIALLY supported** — shape correlates well, but for the largest bias (bulge_hlr), Fisher over-predicts by ~2×, indicating non-quadratic likelihood beyond ~10% bias.

#### Full Fisher (vs multi-start optimizer recovery)

n=5 valid params from prior session's reported optimizer values:

| Param | truth | optimizer | Δθ_obs | Δθ_full (pred) |
|---|---|---|---|---|
| cosi | 0.700 | 0.755 | +0.055 | +0.085 |
| theta_int | 0.785 | 0.165 | -0.620 | -0.0001 |
| bulge_frac | 0.250 | 0.139 | -0.111 | -0.003 |
| disk_rscale | 2.000 | 1.804 | -0.196 | +0.215 (wrong sign!) |
| bulge_hlr | 0.800 | 0.373 | -0.427 | -0.201 |

- Pearson r = **0.456** ✗
- Slope = **0.079** ✗
- RMS rel err = **1.176** ✗

**Verdict: REFUTED** — the linear analysis cannot predict the multivariate optimizer bias because the optimizer escapes the local quadratic and finds a different basin (the bulge_frac→0 / theta_int rotation attractor cited in the prior handoff). For disk_rscale the Fisher prediction even has the wrong SIGN — a clean signature of non-quadratic / multi-modal behavior.

#### Bound-hit re-scan

Three params hit the slice scan boundary even at ±50% width:
- `cosi`: peak at 0.99 (physical bound). Conditional MLE preference: cosi → 1 (face-on). Δ ≥ +0.29.
- `disk_h_over_r`: peak at 0.05 (lower scan bound). Δ ≤ -0.05.
- `bulge_h_over_hlr`: peak at 0.15 (lower scan bound). Δ ≤ -0.15.

Bound-hit params excluded from Pearson `r`. They confirm the bias is large enough to push some params to the edges of physical / scan space.

### E7 — shear & cosi bias sensitivity sweep (`out/e7_sensitivity.png`, `out/e7_sensitivity.csv`)

Diag Fisher predictions for {Δcosi, Δg1, Δg2} swept across 5 knobs at experiment default (g1=0.02 WL regime).

#### Headline numbers — shear bias is catastrophic across the explored space

The WL target for per-galaxy shear bias is roughly |Δg| ≤ 1e-3. **|Δg2| is consistently 50-250× above this target across the entire knob space**:

| knob | range of |Δg2| |
|---|---|
| PSF FWHM ∈ [0.05, 0.50] | 0.033 — 0.18 |
| pixel_scale ∈ [0.05, 0.30] | 0.11 — 0.24 |
| bulge_hlr ∈ [0.4, 2.0] | 0.090 — 0.13 |
| bulge_frac ∈ [0.05, 0.95] | **0.013 — 0.24** ← B/T-dependent |
| true_cosi ∈ [0.3, 0.95] | 0.036 — 0.21 |

|Δg1| is smaller (~1-5e-3) but still typically above the 1e-3 target.

#### B/T (bulge fraction) is the dominant knob

The single strongest dependency is on `bulge_frac`:

| bulge_frac | Δcosi | Δg2 |
|---|---|---|
| 0.05 (pure-disk) | -0.002 | +0.013 |
| 0.10 | +0.001 | +0.040 |
| 0.25 (test default) | +0.066 | +0.130 |
| 0.50 | -0.226 | +0.204 |
| 0.75 | -0.150 | +0.230 |
| 0.95 (pure-bulge) | -0.140 | +0.240 |

For pure-disk galaxies (B/T=0.05), |Δg2|=0.013 — 13× the WL target but the smallest in the sweep. As B/T grows above ~0.1 the bias rapidly worsens.

#### Other notable patterns

- **PSF FWHM**: sharper PSF → larger bias (cusp not smoothed enough). At FWHM=0.5, |Δcosi| < 0.025 — close to the cosi target — but |Δg2|=0.033 still ≫ target.
- **pixel_scale**: the finest pixel (0.05) blows up |Δcosi| to -1.19, presumably a numerical/aliasing instability at the extreme. Test value (0.15) gives moderate bias; coarser pixels (≥ 0.2) reduce |Δcosi| but |Δg2| stays large.
- **bulge_hlr**: |Δcosi| peaks at hlr=0.8-1.2 and drops at large hlr. |Δg2| is roughly constant 0.09-0.13.
- **true_cosi**: |Δcosi| peaks dramatically at cosi=0.85 (Δcosi = +0.55!) — possibly a numerical sensitivity near face-on where the sech² LOS integral becomes nearly degenerate with the radial profile. |Δg2| highest at intermediate inclinations.

## Verdict and interpretation

**E6 verdict** — does the per-pixel residual quantitatively explain the parameter bias?

- For the 1D conditional (slice peak) regime with biases ≤ ~5%: **YES** (diag Fisher matches slice peaks at slope ≈ 1).
- For larger biases (~15% on bulge_hlr) or multivariate joint MLE: **NO** — the linearization breaks. Hessian indefiniteness at θ_true confirms that the model bias is so large that the noise-free NLL is non-quadratic / multi-modal in this neighborhood.

**E7 verdict** — is there a "safe" WL regime?

- For `bulge_frac ≤ 0.05`: |Δg2| ~ 0.013 — best in sweep but still 13× the target.
- **No combination of the explored knobs lands in the |Δg|<1e-3 WL-target regime.** The Miller-Pasha n=4 emulator's per-pixel bias produces a per-galaxy shear bias of order 1-25%, and an emulator-side improvement is required before this composite path is usable for production weak-lensing inference on bulge-bearing galaxies.
- For pure-disk subsamples (B/T < 0.05), the bias is much smaller — pure-disk WL inference may be tractable as a near-term science case.

## Methodology notes

1. **Noise-free Fisher with SNR=1000 variance**: deterministic prediction, isolates the model bias from noise scatter. Justified because at SNR=1000, per-pixel bias (~2-4% of central) dominates noise std (~0.1% of central) by ~30:1.
2. **Slice peaks recomputed on noise-free likelihood** (not from prior session's noisy seed=42 numbers) — apples-to-apples comparison with diag Fisher.
3. **Optimizer comparison uses prior session's reported numbers** — the multi-start L-BFGS-B run is expensive to re-execute; the prior numbers serve as a fixed baseline. Caveat: those were noisy data with seed=42; the comparison is approximate but the magnitudes dwarf the SNR=1000 noise so the conclusions hold.
4. **Pearson r + slope + RMS rel err triple metric**: r alone tests *shape*, slope tests *calibration*, RMS rel err tests *magnitude*. All three are required for a "supported" verdict.

## What was NOT done (anti-scope)

- No emulator code changes
- No xfail decorator changes
- No tolerance changes
- No SNR=50 catastrophic-mode investigation
- No proposed fix for the n=4 inclined emulator bias
- E1.5b oversample sweep flagged "emulator-intrinsic, not aliasing" — no fix proposed; the result is the result.

## Outputs

All in `out/`:

- `e0a_radial_profiles.png`, `e0b_jacobian_images.png`, `e0b_scores.csv`, `e0c_scalars.csv`
- `e1_residual_maps.png`, `e1_residual_scalars.csv`
- `e1p5a_radial_average.csv`, `e1b_oversample_convergence.png`, `e1p5b_oversample.csv`
- `e2_bulge_only_residual.png`, `e2_bulge_only_scalars.csv`
- `e3_cosi_sweep.png`, `e3_cosi_sweep.csv`
- `e4_thickness_sweep.png`, `e4_thickness_sweep.csv`
- `e5_kspace_residual.png`, `e5_kspace.csv`
- `e6_fisher_table.png`, `e6_fisher_table.csv`, `e6_fisher_metrics.csv`, `e6_hessian_spectrum.png`
- `e7_sensitivity.png`, `e7_sensitivity.csv`
