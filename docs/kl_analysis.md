# Modeling Complexity Strategy for kl_pipe

## Context

The pipeline currently fits velocity (arctan RC) + intensity (3D inclined exponential) through 5-plane coordinate transforms to extract weak lensing shear from galaxy kinematics. TNG50 testing reveals **10-30% model-data amplitude mismatch** — consistent with Donet & Wittman (2023) finding gamma_cross noise floor ~0.08 from model misspecification alone on TNG100. This model-induced shear bias likely dominates over statistical uncertainty at Roman S/N.

The goal is: **accurate shear estimates** (|m| < 0.01, |c| < 0.001). Everything below is evaluated against this goal.

---

## Audit Summary

### What works well
- Stateless pure-functional JAX architecture — clean extensibility
- 5-plane transform chain is correct and well-tested
- 3D sech² intensity model matches GalSim to ~1e-5 (k-space path)
- K-space pixel integration (sinc + wrap) achieves <0.02% vs GalSim; PSF fused in single FFT pass
- Three-tier test hierarchy (likelihood slices → optimizer → MCMC) with SNR-dependent tolerances
- NumPyro NUTS w/ Z-score reparam handles 14-dim joint posteriors robustly

### Where it breaks
| Problem | Evidence | Shear impact |
|---------|----------|-------------|
| Arctan RC too rigid | TNG residuals 10-30%; can't model bars/warps/multi-component mass | Dominant systematic (Donet & Wittman gamma_cross ~0.08) |
| Forced shared PA | TNG gas-stellar misalignment 10-40° | PA-shear degeneracy biases g1/g2 |
| No non-circular motions | cos(phi) projection only; m=2 harmonics alias to shear | Unmodeled streaming → additive shear bias |
| Single exponential intensity | No bulge/clumps; n=1 fixed | Biases shared geometric params (cosi, theta_int) |
| Gaussian-only noise | No pixel correlations from drizzling | Overconfident posteriors → underestimated shear uncertainty |
| No masks | Can't handle bad/missing pixels | Blocks real data usage |
| cosi→0 singularity | 1/cos(i) diverges; epsilon=1e-10 guard biases gradients | Edge-on galaxies unfittable |
| No shear calibration tests | m,c bias never measured directly | Can't quantify if changes help shear |

### Key degeneracies
- **cosi-vcirc**: Fundamental (v_LOS = vcirc × sin(i) × cos(phi)). Only breakable with intensity constraints on cosi.
- **shear-PA**: theta_int rotation mixes with g1/g2. Split PA partially breaks this.
- **m=2 harmonics-shear**: sin(2phi) non-circular motions are near-degenerate with g2. Dangerous.
- **flux-h_over_r**: Vertical thickness unmeasurable at low inclination.

---

## Literature-Informed Approaches

### Approaches ranked by shear-recovery impact / effort

#### Tier 1: High impact, feasible now

| # | Approach | What it does | Shear benefit | Effort |
|---|----------|-------------|---------------|--------|
| **1** | **Shear calibration test infrastructure** | Measure m,c bias directly: g_meas = (1+m)g_true + c | Enables quantifying all other improvements | Low |
| **2** | **Mask support** (already on roadmap) | Exclude bad/missing pixels from likelihood | Unblocks real data | Low |
| **3** | **Polyex rotation curve** | v(R) = V0(1-exp(-R/r))(1+alpha*R/r); adds 1 param (outer slope) | Reduces RC shape mismatch; Sofue (2016) | Low |
| **4** | **Split kinematic/morphological PA** | vel_theta_int ≠ int_theta_int in joint model | Breaks PA-shear degeneracy; TNG shows 10-40° offset | Low |
| **5** | **Intensity-dependent velocity variance** | variance_vel(x,y) ∝ 1/I(x,y) | Correct weighting; already supported by likelihood signature | Low |

#### Tier 2: High impact, moderate effort

| # | Approach | What it does | Shear benefit | Effort |
|---|----------|-------------|---------------|--------|
| **6** | **Kinematic metacalibration** | Self-calibrate shear: R = d(g_hat)/d(g_true), then g_cal = g_hat/R | Calibrates ALL bias sources simultaneously; Sheldon & Huff (2017) adapted to KL | Med-High |
| **7** | **Pixel covariance likelihood** | Fourier-space chi2 with noise power spectrum P(k) | Honest uncertainties for drizzled data; Gurvich & Mandelbaum (2016) | Med |
| **8** | **MGE intensity model** | Sum of N Gaussians: analytic FT, analytic deprojection | Better geometric param constraints in joint fit; Cappellari (2002) | Med |

#### Tier 3: Research frontier

| # | Approach | What it does | Shear benefit | Effort |
|---|----------|-------------|---------------|--------|
| **9** | **MIRoRS symmetry restoration** | Find shear that restores velocity field symmetry — no parametric model | Avoids all parametric model bias; Hopp & Wittman (2024) | High |
| **10** | **Harmonic velocity decomposition** | v_LOS = Σ(c_m cos(mφ) + s_m sin(mφ)); captures bars/warps | Models non-circular motions BUT m=2 degeneracy with shear | High |
| **11** | **SBI with normalizing flows** | Train NPE on TNG-like simulations → amortized posterior | Implicitly handles misspecification; sbijax/FlowJAX (JAX-native) | High |
| **12** | **flowMC sampling** | NF-enhanced MCMC for multi-modal posteriors | Better exploration of banana-shaped cosi-vcirc degeneracy; Wong+ (2023) | Med |

---

## Recommended Strategy (Phased)

### Phase 0: Measurement infrastructure (prerequisite)
**Build shear calibration test framework** — without this, we can't quantify if anything helps.

- New test module: `tests/test_shear_calibration.py`
- Generate synthetic galaxies at multiple input shears g_true ∈ [-0.1, 0.1]
- Fit each, record g_recovered
- Linear regression: g_recovered = (1+m) × g_true + c
- Pass criteria: |m| < 0.01, |c| < 0.001 (Roman WL science req)
- Run for: arctan model (baseline), then each improvement
- Also run on TNG50 galaxies with known zero shear → any recovered shear = bias from model misspecification

Files: new `tests/test_shear_calibration.py`, extend `tests/test_utils.py` with `measure_shear_bias()` helper

### Phase 1: Low-hanging fruit (unblocks real data + reduces known biases)

**1a. Mask support in likelihoods**
- Add `mask_vel`, `mask_int` (boolean 2D arrays) to all 3 likelihood functions
- Masked pixels: set residual to 0 before chi2 sum; adjust n_data = sum(mask)
- Freeze mask via `functools.partial` for JIT
- Files: `kl_pipe/likelihood.py`, `kl_pipe/sampling/task.py`
- Validation: unit tests (mask-all → constant logL; mask-half → half chi2)

**1b. Polyex rotation curve model**
- New `PolyexVelocityModel` subclass with `vel_alpha` param (outer RC slope)
- v(R) = vcirc × (1 - exp(-R/rscale)) × (1 + alpha × R/rscale)
- Register as `'polyex'` in factory
- Files: `kl_pipe/velocity.py`
- Validation: unit tests, likelihood slices, optimizer recovery, TNG comparison (arctan vs polyex residual chi2)

**1c. Split kinematic/morphological PA**
- New `SplitPAVelocityModel` subclass with `vel_theta_int` replacing `theta_int`
- New `SplitPAIntensityModel` subclass with `int_theta_int` replacing `theta_int`
- When combined in KLModel, `theta_int` is no longer shared → 1 extra param
- Files: `kl_pipe/velocity.py`, `kl_pipe/intensity.py`, possibly `kl_pipe/model.py` (KLModel shared-param logic)
- Validation: TNG fit with shared vs split PA; compare g1/g2 posterior widths and bias

**1d. Intensity-dependent velocity variance**
- Generate variance_vel = sigma_v² × (I_max / I(x,y)) from the intensity map
- Bright pixels → low variance (high weight); faint pixels → high variance
- Likelihood already accepts 2D variance arrays; just need to generate them
- Files: `kl_pipe/noise.py` (utility function), test helpers
- Validation: compare parameter recovery with constant vs intensity-dependent variance

### Phase 2: Calibration (the single highest-impact item for shear)

**2a. Kinematic metacalibration**
- Apply small artificial shear δg to observed data (both velocity + intensity maps)
- Re-fit at g ± δg; compute response R = Δg_hat / (2δg)
- Calibrated shear = g_hat / R
- **Key challenge**: applying shear to velocity map requires coordinate grid transformation + interpolation. The velocity field transforms as v_LOS(x',y') where (x',y') = M(δg) × (x,y). Interpolation on JAX grids.
- **PSF correction**: for small δg (~0.01), PSF change is second-order — test empirically
- Files: new `kl_pipe/metacal.py`
- Validation: apply to synthetic data with known g → metacal-corrected m,c should be < 0.01/0.001; apply to TNG with g=0 → bias should decrease

**NOTE**: This is uncharted territory for KL. Metacalibration has only been applied to photometric shapes (Sheldon & Huff 2017). Applying it to velocity fields requires careful theoretical work on how v_LOS transforms under lensing. The coordinate grid shear is straightforward; whether the v_LOS scalar field is invariant under the coordinate transform needs verification.

### Phase 3: Advanced models (if Phase 0 shows model misspecification still dominates)

**3a. MGE intensity model** — sum of N Gaussians
- Analytic k-space FT (product of Gaussians is trivial)
- Reuse `_render_kspace` structure; each Gaussian adds 2 params (amplitude, sigma)
- Prior design: ordered sigma_1 < sigma_2 < ... to break label switching
- Files: new class in `kl_pipe/intensity.py`, register in factory

**3b. Harmonic velocity decomposition** — add m=1,2,3 Fourier terms
- v_LOS += Σ(c_m cos(mφ) + s_m sin(mφ)) in disk plane
- **WARNING**: m=2 term near-degenerate with g2 at low inclination. Must either: (a) use strong physical prior on m=2 amplitude, or (b) fit m=2 jointly with shear and accept wider posteriors
- Files: extend `VelocityModel.__call__` in `kl_pipe/model.py` or new subclass

**3c. Pixel covariance likelihood** — Fourier-space chi2
- chi2 = Σ_k |R_hat(k)|² / P(k) where P(k) = noise power spectrum
- For white noise: P(k) = const → reduces to standard chi2
- For drizzled: P(k) from drizzle kernel autocorrelation
- Files: `kl_pipe/likelihood.py`

### Phase 4: Model-independent shear (research)

**4a. MIRoRS-style symmetry estimator**
- Separate code path, not a Model subclass
- Objective: find g that maximizes velocity field reflection symmetry along major axis + anti-symmetry along minor axis
- Complements parametric pipeline; provides independent shear estimate for cross-validation
- Files: new `kl_pipe/symmetry.py`

---

## What NOT to do (and why)

| Approach | Why skip | Reference |
|----------|----------|-----------|
| Bulge+disk decomposition | KL targets are disk galaxies; bulge-dominated = no RC; doubles param count for minimal shear benefit | Haussler+ (2018) shows bulge params poorly recovered when barely resolved |
| Free Sersic index | n~1 is correct for KL target population; n adds degenerate params | |
| Full 3DBarolo tilted-ring | Requires 3D spectral cubes, not 2D maps; overkill for grism R~600 | Di Teodoro & Fraternali (2015) |
| Asymmetric drift correction | V/sigma > 10 for Roman KL targets; correction < 1% | Only matters at V/sigma < 5 |
| Chromatic PSF (now) | Important eventually but model misspecification dominates over PSF chromatic bias at current stage | Defer to medium-term roadmap |

---

## Validation Hierarchy (extended)

| Tier | Test | What it validates | Where |
|------|------|-------------------|-------|
| 0 (NEW) | **Shear calibration** | m,c bias directly | `test_shear_calibration.py` |
| 1 | Likelihood slices | Forward model pixel-level correctness | `test_likelihood_slices.py` |
| 2 | Optimizer recovery | Gradient-based param recovery | `test_optimizer_recovery.py` |
| 3 | Sampling diagnostics | Full MCMC convergence + posteriors | `test_sampling_diagnostics.py` |
| 4 | TNG diagnostics | Model-data mismatch on realistic galaxies | `test_tng_sampling_diagnostics.py` |

**For each new model/improvement**, run tiers 0-2 minimum. Tier 3-4 for production candidates.

---

## Open Questions

1. **What fraction of TNG residual is RC-shape vs non-circular?** — Answerable by fitting Polyex to TNG and comparing residuals to arctan. If residuals don't drop substantially, the problem is non-circular motions, not RC shape.

2. **Can velocity maps be metacalibrated?** — v_LOS is a scalar field on the sky; under lensing, pixel positions transform but v_LOS values don't change (they're spectroscopic measurements). So interpolating v_LOS(M(δg)·x) should be correct. But the PSF-convolved velocity field IS affected because PSF acts in the image plane. Needs careful derivation.

3. **Does split PA help or hurt for well-aligned galaxies?** — Extra DOF adds noise when true offset is <5°. Could use model comparison (Nautilus evidence) to decide per-galaxy.

4. **m=2 harmonic vs shear degeneracy**: Is there ANY prior or auxiliary data that breaks this, or is it fundamentally degenerate? If degenerate, harmonic decomposition cannot be used for KL — it destroys the shear signal.

5. **Is the Donet & Wittman gamma_cross~0.08 floor reducible?** — Their model was simpler than ours (no 3D profile, no PSF). Our pipeline may already do better. Need to replicate their experiment with kl_pipe on TNG50.

6. **Polyex vel_alpha prior**: What range is physical? TNG50 RC outer slopes span ~[-0.1, 0.3]. Is this tight enough to be informative?
