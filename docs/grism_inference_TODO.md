# Grism Inference — Status & Deferred Work

Status: Phases 1-3 all shipped. The current public API is `SourceModel` +
the unified `InferenceTask.from_obs(...)` factory; `KLModel`, `SpectralModel`,
and the typed `from_*_obs`/`from_*_model` factories are deleted. The sections
below are kept as a build record; the Phase-1/2 API names refer to code that
Phase 3 has since replaced.

## Phase 1: shipped

### Likelihood Functions

- ✓ `_log_likelihood_grism(theta, obs: GrismObs, kl_model: KLModel)` — chi-squared on 2D dispersed image (`kl_pipe/likelihood.py`).
- ✓ `_log_likelihood_joint_photometry_grism(theta, obs_int, obs_grism, kl_model)` — combined imaging + grism likelihood (sum of independent log-likes).

### JIT helpers

- ✓ `create_jitted_likelihood_grism(kl_model, obs_grism)` (`kl_pipe/likelihood.py`).
- ✓ `create_jitted_likelihood_joint_photometry_grism(kl_model, obs_int, obs_grism, render_config_int=None)`.

### InferenceTask Factories

Phase 1 shipped typed `from_grism_obs` / `from_joint_photometry_grism_obs`
factories. Both are deleted; Phase 3 replaced them with the unified
`InferenceTask.from_obs(source, priors, image_obs=, grism_obs=, velocity_obs=)`
(`kl_pipe/sampling/task.py`).

### Tests

- ✓ `tests/test_grism_likelihood.py` — unit tests (eval / JIT / grad / factory) + likelihood slice tests for `Halpha.flux` + `vcirc` + `Halpha.dispersion` + one smoke optimizer-recovery test. (Slice tests are now noiseless three-gate; see `docs/plans/TEST_TIER_REDESIGN.md`.)

## Phase 2: shipped — LSF refactor

- ✓ Drop `σ_inst` absorption from `SpectralModel.build_cube`. `sigma_eff` is now `vel_disp` only.
- ✓ Delete `roman_grism_R`, `SpectralConfig.lsf_mode`, `SpectralConfig.R_func`, `convolve_spectral` stub.
- ✓ Mirror change in numpy reference `kl_pipe/synthetic.py:generate_datacube_3d`.
- ✓ `vel_dispersion` recoverable in grism likelihood slice + smoke recovery tests.
- ✓ `tests/test_spectral_resolution.py` codifies the empirical gate: PSF+dispersion alone reproduces Roman `R = 461·λ_μm` within ±5%.

Empirical evidence drove the refactor: see `experiments/sweverett/lsf_gate_test/`. PSF+dispersion alone gives `R_measured/R_spec = 1.035` at λ_obs ≈ 1.30 μm; σ_inst-active broadened the detector FWHM by 48% on top — classic double-counting signature.

## Phase 1 limitations — resolved in Phase 3

1. **Photometric and emission centroids** — Phase 1 shared a single intensity
   centroid across broadband and grism. Resolved: per-component dotted
   centroids (`F087.x0`, `Halpha.x0`, `vel.x0`) via `SourceModel`.

2. **Joint photometry+grism flux weighting** — the Phase-1 `flux_theta_override`
   coupling is resolved by explicit `flux_weight_key` binding on `VelocityObs`
   (`kl_pipe/observation.py`, consumed in `kl_pipe/source.py`).

3. **Grid-adequacy validation for GrismObs** — resolved: `_check_source_priors_fit_obs`
   (`kl_pipe/sampling/task.py`) runs per channel, grism included.

## Phase 3: shipped — SourceModel refactor

See **`docs/plans/phase3_sourcemodel_refactor.md`** for the original plan. Shipped:

- `KLModel` replaced by `SourceModel(velocity_model, broadband_models, emission_lines)` (`kl_pipe/source.py`).
- Per-component centroids (`F087.x0`, `Halpha.x0`, `vel.x0`) — fixed the Phase 1 centroid coupling.
- Multi-line emission with shared or independent spatial profiles via `EmissionLine(intensity= | intensity_key=)` (`kl_pipe/lines.py`).
- Multi-band photometry via `broadband_models` dict.
- Per-line continuum via `EmissionLine.continuum`.
- Unified `InferenceTask.from_obs(source, priors, image_obs=, grism_obs=, velocity_obs=)` factory replaced all typed factories.
- Hard break — `KLModel`, `SpectralModel`, and the legacy `from_*_obs`/`from_*_model` factories are deleted.

PA-velocity degeneracy break (Outini & Copin 2020, Eq. 1) — handled via joint photometry+grism with shared geometric priors.

## Not on the Phase 1–3 roadmap

These were sketched in earlier planning but are not part of the current three-phase plan; revisit if science demands:

- `log_likelihood_cube` (3D datacube likelihood, IFU-style) — kl_pipe targets dispersed-grism inference; an explicit cube likelihood is not needed for current pipeline scope.
- `SyntheticGrism` class — current tests use direct `kl_model.render_grism(theta_true) + noise`. A dedicated synthetic class can be added if/when test patterns demand reuse.
- Sampling diagnostics (numpyro NUTS trace plots, corner plots) for the grism channel — current tests ship optimizer-based smoke recovery only; full sampling infrastructure can be wired up when production runs begin.

## Comparison with geko

geko's slitless-grism kinematics fitter (`astro-geko` v1.0.0) was audited during Phase 1 planning. Notes:

- geko also absorbs σ_inst into the line profile (`grism.py:711-714`) — same double-count kl_pipe just dropped. Post-Phase 2, kl_pipe and geko will systematically disagree on cube-level line widths; the detector-level images should still roughly agree if σ_inst was indeed double-counting (gate test confirms it was). Tracked in #50.
- geko has dead `compute_lsf_new` and `full_kernel` paths (`grism.py:564-629, 670`) suggesting an abandoned refactor toward unified PSF·LSF kernel — neither approach is in production geko.
- geko fits grism-only (priors from external PySersic fits on imaging); no joint phot+grism inference. kl_pipe Phase 1 ships joint phot+grism; Phase 3 expands it.

Cross-validation `make render-validation-*` needs re-baselining post-Phase 2; tracked in #50.
