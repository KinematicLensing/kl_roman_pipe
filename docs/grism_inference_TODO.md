# Grism Inference — Status & Deferred Work

## Phase 1: shipped

### Likelihood Functions

- ✓ `_log_likelihood_grism(theta, obs: GrismObs, kl_model: KLModel)` — chi-squared on 2D dispersed image (`kl_pipe/likelihood.py`).
- ✓ `_log_likelihood_joint_photometry_grism(theta, obs_int, obs_grism, kl_model)` — combined imaging + grism likelihood (sum of independent log-likes).

### JIT helpers

- ✓ `create_jitted_likelihood_grism(kl_model, obs_grism)` (`kl_pipe/likelihood.py`).
- ✓ `create_jitted_likelihood_joint_photometry_grism(kl_model, obs_int, obs_grism, render_config_int=None)`.

### InferenceTask Factories

- ✓ `InferenceTask.from_grism_obs(model, priors, obs, meta_pars=None)` (`kl_pipe/sampling/task.py`).
- ✓ `InferenceTask.from_joint_photometry_grism_obs(model, priors, obs_int, obs_grism, meta_pars=None)`.

### Tests

- ✓ `tests/test_grism_likelihood.py` — unit tests (eval / JIT / grad / factory) + likelihood slice tests for `Ha_flux` at SNR ∈ {100, 1000} + `vcirc` + `vel_dispersion` at SNR=1000 + one smoke optimizer-recovery test (Ha_flux + vcirc + vel_dispersion).

## Phase 2: shipped — LSF refactor

- ✓ Drop `σ_inst` absorption from `SpectralModel.build_cube`. `sigma_eff` is now `vel_disp` only.
- ✓ Delete `roman_grism_R`, `SpectralConfig.lsf_mode`, `SpectralConfig.R_func`, `convolve_spectral` stub.
- ✓ Mirror change in numpy reference `kl_pipe/synthetic.py:generate_datacube_3d`.
- ✓ `vel_dispersion` recoverable in grism likelihood slice + smoke recovery tests.
- ✓ `tests/test_spectral_resolution.py` codifies the empirical gate: PSF+dispersion alone reproduces Roman `R = 461·λ_μm` within ±5%.

Empirical evidence drove the refactor: see `experiments/sweverett/lsf_gate_test/`. PSF+dispersion alone gives `R_measured/R_spec = 1.035` at λ_obs ≈ 1.30 μm; σ_inst-active broadened the detector FWHM by 48% on top — classic double-counting signature.

## Phase 1 known limitations still open (Phase 3)

1. **Photometric and emission centroids are shared** — both broadband image rendering and grism cube assembly use `kl_model.intensity_model`'s `int_x0`/`int_y0`. If the photometric and grism observations have independent astrometric solutions this is incorrect. See **`docs/plans/phase3_sourcemodel_refactor.md`**.

2. **Joint photometry+grism rendering shares `flux_theta_override` mechanism** — velocity PSF flux weighting uses the intensity model's params in the velocity grid's frame; latent bug if grids are not aligned. Phase 3 resolves via explicit `flux_weight_key` binding on VelocityObs.

3. **`_check_priors_fit_obs_rc` skipped for GrismObs** — the grid-adequacy validation that broadband inference runs is not run for grism. Phase 3 generalizes the validation to obs-type-aware.

## Deferred — Phase 3: SourceModel refactor

See **`docs/plans/phase3_sourcemodel_refactor.md`** for the full plan. Summary:

- Replace `KLModel` with `SourceModel(velocity_model, broadband_models, emission_lines)`.
- Per-component centroids (`F087.x0`, `Halpha.x0`, `vel.x0`) — fixes the Phase 1 centroid coupling.
- Multi-line emission with shared or independent spatial profiles via `EmissionLine(intensity= | intensity_key=)`.
- Multi-band photometry via `broadband_models` dict.
- Per-line continuum via `EmissionLine.continuum`.
- Unified `InferenceTask.from_obs(source, priors, image_obs=, grism_obs=, velocity_obs=)` factory replaces all 5+ typed factories.
- Hard break — `KLModel`, `SpectralModel`, and the legacy `from_*_obs`/`from_*_model` factories are deleted.

PA-velocity degeneracy break design (Outini & Copin 2020, Eq. 1) — handled in Phase 3 via joint photometry+grism with the SourceModel architecture: shared geometric priors break the degeneracy.

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
