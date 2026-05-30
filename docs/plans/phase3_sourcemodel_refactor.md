# Phase 3 — SourceModel refactor

Self-contained execution plan for the `se/source-model` branch. Replaces `KLModel(velocity_model, intensity_model, spectral_model)` with `SourceModel(velocity_model, broadband_models, emission_lines)` and unifies the inference factories into a single `InferenceTask.from_obs(...)`.

## Context

Phase 1 + 2 of the grism inference work ship under `KLModel`, which couples photometric (broadband stellar continuum) and emission-line (ionized gas) spatial distributions into a single `intensity_model`. Multi-line emission, multi-band photometry, per-line continuum, and per-component centroids are blocked by this coupling. PR #27 is in review for the Phase 1+2 work; Phase 3 is the planned follow-up.

This is a **hard break** refactor: deletes `KLModel`, `SpectralModel`, and the entire `from_*_obs` / `from_*_model` family in one PR (staged as 9 atomic commits). Phase 1 collaborators update their code at the cutover; Phase 1 was shipped with no API-stability obligation.

## Progress (as of 2026-05-28)

Branch: `se/source-model` (from `se/grism-core` HEAD `ccfb219`).

| Status | Commit | SHA | Notes |
|---|---|---|---|
| done | 1. SourceModel + EmissionLine + LINE_LAMBDAS + WCS rotation | `22617e4` | Includes Step 0 refresh of `SOURCE_MODEL_EXAMPLES.py`; `EmissionLine` + `LINE_LAMBDAS` later moved to `kl_pipe/lines.py` (see 1.5). `GrismPars.dispersion_angle_detector` rename with legacy alias landed here. |
| done | 1.5. refactor: move EmissionLine + LINE_LAMBDAS to `kl_pipe/lines.py` | `879415f` | Not in original plan -- pulled out to avoid name collision with the back-compat `EmissionLine = _LegacyEmissionLine` alias in `kl_pipe/spectral.py`. |
| done | 2. SourceModel rendering methods + per-obs rotation | `9704e43` | `render_broadband` / `render_velocity` / `build_cube` / `render_grism` all on SourceModel. Theta routing helpers `_strip_param_prefix` / `_lookup_param` / `_build_component_theta` / `_apply_obs_rotation` are module-level in `source.py`. |
| done | 3. `InferenceTask.from_obs` + per-channel `_log_likelihood_*` primitives | `9e3bb42` | SourceModel branch added to `InferenceTask.__post_init__` / `_build_full_theta` / `parameter_names`; legacy KLModel path unchanged. |
| done | 4. obs-type-aware `_check_priors_fit_obs_rc` + GrismObs→ImageObs `RenderConfig` parity (cleanup A, expanded) | `bfbe8c6` | Originally scoped to just the check; expanded after audit revealed GrismObs lacked `render_config` infrastructure. Added `render_config` + `psf` fields to GrismObs, threaded through `build_grism_obs`; uniform check path on `obs.render_config`. VelocityObs flux-weight auto-derive deferred; 2-step user pattern locked. |
| done | 5. grism cube-slice bandwidth-aware `for_grism_priors` + velocity-gradient bandwidth + Minkowski-sum math + flux/pixel convention enforcement + units doc + convergence tests | `0f820a3` | Implemented Minkowski-sum bandwidth bound `k_I + k_G` (not the originally-drafted product walk -- empirically underestimated). New: `VelocityModel.grad_bandwidth(params)` (arctan rotation curve default); `compute_effective_maxk_grism` in render.py; `RenderConfig.for_grism_priors` classmethod; `for_grism_priors(source, priors, ...)` in source.py; extended `_extract_worst_case_params` rules for `vcirc`/`vel_rscale`; moved `_component_priors_for_intensity` from sampling/task.py to source.py. Also surfaced + fixed a pre-existing flux non-conservation bug in `_apply_post_dispersion_pixel_response` (output scaled as N², now correctly returns flux/pixel via `mean × coarse_area`); new tracked `docs/units_and_conventions.md` + CLAUDE.md table updates. Tests: `tests/test_grism_bandwidth.py` with 4 tests (full-pipeline convergence, default-accuracy, cube-slice convergence isolated, visual diagnostics). LaTeX: `docs/notes/grism_cube_bandwidth.tex` (gitignored). |
| done | 6. move `_apply_post_dispersion_pixel_response` to `kl_pipe/grism.py` (cleanup B) | `bbb2085` | Mechanical move as scoped. New `kl_pipe/grism.py` (parallel to `kl_pipe/dispersion.py`); function body unchanged; import path updated at three callsites (SourceModel.render_grism, KLModel.render_grism, tests/test_cube_psf.py). 66 grism-touching tests pass. Frees `kl_pipe/model.py` of grism-specific plumbing before KLModel deletion in commit 10. |
| done | 7. top-level rc-derivation builders in render.py + obs-symmetric API (cleanup E, expanded) | `d9d8f0a` | Original "centralize 2 callsites" framing was misleading -- audit revealed both targets live in legacy factories deleted in commit 10. Pivoted to a real ergonomics gap: the user-facing API had `for_grism_priors` in `source.py` but no broadband analogue. Shipped: `build_image_render_config` and `build_grism_render_config` top-level functions in `kl_pipe/render.py` (parallel to `build_*_obs` factories); deleted public `for_grism_priors` from `source.py` (logic moved); `RenderConfig.for_priors` / `for_grism_priors` classmethods stay as internal primitives. Updated `_check_source_priors_fit_obs` GrismObs branch + observation.py docstrings to point at new entry points + SOURCE_MODEL_EXAMPLES.py with explicit rc-derivation recipe. 5 new unit tests covering happy paths + missing-band / missing-velocity / missing-line / missing-dispersion error cases. 86 tests pass in touched files. |
| **likely skip** | 8. `Pars/SampledPars/MetaPars` dotted-key support (cleanup D, conditional) | -- | (renumbered from prev. 7) Audit established `PriorDict` absorbs dotted keys with no changes; commits 1-3 confirm. Default expectation is to skip. |
| pending | 9. migrate tests + tutorials to SourceModel API + drop `int_`/`vel_` from class tuples | -- | (renumbered from prev. 8) Bulk of remaining work (~46 KLModel sites, ~29 factory calls, ~500 hardcoded parameter strings across ~25 test files, 3 tutorials). |
| pending | 10. delete `KLModel` + `SpectralModel` + legacy factories + `flux_theta_override` + `shared_pars` | -- | (renumbered from prev. 9) Mechanical deletion (~500 LOC across many files). |

**Already-shipped capabilities (commits 1-3):**

- `SourceModel(velocity_model, broadband_models, emission_lines)` with `__post_init__` validation (all-empty, disjoint keys, intensity_key / continuum_key / dispersion_key cross-references, lambda_rest auto-resolve).
- `EmissionLine(intensity, intensity_key, continuum, continuum_key, dispersion_key, lambda_rest)` in `kl_pipe/lines.py`.
- `LINE_LAMBDAS` registry in `kl_pipe/lines.py`.
- `kl_pipe/coordinates.py` with `image_rotation_from_wcs` + `rotate_shear`.
- `image_rotation` precomputed on all three obs types from `image_pars.wcs`.
- `ImageObs.broadband_key` + `VelocityObs.flux_weight_key`.
- `GrismPars.dispersion_angle_detector` canonical field + legacy `dispersion_angle` alias.
- SourceModel render methods: `render_broadband`, `render_velocity`, `build_cube`, `render_grism` -- each applies celestial-to-detector rotation per-obs via `obs.image_rotation`.
- SourceModel-aware likelihood primitives in `kl_pipe/likelihood.py`: `_log_likelihood_broadband_source` / `_log_likelihood_grism_source` / `_log_likelihood_velocity_source` / `_log_likelihood_total_source` + `create_jitted_likelihood_from_obs`.
- `InferenceTask.from_obs(source, priors, *, image_obs, grism_obs, velocity_obs, meta_pars, spectral_oversample)` with source-to-obs validation.
- `kl_pipe/source.py`, `kl_pipe/coordinates.py`, `kl_pipe/lines.py` set `jax_enable_x64=True` at import time (matches `kl_pipe/psf.py` defensive contract).

**Test count delta:** `make test-basic` baseline (post-Phase-2): 659 passed. After commit 3: **749 passed**, 0 failed, 140 skipped, 124 deselected. +90 new tests in:
- `tests/test_coordinates.py` (8)
- `tests/test_obs_image_rotation.py` (17)
- `tests/test_source.py` (28)
- `tests/test_source_render.py` (24)
- `tests/test_from_obs.py` (13)

**Drift from plan as written:**

- `SOURCE_MODEL_EXAMPLES.py` is **kept** as a maintained pseudocode reference at the repo root (tracked since commit 1) -- the plan originally said "rm" in commit 1, but the user redirected to keep it.
- `EmissionLine` and `LINE_LAMBDAS` live in **`kl_pipe/lines.py`**, not in `kl_pipe/source.py`. See commit `879415f`. `source.py` imports them.
- `VelocityModel.PARAMETER_NAMES` did NOT add `dispersion`. Per emission line dispersion design: `<line>.dispersion` lives in priors; `EmissionLine.dispersion_key` provides cross-line sharing analogous to `intensity_key` / `continuum_key`. `vel.` namespace stays reserved for params that `VelocityModel.__call__` actually reads.
- Plan's `_build_full_theta` for SourceModel is identity (theta_sampled passed directly to likelihood_fn); `InferenceTask.parameter_names` returns `tuple(priors.sampled_names)` for SourceModel.
- Commit 4 expanded to bring GrismObs to ImageObs `RenderConfig` parity (originally scoped as cleanup E / commit 6 work). Triggered by audit during execution: GrismObs had no `render_config` field, `build_grism_obs` had no `render_config` parameter, and no callsite anywhere computed `RenderConfig.for_priors` for grism. Commit 7's "centralize 2 production sites" scope is now likely near-empty; re-evaluate before executing.
- Commit 5 expanded beyond the originally-scoped bandwidth machinery to also include a pre-existing flux non-conservation bug fix in `_apply_post_dispersion_pixel_response` (output scaled as N² instead of returning flux/pixel) and a new tracked `docs/units_and_conventions.md` doc. Surfaced empirically while writing the convergence test for `for_grism_priors`: predicted oversample produced rel L2 ≈ 0.96 against high-N reference, traced to N² flux scaling, not the bandwidth prediction.
- Commit 5 chose Minkowski-sum bandwidth bound (`k_I + k_G`) over the originally-drafted product walk (`|FT[I]| × |FT[G]| × |FT[PSF]|`). The product walk was empirically UNSAFE in the no-PSF regime, underestimating cube-slice bandwidth by 30-45%. Reason: for inclined galaxies + rotating disks the FT[I] and FT[G] envelopes live on orthogonal k-directions (minor vs major axis), violating the product walk's "aligned envelopes" assumption. The LaTeX doc §3/§5 inconsistency was corrected post-hoc.

**Deferred items surfaced during commits 4-5:**

- VelocityObs flux-weight `RenderConfig.for_priors` auto-derivation (when `flux_weight_key` references an emission line's intensity model). Out of commit 4-5 scope.
- Single-call `build_obs(model, priors, ...)` helper that bundles `RenderConfig.for_priors` + `build_*_obs` into one user-facing call. Decision: 2-step pattern (user calls `for_priors`, passes `render_config=rc` to `build_*_obs`) is sufficient with proper documentation + tutorial. Track as ergonomic improvement if friction becomes apparent later.
- Hook + memory implementation for the "stop on unexpected event requiring judgment" guardrail discussed in commit 4. Trigger: output-pattern matching `(FAILED|Traceback|Error:|AssertionError)`. Action: inject structured reminder ("Stop. Was this trivial-mechanical? If yes proceed. If no, present options to user and halt.") plus a redundant memory entry codifying the rule. Track for follow-up.
- KLModel.render_grism legacy non-obs path (kl_pipe/model.py:1024) returns dispersed image in SB units (not flux/pixel) -- pre-existing divergence flagged in a code comment, deleted with KLModel in commit 10.
- Tighter Approach C (2D numerical convolution of FT[I] ⊛ FT[G]) for grism bandwidth -- would tighten the Minkowski overestimate (3x in no-PSF, ~35% in with-PSF), but the current Minkowski bound is safe and acceptable. Defer unless production performance demands tighter sizing.

## Architecture (locked)

```python
@dataclass
class SourceModel:
    """A source can populate any subset of {velocity, broadband, emission}.
    All-empty is a loud error; otherwise everything else is free to be empty."""
    velocity_model:   Optional[VelocityModel] = None
    broadband_models: Dict[str, IntensityModel] = field(default_factory=dict)
    emission_lines:   Dict[str, EmissionLine]   = field(default_factory=dict)

    def __post_init__(self):
        # require at least one component (no-op SourceModel is meaningless)
        if (self.velocity_model is None
                and not self.broadband_models
                and not self.emission_lines):
            raise ValueError(
                "SourceModel requires at least one of velocity_model, "
                "broadband_models, or emission_lines to be non-empty"
            )
        # collision check between broadband and emission line names
        overlap = set(self.broadband_models) & set(self.emission_lines)
        if overlap:
            raise ValueError(
                f"broadband_models and emission_lines keys must be disjoint; collision: {overlap}"
            )
        # auto-resolve lambda_rest from LINE_LAMBDAS by dict key
        for name, line in self.emission_lines.items():
            if line.lambda_rest is None:
                if name not in LINE_LAMBDAS:
                    raise ValueError(
                        f"emission line '{name}' has no lambda_rest and is not in LINE_LAMBDAS. "
                        f"Provide lambda_rest=... explicitly."
                    )
                line.lambda_rest = LINE_LAMBDAS[name]


@dataclass
class EmissionLine:
    intensity:      Optional[IntensityModel] = None
    intensity_key:  Optional[str]            = None    # reference another line's intensity
    continuum:      Optional[IntensityModel] = None
    continuum_key:  Optional[str]            = None    # reference another line's continuum
    dispersion_key: Optional[str]            = None    # reference another line's dispersion
    lambda_rest:    Optional[float]          = None    # nm, vacuum; auto from LINE_LAMBDAS

    def __post_init__(self):
        if (self.intensity is None) == (self.intensity_key is None):
            raise ValueError("EmissionLine: exactly one of 'intensity' or 'intensity_key' must be set")
        if self.continuum is not None and self.continuum_key is not None:
            raise ValueError("EmissionLine: at most one of 'continuum' or 'continuum_key' may be set")
        # dispersion_key: when set, this line's dispersion is shared from another line
        # (the referenced line's <line>.dispersion in priors); when None, this line has
        # its own <line>.dispersion prior. Validated against source.emission_lines by SourceModel.
```

### Component physical meaning

| Component | Physics | Consumed by |
|---|---|---|
| `velocity_model` | 3D kinematic velocity field — rotation curve V(R), velocity dispersion σ_v | `VelocityObs` (LOS projection); `GrismObs` (Doppler shift in cube assembly) |
| `broadband_models[band]` | Spatial light distribution at a Roman broadband filter; stellar continuum integrated over the filter | `ImageObs` (one obs per band) |
| `emission_lines[line].intensity` | Spatial distribution of ionized-gas emission at the line wavelength | `GrismObs` (cube assembly per line); `VelocityObs` (flux weighting) |
| `emission_lines[line].continuum` | Stellar continuum near the emission line — adds background under the line. Optional. | `GrismObs` (continuum in cube assembly) |
| `emission_lines[line].lambda_rest` | Rest-frame line wavelength (nm, vacuum) | Doppler shift per pixel |

### Line naming convention

Vacuum wavelengths in nm. Singlets use canonical name; doublets/multiplets carry a wavelength-integer suffix.

```python
LINE_LAMBDAS = {
    'Lyalpha':  121.567,  'CIV':      154.95,   'CIII':     190.9,
    'MgII':     279.8,    'Hbeta':    486.13,   'Hgamma':   434.05,
    'Halpha':   656.28,
    # doublets / multiplets — wavelength integer suffix
    'OII3726':  372.60,   'OII3728':  372.88,   'OII3727':  372.74,  # blended
    'OIII4959': 495.89,   'OIII5007': 500.68,
    'NII6548':  654.81,   'NII6584':  658.35,
    'SII6717':  671.65,   'SII6731':  673.08,
}
```

To use a line not in the registry, set `EmissionLine(..., lambda_rest=<nm>)` explicitly.

### Parameter naming convention

The prior/sampler layer carries full dotted names. **SourceModel is the only layer that knows about the dotted namespace; the underlying `VelocityModel` / `IntensityModel` classes operate on flat theta arrays matching their own `PARAMETER_NAMES` and never see dotted keys.**

| Component | Param form | Example |
|---|---|---|
| Velocity | `vel.<param>` | `vel.vcirc`, `vel.v0`, `vel.rscale`, `vel.x0`, `vel.y0` |
| Broadband | `<filter>.<param>` | `F087.flux`, `F087.rscale`, `F087.bulge_frac`, `F087.x0` |
| Emission line intensity | `<line>.<param>` | `Halpha.flux`, `Halpha.rscale`, `Halpha.x0`, `Halpha.dispersion` |
| Continuum under a line | `<line>.cont.<param>` | `Halpha.cont.flux`, `Halpha.cont.rscale` |
| Shared geometry | top-level | `g1`, `g2`, `cosi`, `theta_int` |
| Source spectroscopy | top-level | `z` |

Rules:

- `vel.` prefix always used for the singular velocity model (avoids ambiguity — `x0` alone would be unclear).
- Filter names (`F087`, `F184`) and line names (`Halpha`, `NII6584`) are themselves the namespace. No `bb.` / `em.` prefix.
- Filter and line names must be disjoint (validated at construction).
- Component-prefixed params drop `int_` / `vel_` (class tuples drop the prefix too — see "Class-tuple rename" below).
- `intensity_key` shares the spatial profile only; `<line>.flux` is always per-line.
- `continuum_key` similarly: shares spatial; `<line>.cont.flux` is always per-line.

**Class-tuple rename (drop `int_` / `vel_`):** Today `InclinedExponentialModel.PARAMETER_NAMES = ('flux', 'int_rscale', 'int_h_over_r', 'int_x0', 'int_y0', ...)`. After Phase 3: `('flux', 'rscale', 'h_over_r', 'x0', 'y0', ...)`. Today `CenteredVelocityModel.PARAMETER_NAMES = ('cosi', 'theta_int', 'g1', 'g2', 'v0', 'vcirc', 'vel_rscale')`. After Phase 3: `('cosi', 'theta_int', 'g1', 'g2', 'v0', 'vcirc', 'rscale')`. The prefix existed only because `KLModel` flattened everything into one array; the dotted namespace at the SourceModel layer now disambiguates `vel.rscale` from `F087.rscale`. **Note:** the old `vel_dispersion` parameter does NOT move to VelocityModel — see "Velocity dispersion" section below.

### Velocity dispersion: per emission line

Post-Phase-2, what was `SpectralModel.PARAMETER_NAMES`'s `vel_dispersion` represents the **intrinsic kinematic velocity dispersion of the emitting gas** (km/s), not instrumental LSF — Phase 2's gate test established that PSF + dispersion geometry already reproduces the Roman spec R, so the surviving dispersion term is purely intrinsic gas kinematics.

Physically, different ionization species can have measurably different intrinsic dispersions (Hα in HII regions vs [OIII] in narrow-line regions), so this parameter is **per emission line**:

- Priors expose `<line>.dispersion` per emission line in `source.emission_lines` (e.g. `'Halpha.dispersion': Uniform(20, 150)`, `'OIII5007.dispersion': 50.0`).
- Units: km/s. `SourceModel.build_cube` reads each line's dispersion and applies `sigma_lambda = lam_obs * <line>.dispersion / C_KMS` for that line's broadening kernel.
- Velocity-only inference (no emission lines): no dispersion parameter exists — velocity rendering doesn't use it.
- VelocityModel.PARAMETER_NAMES does NOT include dispersion. The `vel.` namespace is reserved for params that `VelocityModel.__call__` actually reads.

**Sharing dispersion across lines** uses the `dispersion_key` field on `EmissionLine` — exactly parallel to `intensity_key` / `continuum_key`:

```python
emission_lines = {
    'Halpha':  EmissionLine(intensity=InclinedExponential()),
    'NII6584': EmissionLine(
        intensity_key='Halpha',     # share spatial profile
        dispersion_key='Halpha',    # share intrinsic dispersion
    ),
}
# priors include 'Halpha.dispersion' but NOT 'NII6584.dispersion'
```

`SourceModel.build_cube` resolves per line: if `line.dispersion_key is not None`, use `<referenced_line>.dispersion` from theta; else use this line's own `<line>.dispersion`. `SourceModel.__post_init__` validates that `dispersion_key` references an existing emission line; loud error otherwise.

Three common patterns fall out:
- All lines independent: each `<line>.dispersion` is its own prior.
- All lines share one value: one line owns `dispersion`, others use `dispersion_key`. Single sampled parameter constrains all line widths.
- Mixed (e.g. Hα/[NII]/[SII] share for typical SFG; [OIII] independent for an AGN narrow-line component): each line declares its key (or not) individually.

**Resolution order in SourceModel** (when building a component's flat theta): look for `<component>.<param>` first; fall back to top-level `<param>`. Today `cosi` lives top-level (shared); a future user wanting per-component inclination just adds `F087.cosi` to the priors and it routes to that component only. **Zero new infrastructure required — designed-in extensibility.**

**`shared_pars` is gone.** Shared geometry lives once at the top of the priors; SourceModel injects it into whichever component's theta needs it. No dedup set, no `_shared_pars` validation, no `shared_pars=` kwarg anywhere.

### Obs-to-component binding

- **`ImageObs`** binds via explicit `broadband_key` kwarg referencing a key in `source.broadband_models`. Construction-time validation: missing key → loud error.
- **`GrismObs`** auto-includes ALL `source.emission_lines` for cube assembly. Lines whose observed wavelength `λ_rest * (1 + z)` falls outside the grism's wavelength coverage contribute zero (visible by inspection of the rendered image). Requires `source.velocity_model` is set (Doppler shift needs it) and `source.emission_lines` is non-empty.
- **`VelocityObs`** requires `source.velocity_model` is set (binds implicitly). `flux_weight_key: Optional[str] = None`:
  - If set, must reference a key in `source.emission_lines`; SourceModel computes the flux map at render time from that emission line and uses it for PSF flux-weighting (replaces today's `flux_image` mechanism).
  - If `None`, velocity is rendered without flux weighting — this is the velocity-only inference path (mirrors today's `from_velocity_obs` behavior when no flux_image is provided).

### Supported inference patterns

The SourceModel + `from_obs` API explicitly supports — and tests must cover — all the inference shapes the existing `from_*_obs` / `from_*_model` family covers:

| Pattern | SourceModel populated | Obs provided | Today's factory |
|---|---|---|---|
| Velocity-only (IFU / kinematics only) | `velocity_model` | `velocity_obs` (no `flux_weight_key`) | `from_velocity_obs`, `from_velocity_model` |
| Intensity-only single-band | `broadband_models={'F087': ...}` | `image_obs={'F087': obs}` | `from_intensity_obs`, `from_intensity_model` |
| Intensity-only multi-band | `broadband_models={'F087': ..., 'F184': ...}` | `image_obs={'F087': ..., 'F184': ...}` | (not supported today) |
| Joint vel+int | `velocity_model + broadband_models + emission_lines` | `velocity_obs + image_obs` (or grism) | `from_joint_obs`, `from_joint_model` |
| Grism-only | `velocity_model + emission_lines` | `grism_obs` | `from_grism_obs` |
| Joint phot+grism | `velocity_model + broadband_models + emission_lines` | `image_obs + grism_obs` | `from_joint_photometry_grism_obs` |
| Multi-roll grism | same as above | `grism_obs` is a Dict of obs at different rolls | (not supported today) |

All-empty SourceModel or `from_obs` with no obs at all → loud error.

### Centroid handling

Per-component centroids only. Each broadband/emission/velocity component has its own canonical centroid in the parameter space (`F087.x0/y0`, `Halpha.x0/y0`, `vel.x0/y0`). No per-Obs centroid offset layer. Assumes coadd / stacked-image workflow (multiple visits combined into a single WCS-defined image before inference).

If multi-visit unstacked inference becomes a need: construct multiple components with the same spatial profile but independent centroid params (`F087_visit1`, `F087_visit2`). Defer until concrete.

### Multi-observation handling (roll angles, WCS, celestial vs detector frame)

**Physics.** For Roman, the grism's dispersion direction is fixed in the **detector frame** (G150 and P127 disperse along the detector +x axis). What varies between observations of the same source is the **celestial-to-detector rotation** of the telescope, not the dispersion direction. So:

- Sampled top-level params (`theta_int`, `g1`, `g2`) live explicitly in **celestial frame**. One galaxy = one intrinsic PA in celestial coords; one source = one celestial shear. These are shared across all obs; that's where joint constraint power comes from.
- Each obs (ImageObs, GrismObs, VelocityObs) carries an **astropy WCS** that encodes its celestial→detector rotation. SourceModel rotates the celestial-frame sampled params into the detector frame for each obs at render time.
- `GrismPars.dispersion_angle` is renamed to `dispersion_angle_detector` and stays detector-frame (constant for Roman = 0). Per-obs variation in dispersion-vs-galaxy angle is captured entirely through each obs's WCS rotation.

**Existing infrastructure already in place** (verified during audit):

- `kl_pipe/parameters.py:ImagePars` already carries `self.wcs` (astropy WCS). Default constructor (`ImagePars(shape, pixel_scale, indexing)`) builds an identity-rotation WCS — so `image_rotation = 0` for free in existing tests.
- All three obs types reach the WCS via `obs.image_pars.wcs` (or `obs.grism_pars.image_pars.wcs` for grism).
- **No existing rendering code reads `image_pars.wcs`** — WCS today is carried metadata only. Adding rotation-aware rendering is a strictly additive code path: nothing changes for obs with default WCS.

**Implementation.** At obs construction time (in `build_image_obs` / `build_grism_obs` / `build_velocity_obs`), precompute `image_rotation: float` (radians) from the WCS PC/CD matrix and freeze it on the obs as a derived field (alongside the existing precomputed `pixel_response_fft`). astropy WCS objects are not JAX-traceable; the precomputed scalar is.

```python
# kl_pipe/coordinates.py — JAX-friendly slim equivalent of kl-tools OrientedAngle
def image_rotation_from_wcs(wcs):
    """Extract celestial→detector rotation (radians) from astropy WCS CD/PC matrix.
    Called once at obs construction; result frozen on the obs as a JAX scalar."""
    if wcs.wcs.has_pc():
        R = wcs.wcs.get_pc()
    elif wcs.wcs.has_cd():
        R = wcs.wcs.get_cd()
    else:
        raise ValueError("WCS must have a CD or PC matrix")
    theta = float(np.arctan2(R[1, 0], R[0, 0]))
    if np.linalg.det(R) < 0:
        theta += np.pi
    return theta

def rotate_shear(g1, g2, phi):
    """Spin-2 shear rotation by phi radians."""
    c, s = jnp.cos(2.0 * phi), jnp.sin(2.0 * phi)
    return g1 * c + g2 * s, -g1 * s + g2 * c
```

At render time, SourceModel applies the per-obs rotation when building the flat theta for the model classes:

- `theta_int_det = theta_int_celestial - obs.image_rotation` (matches `OrientedAngle._sky2cartesian` from kl-tools; we use the term "celestial" instead of "sky" but the math is identical).
- `(g1_det, g2_det) = rotate_shear(g1_celestial, g2_celestial, obs.image_rotation)`.

These detector-frame values are what get threaded to `VelocityModel.__call__` / `IntensityModel.render_image` for THIS obs. Model classes still don't know about WCS / rotation / namespacing — they receive a flat theta in their own (detector-frame, namespace-naive) world. `kl_pipe/transformation.py` (5-plane chain) is unchanged.

**Sign convention.** Celestial→detector: subtract `image_rotation` (matches OrientedAngle `_sky2cartesian`). Mirroring (det(R) < 0): add π to `image_rotation`. Documented at the top of `coordinates.py`.

**API consequence.** `from_obs(grism_obs=...)` widens to `Dict[str, GrismObs]` (parallel to `image_obs`), keyed by an identifier like `{'roll0': obs0, 'roll90': obs90, 'roll180': obs180}`. Each obs carries its own WCS; SourceModel rotates per-obs independently. The per-channel likelihood sums over all entries.

**Backward compatibility.** Three explicit protections:

- Default `ImagePars(shape=..., pixel_scale=0.1, indexing='ij')` → identity WCS → `image_rotation = 0` → existing tests' theta values map identically into the model.
- Regression test `test_default_wcs_rotation_is_zero` asserts each obs type with the default constructor has `obs.image_rotation == 0.0`. Catches accidental drift in `ImagePars` default.
- Equivalence test `test_zero_roll_equivalence` renders a source with default-WCS obs and confirms output matches pre-Phase-3 (KLModel) output for the same theta — bit-for-bit modulo numerical noise. Catches accidental rotation-on-zero bugs.

**The per-component override trick (`F087.cosi` overriding top-level `cosi`)** remains available for the case it's actually meant — genuinely independent geometry between two intensity components, not multi-obs of the same source.

## Worked PriorDict examples

### Example A — minimal: F087 imaging + G150 grism with Hα only, z=1

```python
source = SourceModel(
    velocity_model   = CenteredVelocityModel(),
    broadband_models = {'F087': InclinedExponential()},
    emission_lines   = {'Halpha': EmissionLine(intensity=InclinedExponential())},
)

priors = PriorDict({
    # velocity (vel.dispersion does NOT exist — dispersion is per emission line)
    'vel.vcirc':       Uniform(80, 300),
    'vel.v0':          0.0,
    'vel.rscale':      Uniform(0.05, 1.0),
    'vel.x0':          Uniform(-0.2, 0.2),
    'vel.y0':          Uniform(-0.2, 0.2),
    # broadband F087
    'F087.flux':       Uniform(1, 200),
    'F087.rscale':     Uniform(0.05, 1.0),
    'F087.h_over_r':   0.15,
    'F087.x0':         Uniform(-0.2, 0.2),
    'F087.y0':         Uniform(-0.2, 0.2),
    # emission Halpha
    'Halpha.flux':       Uniform(1, 100),
    'Halpha.rscale':     Uniform(0.05, 1.0),
    'Halpha.h_over_r':   0.15,
    'Halpha.x0':         Uniform(-0.2, 0.2),
    'Halpha.y0':         Uniform(-0.2, 0.2),
    'Halpha.dispersion': Uniform(20, 150),    # km/s, intrinsic gas dispersion
    # shared (celestial-frame)
    'g1':         Uniform(-0.1, 0.1),
    'g2':         Uniform(-0.1, 0.1),
    'cosi':       Uniform(0.1, 0.99),
    'theta_int':  Uniform(0, jnp.pi),
    'z':          1.0,
})
```

### Example B — production-typical: F087 + Hα+[NII] grism sharing spatial profile

```python
source = SourceModel(
    velocity_model   = OffsetVelocityModel(),
    broadband_models = {'F087': InclinedExponential()},
    emission_lines   = {
        'Halpha':  EmissionLine(intensity=InclinedExponential()),
        'NII6584': EmissionLine(intensity_key='Halpha'),   # shared spatial profile
    },
)
source = SourceModel(
    velocity_model   = OffsetVelocityModel(),
    broadband_models = {'F087': InclinedExponential()},
    emission_lines   = {
        'Halpha':  EmissionLine(intensity=InclinedExponential()),
        # share both spatial profile AND dispersion with Halpha:
        'NII6584': EmissionLine(intensity_key='Halpha', dispersion_key='Halpha'),
    },
)
# priors as in A, plus:
#   'NII6584.flux': Uniform(0.05, 50)   # only flux per-line; spatial + dispersion shared
# 'NII6584.dispersion' is NOT in priors — resolved via dispersion_key='Halpha'
```

### Example C — stretch: F087 (bulge+disk) + F184 (disk) + Hα (Sersic) + [OIII] (exp) + per-line continuum, z=1.5

```python
source = SourceModel(
    velocity_model   = OffsetVelocityModel(),
    broadband_models = {
        'F087': BulgeDiskSersic(),
        'F184': InclinedExponential(),
    },
    emission_lines   = {
        'Halpha':   EmissionLine(intensity=InclinedSersic(), continuum=InclinedExponential()),
        'OIII5007': EmissionLine(intensity=InclinedExponential(), continuum_key='Halpha'),
    },
)
# priors include 'F087.bulge_frac', 'F087.bulge_rscale', 'F087.bulge_n_sersic', etc.
# plus 'Halpha.cont.flux', 'Halpha.cont.rscale', etc.; OIII5007 inherits Halpha.cont spatial via continuum_key
```

## Unified inference factory: `InferenceTask.from_obs`

Replaces the entire `from_*_obs` / `from_*_model` family (8 factories today).

```python
@classmethod
def from_obs(
    cls,
    source: SourceModel,
    priors: PriorDict,
    *,
    image_obs:    Optional[Dict[str, ImageObs]] = None,
    grism_obs:    Optional[Dict[str, GrismObs]] = None,
    velocity_obs: Optional[VelocityObs] = None,
    meta_pars:    Optional[Dict] = None,
) -> 'InferenceTask':
    """Build inference task from any combination of observation channels.

    Validation:
      - At least one obs must be provided (otherwise raise — there's nothing to fit).
      - For each entry in image_obs: the key must exist in source.broadband_models.
      - grism_obs (Dict, possibly empty): if non-empty, requires source.emission_lines
        non-empty AND source.velocity_model is set (Doppler needs it).
      - velocity_obs: requires source.velocity_model is set. If
        velocity_obs.flux_weight_key is not None, it must reference a key in
        source.emission_lines. flux_weight_key=None is allowed and means
        unweighted velocity rendering (velocity-only inference path).
    """
```

Internal likelihood is the sum of per-channel primitives:

```python
def _log_likelihood_total(theta, source, image_obs, grism_obs, velocity_obs):
    log_l = 0.0
    if image_obs is not None:
        for band_key, obs in image_obs.items():
            log_l += _log_likelihood_broadband(theta, source, obs, band_key)
    if grism_obs is not None:
        for grism_key, obs in grism_obs.items():
            log_l += _log_likelihood_grism(theta, source, obs)
    if velocity_obs is not None:
        log_l += _log_likelihood_velocity(theta, source, velocity_obs)
    return log_l
```

JIT-compiled with all obs frozen via partial. The `if obs is not None` branches and dict iteration resolve at trace time (structurally fixed).

## Workflow examples (new API)

```python
# A — grism-only, single roll
task = InferenceTask.from_obs(source, priors, grism_obs={'roll0': obs_grism})

# B — production: F087 + Hα+[NII] grism (single roll)
task = InferenceTask.from_obs(
    source, priors,
    image_obs = {'F087': obs_f087},
    grism_obs = {'roll0': obs_grism},
)

# C — multi-band + grism
task = InferenceTask.from_obs(
    source, priors,
    image_obs = {'F087': obs_f087, 'F184': obs_f184},
    grism_obs = {'roll0': obs_grism},
)

# D — broadband-only single band (no kinematics, no emission)
source_d = SourceModel(broadband_models={'F087': InclinedExponential()})
task = InferenceTask.from_obs(source_d, priors, image_obs={'F087': obs_f087})

# D2 — broadband-only multi-band
source_d2 = SourceModel(broadband_models={'F087': InclinedExponential(), 'F184': InclinedExponential()})
task = InferenceTask.from_obs(source_d2, priors, image_obs={'F087': obs_f087, 'F184': obs_f184})

# E — velocity-only (IFU / kinematics, no intensity context)
source_e = SourceModel(velocity_model=OffsetVelocityModel())
# obs_velocity built with flux_weight_key=None → no flux weighting
task = InferenceTask.from_obs(source_e, priors, velocity_obs=obs_velocity)

# E2 — velocity + photometry (IFU + broadband)
task = InferenceTask.from_obs(source, priors,
    image_obs    = {'F087': obs_f087},
    velocity_obs = obs_velocity,
)

# F — multi-roll grism (new): 2 broadband + 2 grism rolls
task = InferenceTask.from_obs(
    source, priors,
    image_obs = {'F087': obs_f087, 'F184': obs_f184},
    grism_obs = {'roll0': obs_g0, 'roll90': obs_g90},
)

# G — kitchen sink: 4-band + 3-roll grism + velocity
task = InferenceTask.from_obs(
    source, priors,
    image_obs    = {'F087': obs_f087, 'F129': obs_f129, 'F158': obs_f158, 'F184': obs_f184},
    grism_obs    = {'roll0': obs_g0, 'roll90': obs_g90, 'roll180': obs_g180},
    velocity_obs = obs_velocity,
)
```

## Commit chain (one PR, 10 atomic commits, dual-API through commit 9)

Each commit 1–9 compiles and `make test-basic` passes (bisectable). Commit 10 is the hard delete.

**Note: chain renumbered mid-execution to insert commit 5 (grism cube-slice bandwidth-aware
`for_grism_priors`). See Progress table at top for current numbering. Original text below
left intact for audit context; numbers shift by +1 starting at commit 5.**

```
1. feat: LINE_LAMBDAS + new EmissionLine + SourceModel + WCS-derived rotation infrastructure
   - new kl_pipe/source.py (or extend model.py): SourceModel, EmissionLine, LINE_LAMBDAS
   - new kl_pipe/coordinates.py: image_rotation_from_wcs, rotate_shear (slim JAX-friendly utilities)
   - kl_pipe/spectral.py: rename old EmissionLine → _LegacyEmissionLine (kept compiling for SpectralModel)
   - kl_pipe/observation.py: add ImageObs.broadband_key (Optional[str]=None), VelocityObs.flux_weight_key (Optional[str]=None), precompute obs.image_rotation from image_pars.wcs at build_*_obs time
   - kl_pipe/dispersion.py: rename GrismPars.dispersion_angle → dispersion_angle_detector (detector frame, constant for Roman = 0). Keep old name as a deprecation alias through commit 8.
   - kl_pipe/velocity.py: NO changes in commit 1 (dispersion stays as `vel_dispersion` on SpectralModel during dual-API window; per-line `<line>.dispersion` lookup added in commit 2 inside SourceModel.build_cube)
   - SOURCE_MODEL_EXAMPLES.py: keep at repo root as a maintained pseudocode reference for collaborators (refreshed in Step 0 below; updated throughout Phase 3 as the API evolves)

2. feat: SourceModel rendering methods + celestial-to-detector rotation per obs
   - build_cube(theta, cube_pars, plane) — analog of SpectralModel.build_cube but using emission_lines dict
   - render_broadband(theta, obs, band_key) — uses broadband_models[band_key]; rotates celestial-frame theta_int + (g1, g2) by obs.image_rotation before threading into the intensity model
   - render_grism(theta, obs) — uses build_cube + dispersion; same per-obs rotation; dispersion_angle_detector stays detector-frame
   - render_velocity(theta, obs) — uses velocity_model + flux_weight_key for PSF weighting; same per-obs rotation

3. feat: InferenceTask.from_obs unified factory + per-channel _log_likelihood primitives
   - _log_likelihood_broadband, _log_likelihood_grism, _log_likelihood_velocity (SourceModel-aware)
   - _log_likelihood_total dispatch sum
   - from_obs validates source ↔ obs binding; legacy from_*_obs factories untouched

4. refactor: generalize _check_priors_fit_obs_rc to obs-type-aware (cleanup A)
   - Per-obs validation: ImageObs uses existing check; GrismObs uses fine-grid-based analogue; VelocityObs no-op (no k-space grid)
   - Called once inside from_obs for each channel

5. refactor: move _apply_post_dispersion_pixel_response to kl_pipe/grism.py (cleanup B)
   - Create kl_pipe/grism.py; move helper from model.py:432-479
   - Update callsite in render_grism

6. refactor: centralize RenderConfig.for_priors callsites in from_obs (cleanup E)
   - Audit-flagged inconsistency: 2 production sites in task.py today
   - Consolidate into from_obs prelude

7. refactor: Pars/SampledPars/MetaPars dotted-key support (cleanup D — conditional)
   - Only execute if commit 3 reveals blockage; PriorDict already absorbs dotted keys (verified during audit)
   - Default expectation: skip; mark commit 7 as a no-op or fold into commit 3

8. refactor: migrate tests + tutorials to SourceModel API + drop int_/vel_ from class tuples
   - All ~46 KLModel(...) sites → SourceModel(...)
   - All ~29 from_*_obs/from_*_model calls → from_obs(...)
   - All hardcoded parameter-name strings → dotted form (mechanical sed)
   - Fixture dicts (_VEL_PARS, _INT_PARS, _SPEC_PARS) → per-component dicts
   - kl_pipe/velocity.py + intensity.py: drop int_/vel_ from PARAMETER_NAMES; _rename_param simplifies
   - Tutorials: quickstart.md, grism.md, sampling.md rewritten
   - This commit is where the class-tuple rename lands (post-migration, pre-deletion)

9. feat: delete KLModel + SpectralModel + legacy factories + flux_theta_override + shared_pars
   - kl_pipe/model.py: delete KLModel
   - kl_pipe/spectral.py: delete SpectralModel, LineSpec, _LegacyEmissionLine
   - kl_pipe/sampling/task.py: delete from_velocity_model, from_intensity_model, from_joint_model, from_velocity_obs, from_intensity_obs, from_joint_obs, from_grism_obs, from_joint_photometry_grism_obs
   - kl_pipe/likelihood.py: delete old _log_likelihood_* primitives + flux_theta_override plumbing
   - kl_pipe/intensity.py: delete _DEFAULT_SHARED + shared_pars kwarg from CompositeIntensityModel (single-source-of-truth in SourceModel)
```

## Critical files

**New:**
- `kl_pipe/source.py` — `SourceModel`, `EmissionLine`, `LINE_LAMBDAS`, per-channel render methods. (Decision deferred to commit-1 time: extend `model.py` vs new file. Lean toward new file — `model.py` is already 1100+ lines.)
- `kl_pipe/grism.py` — `_apply_post_dispersion_pixel_response` + future grism-only helpers (commit 5).
- `kl_pipe/coordinates.py` — `image_rotation_from_wcs(wcs)`, `rotate_shear(g1, g2, phi)`. JAX-friendly slim equivalent of kl-tools `OrientedAngle`. Created in commit 1.

**Heavily modified:**
- `kl_pipe/model.py` — delete `KLModel` (commit 9); `Model` / `VelocityModel` / `IntensityModel` ABCs stay.
- `kl_pipe/spectral.py` — delete `SpectralModel`, `LineSpec`, `_LegacyEmissionLine` (commit 9). `CubePars` stays.
- `kl_pipe/intensity.py` — drop `int_` from all subclass `PARAMETER_NAMES` (commit 8). Simplify `_rename_param` (line 2697): strip-`int_` step disappears.
- `kl_pipe/velocity.py` — drop `vel_` from both subclass `PARAMETER_NAMES` (commit 8). `dispersion` is NOT added here — it lives per-line in `source.emission_lines` and is consumed by `SourceModel.build_cube`.
- `kl_pipe/observation.py` — add `broadband_key` to `ImageObs` (commit 1), `flux_weight_key` to `VelocityObs` (commit 1). Update factory functions; precompute `image_rotation` from `image_pars.wcs`.
- `kl_pipe/likelihood.py` — new SourceModel-aware primitives (commit 3); delete old primitives + `flux_theta_override` (commit 9).
- `kl_pipe/sampling/task.py` — add `from_obs` (commit 3); generalize `_check_priors_fit_obs_rc` (commit 4); centralize `RenderConfig.for_priors` (commit 6); delete legacy factories (commit 9).

**Drift items surfaced during audit** (not in original design doc, all fixed in commit 1):
1. `LINE_LAMBDAS` registry does not exist — add to `kl_pipe/source.py` (or dedicated `kl_pipe/lines.py`).
2. `EmissionLine` already exists at `kl_pipe/spectral.py:48` with a different shape — rename old to `_LegacyEmissionLine`; define new in `kl_pipe/source.py`.
3. `LineSpec` (spectral.py:39) is Phase-1 scaffolding — deleted with `SpectralModel` in commit 9.
4. `ImageObs.broadband_key`, `VelocityObs.flux_weight_key` fields do not exist — add in commit 1 (default `None` for legacy compatibility).
5. `vel_dispersion` lives on `SpectralModel.PARAMETER_NAMES` today — migrate to `VelocityModel.PARAMETER_NAMES` as `dispersion` (added commit 1; removed from SpectralModel in commit 8; SpectralModel deleted commit 9).
6. `kl_pipe/grism.py` does not exist — created in commit 5.

## Existing functions to reuse

- `kl_pipe/priors.py:PriorDict` — already string-agnostic; absorbs dotted keys with no parser changes. `get_param_bounds` works unchanged.
- `kl_pipe/render.py:RenderConfig.for_priors` — used as-is; cleanup E centralizes callsites but doesn't change the API.
- `kl_pipe/intensity.py:CompositeIntensityModel._build_parameter_structure` — keeps its intra-component dedup; only the strip-`int_` step in `_rename_param` simplifies.
- `kl_pipe/observation.py:build_image_obs` / `build_velocity_obs` / `build_grism_obs` — extend signatures with the new key fields; keep behavior.
- All `IntensityModel` subclasses (`maxk`, `stepk`, `_ft_envelope`) — unchanged.
- `kl_pipe/parameters.py:ImagePars.wcs` — already exists; default constructor produces identity-rotation WCS. SourceModel-era rendering reads this for the first time.

## New tests + visual diagnostics

### Unit tests (commit 1–3)

- `tests/test_source.py` (new) — `SourceModel` / `EmissionLine` construction:
  - Disjoint-name validation: `SourceModel(broadband_models={'X': ...}, emission_lines={'X': ...})` raises.
  - `EmissionLine` exactly-one-of validation: `intensity` AND `intensity_key` both set → raises; both unset → raises.
  - `LINE_LAMBDAS` auto-resolution: `EmissionLine(intensity=...)` keyed `'Halpha'` gets `lambda_rest=656.28`; keyed `'NotInRegistry'` without explicit `lambda_rest` raises.
  - `continuum` / `continuum_key` mutual exclusivity.
  - Pytree registration: `jax.tree_util.tree_flatten(source)` → expected children + aux.

### Inference-task setup + likelihood evaluation tests (commit 3)

- `tests/test_from_obs.py` (new) — `InferenceTask.from_obs` for each example configuration. Each evaluates the likelihood at synthetic dummy theta and asserts finite + gradient computable via `jax.grad`:
  - **Velocity-only** (`SourceModel(velocity_model=...)`, no intensity/emission): `velocity_obs` with `flux_weight_key=None`. Mirrors today's `from_velocity_obs` path.
  - **Intensity-only single-band**: `SourceModel(broadband_models={'F087': ...})`, `image_obs={'F087': ...}`. Mirrors today's `from_intensity_obs`.
  - **Intensity-only multi-band**: `SourceModel(broadband_models={'F087': ..., 'F184': ...})`, `image_obs={'F087': ..., 'F184': ...}`. New capability.
  - **Example A**: grism-only, single roll.
  - **Example B**: F087 + Hα+[NII] grism.
  - **Example C**: multi-band + multi-line + per-line continuum.
  - **Example F**: 2-broadband + 2-roll grism; assert each grism obs contributes to the total log-likelihood (zero out one, observe change).
  - **Velocity + photometry joint**: mirrors today's `from_joint_obs`.
  - **Validation errors fire loudly** (each must raise with a clear message):
    - `SourceModel()` with all components empty → raises.
    - `from_obs(source, priors)` with no obs at all → raises.
    - `from_obs` with `image_obs` key not in `source.broadband_models` → raises.
    - `from_obs` with `grism_obs` non-empty but `source.emission_lines` empty → raises.
    - `from_obs` with `grism_obs` non-empty but `source.velocity_model is None` → raises.
    - `from_obs` with `velocity_obs` but `source.velocity_model is None` → raises.
    - `from_obs` with `velocity_obs.flux_weight_key` referencing a non-existent emission line → raises.

### WCS / rotation regression tests (commit 1–2)

- `tests/test_coordinates.py` (new) — `image_rotation_from_wcs` and `rotate_shear`:
  - Default `ImagePars(shape, pixel_scale)` WCS → `image_rotation = 0.0`.
  - WCS with PC matrix at 90° rotation → `image_rotation = pi/2` (within tolerance).
  - WCS with mirroring (det(R) < 0) → image_rotation receives +π adjustment.
  - `rotate_shear(g1, g2, phi)` round-trip: `rotate_shear(*rotate_shear(g1, g2, phi), -phi) == (g1, g2)`.

- `tests/test_obs_image_rotation.py` (new) — obs construction wires rotation correctly:
  - `test_default_wcs_rotation_is_zero` — `build_image_obs(image_pars)` / `build_grism_obs(grism_pars)` / `build_velocity_obs(image_pars)` with the default-pixel_scale ImagePars all have `obs.image_rotation == 0.0`. Regression guard against drift in ImagePars default.
  - `test_custom_wcs_rotation_propagates` — passing a custom WCS with 30° rotation through to `obs.image_rotation` returns `pi/6` (within tolerance).
  - `test_zero_roll_equivalence` — render a source via SourceModel with default-WCS obs at fixed theta; assert output matches pre-Phase-3 KLModel output at the same theta, bit-for-bit modulo numerical noise. Catches accidental rotation-on-zero bugs.

### Celestial-vs-detector frame visual diagnostic (commit 2)

- `tests/test_celestial_vs_detector.py` (new) — single-source visual demonstration that celestial-frame `(theta_int, g1, g2)` rotate consistently into the detector frame across obs.
  - **Setup:** one `InclinedExponential` broadband source; non-zero `theta_int_celestial`, non-zero `(g1_celestial, g2_celestial)`. Low-noise / noiseless rendering. Two obs:
    - `obs_celestial`: default WCS (identity rotation → detector frame = celestial frame).
    - `obs_rolled`: WCS with PC matrix at e.g. `phi = 30°` (non-trivial celestial→detector rotation).
  - **Render both** through `SourceModel.render_broadband` using the same celestial-frame priors. The rendered images differ — the rolled obs shows the same source rotated by `-phi` in the detector pixel grid.
  - **2-panel figure** saved to `tests/out/celestial_vs_detector/`:
    - Panel A: `obs_celestial` rendering, with annotations:
      - North/East arrows from the WCS (in panel A, North = +y, East = -x by convention).
      - Celestial-frame `(g1, g2)` direction vector overlaid (drawn as a short bar showing major-axis tilt).
      - Galaxy `theta_int_celestial` major-axis line overlaid.
    - Panel B: `obs_rolled` rendering, with annotations:
      - North/East arrows rotated by `phi` relative to panel A's pixel frame.
      - Celestial-frame `(g1, g2)` direction overlaid (the *same* shear vector, but in panel B's pixel coords it points in a different pixel direction because the detector is rotated).
      - Galaxy `theta_int_celestial` major-axis line overlaid (also different pixel orientation than in panel A).
    - **Consistency check shown visually:** the source's apparent major axis in panel A relative to North-East matches the apparent major axis in panel B relative to North-East. Likewise the shear orientation in panel A relative to North-East matches in panel B.
  - **Numerical assertions:**
    - `(g1_det, g2_det) = rotate_shear(g1_celestial, g2_celestial, obs.image_rotation)` matches `rotate_shear` round-trip from celestial → detector → celestial.
    - Rendering `obs_celestial` with `(g1_celestial, g2_celestial)` via SourceModel produces same image (within ε) as rendering directly in detector frame using `(g1_celestial, g2_celestial)` (since image_rotation = 0 for celestial-frame obs).
    - Rendering `obs_rolled` via SourceModel with celestial-frame `(g1, g2, theta_int)` produces same image as rendering directly in detector frame using rotated values `(g1_det, g2_det, theta_int - phi)` — proves the SourceModel rotation step is internally consistent.
  - **Marker:** standard test (not `slow`); runtime ~3-5s.

### Multi-roll-angle integration test + visual diagnostic (commit 8 or 9)

- `tests/test_multi_roll_grism.py` (new) — optimizer-driven joint fit:
  - **Setup:** synthetic source with `OffsetVelocityModel` + `F087` and `F184` broadband + Hα emission. 2 broadband + 2 grism observations at non-trivial roll angles, encoded via per-obs WCS PC matrices (e.g. `phi_obs = 0` and `phi_obs = pi/2`). `GrismPars.dispersion_angle_detector = 0` on all grism obs (Roman convention). Roman-like PSF + pixel scale. SNR ~ 200.
  - **Run:** `scipy.optimize.minimize` with `jax.grad` from true-pars perturbed start (`L-BFGS-B`).
  - **Assertions:**
    - `vel.vcirc`, `theta_int` (celestial-frame), `cosi`, `g1`, `g2` (celestial-frame) recover within `TestConfig.optimizer` tolerances.
    - `F087.flux`, `F184.flux`, `Halpha.flux` recover within tolerance.
    - Likelihood at recovered theta ≥ likelihood at any single-obs subset (joint info adds).
    - **Per-obs rotation check:** rotating an obs's WCS by an additional `δφ` and shifting the source theta_int by `+δφ` produces identical likelihood (rotation invariance regression).
  - **Visual diagnostic** (saved to `tests/out/multi_roll_grism/`):
    - 6-panel figure: top row = 2 broadband (data, model, residual); bottom row = 2 grism (data, model, residual). North/East arrows overlaid on each.
    - Recovery summary table: true vs fit per parameter with σ.
    - Posterior-uncertainty-vs-shrinkage plot: how much does each parameter's posterior tighten with each added obs (data1, +data2, +data3, +data4)?
  - **Marker:** `@pytest.mark.slow` (optimizer test; runtime ~30–60s).

### Per-line continuum + intensity_key + dispersion_key sharing test (commit 2 or 3)

- `tests/test_emission_line_sharing.py` (new) — verify the `intensity_key` / `continuum_key` / `dispersion_key` mechanisms:
  - Construct `{'Halpha': EmissionLine(intensity=Exp()), 'NII6584': EmissionLine(intensity_key='Halpha')}`.
  - At identical theta, render both lines; their **spatial** profiles match exactly (FT-level equality assert).
  - Their **flux** values can differ via independent `Halpha.flux` and `NII6584.flux` priors.
  - Same for `continuum_key`.
  - For `dispersion_key`: build cube; assert NII6584's line-broadening sigma_lambda matches Halpha's (modulo λ_obs scaling); confirm `NII6584.dispersion` is NOT in `priors.sampled_names`.
  - Validation error: `dispersion_key='NonExistent'` raises at SourceModel construction.

### Per-component centroid recovery (commit 8)

- `tests/test_per_component_centroids.py` (new) — regression for Phase 1's shared-centroid coupling:
  - Synthetic source: `F087.x0 = 0.1`, `Halpha.x0 = -0.1` (intentionally different — would be impossible to recover with KLModel's shared centroid).
  - Joint phot+grism fit recovers both centroids within tolerance.
  - Asserts the new architecture solves the Phase 1 limitation #1.

### Migration regression tests (commit 8)

- All existing tests pass under the SourceModel API at commit 8 (~18–20 test files migrated). Migration is mechanical: `KLModel(...)` → `SourceModel(...)`, factory call signatures change, parameter-name strings dotted.
- Specifically retain: `tests/test_grism_likelihood.py`, `tests/test_grism_core.py`, `tests/test_datacube.py`, `tests/test_lsf_gate.py`, `tests/test_optimizer_recovery.py`, `tests/test_likelihood_slices.py` semantics.

### Tutorial executable verification (commit 8)

- `make test-tutorials` against rewritten `quickstart.md`, `grism.md`, `sampling.md` — every code cell runs end-to-end.

## Verification

- Each commit 1–8 passes `make test-basic` independently (bisectable chain).
- After commit 9, full `make test` is green (includes TNG).
- `make test-tutorials` against rewritten tutorials.
- Geko cross-validation harness still green (no regression in `make render-validation-*`); Issue #50 follow-up tracked separately.
- `/test-integrity` pre-merge sweep (per CLAUDE.md rule 8) catches any tolerance loosening.
- Pre-merge sanity: bench joint-fit runtime to ensure no >2× regression from current Phase 1 grism likelihood JIT time.

## Open implementation questions (defer to execution)

These were noted during planning and re-examined during audit; none block planning:

1. **JAX pytree registration for `SourceModel` and `EmissionLine`** — `broadband_models` and `emission_lines` dicts need explicit aux/children split. Pattern follows existing `ImageObs` / `GrismObs` registrations.
2. **Cleanup D conditional** — audit confirms `PriorDict` absorbs dotted keys today without modification. Cleanup D is not strictly required; default expectation is to skip. Mark commit 7 as no-op if so.
3. **Where `LINE_LAMBDAS` lives** — `kl_pipe/source.py` (alongside `SourceModel`) or dedicated `kl_pipe/lines.py`. Pick at commit-1 time. Lean toward source.py for now (small registry).
4. **Where `SourceModel` lives** — extend `kl_pipe/model.py` or new `kl_pipe/source.py`. Lean toward new file (cleaner separation).
5. **`docs/plans/phase3_sourcemodel_refactor.md`** — keep gitignored at PR open (matches PR #41 / #43 precedent; PR description summarizes design self-contained). Override at PR time if reviewers want it tracked.

## Post-plan actions (after ExitPlanMode)

1. **Create branch:** `git checkout -b se/source-model` from current `se/grism-core` HEAD.

2. **Step 0 — refresh `SOURCE_MODEL_EXAMPLES.py` (first action on the new branch).**
   The file is a pseudocode reference for collaborators showing the SourceModel API in use. It is not meant to execute — it illustrates design. The current copy is from an earlier draft of the plan and has drifted; the refresh covers:

   **Bug fixes:**
   - `from types import Optional, Dict` → `from typing import Optional, Dict` (current import is wrong; `types` doesn't have `Optional`).
   - Fix indentation throughout (currently has stray leading-space lines as if inside a class/function).
   - Fix the `from_obs` signature definition — `@classmethod` at module level isn't valid Python.

   **Content updates to match the locked plan:**
   - Per-line `<line>.dispersion` (km/s); `vel.dispersion` does NOT exist.
   - `dispersion_key` field on `EmissionLine` shown in at least one example.
   - `from_obs(grism_obs=Dict[str, GrismObs])` — widened to Dict (parallel to `image_obs`).
   - `flux_weight_key` on `VelocityObs` shown explicitly.
   - `broadband_key` on `ImageObs` shown explicitly.
   - All priors examples updated to celestial-frame `g1`, `g2`, `theta_int` with a one-line comment noting the convention.
   - At least one example constructs an obs with a non-trivial WCS PC matrix to demonstrate multi-roll-angle handling (e.g. Workflow F).

   **New examples to add:**
   - Velocity-only inference (Workflow E from the plan): `SourceModel(velocity_model=...)`, `from_obs(velocity_obs=...)` with `flux_weight_key=None`.
   - Intensity-only multi-band (Workflow D2): `SourceModel(broadband_models={'F087': ..., 'F184': ...})`, `from_obs(image_obs={'F087': ..., 'F184': ...})`.
   - Multi-roll grism (Workflow F): 2 broadband + 2 grism rolls, with the WCS construction for each grism obs.
   - Dispersion sharing pattern: `'NII6584': EmissionLine(intensity_key='Halpha', dispersion_key='Halpha')`.

   Commit this refreshed file as part of commit 1 with `git add -f SOURCE_MODEL_EXAMPLES.py` (or move to `docs/source_model_examples.py` if a cleaner location is preferred at commit-1 time).

3. **Begin commit 1** — new `kl_pipe/source.py`, `LINE_LAMBDAS`, `SourceModel`, `EmissionLine` (with `dispersion_key`), `kl_pipe/coordinates.py`, obs-field additions, refreshed `SOURCE_MODEL_EXAMPLES.py` checked in.
