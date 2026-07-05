# Datacube + Grism Rendering: Test Coverage

Honest inventory of what is and is not gated for the `kl_pipe`
datacube-assembly + grism-dispersion + PSF + pixel-readout stack, verified
against the actual test files (not assumed) as of 2026-07-05, `se/speedups`
@ commit including the GalSim-chromatic reference gate.

"Gated" means an automated test asserts a numerical tolerance against
either (a) an independently-derived closed-form/analytic answer, (b) an
independent external codebase (geko), or (c) an independent from-scratch
numerical construction (the GalSim-chromatic reference render). Internal
self-consistency tests (e.g., two `kl_pipe` code paths agreeing with each
other) are noted separately -- they catch regressions but cannot rule out a
shared bug in both paths.

## Coverage table

| Element | Gated by | Independence | Notes |
|---|---|---|---|
| Spectral bin amplitude (line flux per coarse wavelength bin) | `tests/test_spectral_methods.py::TestValueEquivalence` (erf vs converged midpoint-oversample limit), `TestGradients` (AD vs finite-difference) | Closed-form (erf is the exact Gaussian CDF bin integral; the midpoint path independently converges to it) | `erf` is the default `spectral_method`. Continuum exactness also covered here (`test_erf_line_flux_conservation`-adjacent tests). |
| Pixel readout (post-dispersion BoxPixel sinc + coarse-pixel-center sampling) | `tests/test_pixel_readout.py::TestClosedFormGate` | Closed-form (Gaussian source, erf box-integral reference) | Also pins flux conservation (`test_flux_conservation`) and the historical mean-bin bug (`TestMeanBinCanary`, a regression canary, not a correctness gate). |
| Velocity-entanglement / `n_lambda` resolution | `tests/test_pixel_readout.py::TestEntanglementCanary` (default-vs-refined deviation window, in-suite) + `tests/test_galsim_reference.py::test_dynamic_scene_refined_n_lambda` (refined-`n_lambda` pathway vs independent GalSim-chromatic render) | Canary = internal (frozen deviation window, not a ground truth); GalSim-chromatic gate = independent | The canary catches a silent change to default behavior; the GalSim gate confirms the *refined* pathway (once entanglement quantization is resolved) is actually correct, not just internally self-consistent. Neither individually proves the *default* `n_lambda` pathway is numerically correct -- only that its deviation from the refined pathway is a known, frozen quantity. |
| Dispersal interpolation (`disperse_cube` bilinear sub-pixel shift), integer-shift case | `tests/test_grism_core.py::TestAnalytical` (multiple, dispersion chosen so pixel offsets are integers), `tests/test_grism_shared_cube.py` point-source 90-degree-rotation tests | Closed-form (bilinear is exact at integer offsets, so these reduce to an exact analytic answer) | Only tests the degenerate case where interpolation error is exactly zero by construction. |
| Dispersal interpolation, general fractional-shift case | `tests/test_galsim_reference.py::test_static_scene`, indirectly | Independent (GalSim-chromatic), but not isolated | The static-scene floor (1.678% max\|diff\|/peak) is understood to be dominated by this bilinear-shift bias at realistic (~2.2 fine-pixel) shifts, but the gate bounds the *combined* floor (interpolation + quintic-interpolant + other numerics), not this term alone. **No dedicated, isolated test exists** that gates the fractional-shift interpolation error against a closed-form or independent reference at a chosen shift magnitude. |
| PSF pathway (`post_dispersion` vs `per_slice` ordering) | `tests/test_grism_psf_mode.py` (equivalence tests, `L1(A-B)/F_tot <= C * folding_threshold`) | Internal self-consistency (both `kl_pipe` code paths) | Mathematically the two orderings are identical for a wavelength-independent PSF, so this is a correctness-preserving-refactor gate, not an external-accuracy gate. External PSF accuracy for the *broadband* case (no dispersion) is separately covered in `tests/test_intensity.py`/`test_psf.py` (GalSim regression). |
| Flux conservation | `tests/test_cube_psf.py::test_cube_psf_flux_conservation`, `tests/test_grism_core.py::test_disperse_flux_conservation`/`test_flux_conservation`, `tests/test_pixel_readout.py::TestClosedFormGate::test_flux_conservation`, `tests/test_galsim_reference.py` (raw flux-ratio check, both scenes) | Mix: closed-form (pixel readout) + independent (GalSim-chromatic flux ratio) + internal (cube/dispersion sum checks) | Well covered from multiple independent angles; this is the axis least likely to hide a bug. |
| Shared-cube multi-roll (rotated-sampling dispersion for multiple position angles sharing one cube) | `tests/test_grism_shared_cube.py::TestFisherGate` (Fisher-projected posterior-shift bound) | Internal (compares shared-cube fast path to per-roll reference path within `kl_pipe`) | The metric that originally exposed the truncation-term bias driving this pathway's design; not checked against an external code or independent render. |
| Continuum dispersal | `tests/test_spectral_methods.py`, `tests/test_grism_core.py::TestContinuumModel`, `tests/test_grism_likelihood.py`, `tests/test_datacube.py`, `tests/test_cube_psf.py`, `tests/test_grism_psf_mode.py` (all internal correctness/exactness tests) | **None independent** | Neither the geko cross-code tier (`continuum: 0.0` fixed in all 28 combos) nor the GalSim-chromatic reference (`oracle_physics`/`galsim_reference.physics` has no continuum term) cross-checks continuum dispersal against anything outside `kl_pipe`. Gap; severity moderate (continuum is used in production joint phot+grism fits per `test_grism_likelihood.py`). Next step: extend the GalSim-chromatic reference with a flat-SED chromatic component (straightforward -- same `ChromaticSum` machinery, no isovelocity binning needed since continuum has no line-center Doppler structure to resolve) or extend geko's `continuum` param off zero. |
| Sheared scenes (`g1`, `g2` != 0) through the grism pipeline | `tests/test_flagship.py` (full joint phot+grism recovery at `g1=0.02, g2=-0.01`), presumably `tests/test_likelihood_slices.py` grism-inclusive slices | **None independent** | Geko has no shear model at all (`test_params.yaml` fixes `g1=g2=0` for all 28 combos). `galsim_reference.kl_pipe_scene.true_pars_dotted` also fixes `g1=g2=0`. Shear/centroid transform correctness for the *grism* pathway rests entirely on internal tests plus the (non-grism) `source2gal`/`cen2source` unit tests. Severity: moderate-to-high given shear recovery is the pipeline's primary science deliverable; mitigated by shear being validated independently for the *imaging* pathway (`test_intensity.py` GalSim regressions include shear) -- the residual risk is specifically in how shear interacts with dispersion/cube assembly, not the shear transform itself. |
| Off-center sources (`x0`, `y0` != 0) | `tests/test_grism_shared_cube.py` (off-center point-source tests, rotation-consistency) | Internal | Geko's 28 combos have no `x0`/`y0` override; `galsim_reference` hardcodes `x0=y0=0`. Lower severity than shear -- centroid handling is geometrically simple and shares code with the well-tested imaging pathway. |
| Multi-line / doublets | `tests/test_source.py::test_doublet_auto_resolved`, `tests/test_grism_core.py` (Halpha + NII6584 two-line test) | Internal | No external or independent-render cross-check of multi-line cube assembly (shared wavelength grid sizing across separated lines, potential slice-budget interactions). Severity low-moderate; most near-term science targets are single-line. |
| Wavelength-dependent PSF | **Not implemented** (tracked as issue #51; `GrismObs` docstring explicitly notes "one shared PSF ... across all emission lines," real instruments have wavelength-dependent PSFs) | N/A | This is a feature gap, not a test gap -- there is nothing to test yet. Flagged here so it is not mistaken for an oversight. |
| Non-Gaussian PSF/LSF in the grism pathway | None in any `test_grism_*.py`/`test_cube_psf.py`/`test_lsf_gate.py` file (all grism PSFs are `galsim.Gaussian`) | N/A within grism scope | Moffat and Airy PSFs *are* regression-tested for the broadband imaging pathway (`tests/test_psf.py`), so the PSF-convolution machinery itself is validated against non-Gaussian kernels -- but never carried through dispersion. `tests/test_lsf_gate.py` validates the Roman spectral-resolution spec but also uses a Gaussian PSF. Severity low: the grism PSF-convolution code path (`_apply_post_dispersion_pixel_response`, FFT convolution) does not special-case PSF shape, so risk is concentrated in edge cases (very sharp/cuspy kernels aliasing on the fine grid) rather than a likely silent bug. |
| Throughput / wavelength-dependent sensitivity (`GrismPars.throughput`) | **No test references it at all** (`grep -rl throughput tests/` finds nothing) | N/A | The field exists and is wired through `disperse_cube` (defaults to flat/`None` = 100%), but zero tests exercise a non-flat throughput function. This is a real, currently-silent gap: a bug in how `throughput` is applied (indexing, normalization, `dlam` weighting) would not be caught by any existing test. Severity moderate if throughput is used in any current or planned fit. |
| Roll-angle / dispersion-angle sweeps vs external code | `tests/test_grism_core.py::TestDispersion` (`dispersion_angle=0, pi/2`, others), `tests/test_grism_shared_cube.py` (rotation-consistency, cos/sin sign conventions) | Internal only | The geko cross-code tier explicitly removed its dispersion-angle sweep (`tests/test_grism_validation.py` docstring: "tests kl_pipe-internal cos/sin decomposition, retained in test_grism_core.py"). `galsim_reference.render.render_galsim_reference` also hardcodes dispersion along +x only (`shift_fn` returns `(dx, 0.0)`). No angle other than 0/90 degrees is checked against anything outside `kl_pipe`. |

## Summary of gaps by severity

**Moderate-to-high, worth prioritizing:**
- Sheared scenes through the grism pathway (no independent check at all; shear recovery is the pipeline's core science output).
- Throughput (silently zero test coverage of any kind, not just "no external reference").

**Moderate:**
- Continuum dispersal (no independent reference; used in production joint fits).
- Fractional-shift dispersal interpolation error is bounded only in combination with other floor terms, not isolated.

**Low-to-moderate:**
- Off-center sources, multi-line/doublets, roll-angle sweeps vs external code -- internally tested, plausible-but-unconfirmed against anything external.

**Not a test gap (feature gap or genuinely out of scope):**
- Wavelength-dependent PSF: not implemented (issue #51).
- Non-Gaussian PSF/LSF in the grism pathway: PSF-shape robustness is validated for imaging; grism-specific risk is judged low given the convolution code is PSF-shape-agnostic.

## Relationship to the two cross-code/reference tiers

Neither tier alone would close all of the above gaps even if fully
extended -- see `docs/validation/galsim_reference_gate.md` for what the
GalSim-chromatic reference currently covers, and the (superseded)
`docs/validation/oracle_promotion_audit.md` promotion audit (primary
checkout, to be deleted after merge) for the original geko-vs-reference
comparison this table's continuum/shear/off-center/multi-line rows are
partly drawn from.
