# GalSim-Chromatic Reference Gate

## What it is

An independent reference render of a dispersed slitless-grism image, used to
cross-check `kl_pipe`'s `render_grism` pipeline (cube assembly + dispersion
+ PSF + pixel readout) pixel-by-pixel.

Implementation: `scripts/validation/galsim_reference/`.

- `physics.py` -- numpy-only intensity (3D inclined-exponential LOS
  quadrature, dense trapezoidal rule) and velocity (arctan rotation curve)
  fields. No `kl_pipe` import.
- `render.py` -- partitions the velocity field into `n_v` isovelocity-bin
  channel images, wraps each as a `galsim.InterpolatedImage * SED`
  (Gaussian line, Doppler-centered per bin), sums via
  `galsim.ChromaticSum`, and disperses with
  `ChromaticObject.shift(callable_of_wavelength)` -- a callable that maps
  wavelength to a spatial (dx, dy) shift, letting GalSim integrate the
  wavelength-dependent shift and PSF convolution together in
  `drawImage`.
- `kl_pipe_scene.py` -- builds the matching `kl_pipe` `SourceModel` +
  `GrismObs` scene and renders it via `render_grism`.

This construction shares no code with `kl_pipe`'s (x, y, lambda)
cube-assembly + `disperse_cube` pathway (`kl_pipe/spectral.py`,
`kl_pipe/dispersion.py`, `kl_pipe/grism.py`): rather than building a
discrete wavelength-slice datacube and shifting slices by whole/fractional
coarse pixels, it sums continuous chromatic objects and lets GalSim handle
the wavelength integral internally. Agreement between the two therefore
pins pipeline *mechanics* (isovelocity/wavelength discretization, dispersion
shift, PSF pathway, pixel readout), not just a shared parametric-model
implementation -- that role is filled by the geko cross-code tier
(`tests/test_grism_validation.py`), which uses a fully external codebase but
shares `kl_pipe`'s coarse-`n_lambda` dispersion-grid limitation and would
not have caught the bugs below.

This reference render was originally developed as
`experiments/sweverett/galsim_grism_oracle/` and found two real bugs (one
already fixed, one an open limitation, both below) before being promoted
into the test suite.

## Test gate: `tests/test_galsim_reference.py`

Marker: `galsim_reference` (registered in `pyproject.toml`). No external
environment or reference data required -- both renders happen in-process.

Run:

```
make test-galsim-reference
# or
conda run -n klpipe pytest tests/test_galsim_reference.py -v -m galsim_reference
```

Runtime: ~20-25s for all four scenes (well under the 120s `slow`-marker
threshold), so the gate runs by default in `make test` / `make test-basic`
/ `make test-extended` -- it is not excluded by any marker filter.

### Scenes gated

All: inclined exponential disk (cosi=0.6, theta_int=pi/4), Halpha line
only, z=1.0, no continuum/off-center. Continuum and off-center remain
unextended; see the gap table in
`docs/validation/rendering_test_coverage.md`.

1. **Static** (`vcirc=0`): v_los is spatially uniform (`=v0=10 km/s`), so
   there is no velocity-entanglement effect to resolve; kl_pipe's
   *default* `n_lambda` is used.
2. **Dynamic** (`vcirc=200`): a real rotation curve, spatially-varying
   Doppler shift, rendered with a caller-tuned, refined `n_lambda=251`.
   This tests demonstrated-achievable accuracy on the entanglement
   pathway, not default behavior.
3. **Sheared dynamic** (`vcirc=200`, `n_lambda=251`, `g1=0.05, g2=0.03`,
   ~2.5x the flagship shear): reduced shear through the full grism
   pathway. The reference applies GalSim's own area-preserving `.shear()`
   to the isovelocity channel images (shearing intensity and velocity
   structure together) -- the same transform convention as `kl_pipe`'s
   `cen2source`, implemented independently. Measured agreement sits at or
   below the unsheared dynamic floor (0.428%/0.019% vs 0.534%/0.030%);
   flagship-value shear `g=(0.02, -0.01)` measured 0.487%/0.025%, same
   conclusion. Closes the "sheared grism scenes have no independent
   cross-check" gap.
4. **Ramp-throughput static** (`vcirc=0`, default `n_lambda`,
   `T(lambda) = 0.5 + 0.02*(lambda - lambda_ref)`, a factor ~3 across the
   window): wavelength-dependent `GrismPars.throughput` vs the same
   T(lambda) applied as the reference's GalSim draw bandpass. The ramp
   changes the kl_pipe image by ~50% of peak vs flat (asserted in-test),
   so a silently-ignored or index-misaligned throughput cannot pass.
   Measured agreement sits at the flat static floor (1.715%/0.061% vs
   1.678%/0.060%). First test coverage of `GrismPars.throughput`
   anywhere in the suite; per-slice indexing is additionally pinned
   closed-form by `tests/test_grism_core.py::TestDispersion`
   throughput tests (one-hot slice selection, ramp orientation,
   loop-vs-operator pathway agreement at integer offsets).

kl_pipe's *default*-`n_lambda` regression on the dynamic case (under-
resolving the spatially-varying Doppler field: max|diff|/peak jumps from
0.53% at `n_lambda=251` to ~6.4% at the default) is intentionally **not**
re-tested here -- it is already pinned in-suite by
`tests/test_pixel_readout.py::TestDefaultWavelengthGridDeviation`, which freezes the
measured default-vs-refined deviation window so a silent change (better or
worse) trips a test.

### Frozen tolerances (measure-then-freeze, ~3x headroom)

Both images are normalized to unit total flux before differencing (matches
GalSim's exact but arbitrarily-scaled SED normalization vs `kl_pipe`'s flux
convention -- absolute flux agreement is checked separately via the raw
flux ratio).

| Scene | Metric | Measured (2026-07-05, this gate, float64) | Frozen bound |
|---|---|---|---|
| Static (`vcirc=0`) | max\|diff\|/peak | 1.678% | < 5.0% |
| Static | mean\|diff\|/peak | 0.060% | < 0.2% |
| Static | \|flux ratio - 1\| | 0.343% | < 0.5% |
| Dynamic (`vcirc=200`, `n_lambda=251`) | max\|diff\|/peak | 0.534% | < 1.5% |
| Dynamic | mean\|diff\|/peak | 0.030% | < 0.1% |
| Dynamic | \|flux ratio - 1\| | 0.399% | < 0.5% |
| Sheared dynamic (`g1=0.05, g2=0.03`) | max\|diff\|/peak | 0.428% | < 1.5% |
| Sheared dynamic | mean\|diff\|/peak | 0.019% | < 0.1% |
| Sheared dynamic | \|flux ratio - 1\| | 0.247% | < 0.5% |
| Ramp-throughput static | max\|diff\|/peak | 1.715% | < 5.0% |
| Ramp-throughput static | mean\|diff\|/peak | 0.061% | < 0.2% |
| Ramp-throughput static | \|flux ratio - 1\| | 0.159% | < 0.5% |

**The two scenes are frozen at different bounds because they have
different, independently-understood floors** -- this is not a case of
picking a bound to make a test pass. The static scene's higher residual is
dominated by `disperse_cube`'s bilinear sub-pixel shift interpolation bias
(~0.8-1% for an exponential profile at ~2.2 fine-pixel shifts, measured
directly by isolating the shift step alone during the original
investigation) plus GalSim's own quintic-interpolant floor; refining
`n_lambda` does **not** reduce it (a uniform-velocity control scene stayed
flat at 0.7-1.7% across `n_lambda` in {25, 51, 101, 251, 1001} during the
original investigation, confirming the floor is unrelated to wavelength-
grid coarseness). The dynamic scene's 0.534%/0.030% pair reproduces the
original investigation's numbers exactly and is the load-bearing
measurement: it demonstrates that once `n_lambda` is refined, the
entanglement-driven residual falls to the same numerical floor as the
static scene's non-entanglement floor -- i.e., no further undiagnosed bug
remains once refined.

Flux-ratio tolerance (0.5%) is shared across all scenes: flux is
conserved through dispersion/PSF/pixel-readout by construction (throughput-
weighted on both sides in the ramp scene), so this is a much harder
physical constraint than shape agreement and every scene sits comfortably
under it (0.16-0.40%).

The sheared and ramp-throughput scenes are frozen at their parent scenes'
bounds (dynamic and static respectively) because their measured floors are
statistically identical to the parents' -- shear and throughput introduce
no additional disagreement of their own.

## Bugs found during development (context)

1. **Fixed** (`kl_pipe/grism.py::_apply_post_dispersion_pixel_response`,
   merged prior to promotion): the post-dispersion readout mean-binned the
   sinc-integrated fine field over the whole oversample block instead of
   sampling the coarse-pixel-aligned fine center, an unintended second box
   convolution. Biased compact-source peaks low by several percent while
   conserving flux exactly (which is why flux-only checks never caught
   it). Pinned by `tests/test_pixel_readout.py::TestMeanBinReadoutBug`.
2. **Open limitation, not a bug**: `GrismPars.to_cube_pars`'s default
   `n_lambda` spaces wavelength slices by ~1 dispersion pixel. For a
   spatially uniform velocity this is fine (erf-exact per-bin amplitude
   regardless of bin width). For a real rotation curve, different sky
   regions with different `v_los` get pooled into the same coarse
   wavelength bin and dispersed to that bin's single nominal shift,
   quantizing dispersed *position* by up to half a bin width. Not fixed by
   `spectral_method` (that only changes intra-bin amplitude sampling, not
   bin count). Practical implication: users whose `vcirc*sin(i)` span is
   comparable to or smaller than one dispersion pixel's velocity width
   should pass an explicit finer `n_lambda`. Pinned (as a frozen deviation
   window, not a pass/fail correctness gate) by
   `tests/test_pixel_readout.py::TestDefaultWavelengthGridDeviation`.

## Superseded document

The promotion plan and its measurement audit lived at
`docs/validation/oracle_promotion_audit.md` in the primary checkout
(untracked). That document should be deleted after this gate is merged --
its content is now split between this file and
`docs/validation/rendering_test_coverage.md`.
