"""GalSim-chromatic reference gate for dispersed grism rendering.

Cross-checks ``kl_pipe``'s ``render_grism`` against an independent
construction: ``scripts/validation/galsim_reference`` builds isovelocity-bin
channel images from a from-scratch numpy LOS-quadrature intensity model and
arctan rotation curve, wraps each channel as a ``galsim.ChromaticObject``
with a Gaussian-line SED, and disperses via
``ChromaticObject.shift(callable_of_wavelength)`` -- entirely independent of
``kl_pipe``'s (x, y, lambda) cube-assembly + ``disperse_cube`` code path.
Both sides share no code; agreement pins the pipeline's cube/dispersion/PSF/
pixel-readout mechanics, not just a shared parametric-model implementation
(that role is filled by the geko cross-code tier, ``test_grism_validation.py``).

Four scenes, all Halpha line only, no continuum/off-center (see
``docs/validation/galsim_reference_gate.md``):

1. Static (``vcirc=0``): sanity floor, decoupled from any velocity-driven
   wavelength-grid effect. Uses kl_pipe's *default* ``n_lambda``.
2. Dynamic (``vcirc=200``): real rotation curve, spatially-varying Doppler
   shift. Uses a caller-tuned, refined ``n_lambda=251`` -- this pathway is
   demonstrated-achievable accuracy, not default behavior. kl_pipe's
   *default*-``n_lambda`` regression (under-resolving the spatially-varying
   Doppler field, ~6.4% max|diff|/peak vs a refined grid) is already pinned
   in-suite by ``tests/test_pixel_readout.py::TestDefaultWavelengthGridDeviation`` and
   is NOT re-tested here.
3. Sheared dynamic (``g1=0.05, g2=0.03``): reduced shear through the full
   grism pathway vs GalSim's own ``.shear()`` on the reference channels.
4. Ramp-throughput static: wavelength-dependent ``GrismPars.throughput``
   vs the same T(lambda) applied as the reference's draw bandpass.

All tolerances are measure-then-freeze (repo convention: measured value
recorded in a comment, frozen bound at ~3x headroom). The two scenes have
DIFFERENT floors and are frozen separately -- the static scene's residual is
dominated by ``disperse_cube``'s bilinear sub-pixel shift interpolation
(unaffected by ``n_lambda``, per the promotion audit's elimination sweep:
a uniform-shift control stayed flat at 0.7-1.7% across ``n_lambda`` in
{25, 51, 101, 251, 1001}), while the dynamic scene's residual is the
n_lambda-refinement floor once the entanglement effect above is removed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# add scripts/validation to path for the galsim_reference package (mirrors
# tests/test_grism_validation.py's import pattern for scripts/validation/utils.py)
_SCRIPTS_DIR = Path(__file__).parent.parent / 'scripts' / 'validation'
sys.path.insert(0, str(_SCRIPTS_DIR))

from galsim_reference.kl_pipe_scene import build_kl_pipe_scene, render_kl_pipe_grism
from galsim_reference.physics import GalaxyParams
from galsim_reference.render import GalSimReferenceConfig, render_galsim_reference

Z = 1.0
LAMBDA_REST_HALPHA = 656.28  # nm, Halpha vacuum rest wavelength
LAMBDA_REF = LAMBDA_REST_HALPHA * (1.0 + Z)
PIXEL_SCALE = 0.11  # arcsec, Roman-like
SHAPE = (32, 32)
PSF_FWHM = 0.18  # arcsec, Gaussian
DISPERSION_NM_PER_PIX = 1.1
OVERSAMPLE = 5

# shared galaxy truth params (matches kl_pipe_scene.true_pars_dotted, minus vcirc)
_BASE_GALAXY_KWARGS = dict(
    cosi=0.6,
    theta_int=np.pi / 4,
    flux=100.0,
    rscale=0.25,
    h_over_r=0.1,
    v0=10.0,
    vel_rscale=0.3,
    sigma_v=50.0,
    z=Z,
    lambda_rest=LAMBDA_REST_HALPHA,
)

_REFERENCE_CFG = GalSimReferenceConfig(n_v=64, pts_per_sigma=20)


def _render_both(
    vcirc: float,
    n_lambda: int | None,
    g1: float = 0.0,
    g2: float = 0.0,
    throughput_fn=None,
    dispersal_method: str = 'slice',
):
    """Render the same scene through kl_pipe and the GalSim-chromatic
    reference. Returns (kl_pipe_image, reference_image), both flux/pixel.
    """
    scene = build_kl_pipe_scene(
        z=Z,
        vcirc=vcirc,
        pixel_scale=PIXEL_SCALE,
        shape=SHAPE,
        psf_fwhm=PSF_FWHM,
        dispersion_nm_per_pix=DISPERSION_NM_PER_PIX,
        oversample=OVERSAMPLE,
        n_lambda=n_lambda,
        g1=g1,
        g2=g2,
        throughput_fn=throughput_fn,
        dispersal_method=dispersal_method,
    )
    kl_image = render_kl_pipe_grism(scene)

    p = GalaxyParams(vcirc=vcirc, g1=g1, g2=g2, **_BASE_GALAXY_KWARGS)
    out = render_galsim_reference(
        p,
        coarse_pixel_scale=PIXEL_SCALE,
        coarse_shape=SHAPE,
        psf_fwhm=PSF_FWHM,
        dispersion_nm_per_pix=DISPERSION_NM_PER_PIX,
        lambda_ref=LAMBDA_REF,
        cfg=_REFERENCE_CFG,
        throughput_fn=throughput_fn,
    )
    return kl_image, out['image']


def _ramp_throughput(lam):
    """Linear throughput ramp, factor ~3 across the wavelength window
    (positive everywhere the cube and reference windows reach)."""
    return 0.5 + 0.02 * (np.asarray(lam) - LAMBDA_REF)


def _assert_agreement(kl_image, ref_image, max_tol, mean_tol, flux_tol, label):
    """Normalized-shape + raw-flux agreement checks, matching the
    promotion audit's metric definitions exactly (both images normalized
    to unit total flux before differencing).
    """
    kl_norm = kl_image / kl_image.sum()
    ref_norm = ref_image / ref_image.sum()
    diff = ref_norm - kl_norm
    peak = kl_norm.max()

    max_over_peak = float(np.abs(diff).max() / peak)
    mean_over_peak = float(np.abs(diff).mean() / peak)
    flux_rel_diff = float(abs(ref_image.sum() / kl_image.sum() - 1.0))

    assert max_over_peak < max_tol, (
        f'{label}: max|diff|/peak={max_over_peak:.4%} exceeds frozen bound '
        f'{max_tol:.2%}'
    )
    assert mean_over_peak < mean_tol, (
        f'{label}: mean|diff|/peak={mean_over_peak:.4%} exceeds frozen bound '
        f'{mean_tol:.2%}'
    )
    assert flux_rel_diff < flux_tol, (
        f'{label}: raw flux ratio deviates from unity by {flux_rel_diff:.4%}, '
        f'exceeds frozen bound {flux_tol:.2%}'
    )


@pytest.mark.galsim_reference
class TestGalSimReferenceGate:
    """Cross-check kl_pipe render_grism against the GalSim-chromatic
    reference render at two scenes (static, velocity-entangled)."""

    # flux is exactly conserved through dispersion/PSF/pixel-readout on
    # both sides; measured deviation from unity 0.34-0.40% across both
    # scenes below (2026-07-05, this gate, float64). Frozen with ~1.3x
    # headroom -- flux conservation is a much harder physical constraint
    # than shape agreement and should stay tight.
    FLUX_TOL = 0.005

    def test_static_scene(self):
        # vcirc=0: v_los is spatially uniform (=v0), so the velocity-
        # entanglement pathway (pinned in test_pixel_readout.py) does not
        # apply; default n_lambda is used. Measured (2026-07-05, float64,
        # this gate, oversample=5): max|diff|/peak=1.678%,
        # mean|diff|/peak=0.060%. Floor is dominated by disperse_cube's
        # bilinear sub-pixel shift interpolation bias (~0.8-1% for an
        # exponential profile at ~2.2 fine-pixel shifts, per the promotion
        # audit's stage3a isolation) -- NOT fixed by refining n_lambda
        # (audit's uniform-shift control stayed flat at 0.7-1.7% across
        # n_lambda in {25,...,1001}), so this scene is frozen at its own,
        # higher bound rather than reusing the dynamic scene's tighter one.
        kl_image, ref_image = _render_both(vcirc=0.0, n_lambda=None)
        _assert_agreement(
            kl_image,
            ref_image,
            max_tol=0.05,  # ~3x headroom over measured 1.678%
            mean_tol=0.002,  # ~3x headroom over measured 0.060%
            flux_tol=self.FLUX_TOL,
            label='static scene (vcirc=0)',
        )

    def test_sheared_dynamic_scene(self):
        # reduced shear through the full grism pathway (cube assembly +
        # dispersion + PSF + readout) vs the independent reference, which
        # applies GalSim's own .shear() to the isovelocity channels --
        # kl_pipe's cen2source is the same area-preserving transform, but
        # the two implementations share no code. g=(0.05, 0.03) is ~2.5x
        # the flagship shear, chosen to amplify any convention bug well
        # above the unsheared floor. Measured (2026-07-05, float64, this
        # gate, oversample=5, n_lambda=251): max|diff|/peak=0.428%,
        # mean|diff|/peak=0.019% -- at/below the unsheared dynamic floor
        # (0.534%/0.030%), so shear introduces no additional disagreement.
        # Flagship values g=(0.02, -0.01) measured 0.487%/0.025%, same
        # conclusion. Frozen at the dynamic scene's bounds.
        kl_image, ref_image = _render_both(vcirc=200.0, n_lambda=251, g1=0.05, g2=0.03)
        _assert_agreement(
            kl_image,
            ref_image,
            max_tol=0.015,
            mean_tol=0.001,
            flux_tol=self.FLUX_TOL,
            label='sheared dynamic scene (g1=0.05, g2=0.03)',
        )

    def test_throughput_ramp_static_scene(self):
        # wavelength-dependent throughput: kl_pipe applies per-slice
        # T(lambda_k) * dlam * quad_weight_k in disperse_cube; the
        # reference applies the same T(lambda) as the GalSim draw
        # bandpass (continuous SED x bandpass integration). The ramp
        # spans a factor ~3 across the wavelength window and changes the
        # kl_pipe image by ~50% of peak vs flat throughput, so a
        # silently-ignored or misaligned throughput cannot pass. Measured
        # (2026-07-05, float64, this gate, oversample=5, default
        # n_lambda): max|diff|/peak=1.715%, mean|diff|/peak=0.061%,
        # flux-ratio dev 0.159% -- at the flat static-scene floor
        # (1.678%/0.060%), so throughput adds no disagreement. Frozen at
        # the static scene's bounds.
        kl_ramp, ref_ramp = _render_both(
            vcirc=0.0, n_lambda=None, throughput_fn=_ramp_throughput
        )
        # kl-only flat control for the self-check (no reference render needed)
        flat_scene = build_kl_pipe_scene(
            z=Z,
            vcirc=0.0,
            pixel_scale=PIXEL_SCALE,
            shape=SHAPE,
            psf_fwhm=PSF_FWHM,
            dispersion_nm_per_pix=DISPERSION_NM_PER_PIX,
            oversample=OVERSAMPLE,
        )
        kl_flat = np.asarray(render_kl_pipe_grism(flat_scene))
        ramp_change = float(np.abs(kl_ramp - kl_flat).max() / kl_flat.max())
        assert ramp_change > 0.2, (
            f'throughput self-check: ramp changed the kl_pipe image by only '
            f'{ramp_change:.1%} of peak (expected ~50%); the throughput array '
            f'is not reaching disperse_cube'
        )
        _assert_agreement(
            kl_ramp,
            ref_ramp,
            max_tol=0.05,
            mean_tol=0.002,
            flux_tol=self.FLUX_TOL,
            label='ramp-throughput static scene (vcirc=0)',
        )

    def test_dynamic_scene_refined_n_lambda(self):
        # vcirc=200: real rotation curve, spatially-varying Doppler shift.
        # n_lambda=251 is a caller-tuned refinement of the wavelength grid
        # (kl_pipe's *default* n_lambda under-resolves this case; that
        # regression is pinned separately by
        # test_pixel_readout.py::TestDefaultWavelengthGridDeviation, not here).
        # Measured (2026-07-05, float64, this gate, oversample=5,
        # n_v=64, pts_per_sigma=20): max|diff|/peak=0.534%,
        # mean|diff|/peak=0.030% -- reproduces the promotion audit
        # exactly. Frozen at the audit's proposed ~3x-headroom bounds.
        kl_image, ref_image = _render_both(vcirc=200.0, n_lambda=251)
        _assert_agreement(
            kl_image,
            ref_image,
            max_tol=0.015,  # ~2.8x headroom over measured 0.534%
            mean_tol=0.001,  # ~3.3x headroom over measured 0.030%
            flux_tol=self.FLUX_TOL,
            label='dynamic scene (vcirc=200, n_lambda=251)',
        )

    def test_dynamic_scene_analytic_dispersal(self):
        # same dynamic scene through the closed-form dispersal path
        # (dispersal_method='analytic'; no wavelength grid for the line),
        # against the same independent GalSim-chromatic reference. The
        # analytic path is the n_lambda -> infinity limit of the slice
        # method, so it must sit at or below the refined-grid floor.
        # Measured (2026-07-06, float64, this gate, oversample=5):
        # max|diff|/peak=0.206%, mean|diff|/peak=0.019% -- well below the
        # n_lambda=251 slice floor (0.534%/0.030%): removing slice
        # quantization also removes part of the shift-interpolation
        # disagreement. Frozen at ~3x measured.
        kl_image, ref_image = _render_both(
            vcirc=200.0, n_lambda=None, dispersal_method='analytic'
        )
        _assert_agreement(
            kl_image,
            ref_image,
            max_tol=0.006,
            mean_tol=0.0006,
            flux_tol=self.FLUX_TOL,
            label='dynamic scene (vcirc=200, analytic dispersal)',
        )

    def test_throughput_ramp_analytic_dispersal(self):
        # ramp throughput through the analytic path: the line weight is
        # T interpolated at each spaxel's observed wavelength instead of
        # per-slice T(lambda_k). Measured (2026-07-06, float64, this
        # gate, oversample=5): max|diff|/peak=0.448%, mean=0.025% --
        # well below the slice-path static floor (1.715%/0.061%), same
        # interpolation-floor reduction as the dynamic scene. Frozen at
        # ~3x measured.
        kl_image, ref_image = _render_both(
            vcirc=0.0,
            n_lambda=None,
            throughput_fn=_ramp_throughput,
            dispersal_method='analytic',
        )
        _assert_agreement(
            kl_image,
            ref_image,
            max_tol=0.015,
            mean_tol=0.001,
            flux_tol=self.FLUX_TOL,
            label='ramp-throughput static scene (analytic dispersal)',
        )
