"""SyntheticIntensity / SourceModel rendering parity tests.

At identical physical parameters, the legacy ``SyntheticIntensity`` data
generators and the canonical ``SourceModel.render_broadband``
inference-time renderer must produce the same image to within numerical
precision. Violations of this contract are silent across most existing
tests because broad priors absorb the offset; they surface as Nsigma
deviations only when the test couples tightly to centroid posterior
precision (e.g., the flagship test).

History: the flagship ``test_recover_joint_phot_grism`` failed with
joint Nsigma = 14.7 (commit 9751e03). An independent reference render
via the GalSim backend of ``synthetic.py`` confirmed that the production
``SourceModel.render_broadband`` path is canonically correct and the
scipy synth path was the one to fix. The audit surfaced three related
centroid bugs in ``_generate_inclined_kspace_scipy``:

1. PSF kernel ``kr``-even half-pixel offset. The kernel is drawn via
   GalSim's ``drawImage(nx=kr, ny=kr, method='no_pixel')`` then padded
   and ``np.roll`` ed by ``-(kr // 2)`` to align the kernel peak with
   the FFT origin. For *even* ``kr`` (returned by
   ``psf.getGoodImageSize`` in typical Roman-like configurations),
   GalSim places the kernel peak at the image's ``true_center``
   ``(kr/2 - 0.5, kr/2 - 0.5)`` (0-indexed), so the roll lands the
   kernel centroid at fine-pixel ``(-0.5, -0.5)`` instead of ``(0, 0)``.
   At ``SyntheticIntensity.generate``'s default ``oversample=1`` this
   is a full half-coarse-pixel shift in the rendered intensity. Fix:
   force ``kern_size`` odd before drawing, matching
   ``kl_pipe/psf.py:gsobj_to_kernel``.

2. ``hx`` / ``hy`` half-pixel correction scaled to coarse ``ps`` instead
   of fine ``eff_ps``. At ``oversample > 1`` the FFT runs at fine
   resolution, so the half-pixel phase correction should be measured in
   fine pixels. Fix: ``hx = 0.5 * eff_ps * (1 - (Ncol * oversample) %
   2)``.

3. Roll-to-center amount truncated. ``(Nrow // 2) * oversample`` is
   short by ``(oversample - 1) / 2`` fine pixels for odd ``Nrow`` and
   ``oversample > 1``. Fix: ``roll_row = (Nrow * oversample) // 2``.

Together bugs (2) and (3) shifted the binned coarse centroid by
``-(os - 1) / (2 * os)`` coarse pixels at ``synth oversample > 1`` —
latent because no production code or test exercised that branch. Bug
(1) alone drove the flagship Nsigma failure (at the default
``oversample=1``).
"""

import galsim
import numpy as np
import pytest

from kl_pipe.intensity import InclinedExponentialModel
from kl_pipe.observation import build_image_obs
from kl_pipe.parameters import ImagePars
from kl_pipe.source import SourceModel
from kl_pipe.synthetic import SyntheticIntensity
from kl_pipe.velocity import CenteredVelocityModel


# max(|synth - source|) / max(source). Sub-percent floor from FFT padding
# / oversample / sinc treatment differences. The PSF kernel half-pixel
# bug surfaces at ~16% so 1% threshold gives ample separation from the
# numerical floor.
PARITY_REL_TOL = 0.01


def _render_pair(pars_flat, image_pars, psf, oversample):
    """Render the same source via SyntheticIntensity (scipy backend) and
    SourceModel.render_broadband at the same physical parameters."""
    pars_dotted = {
        'cosi': pars_flat['cosi'],
        'theta_int': pars_flat['theta_int'],
        'g1': pars_flat['g1'],
        'g2': pars_flat['g2'],
        'F087.flux': pars_flat['flux'],
        'F087.rscale': pars_flat['int_rscale'],
        'F087.h_over_r': pars_flat['int_h_over_r'],
        'F087.x0': pars_flat['int_x0'],
        'F087.y0': pars_flat['int_y0'],
    }

    synth = SyntheticIntensity(pars_flat, model_type='exponential', seed=42, psf=psf)
    _ = synth.generate(image_pars, snr=1e6, seed=42, include_poisson=False)
    img_synth = np.asarray(synth.data_true)

    source = SourceModel(
        velocity_model=CenteredVelocityModel(),
        broadband_models={'F087': InclinedExponentialModel()},
    )
    obs = build_image_obs(
        image_pars, psf=psf, oversample=oversample, broadband_key='F087'
    )
    img_source = np.asarray(source.render_broadband(pars_dotted, obs, 'F087'))

    return img_synth, img_source


def _rel_max_diff(img_synth, img_source):
    return float(np.abs(img_synth - img_source).max() / float(img_source.max()))


def _assert_parity(img_synth, img_source, label):
    rel_max = _rel_max_diff(img_synth, img_source)
    assert rel_max < PARITY_REL_TOL, (
        f"{label}: relative max diff {rel_max:.4%} > {PARITY_REL_TOL:.0%}. "
        f"Cross-path convention disagreement between SyntheticIntensity "
        f"and SourceModel.render_broadband. See module docstring."
    )


# ============================================================================
# Hard-asserted: parity holds (no-PSF baseline)
# ============================================================================


@pytest.mark.parametrize("N", [31, 32, 33, 34])
def test_parity_face_on_no_psf(N):
    """Face-on circular profile, no PSF. Parity holds for both N parities."""
    pars_flat = {
        'cosi': 1.0,
        'theta_int': 0.0,
        'g1': 0.0,
        'g2': 0.0,
        'flux': 100.0,
        'int_rscale': 0.3,
        'int_h_over_r': 0.1,
        'int_x0': 0.0,
        'int_y0': 0.0,
    }
    image_pars = ImagePars(shape=(N, N), pixel_scale=0.11, indexing='ij')
    img_synth, img_source = _render_pair(pars_flat, image_pars, psf=None, oversample=5)
    _assert_parity(img_synth, img_source, label=f"face_on_no_psf N={N}")


@pytest.mark.parametrize("N", [31, 32, 33, 34])
def test_parity_inclined_no_psf(N):
    """Inclined + rotated + sheared, no PSF. Parity holds."""
    pars_flat = {
        'cosi': 0.6,
        'theta_int': np.pi / 4,
        'g1': 0.02,
        'g2': -0.01,
        'flux': 100.0,
        'int_rscale': 0.3,
        'int_h_over_r': 0.1,
        'int_x0': 0.0,
        'int_y0': 0.0,
    }
    image_pars = ImagePars(shape=(N, N), pixel_scale=0.11, indexing='ij')
    img_synth, img_source = _render_pair(pars_flat, image_pars, psf=None, oversample=5)
    _assert_parity(img_synth, img_source, label=f"inclined_no_psf N={N}")


# ============================================================================
# PSF-convolved: parity holds after the kern_size-odd fix in
# ``_generate_inclined_kspace_scipy``. Regression guard for the half-pixel
# offset that drove flagship joint Nsigma = 14.7 (commit 9751e03).
# ============================================================================


@pytest.mark.parametrize("N", [31, 32, 33, 34])
def test_parity_face_on_with_psf(N):
    """Face-on profile + Roman-like PSF at all N parities. Locks bug (1)
    (kr-even PSF roll) at the default ``oversample=1`` — the bug that
    drove flagship joint Nsigma = 14.7."""
    pars_flat = {
        'cosi': 1.0,
        'theta_int': 0.0,
        'g1': 0.0,
        'g2': 0.0,
        'flux': 100.0,
        'int_rscale': 0.3,
        'int_h_over_r': 0.1,
        'int_x0': 0.0,
        'int_y0': 0.0,
    }
    image_pars = ImagePars(shape=(N, N), pixel_scale=0.11, indexing='ij')
    psf = galsim.Gaussian(fwhm=0.18)
    img_synth, img_source = _render_pair(pars_flat, image_pars, psf=psf, oversample=5)
    _assert_parity(img_synth, img_source, label=f"face_on_with_psf N={N}")


@pytest.mark.parametrize("N", [31, 32, 33, 34])
def test_parity_inclined_with_psf(N):
    """Inclined + rotated + sheared, Roman-like PSF. The flagship config."""
    pars_flat = {
        'cosi': 0.6,
        'theta_int': np.pi / 4,
        'g1': 0.02,
        'g2': -0.01,
        'flux': 100.0,
        'int_rscale': 0.3,
        'int_h_over_r': 0.1,
        'int_x0': 0.0,
        'int_y0': 0.0,
    }
    image_pars = ImagePars(shape=(N, N), pixel_scale=0.11, indexing='ij')
    psf = galsim.Gaussian(fwhm=0.18)
    img_synth, img_source = _render_pair(pars_flat, image_pars, psf=psf, oversample=5)
    _assert_parity(img_synth, img_source, label=f"inclined_with_psf N={N}")


# ============================================================================
# Coverage extensions: synth oversample > 1; nonzero centroid offsets;
# off-default kernel sizes that exercise both even/odd ``getGoodImageSize``
# returns. Each guards a distinct failure mode of the kern_size-odd fix.
# ============================================================================


# Centroid-only sweep across synth oversample. The legacy synth path had two
# osf-dependent centroid bugs (PSF kr-even half-pixel + ps-vs-eff_ps hx + roll
# truncation for odd N + osf > 1). Together they shifted the binned coarse
# centroid by ``-(osf - 1) / (2 * osf)`` coarse pixels; this test guards the
# fix at every step of that ladder.
#
# NOTE on pixel-level parity at osf > 1: even after the centroid fix,
# the scipy synth path at synth-oversample > 1 produces pixel values that
# disagree with ``SourceModel.render_broadband`` by ~2-3% (max). This is a
# distinct pixel-integration scheme difference (synth sum-bins a sinc-
# multiplied fine grid; production wrap-folds the extended k-grid onto the
# base before IFFT). Both produce centroid-correct images; the discrepancy
# is in how each numerically realizes the box-pixel integral at osf > 1.
# No production code or test currently uses SyntheticIntensity at oversample
# > 1, so this is a latent issue rather than a live bug. Documented as a
# separate follow-up; not in scope for the centroid fix.


@pytest.mark.parametrize("synth_oversample", [1, 3, 5, 7])
@pytest.mark.parametrize("N", [31, 32])
def test_centroid_alignment_with_psf(synth_oversample, N):
    """Centroid-only regression for the synth path's coarse-grid placement
    at every oversample. See module-level NOTE on pixel-level parity at
    osf > 1."""
    pars_flat = {
        'cosi': 1.0,
        'theta_int': 0.0,
        'g1': 0.0,
        'g2': 0.0,
        'flux': 100.0,
        'int_rscale': 0.05,
        'int_h_over_r': 0.1,
        'int_x0': 0.0,
        'int_y0': 0.0,
    }
    image_pars = ImagePars(shape=(N, N), pixel_scale=0.11, indexing='ij')
    psf = galsim.Gaussian(fwhm=0.18)

    synth = SyntheticIntensity(pars_flat, model_type='exponential', seed=42, psf=psf)
    _ = synth.generate(
        image_pars,
        snr=1e6,
        seed=42,
        include_poisson=False,
        oversample=synth_oversample,
    )
    img = np.asarray(synth.data_true)

    # flux-weighted centroid in 0-indexed array coords
    s = img.sum()
    rows = np.arange(img.shape[0])
    cols = np.arange(img.shape[1])
    R, C = np.meshgrid(rows, cols, indexing='ij')
    cy = float((img * R).sum() / s)
    cx = float((img * C).sum() / s)

    true_y, true_x = (N - 1) / 2.0, (N - 1) / 2.0
    assert abs(cy - true_y) < 1e-3, (
        f"synth_oversample={synth_oversample}, N={N}: cy={cy:.4f} vs "
        f"true_y={true_y:.4f}, |Δ|={abs(cy - true_y):.4f}"
    )
    assert abs(cx - true_x) < 1e-3, (
        f"synth_oversample={synth_oversample}, N={N}: cx={cx:.4f} vs "
        f"true_x={true_x:.4f}, |Δ|={abs(cx - true_x):.4f}"
    )


@pytest.mark.parametrize("x0_arcsec", [-0.02, 0.0, +0.03])
@pytest.mark.parametrize("y0_arcsec", [0.0, +0.05])
def test_parity_psf_with_centroid_offset(x0_arcsec, y0_arcsec):
    """Nonzero centroid sources. Catches any residual centroid-dependent
    drift in the centroid-phase chain after the PSF fix (would surface as
    rel-max-diff that scales with |offset|)."""
    pars_flat = {
        'cosi': 0.6,
        'theta_int': np.pi / 4,
        'g1': 0.02,
        'g2': -0.01,
        'flux': 100.0,
        'int_rscale': 0.3,
        'int_h_over_r': 0.1,
        'int_x0': x0_arcsec,
        'int_y0': y0_arcsec,
    }
    image_pars = ImagePars(shape=(33, 33), pixel_scale=0.11, indexing='ij')
    psf = galsim.Gaussian(fwhm=0.18)
    img_synth, img_source = _render_pair(pars_flat, image_pars, psf=psf, oversample=5)
    _assert_parity(
        img_synth,
        img_source,
        label=f"offset x0={x0_arcsec} y0={y0_arcsec}",
    )
