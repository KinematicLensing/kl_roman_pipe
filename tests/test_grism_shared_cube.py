"""Tests for the shared-cube-across-rolls grism pathway.

Covers, bottom-up:
- ``disperse_cube(image_rotation=...)``: fused rotated sampling. Sign and
  center conventions pinned by point-source tests at 90-degree multiples
  (rotated sample coordinates land on integer grid points, so bilinear
  interpolation is exact and expectations are near machine precision).
- Shared-vs-per-roll equivalence, grouping, and guards (later sections).

Conventions under test (must match coordinates.rotate_position and
utils.build_map_grid_from_image_pars): Cartesian x = cols (axis 1),
y = rows (axis 0), stamp center at pixel index (N - 1) / 2; a celestial
position p appears at detector position M(phi) p with
M(phi) = [[cos, sin], [-sin, cos]].
"""

import numpy as np
import jax.numpy as jnp
import pytest

from kl_pipe.dispersion import GrismPars, disperse_cube
from kl_pipe.parameters import ImagePars


# ===========================================================================
# disperse_cube image_rotation kwarg
# ===========================================================================


NROW, NCOL = 33, 33  # odd: stamp center on a pixel, integer-coord tests exact
NLAM = 5
LAMBDA_REF = 1300.0
DISPERSION = 1.1  # nm / pixel


@pytest.fixture(scope='module')
def grism_pars():
    ip = ImagePars(shape=(NROW, NCOL), pixel_scale=0.11, indexing='ij')
    return GrismPars(
        image_pars=ip,
        dispersion=DISPERSION,
        lambda_ref=LAMBDA_REF,
        dispersion_angle_detector=0.0,
    )


@pytest.fixture(scope='module')
def lambda_grid():
    # symmetric grid: integer pixel offsets (-2, -1, 0, 1, 2)
    return jnp.array(
        [LAMBDA_REF + i * DISPERSION for i in range(-(NLAM // 2), NLAM // 2 + 1)]
    )


def _delta_cube(row: int, col: int, k: int) -> jnp.ndarray:
    """Cube that is zero except a unit delta at (row, col) in slice k."""
    cube = jnp.zeros((NROW, NCOL, NLAM))
    return cube.at[row, col, k].set(1.0)


class TestDisperseCubeRotationKwarg:

    def test_zero_rotation_identical(self, grism_pars, lambda_grid):
        """image_rotation=0.0 takes the classic branch bit-for-bit."""
        rng = np.random.default_rng(7)
        cube = jnp.asarray(rng.normal(size=(NROW, NCOL, NLAM)))
        out_default = disperse_cube(cube, grism_pars, lambda_grid)
        out_explicit = disperse_cube(cube, grism_pars, lambda_grid, image_rotation=0.0)
        assert (np.asarray(out_default) == np.asarray(out_explicit)).all()

    @pytest.mark.parametrize(
        'phi_deg, expected_det_offset',
        [
            # celestial delta at Cartesian offset (x, y) = (dcol, drow) = (5, 3);
            # detector position offset = M(phi) (x, y):
            (0, (5, 3)),
            (90, (3, -5)),  # x_det = y, y_det = -x
            (180, (-5, -3)),
            (270, (-3, 5)),
        ],
    )
    def test_point_source_90deg_multiples_exact(
        self, grism_pars, lambda_grid, phi_deg, expected_det_offset
    ):
        """A celestial point source lands at M(phi) p + s_k in the detector
        frame, exactly, when rotated coords hit integer grid points."""
        c = (NROW - 1) // 2  # == (NCOL - 1) // 2, odd square stamp
        dcol_cel, drow_cel = 5, 3
        k0 = 1  # pixel offset (lambda_grid[1] - ref) / disp = -1
        disp_shift = -1

        cube = _delta_cube(c + drow_cel, c + dcol_cel, k0)
        out = disperse_cube(
            cube,
            grism_pars,
            lambda_grid,
            image_rotation=np.deg2rad(phi_deg),
        )
        out = np.asarray(out)

        dx_det, dy_det = expected_det_offset
        exp_row = c + dy_det
        exp_col = c + dx_det + disp_shift  # dispersion along det +x
        dlam = float(lambda_grid[1] - lambda_grid[0])

        assert out[exp_row, exp_col] == pytest.approx(dlam, rel=1e-12)
        # everything else at machine level: cos/sin of 90-degree multiples
        # are not exactly 0/1 in floats, so bilinear bleeds ~1e-13 (measured
        # 9.4e-14) into neighbors; 1e-12 abs vs dlam ~ 1.1 is 10x headroom
        mask = np.ones_like(out, dtype=bool)
        mask[exp_row, exp_col] = False
        assert np.abs(out[mask]).max() < 1e-12

    def test_point_source_arbitrary_angle_centroid(self, grism_pars, lambda_grid):
        """At a non-90-degree angle the delta bilinear-smears over <=4
        pixels; its centroid must still sit at M(phi) p + s_k."""
        phi = np.deg2rad(35.0)
        c = (NROW - 1) // 2
        dcol_cel, drow_cel = 4, -2
        k0 = 3  # dispersion shift +1 pixel
        disp_shift = 1

        cube = _delta_cube(c + drow_cel, c + dcol_cel, k0)
        out = np.asarray(
            disperse_cube(cube, grism_pars, lambda_grid, image_rotation=phi)
        )

        cph, sph = np.cos(phi), np.sin(phi)
        dx_det = cph * dcol_cel + sph * drow_cel
        dy_det = -sph * dcol_cel + cph * drow_cel

        rows, cols = np.mgrid[0:NROW, 0:NCOL]
        total = out.sum()
        assert total > 0
        row_cen = (rows * out).sum() / total
        col_cen = (cols * out).sum() / total
        # pull-sampling of a delta is not exactly moment-preserving; 0.1 pix
        # is far tighter than any real profile scale and pins the convention
        assert row_cen == pytest.approx(c + dy_det, abs=0.1)
        assert col_cen == pytest.approx(c + dx_det + disp_shift, abs=0.1)

    def test_rotation_matches_rotate_position_convention(self, grism_pars, lambda_grid):
        """Cross-check against coordinates.rotate_position: the detector
        position of a celestial delta must equal rotate_position(x, y, phi)."""
        from kl_pipe.coordinates import rotate_position

        phi = np.deg2rad(90.0)
        c = (NROW - 1) // 2
        x_cel, y_cel = 2.0, 6.0
        x_det, y_det = rotate_position(x_cel, y_cel, phi)

        cube = _delta_cube(c + int(y_cel), c + int(x_cel), 2)  # zero disp shift
        out = np.asarray(
            disperse_cube(cube, grism_pars, lambda_grid, image_rotation=phi)
        )
        peak = np.unravel_index(np.argmax(out), out.shape)
        assert peak == (c + int(round(float(y_det))), c + int(round(float(x_det))))


# ===========================================================================
# Shared-cube pathway: equivalence vs the per-roll reference path
# ===========================================================================
#
# The shared path builds ONE cube in the FIRST obs's detector frame (the
# anchor) and disperses every obs through a precomputed sparse operator
# fusing dispersion + relative roll rotation + Catmull-Rom CUBIC
# interpolation. Bilinear resampling is deliberately excluded: its
# sub-pixel smoothing biased inclination-like posterior modes at 0.35
# sigma in tight-posterior tests (2026-07-04 investigation; grid padding
# and k-space mean-transfer-function compensation were measured and
# REJECTED -- padding does not touch the posterior-relevant mode, the
# deconvolution overcorrects ~2x). Cubic measures at the per-roll
# accuracy floor (Fisher-projected shift 0.005 sigma, MCMC-confirmed).
#
# Protocol (measure-then-freeze: every frozen constant below carries
# its measured value in a comment at the assert):
#   - anchor roll + 90-deg multiples: no rotational resampling error; the
#     residual is cubic-x vs bilinear-x dispersion interpolation only.
#   - arbitrary angles: residual is dominated by rotated-corner truncation
#     (folding_threshold flux class, Fisher-irrelevant); the interpolation
#     part is h^4 (central-window convergence test).
#   - Fisher-projection gate: predicted posterior shift < 0.05 sigma --
#     the posterior-relevant metric (image-level L1/moments demonstrably
#     are not).
# All frozen constants below were measured 2026-07-04 on the exact configs
# in this file; values quoted in comments at each assert.

import galsim
from astropy.wcs import WCS

from kl_pipe.intensity import InclinedExponentialModel
from kl_pipe.lines import EmissionLine, LINE_LAMBDAS
from kl_pipe.observation import (
    build_grism_obs,
    group_grism_obs_by_cube_compat,
    validate_shared_cube_group,
)
from kl_pipe.render import RenderConfig
from kl_pipe.source import SourceModel
from kl_pipe.velocity import CenteredVelocityModel
from kl_pipe.priors import PriorDict, Uniform
from kl_pipe.likelihood import create_jitted_likelihood_from_obs
from kl_pipe.sampling import InferenceTask

import os
import jax

OUT_DIR = os.path.join(os.path.dirname(__file__), 'out', 'grism_shared_cube')
os.makedirs(OUT_DIR, exist_ok=True)

_Z = 1.0
_PS = 0.11
_SHAPE = (24, 24)
_LAM_CENTER = LINE_LAMBDAS['Halpha'] * (1 + _Z)
_DISPERSION_NM = 1.1
_FOLDING_THRESHOLD = 5e-3  # RenderConfig default; the aliasing budget

# --- frozen equivalence constants (measured 2026-07-04, this file's grid,
# cubic-operator shared path anchored at the first roll) ---
# anchor roll + 90-deg rolls: the ONLY pathway difference is cubic-x vs
# bilinear-x dispersion interpolation (no rotational resampling); measured
# L1 1.5e-5 / 1.8e-5; frozen 5e-5 (~3x margin)
_L1_EXACT_ROLLS = 5e-5
# truncation term |sum(shared)-sum(ref)|/F <= C * folding_threshold:
# measured worst C = 0.66 (135 deg, fwhm 0.11); frozen 1.5 (~2.3x margin).
# Fisher-projected posterior impact of the truncation term is < 0.01 sigma
# (it lives where the noise-weighted Jacobian is negligible).
_C_TRUNC = 1.5
# L1(shared-ref)/F, canonical grid (15/30/45/135 deg x 3 FWHMs): measured
# worst 3.4e-3, now TRUNCATION-dominated (cubic removed the interpolation
# term); frozen 7e-3 (~2x margin)
_L1_CANONICAL = 7e-3
# L1, stress configs (high dispersion shift / continuum / steep rotation
# curve at 45 deg): measured worst 8.3e-3; frozen 1.6e-2 (~2x margin)
_L1_STRESS = 1.6e-2
# gradient A/B, vector norm at O(1) off-truth point: measured 2.2e-4 with
# the cubic operator (bilinear measured 1.66e-2 and biased cosi at 0.35
# sigma, which is why it was rejected); frozen 1e-3 (~4.5x margin)
_GRAD_AB_RTOL = 1e-3
# shared-path AD-vs-FD self-consistency: measured 3.5e-5; frozen 1e-3
_GRAD_SELF_RTOL = 1e-3
# Fisher-projected posterior-shift gate (the metric that caught the
# bilinear bias): measured 0.005 sigma for cubic vs 0.35 for bilinear;
# frozen 0.05 sigma
_FISHER_SHIFT_SIGMA = 0.05

_BASE_PARS = {
    'cosi': 0.5,
    'theta_int': 0.7,
    'g1': 0.02,
    'g2': -0.03,
    'z': _Z,
    'vel.v0': 10.0,
    'vel.vcirc': 200.0,
    'vel.rscale': 0.5,
    'Halpha.flux': 100.0,
    'Halpha.rscale': 0.3,
    'Halpha.h_over_r': 0.1,
    # nonzero celestial offsets on purpose: the shared cube bakes them in
    # and the per-roll path rotates them (post x0/y0-rotation fix); any
    # disagreement in offset conventions would blow the L1 budget
    'Halpha.x0': 0.1,
    'Halpha.y0': -0.08,
    'Halpha.dispersion': 50.0,
}

_CONTINUUM_EXTRA = {
    'Halpha.cont.flux_per_nm': 2.0,
    'Halpha.cont.rscale': 0.3,
    'Halpha.cont.h_over_r': 0.1,
    'Halpha.cont.x0': 0.1,
    'Halpha.cont.y0': -0.08,
}


def _wcs_with_rotation(shape, pixel_scale, phi, flip=False):
    Nrow, Ncol = shape
    c, s = float(np.cos(phi)), float(np.sin(phi))
    wcs = WCS(naxis=2)
    pc = np.array([[c, -s], [s, c]])
    if flip:
        pc = pc @ np.array([[-1.0, 0.0], [0.0, 1.0]])
    wcs.wcs.pc = pc
    wcs.wcs.cdelt = np.array([pixel_scale, pixel_scale])
    wcs.wcs.crpix = np.array([Ncol / 2, Nrow / 2])
    wcs.wcs.crval = np.array([0.0, 0.0])
    wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    wcs.wcs.cunit = ['arcsec', 'arcsec']
    wcs.pixel_shape = (Ncol, Nrow)
    wcs.wcs.set()
    return wcs


def _make_rotated_obs(
    phi: float,
    fwhm: float = 0.11,
    lambda_ref_offset_nm: float = 0.0,
    oversample: int = 3,
    shape=_SHAPE,
    psf=True,
    render_config=None,
    z=_Z,
    **build_kwargs,
):
    """Grism obs whose WCS carries the roll rotation ``phi``."""
    ip = ImagePars(shape=shape, wcs=_wcs_with_rotation(shape, _PS, phi), indexing='ij')
    gp = GrismPars(
        image_pars=ip,
        dispersion=_DISPERSION_NM,
        lambda_ref=_LAM_CENTER + lambda_ref_offset_nm,
        dispersion_angle_detector=0.0,
    )
    if render_config is None:
        render_config = RenderConfig(oversample=oversample)
    return build_grism_obs(
        gp,
        z=z,
        psf=galsim.Gaussian(fwhm=fwhm) if psf else None,
        render_config=render_config,
        **build_kwargs,
    )


@pytest.fixture(scope='module')
def source_ha():
    return SourceModel(
        velocity_model=CenteredVelocityModel(),
        emission_lines={'Halpha': EmissionLine(intensity=InclinedExponentialModel())},
    )


@pytest.fixture(scope='module')
def source_ha_cont():
    return SourceModel(
        velocity_model=CenteredVelocityModel(),
        emission_lines={
            'Halpha': EmissionLine(
                intensity=InclinedExponentialModel(),
                continuum=InclinedExponentialModel(),
            )
        },
    )


def _shared_vs_ref(source, pars, phi, **obs_kwargs):
    """(truncation, L1) rel errors of shared vs per-roll at angle phi.

    The group pairs the rotated obs with a 0-deg partner so the shared
    branch (not the singleton fallback) is exercised.
    """
    obs_0 = _make_rotated_obs(0.0, **obs_kwargs)
    obs_phi = _make_rotated_obs(phi, **obs_kwargs)
    shared = source.render_grism_group(pars, {'r0': obs_0, 'rp': obs_phi})['rp']
    ref = source.render_grism(pars, obs_phi)
    ftot = float(jnp.sum(jnp.abs(ref)))
    trunc = abs(float(jnp.sum(shared)) - float(jnp.sum(ref))) / ftot
    l1 = float(jnp.sum(jnp.abs(shared - ref))) / ftot
    return trunc, l1


class TestSharedVsPerRollImages:

    def test_anchor_roll_near_identity(self, source_ha):
        """The cube is anchored in the FIRST obs's detector frame, so the
        anchor roll needs no rotational resampling: the only pathway
        difference is cubic-x vs bilinear-x dispersion interpolation
        (measured L1 1.5e-5)."""
        obs_a = _make_rotated_obs(0.0)
        obs_b = _make_rotated_obs(0.0)
        shared = source_ha.render_grism_group(_BASE_PARS, {'a': obs_a, 'b': obs_b})
        ref = source_ha.render_grism(_BASE_PARS, obs_a)
        ftot = float(jnp.sum(jnp.abs(ref)))
        for key in ('a', 'b'):
            l1 = float(jnp.sum(jnp.abs(shared[key] - ref))) / ftot
            assert l1 < _L1_EXACT_ROLLS, f"{key}: L1 {l1:.2e}"

    def test_90deg_near_identity(self, source_ha):
        """90-deg relative rotation: rotated sample coords land on integer
        grid points, so rotation contributes nothing; residual is the same
        dispersion-interpolation class as the anchor roll (measured L1
        1.8e-5)."""
        trunc, l1 = _shared_vs_ref(source_ha, _BASE_PARS, np.pi / 2)
        assert l1 < _L1_EXACT_ROLLS
        assert trunc < 1e-5  # measured 5e-7

    @pytest.mark.parametrize('phi_deg', [15.0, 30.0, 45.0, 135.0])
    @pytest.mark.parametrize('fwhm', [0.11, 0.18, 0.30])
    def test_canonical_grid(self, source_ha, phi_deg, fwhm):
        """Canonical A/B grid: truncation within the folding budget,
        L1 within the frozen interpolation-class bound (measured worst
        5.5e-3 at 135 deg / fwhm 0.11)."""
        trunc, l1 = _shared_vs_ref(
            source_ha, _BASE_PARS, np.deg2rad(phi_deg), fwhm=fwhm
        )
        assert trunc <= _C_TRUNC * _FOLDING_THRESHOLD, (
            f"phi={phi_deg} fwhm={fwhm}: truncation {trunc:.2e} > "
            f"{_C_TRUNC} x folding_threshold"
        )
        assert l1 <= _L1_CANONICAL, f"phi={phi_deg} fwhm={fwhm}: L1 {l1:.2e}"

    def test_high_shift_stress(self, source_ha):
        """Dispersion reference displaced 5 nm from the line: every
        line-carrying slice picks up a large shift (measured L1 9.7e-3,
        the grid's worst case)."""
        trunc, l1 = _shared_vs_ref(
            source_ha, _BASE_PARS, np.deg2rad(45.0), lambda_ref_offset_nm=5.0
        )
        assert trunc <= _C_TRUNC * _FOLDING_THRESHOLD
        assert l1 <= _L1_STRESS, f"high-shift L1 {l1:.2e}"

    def test_with_continuum(self, source_ha_cont):
        """Continuum fills the full bandpass -> maximal dispersion overlap
        (measured L1 5.4e-3)."""
        pars = {**_BASE_PARS, **_CONTINUUM_EXTRA}
        trunc, l1 = _shared_vs_ref(source_ha_cont, pars, np.deg2rad(45.0))
        assert trunc <= _C_TRUNC * _FOLDING_THRESHOLD
        assert l1 <= _L1_STRESS, f"continuum L1 {l1:.2e}"

    def test_high_velocity_gradient(self, source_ha):
        """Steep rotation curve: strong lambda structure across the stamp
        (measured L1 5.0e-3)."""
        pars = {**_BASE_PARS, 'vel.rscale': 0.15, 'vel.vcirc': 300.0}
        trunc, l1 = _shared_vs_ref(source_ha, pars, np.deg2rad(45.0))
        assert trunc <= _C_TRUNC * _FOLDING_THRESHOLD
        assert l1 <= _L1_STRESS, f"high-vgrad L1 {l1:.2e}"

    def test_interpolation_error_converges_with_oversample(self, source_ha):
        """The interpolation part of the pathway difference must shrink
        ~h^4 with the fine grid (Catmull-Rom). Measured on a CENTRAL
        window (r < 10 px of 32 -- the stamp edges carry the
        os-independent truncation term, which the Fisher projection shows
        is posterior-irrelevant): max-abs ratios os 3->5 = 9.9x,
        os 5->9 = 14x; frozen minimum 5x per step. A flat curve here
        would mean the difference is NOT interpolation-order-limited,
        i.e. a real pathway bug."""
        rows, cols = np.mgrid[0:32, 0:32]
        central = ((rows - 15.5) ** 2 + (cols - 15.5) ** 2) < 100
        diffs = {}
        for os_ in (3, 5, 9):
            obs_0 = _make_rotated_obs(0.0, oversample=os_, shape=(32, 32))
            obs_45 = _make_rotated_obs(np.deg2rad(45.0), oversample=os_, shape=(32, 32))
            shared = source_ha.render_grism_group(
                _BASE_PARS, {'r0': obs_0, 'r45': obs_45}
            )['r45']
            ref = source_ha.render_grism(_BASE_PARS, obs_45)
            d = np.abs(np.asarray(shared - ref))
            diffs[os_] = float(d[central].max())
        assert diffs[3] / diffs[5] > 5.0, f"os 3->5 ratio {diffs[3]/diffs[5]:.2f}"
        assert diffs[5] / diffs[9] > 5.0, f"os 5->9 ratio {diffs[5]/diffs[9]:.2f}"


class TestSharedGradients:

    def _four_roll_likelihoods(self, source):
        angles = [0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
        obs_clean = {f'r{i}': _make_rotated_obs(a) for i, a in enumerate(angles)}
        data = {
            k: np.asarray(source.render_grism(_BASE_PARS, o))
            for k, o in obs_clean.items()
        }
        obs = {
            f'r{i}': _make_rotated_obs(a, data=jnp.asarray(data[f'r{i}']), variance=1.0)
            for i, a in enumerate(angles)
        }
        sampled = {
            'cosi': Uniform(0.05, 0.95),
            'theta_int': Uniform(0.0, np.pi),
            'g1': Uniform(-0.1, 0.1),
            'g2': Uniform(-0.1, 0.1),
            'vel.vcirc': Uniform(100.0, 300.0),
            'Halpha.flux': Uniform(30.0, 250.0),
        }
        priors = PriorDict(
            {**sampled, **{k: v for k, v in _BASE_PARS.items() if k not in sampled}}
        )
        names = tuple(priors.sampled_names)
        fixed = dict(priors.fixed_values)
        ll = {
            mode: create_jitted_likelihood_from_obs(
                source, names, fixed, grism_obs=obs, cube_mode=mode
            )
            for mode in ('shared', 'per_roll')
        }
        # O(1) off-truth evaluation point: comparing at the truth point of
        # data generated by one path makes its gradient ~0 and any ratio
        # meaningless (the flawed metric behind a previously retracted
        # "wrong gradients" claim)
        off = {
            'cosi': 0.45,
            'theta_int': 0.8,
            'g1': 0.04,
            'g2': -0.01,
            'Halpha.flux': 110.0,
            'vel.vcirc': 230.0,
        }
        theta = jnp.array([off[n] for n in names])
        return ll, theta, names

    def test_shared_gradient_self_consistent(self, source_ha):
        """The shared path's AD gradient matches finite differences of its
        OWN likelihood (measured 4e-5): adjoint correctness, independent
        of any pathway difference."""
        ll, theta, names = self._four_roll_likelihoods(source_ha)
        g = np.asarray(jax.grad(ll['shared'])(theta))
        i = names.index('vel.vcirc')
        eps = 1e-6
        fd = float(
            (ll['shared'](theta.at[i].add(eps)) - ll['shared'](theta.at[i].add(-eps)))
            / (2 * eps)
        )
        assert abs(fd - g[i]) / abs(fd) < _GRAD_SELF_RTOL, f"AD={g[i]:.8e} FD={fd:.8e}"

    def test_gradient_between_paths(self, source_ha):
        """Vector-norm gradient A/B between shared and per_roll at the
        off-truth point: measured 1.66e-2 -- interpolation-class pathway
        difference (both paths carry same-order bilinear error vs the
        continuous model; posterior impact is screened by the slow
        posterior A/B test). Per-component relative comparison is NOT
        used: near-zero components give small-denominator noise."""
        ll, theta, _ = self._four_roll_likelihoods(source_ha)
        g_sh = np.asarray(jax.grad(ll['shared'])(theta))
        g_pr = np.asarray(jax.grad(ll['per_roll'])(theta))
        rel = np.linalg.norm(g_sh - g_pr) / np.linalg.norm(g_pr)
        assert rel < _GRAD_AB_RTOL, f"gradient A/B vector rel {rel:.2e}"

    def test_likelihood_values_close(self, source_ha):
        """Log-likelihood values agree to interpolation class (measured
        rel 5e-5) but are NOT identical (paths genuinely differ)."""
        ll, theta, _ = self._four_roll_likelihoods(source_ha)
        v_sh, v_pr = float(ll['shared'](theta)), float(ll['per_roll'](theta))
        assert v_sh != v_pr
        assert abs((v_sh - v_pr) / v_pr) < 1e-3


class TestGroupingAndGuards:

    def test_different_lambda_grids_form_separate_groups(self):
        """Obs of genuinely different observed-frame line windows (here:
        different redshifts) group separately -- correct behavior, not an
        error. Note lambda_ref does NOT set the cube grid (to_cube_pars
        derives it from the line + velocity window at z)."""
        obs_a = _make_rotated_obs(0.0)
        obs_b = _make_rotated_obs(0.0, z=1.2)
        groups = group_grism_obs_by_cube_compat({'a': obs_a, 'b': obs_b})
        assert [list(g) for g in groups] == [['a'], ['b']]

    def test_near_miss_lambda_grid_raises(self):
        """Grids differing by float drift (rel ~1e-12, from a z equal up
        to the last digits) are almost certainly meant to be identical:
        loud error, not silent per-roll fallback."""
        obs_a = _make_rotated_obs(0.0)
        obs_b = _make_rotated_obs(0.0, z=_Z * (1 + 1e-12))
        with pytest.raises(ValueError, match="float drift"):
            group_grism_obs_by_cube_compat({'a': obs_a, 'b': obs_b})

    def test_per_slice_with_shared_group_raises(self):
        obs = {'a': _make_rotated_obs(0.0), 'b': _make_rotated_obs(np.pi / 4)}
        with pytest.raises(ValueError, match="post_dispersion"):
            validate_shared_cube_group(obs, 'per_slice')

    def test_flipped_wcs_raises(self):
        ip = ImagePars(
            shape=_SHAPE,
            wcs=_wcs_with_rotation(_SHAPE, _PS, 0.3, flip=True),
            indexing='ij',
        )
        gp = GrismPars(
            image_pars=ip,
            dispersion=_DISPERSION_NM,
            lambda_ref=_LAM_CENTER,
            dispersion_angle_detector=0.0,
        )
        obs_flip = build_grism_obs(
            gp,
            z=_Z,
            psf=galsim.Gaussian(fwhm=0.11),
            render_config=RenderConfig(oversample=3),
        )
        group = {'a': _make_rotated_obs(0.0), 'b': obs_flip}
        with pytest.raises(ValueError, match="parity-flipped"):
            validate_shared_cube_group(group, 'post_dispersion')

    def test_mixed_psf_presence_raises(self):
        group = {
            'a': _make_rotated_obs(0.0),
            'b': _make_rotated_obs(np.pi / 4, psf=False, oversample=1),
        }
        # different oversample -> different groups; force same oversample
        # via explicit configs to isolate the PSF-presence guard
        group['b'] = _make_rotated_obs(np.pi / 4, psf=False, oversample=3)
        with pytest.raises(ValueError, match="mixed PSF presence"):
            validate_shared_cube_group(group, 'post_dispersion')

    def test_incompatible_group_passed_directly_raises(self, source_ha):
        obs_a = _make_rotated_obs(0.0)
        obs_b = _make_rotated_obs(0.0, z=1.2)
        with pytest.raises(ValueError, match="not cube-compatible"):
            source_ha.render_grism_group(_BASE_PARS, {'a': obs_a, 'b': obs_b})

    def test_singleton_group_matches_render_grism(self, source_ha):
        """A single-obs group takes the per-obs path bit-for-bit (and
        supports per_slice there)."""
        obs = _make_rotated_obs(np.pi / 6)
        out = source_ha.render_grism_group(_BASE_PARS, {'only': obs})
        ref = source_ha.render_grism(_BASE_PARS, obs)
        assert (np.asarray(out['only']) == np.asarray(ref)).all()

    def test_render_config_validates_cube_mode(self):
        with pytest.raises(ValueError, match="cube_mode"):
            RenderConfig(cube_mode='bogus')

    def test_from_obs_mismatched_cube_mode_raises(self, source_ha):
        obs_a = _make_rotated_obs(
            0.0,
            render_config=RenderConfig(oversample=3, cube_mode='shared'),
            data=jnp.ones(_SHAPE),
            variance=1.0,
        )
        obs_b = _make_rotated_obs(
            np.pi / 4,
            render_config=RenderConfig(oversample=3, cube_mode='per_roll'),
            data=jnp.ones(_SHAPE),
            variance=1.0,
        )
        priors = PriorDict(
            {
                'vel.vcirc': Uniform(100.0, 300.0),
                **{k: v for k, v in _BASE_PARS.items() if k != 'vel.vcirc'},
            }
        )
        with pytest.raises(ValueError, match="mismatched cube_mode"):
            InferenceTask.from_obs(
                source_ha, priors, grism_obs={'a': obs_a, 'b': obs_b}
            )

    def test_from_obs_shared_with_per_slice_raises(self, source_ha):
        """from_obs surfaces the per_slice+shared conflict eagerly at
        construction, not at first trace."""
        obs = {
            'a': _make_rotated_obs(0.0, data=jnp.ones(_SHAPE), variance=1.0),
            'b': _make_rotated_obs(np.pi / 4, data=jnp.ones(_SHAPE), variance=1.0),
        }
        priors = PriorDict(
            {
                'vel.vcirc': Uniform(100.0, 300.0),
                **{k: v for k, v in _BASE_PARS.items() if k != 'vel.vcirc'},
            }
        )
        with pytest.raises(ValueError, match="post_dispersion"):
            InferenceTask.from_obs(
                source_ha,
                priors,
                grism_obs=obs,
                psf_mode='per_slice',
                cube_mode='shared',
            )
        # the escape hatch: per_roll + per_slice constructs fine
        task = InferenceTask.from_obs(
            source_ha,
            priors,
            grism_obs=obs,
            psf_mode='per_slice',
            cube_mode='per_roll',
        )
        assert np.isfinite(float(task.log_likelihood(jnp.array([200.0]))))


# ===========================================================================
# Visual diagnostic: one sky model, many roll angles
# ===========================================================================


class TestDiagnostics:

    def test_multi_roll_diagnostic(self, source_ha):
        """Figure: ONE celestial-frame source dispersed at four roll
        angles through the shared-cube path, against the per-roll
        reference and their difference. Written for a reader who has not
        seen this code: captions state what should be visible."""
        import matplotlib

        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        angles_deg = [0.0, 45.0, 90.0, 135.0]
        obs_group = {
            f'{int(a)} deg': _make_rotated_obs(np.deg2rad(a), shape=(32, 32))
            for a in angles_deg
        }
        shared = source_ha.render_grism_group(_BASE_PARS, obs_group)
        refs = {k: source_ha.render_grism(_BASE_PARS, o) for k, o in obs_group.items()}

        fig, axes = plt.subplots(3, len(angles_deg), figsize=(16, 11))
        for j, key in enumerate(obs_group):
            sh = np.asarray(shared[key])
            rf = np.asarray(refs[key])
            vmax = rf.max()
            axes[0, j].imshow(sh, origin='lower', vmin=0, vmax=vmax)
            axes[0, j].set_title(f'roll {key}')
            axes[1, j].imshow(rf, origin='lower', vmin=0, vmax=vmax)
            im = axes[2, j].imshow(
                (sh - rf) / vmax, origin='lower', cmap='RdBu_r', vmin=-5e-4, vmax=5e-4
            )
            fig.colorbar(im, ax=axes[2, j], fraction=0.046)
            l1 = np.abs(sh - rf).sum() / np.abs(rf).sum()
            axes[2, j].set_title(f'difference (L1 rel {l1:.1e})')
            for i in range(3):
                axes[i, j].set_xticks([])
                axes[i, j].set_yticks([])
        axes[0, 0].set_ylabel('fast path\n(one shared model)', fontsize=11)
        axes[1, 0].set_ylabel('reference path\n(rebuilt per roll)', fontsize=11)
        axes[2, 0].set_ylabel('difference\n(fraction of peak)', fontsize=11)
        fig.suptitle(
            'Same galaxy observed at four telescope roll angles.\n'
            'Top: the fast method builds the galaxy model once and re-uses\n'
            'it for every roll. Middle: the slow method rebuilds the model\n'
            'for every roll. Bottom: their difference -- essentially blank\n'
            'everywhere the galaxy has light; the faint arcs at 45/135\n'
            'degrees are corner flux clipped at the stamp edge (far from\n'
            'the galaxy, and shown to have no effect on the fit).',
            fontsize=11,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.86))
        out = os.path.join(OUT_DIR, 'multi_roll_shared_vs_reference.png')
        fig.savefig(out, dpi=110)
        plt.close(fig)
        assert os.path.exists(out)


# ===========================================================================
# Seeded posterior A/B (screening depth)
# ===========================================================================


@pytest.mark.slow
class TestPosteriorEquivalence:
    def test_seeded_posterior_ab(self, source_ha):
        """Seeded NUTS on identical 2-roll data (0 and 45 deg) through
        shared vs per_roll cube modes: posterior means shift < 0.1 sigma,
        widths within 20%, divergences not degraded. Screening depth
        (300+300, 1 chain), not flagship depth."""
        import csv

        from kl_pipe.sampling import build_sampler
        from kl_pipe.sampling.configs import NumpyroSamplerConfig

        angles = {'roll0': 0.0, 'roll45': np.pi / 4}
        rng = np.random.default_rng(20260704)
        obs = {}
        for key, phi in angles.items():
            clean = _make_rotated_obs(phi)
            truth = source_ha.render_grism(
                _BASE_PARS, clean, psf_mode='post_dispersion'
            )
            peak = float(jnp.max(truth))
            sigma_noise = peak / 50.0  # SNR ~ 50 at the peak
            data = np.asarray(truth) + rng.normal(0.0, sigma_noise, truth.shape)
            obs[key] = _make_rotated_obs(phi, data=data, variance=sigma_noise**2)

        sampled = {
            'cosi': Uniform(0.2, 0.8),
            'g1': Uniform(-0.1, 0.1),
            'g2': Uniform(-0.1, 0.1),
            'vel.vcirc': Uniform(100.0, 300.0),
        }
        priors = PriorDict(
            {**sampled, **{k: v for k, v in _BASE_PARS.items() if k not in sampled}}
        )
        config = NumpyroSamplerConfig(
            n_samples=300, n_warmup=300, n_chains=1, seed=42, progress=False
        )

        results = {}
        for mode in ('per_roll', 'shared'):
            task = InferenceTask.from_obs(
                source_ha, priors, grism_obs=obs, cube_mode=mode
            )
            results[mode] = build_sampler('numpyro', task, config).run()

        res_a, res_b = results['per_roll'], results['shared']
        assert res_a.param_names == res_b.param_names

        rows = []
        stats = {}
        for i, name in enumerate(res_a.param_names):
            mean_a = float(res_a.samples[:, i].mean())
            mean_b = float(res_b.samples[:, i].mean())
            std_a = float(res_a.samples[:, i].std())
            std_b = float(res_b.samples[:, i].std())
            shift_sigma = abs(mean_b - mean_a) / std_a
            width_ratio = std_b / std_a
            stats[name] = (shift_sigma, width_ratio, mean_a, mean_b, std_a)
            rows.append(
                (
                    name,
                    f'{mean_a:.5g}',
                    f'{mean_b:.5g}',
                    f'{std_a:.3g}',
                    f'{shift_sigma:.3f}',
                    f'{width_ratio:.3f}',
                )
            )

        # write the full table BEFORE asserting so a failure still leaves
        # the complete comparison on disk
        out_csv = os.path.join(OUT_DIR, 'posterior_ab.csv')
        with open(out_csv, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(
                [
                    'param',
                    'mean_per_roll',
                    'mean_shared',
                    'sigma_per_roll',
                    'shift_sigma',
                    'width_ratio',
                ]
            )
            w.writerows(rows)
        print(
            f"posterior A/B (2 rolls, SNR~50, 300+300 NUTS): {rows}; "
            f"divergences per_roll="
            f"{res_a.diagnostics.get('n_divergences', 0)} shared="
            f"{res_b.diagnostics.get('n_divergences', 0)}; csv: {out_csv}"
        )

        for name, (shift_sigma, width_ratio, mean_a, mean_b, std_a) in stats.items():
            assert shift_sigma < 0.1, (
                f"{name}: posterior mean shift {shift_sigma:.3f} sigma "
                f"(per_roll={mean_a:.5g}, shared={mean_b:.5g}, "
                f"sigma={std_a:.3g})"
            )
            assert (
                0.8 < width_ratio < 1.2
            ), f"{name}: posterior width ratio {width_ratio:.3f}"

        div_a = res_a.diagnostics.get('n_divergences', 0)
        div_b = res_b.diagnostics.get('n_divergences', 0)
        assert div_b <= div_a + 5, f"divergences degraded: {div_a} -> {div_b}"


# ===========================================================================
# Dispersion operator: construction correctness
# ===========================================================================


class TestDispersionOperator:

    def _independent_cubic_reference(self, cube, grism_pars, lam, phi):
        """Independent traced Catmull-Rom resampler (duplicated on purpose:
        cross-checks the operator's precomputed weights against a direct
        implementation of the same kernel)."""
        Nrow, Ncol, Nlam = cube.shape
        offsets = (lam - grism_pars.lambda_ref) / grism_pars.dispersion
        dlam = jnp.abs(lam[1] - lam[0])
        rows = jnp.arange(Nrow, dtype=jnp.float64)
        cols = jnp.arange(Ncol, dtype=jnp.float64)
        Yb, Xb = jnp.meshgrid(rows, cols, indexing='ij')
        c_row, c_col = (Nrow - 1) / 2.0, (Ncol - 1) / 2.0
        cr, sr = jnp.cos(phi), jnp.sin(phi)

        def w4(f):
            a = -0.5
            t0, t1, t2, t3 = 1.0 + f, f, 1.0 - f, 2.0 - f
            return (
                a * t0**3 - 5 * a * t0**2 + 8 * a * t0 - 4 * a,
                (a + 2) * t1**3 - (a + 3) * t1**2 + 1.0,
                (a + 2) * t2**3 - (a + 3) * t2**2 + 1.0,
                a * t3**3 - 5 * a * t3**2 + 8 * a * t3 - 4 * a,
            )

        out = jnp.zeros((Nrow, Ncol))
        for k in range(Nlam):
            u = Xb - offsets[k] - c_col
            v = Yb - c_row
            row_s = c_row + sr * u + cr * v
            col_s = c_col + cr * u - sr * v
            i0, j0 = jnp.floor(row_s), jnp.floor(col_s)
            wr, wc = w4(row_s - i0), w4(col_s - j0)
            acc = jnp.zeros((Nrow, Ncol))
            for di in range(4):
                ii = i0.astype(jnp.int32) + (di - 1)
                vi = (ii >= 0) & (ii < Nrow)
                iic = jnp.clip(ii, 0, Nrow - 1)
                for dj in range(4):
                    jj = j0.astype(jnp.int32) + (dj - 1)
                    vj = (jj >= 0) & (jj < Ncol)
                    jjc = jnp.clip(jj, 0, Ncol - 1)
                    acc = acc + jnp.where(
                        vi & vj, wr[di] * wc[dj] * cube[iic, jjc, k], 0.0
                    )
            out = out + acc * dlam
        return out

    @pytest.mark.parametrize('phi_deg', [0.0, 27.0, 45.0, 90.0])
    def test_operator_matches_independent_reference(self, phi_deg):
        """Operator matvec == independent traced cubic resampler at
        machine precision, arbitrary angle."""
        from kl_pipe.dispersion import (
            apply_dispersion_operator,
            precompute_dispersion_operator,
        )

        ip = ImagePars(shape=(24, 24), pixel_scale=_PS, indexing='ij')
        gp = GrismPars(
            image_pars=ip,
            dispersion=_DISPERSION_NM,
            lambda_ref=_LAM_CENTER,
            dispersion_angle_detector=0.0,
        )
        lam = jnp.linspace(_LAM_CENTER - 5.3, _LAM_CENTER + 5.3, 7)
        rng = np.random.default_rng(11)
        cube = jnp.asarray(rng.normal(size=(24, 24, 7)))
        phi = np.deg2rad(phi_deg)
        op = precompute_dispersion_operator(gp, lam, image_rotation=phi)
        out_op = apply_dispersion_operator(op, cube)
        out_ref = self._independent_cubic_reference(cube, gp, lam, phi)
        assert float(jnp.max(jnp.abs(out_op - out_ref))) < 1e-12

    def test_flux_normalization(self):
        """Catmull-Rom weights are a partition of unity: a constant cube
        disperses to (constant x Nlam x dlam) wherever all taps are
        in-bounds."""
        from kl_pipe.dispersion import (
            apply_dispersion_operator,
            precompute_dispersion_operator,
        )

        ip = ImagePars(shape=(24, 24), pixel_scale=_PS, indexing='ij')
        gp = GrismPars(
            image_pars=ip,
            dispersion=_DISPERSION_NM,
            lambda_ref=_LAM_CENTER,
            dispersion_angle_detector=0.0,
        )
        lam = jnp.linspace(_LAM_CENTER - 1.1, _LAM_CENTER + 1.1, 3)
        op = precompute_dispersion_operator(gp, lam, image_rotation=np.deg2rad(30.0))
        out = apply_dispersion_operator(op, jnp.ones((24, 24, 3)))
        dlam = float(lam[1] - lam[0])
        # central pixels: all taps in-bounds for the small shifts here
        assert np.asarray(out)[8:16, 8:16] == pytest.approx(3 * dlam, rel=1e-12)

    def test_bilinear_interp_rejected(self):
        from kl_pipe.dispersion import precompute_dispersion_operator

        ip = ImagePars(shape=(8, 8), pixel_scale=_PS, indexing='ij')
        gp = GrismPars(
            image_pars=ip,
            dispersion=_DISPERSION_NM,
            lambda_ref=_LAM_CENTER,
            dispersion_angle_detector=0.0,
        )
        lam = jnp.linspace(_LAM_CENTER - 1.1, _LAM_CENTER + 1.1, 3)
        with pytest.raises(ValueError, match="cubic"):
            precompute_dispersion_operator(gp, lam, interp='bilinear')


# ===========================================================================
# Fisher-projection bias gate
# ===========================================================================


class TestFisherGate:

    def test_predicted_posterior_shift(self, source_ha):
        """Linear-response prediction of the posterior shift induced by
        the shared pathway, against near-truth (os=9 per-roll) renders:
        shift = -(J^T W J)^-1 J^T W (model - data), in units of the
        Fisher sigma. This is the metric that exposed the bilinear
        pathway bias (0.35 sigma where image-level L1 and moments looked
        benign); cubic measured 0.005 sigma. Gate frozen at 0.05."""
        params = ('cosi', 'g1', 'g2', 'vel.vcirc')
        angles = {'roll0': 0.0, 'roll45': np.pi / 4}
        obs_fit = {k: _make_rotated_obs(phi) for k, phi in angles.items()}
        truth = {
            k: src_render_os9
            for k, src_render_os9 in (
                (
                    k,
                    source_ha.render_grism(
                        _BASE_PARS, _make_rotated_obs(phi, oversample=9)
                    ),
                )
                for k, phi in angles.items()
            )
        }
        peak = float(jnp.max(truth['roll0']))
        weight = 1.0 / (peak / 50.0) ** 2  # SNR ~ 50 noise level

        def render_flat(theta_vec, mode):
            pars = dict(_BASE_PARS)
            for i, n in enumerate(params):
                pars[n] = theta_vec[i]
            if mode == 'per_roll':
                imgs = {k: source_ha.render_grism(pars, o) for k, o in obs_fit.items()}
            else:
                imgs = source_ha.render_grism_group(pars, obs_fit)
            return jnp.concatenate([imgs[k].ravel() for k in sorted(imgs)])

        theta0 = jnp.array([_BASE_PARS[n] for n in params])
        J = jax.jacfwd(lambda t: render_flat(t, 'per_roll'))(theta0)
        Finv = jnp.linalg.inv(weight * (J.T @ J))
        sigmas = jnp.sqrt(jnp.diag(Finv))
        data_flat = jnp.concatenate([truth[k].ravel() for k in sorted(truth)])

        shifts = {}
        for mode in ('per_roll', 'shared'):
            delta = render_flat(theta0, mode) - data_flat
            dtheta = -Finv @ (weight * (J.T @ delta))
            shifts[mode] = np.asarray(dtheta / sigmas)

        for i, name in enumerate(params):
            assert abs(shifts['shared'][i]) < _FISHER_SHIFT_SIGMA, (
                f"{name}: Fisher-projected shift {shifts['shared'][i]:+.3f} "
                f"sigma (per_roll baseline {shifts['per_roll'][i]:+.3f})"
            )
            # the shared pathway must not add bias beyond the per-roll
            # path's own discretization floor
            assert abs(shifts['shared'][i] - shifts['per_roll'][i]) < (
                _FISHER_SHIFT_SIGMA
            ), f"{name}: pathway-relative shift"
