"""Tests for the A3 shared-cube-across-rolls pathway.

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
