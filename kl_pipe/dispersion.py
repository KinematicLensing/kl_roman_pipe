"""
Grism dispersion: project 3D datacube onto 2D dispersed image.

Uses pull-semantics map_coordinates for sub-pixel shifts.
Fully differentiable via JAX bilinear interpolation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Tuple, Optional

import jax.numpy as jnp
import numpy as np

if TYPE_CHECKING:
    from kl_pipe.spectral import CubePars

from kl_pipe.parameters import ImagePars
from kl_pipe.constants import C_KMS


@dataclass(frozen=True)
class GrismPars:
    """Defines grism observation parameters for dispersing a datacube.

    ``dispersion_angle_detector`` is in the detector pixel frame (the
    direction the grism disperses light across detector pixels). For
    Roman G150 / P127 the value is 0 (dispersion along detector +x).
    Per-observation variation in the *celestial* dispersion direction
    arises from the obs's WCS rotation, not this field.

    ``dispersion_angle`` is a backward-compat alias accepted as a kwarg
    on construction (and exposed as a read-only attribute) so existing
    callsites keep working. New code should use ``dispersion_angle_detector``.

    ``throughput`` is optional; None = flat 100% (OK for narrow windows).
    """

    image_pars: ImagePars  # spatial grid of source cutout
    dispersion: float  # nm/pixel (~1.1 for Roman)
    lambda_ref: float  # reference wavelength nm (zero-offset point)
    # exactly one of dispersion_angle_detector / dispersion_angle must be set
    dispersion_angle_detector: Optional[float] = None  # radians, detector frame
    dispersion_angle: Optional[float] = None  # legacy alias (deprecated)
    throughput: Optional[jnp.ndarray] = None  # T(lambda), shape (Nlambda,)

    def __post_init__(self):
        if self.dispersion <= 0:
            raise ValueError(f"dispersion must be > 0, got {self.dispersion}")
        # resolve dispersion_angle / dispersion_angle_detector alias
        det = self.dispersion_angle_detector
        legacy = self.dispersion_angle
        if det is None and legacy is None:
            raise ValueError(
                "GrismPars requires dispersion_angle_detector (or legacy "
                "dispersion_angle) to be set"
            )
        if det is not None and legacy is not None and det != legacy:
            raise ValueError(
                f"GrismPars: dispersion_angle_detector ({det}) and legacy "
                f"dispersion_angle ({legacy}) disagree; pass only one"
            )
        canonical = det if det is not None else legacy
        # frozen dataclass — use object.__setattr__ to populate both fields
        # so reads of either name return the canonical value
        if det is None:
            object.__setattr__(self, 'dispersion_angle_detector', canonical)
        if legacy is None:
            object.__setattr__(self, 'dispersion_angle', canonical)

    def to_cube_pars(
        self,
        z: float,
        velocity_window_kms: float = 3000.0,
        n_lambda: int = None,
        line_lambdas_rest: tuple = None,
    ) -> 'CubePars':
        """Build CubePars centered on the emission line complex at redshift z.

        Parameters
        ----------
        z : float
            Galaxy redshift.
        velocity_window_kms : float
            Half-width of velocity window in km/s. Default 3000.
        n_lambda : int, optional
            Number of wavelength pixels. If None, computed from velocity window
            and dispersion.
        line_lambdas_rest : tuple of float, optional
            Rest-frame wavelengths (nm) of lines to cover. If None, uses
            H-alpha (656.28 nm).
        """
        from kl_pipe.spectral import CubePars

        if line_lambdas_rest is None:
            line_lambdas_rest = (656.28,)

        # observed wavelength range covering all lines + velocity window
        lam_obs = [(lam * (1.0 + z)) for lam in line_lambdas_rest]
        lam_min_line = min(lam_obs)
        lam_max_line = max(lam_obs)

        # velocity window in wavelength units
        lam_center = 0.5 * (lam_min_line + lam_max_line)
        dlam_vel = lam_center * velocity_window_kms / C_KMS

        lam_min = lam_min_line - dlam_vel
        lam_max = lam_max_line + dlam_vel

        if n_lambda is None:
            n_lambda = int(np.ceil((lam_max - lam_min) / self.dispersion)) + 1
            n_lambda = max(n_lambda, 3)

        lambda_grid = jnp.linspace(lam_min, lam_max, n_lambda)
        return CubePars(image_pars=self.image_pars, lambda_grid=lambda_grid)

    @property
    def output_shape(self) -> Tuple[int, int]:
        """Output shape = same as input spatial grid (source cutout)."""
        return (self.image_pars.Nrow, self.image_pars.Ncol)


def disperse_cube(
    cube: jnp.ndarray,
    grism_pars: GrismPars,
    lambda_grid: jnp.ndarray,
    oversample: int = 1,
    image_rotation: float = 0.0,
) -> jnp.ndarray:
    """Project 3D datacube onto 2D dispersed grism image.

    Uses pull-semantics map_coordinates: [Y - dy, X - dx] shifts content
    by (+dy, +dx). Fully differentiable via JAX bilinear interpolation.

    When ``image_rotation`` is nonzero, the input cube is interpreted as
    celestial-frame (built with ``image_rotation=0``) and the roll rotation
    is fused into the sampling coordinates: each slice is sampled at
    ``R(-phi) . (q - s_k - c) + c`` (Cartesian components; ``c`` = stamp
    center, ``s_k`` = detector-frame dispersion shift), so the output is
    the detector-frame dispersed image with dispersion along the detector
    axis set by ``dispersion_angle_detector``. This is an exact change of
    variables from rotating model parameters into the detector frame before
    cube assembly; only the bilinear interpolation error differs (rotated
    sample coordinates are fractional in both axes). Costs zero extra
    interpolation passes. Sign convention matches
    ``coordinates.rotate_position`` (celestial-to-detector).

    The input cube may be at fine spatial resolution (post-PSF,
    pre-pixel-response). When ``oversample > 1``, the cube spatial shape
    is ``(Nrow * oversample, Ncol * oversample, Nlambda)`` and
    ``grism_pars.dispersion`` is in nm per coarse detector pixel; the
    function rescales pixel offsets by ``oversample`` so the
    wavelength-driven shift indexes correctly into the fine grid. The
    output spatial shape matches the input cube's spatial shape (fine
    if ``oversample > 1``); the caller is responsible for applying the
    BoxPixel sinc + sum-bin to coarse detector pixels at the 2D output.

    Parameters
    ----------
    cube : jnp.ndarray
        PSF-convolved datacube, shape ``(Nrow, Ncol, Nlambda)`` or
        ``(Nrow * oversample, Ncol * oversample, Nlambda)`` when
        ``oversample > 1``.
    grism_pars : GrismPars
        Grism parameters. ``dispersion`` is in nm/coarse-pixel.
    lambda_grid : jnp.ndarray
        Wavelength array nm, shape (Nlambda,).
    oversample : int, default 1
        Spatial oversampling factor of the input cube relative to
        ``grism_pars.image_pars``. Pixel offsets are scaled by this
        factor so wavelength shifts map correctly into the fine grid.
    image_rotation : float, default 0.0
        Celestial-to-detector rotation (radians) fused into the sampling
        coordinates (shared-cube pathway). ``0.0`` reproduces the classic
        detector-frame dispersion exactly. Must be a static Python float
        (read from frozen obs aux), not a traced value.

    Returns
    -------
    jnp.ndarray
        Dispersed 2D image, shape matches the spatial dimensions of
        ``cube`` (fine when ``oversample > 1``).
    """
    Nrow, Ncol, Nlam = cube.shape
    angle = grism_pars.dispersion_angle_detector

    # pixel offsets for each wavelength slice relative to reference. The
    # dispersion is in nm per *coarse* detector pixel; if the input cube
    # is at fine resolution (oversample > 1), scale offsets up to fine
    # pixels so the shift indexes the fine grid correctly.
    pixel_offsets = (
        (lambda_grid - grism_pars.lambda_ref) / grism_pars.dispersion * oversample
    )

    cos_a = jnp.cos(angle)
    sin_a = jnp.sin(angle)

    # base pixel coordinates
    rows = jnp.arange(Nrow, dtype=jnp.float64)
    cols = jnp.arange(Ncol, dtype=jnp.float64)
    Y_base, X_base = jnp.meshgrid(rows, cols, indexing='ij')

    # throughput
    throughput = grism_pars.throughput
    if throughput is None:
        throughput = jnp.ones(Nlam)

    # delta_lambda for integration (nm per wavelength pixel). A single-slice
    # cube carries no spectral information to disperse; refuse loudly rather
    # than silently integrating with an arbitrary dlam=1 nm.
    if Nlam < 2:
        raise ValueError(
            f"disperse_cube requires Nlam >= 2 (got Nlam={Nlam}); a "
            f"single-wavelength cube has no spectral axis to disperse."
        )
    dlam = jnp.abs(lambda_grid[1] - lambda_grid[0])

    # accumulate dispersed image. The sequential per-slice loop is intentional:
    # restructuring it as vmap/scan (same algorithm) cut compile ~10x but
    # regressed runtime ~2.5x on CPU (benchmarked), and inference is
    # runtime-dominated. This is a CPU result for same-algorithm variants; a
    # precomputed fixed-dispersion operator and GPU execution are untested and
    # could differ.
    dispersed = jnp.zeros((Nrow, Ncol))

    # rotation center: stamp center in pixel indices, matching the
    # centered-grid convention of build_map_grid_from_image_pars
    # (x = cols, y = rows, center at (N - 1) / 2)
    c_row = (Nrow - 1) / 2.0
    c_col = (Ncol - 1) / 2.0
    cos_r = jnp.cos(image_rotation)
    sin_r = jnp.sin(image_rotation)

    for k in range(Nlam):
        offset_k = pixel_offsets[k]
        dx_k = offset_k * cos_a  # shift along x (cols)
        dy_k = offset_k * sin_a  # shift along y (rows)

        if image_rotation == 0.0:
            # pull semantics: sample source at (Y - dy, X - dx)
            coords = jnp.array([Y_base - dy_k, X_base - dx_k])
        else:
            # shared-cube pathway: rotate the pull coordinates about the
            # stamp center into the celestial frame, x_cel = R(-phi) x_det
            u = X_base - dx_k - c_col
            v = Y_base - dy_k - c_row
            coords = jnp.array(
                [
                    c_row + sin_r * u + cos_r * v,
                    c_col + cos_r * u - sin_r * v,
                ]
            )

        shifted = jax.scipy.ndimage.map_coordinates(
            cube[:, :, k], coords, order=1, mode='constant', cval=0.0
        )

        dispersed = dispersed + shifted * throughput[k] * dlam

    return dispersed


def build_grism_pars_for_line(
    lambda_rest: float,
    redshift: float,
    image_pars: ImagePars = None,
    pixel_scale: float = 0.11,
    Nrow: int = 32,
    Ncol: int = 32,
    dispersion: float = 1.1,
    dispersion_angle: float = 0.0,
) -> GrismPars:
    """Convenience factory for Roman grism centered on a specific line."""
    if image_pars is None:
        image_pars = ImagePars(
            shape=(Nrow, Ncol), pixel_scale=pixel_scale, indexing='ij'
        )
    lambda_ref = lambda_rest * (1.0 + redshift)
    return GrismPars(
        image_pars=image_pars,
        dispersion=dispersion,
        lambda_ref=lambda_ref,
        dispersion_angle=dispersion_angle,
    )


# need jax import for map_coordinates
import jax


# ============================================================================
# Precomputed dispersion operator (shared-cube pathway)
# ============================================================================
#
# Dispersion + roll rotation + interpolation form a FIXED linear map on the
# cube: the sampling coordinates depend only on static geometry (grism pars,
# wavelength grid, oversample, relative roll rotation), never on model
# parameters. Materializing that map as one sparse matrix per roll replaces
# the per-slice interpolation loop with a single matvec whose backward pass
# is the exact transpose. Measured (2026-07-04, M3 Max fp64, 32x32/os=3/
# Nlam=25, 4 rolls): 2.9x faster gradient than the per-roll path, vs 1.2x
# for the fused bilinear loop; bit-identical to running the interpolation
# directly.
#
# Interpolation is Catmull-Rom cubic (Keys a=-0.5). Bilinear sub-pixel
# smoothing at rotated sample coordinates biases inclination-like posterior
# modes at the 0.35-sigma level in tight-posterior tests; cubic measures at
# the per-roll accuracy floor (Fisher-projected shift 0.005 sigma, MCMC
# -0.006 sigma). See docs/plans/PRODUCTION_SPEEDUPS.md Sec 2/A3.


def _catmull_rom_weights(f: np.ndarray) -> np.ndarray:
    """Keys cubic-convolution weights (a=-0.5) for taps at offsets
    (-1, 0, +1, +2) around the floor of the sample coordinate.

    ``f`` is the fractional part in [0, 1). Weights sum to 1 exactly for
    any f (partition of unity), so flux normalization is preserved.
    """
    a = -0.5
    t0, t1, t2, t3 = 1.0 + f, f, 1.0 - f, 2.0 - f
    return np.stack(
        [
            a * t0**3 - 5 * a * t0**2 + 8 * a * t0 - 4 * a,
            (a + 2) * t1**3 - (a + 3) * t1**2 + 1.0,
            (a + 2) * t2**3 - (a + 3) * t2**2 + 1.0,
            a * t3**3 - 5 * a * t3**2 + 8 * a * t3 - 4 * a,
        ]
    )


def precompute_dispersion_operator(
    grism_pars: GrismPars,
    lambda_grid,
    oversample: int = 1,
    image_rotation: float = 0.0,
    interp: str = 'cubic',
):
    """Build the sparse linear operator mapping a (possibly rotated) cube
    to the detector-frame dispersed image.

    The operator ``L`` satisfies
    ``dispersed.ravel() = L @ jnp.transpose(cube, (2, 0, 1)).ravel()``
    (wavelength-major cube flattening); apply via
    ``apply_dispersion_operator``. Rows are detector fine pixels; columns
    are (wavelength slice, cube fine pixel); entries are interpolation
    weights x throughput x dlam. Sample coordinates follow the same
    geometry as ``disperse_cube(image_rotation=...)``: pull semantics,
    rotation about the stamp center, cval=0 outside the cube (taps
    falling outside get weight zero).

    ``image_rotation`` here is the rotation between the frame the cube
    was BUILT in and this observation's detector frame (for a shared cube
    anchored at a reference roll: ``rot_obs - rot_anchor``).

    Construction is host-side numpy (static geometry) and is intended to
    run once per (roll, grid config) at task/likelihood construction; the
    operator is model- and galaxy-independent, so one operator serves
    every galaxy sharing the configuration.

    Parameters
    ----------
    grism_pars : GrismPars
        Dispersion parameters; ``dispersion`` in nm per COARSE pixel.
    lambda_grid : array
        Wavelength grid (nm), shape (Nlambda,).
    oversample : int
        Spatial oversampling of the cube grid relative to
        ``grism_pars.image_pars``.
    image_rotation : float
        Cube-frame-to-detector rotation (radians), fused into the sample
        coordinates. Static Python float.
    interp : str
        Interpolation kernel. Only ``'cubic'`` (Catmull-Rom) is
        supported: bilinear sub-pixel smoothing measurably biases
        inclination-like posterior modes (see module note).

    Returns
    -------
    jax.experimental.sparse.BCOO
        Shape ``(Nrow*Ncol, Nrow*Ncol*Nlambda)`` (fine pixels).
    """
    from jax.experimental import sparse as jsparse

    if interp != 'cubic':
        raise ValueError(
            f"precompute_dispersion_operator supports interp='cubic' only "
            f"(got {interp!r}); bilinear sub-pixel smoothing biases "
            f"inclination-like posterior modes (measured 0.35 sigma in "
            f"tight-posterior tests). Use disperse_cube for the loop path."
        )
    lambda_grid = np.asarray(lambda_grid)
    Nlam = lambda_grid.shape[0]
    if Nlam < 2:
        raise ValueError(
            f"precompute_dispersion_operator requires Nlam >= 2 (got {Nlam})"
        )
    Nrow = grism_pars.image_pars.Nrow * oversample
    Ncol = grism_pars.image_pars.Ncol * oversample

    pixel_offsets = (
        (lambda_grid - grism_pars.lambda_ref) / grism_pars.dispersion * oversample
    )
    angle = grism_pars.dispersion_angle_detector
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    throughput = grism_pars.throughput
    if throughput is None:
        throughput = np.ones(Nlam)
    else:
        throughput = np.asarray(throughput)
    dlam = float(abs(lambda_grid[1] - lambda_grid[0]))

    rows_grid, cols_grid = np.mgrid[0:Nrow, 0:Ncol].astype(np.float64)
    c_row, c_col = (Nrow - 1) / 2.0, (Ncol - 1) / 2.0
    cos_r, sin_r = np.cos(image_rotation), np.sin(image_rotation)

    npix = Nrow * Ncol
    out_rows, in_cols, weights = [], [], []
    row_index = np.arange(npix)
    for k in range(Nlam):
        dx_k = pixel_offsets[k] * cos_a
        dy_k = pixel_offsets[k] * sin_a
        # rotated pull coordinates about the stamp center (matches
        # disperse_cube's image_rotation branch)
        u = cols_grid - dx_k - c_col
        v = rows_grid - dy_k - c_row
        row_s = c_row + sin_r * u + cos_r * v
        col_s = c_col + cos_r * u - sin_r * v

        i0 = np.floor(row_s)
        j0 = np.floor(col_s)
        w_row = _catmull_rom_weights(row_s - i0)  # (4, Nrow, Ncol)
        w_col = _catmull_rom_weights(col_s - j0)
        scale = throughput[k] * dlam
        for di in range(4):
            ii = i0.astype(np.int64) + (di - 1)
            valid_i = (ii >= 0) & (ii < Nrow)
            for dj in range(4):
                jj = j0.astype(np.int64) + (dj - 1)
                valid = valid_i & (jj >= 0) & (jj < Ncol)
                w = w_row[di] * w_col[dj] * scale
                w = np.where(valid, w, 0.0).ravel()
                keep = w != 0.0
                if not keep.any():
                    continue
                flat_in = (
                    k * npix
                    + np.clip(ii, 0, Nrow - 1) * Ncol
                    + np.clip(jj, 0, Ncol - 1)
                ).ravel()
                out_rows.append(row_index[keep])
                in_cols.append(flat_in[keep])
                weights.append(w[keep])

    data = jnp.asarray(np.concatenate(weights))
    coords = jnp.asarray(
        np.stack([np.concatenate(out_rows), np.concatenate(in_cols)], axis=1).astype(
            np.int32
        )
    )
    return jsparse.BCOO((data, coords), shape=(npix, npix * Nlam))


def apply_dispersion_operator(operator, cube: jnp.ndarray) -> jnp.ndarray:
    """Apply a precomputed dispersion operator to a cube.

    ``cube`` has shape (Nrow, Ncol, Nlambda) (fine pixels); returns the
    detector-frame dispersed image, shape (Nrow, Ncol). Linear in the
    cube with constant coefficients, so the autodiff backward pass is the
    exact operator transpose.
    """
    Nrow, Ncol, _ = cube.shape
    flat = jnp.transpose(cube, (2, 0, 1)).ravel()
    return (operator @ flat).reshape(Nrow, Ncol)
