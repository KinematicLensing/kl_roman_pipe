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
        slice_width_kms: float = None,
    ) -> 'CubePars':
        """Build CubePars centered on the emission line complex at redshift z.

        Parameters
        ----------
        z : float
            Galaxy redshift.
        velocity_window_kms : float
            Half-width of velocity window in km/s. Default 3000.
        n_lambda : int, optional
            Number of wavelength pixels. If None (and ``slice_width_kms``
            is None), sized so slices are ~1 dispersion pixel apart --
            fine for exploration, too coarse for production fits (see
            ``build_grism_obs``).
        line_lambdas_rest : tuple of float, optional
            Rest-frame wavelengths (nm) of lines to cover. If None, uses
            H-alpha (656.28 nm).
        slice_width_kms : float, optional
            Wavelength slice width in velocity units (km/s). Mutually
            exclusive with ``n_lambda``; see ``build_grism_obs`` for the
            sizing rule.
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

        if slice_width_kms is not None:
            if n_lambda is not None:
                raise ValueError(
                    "pass either n_lambda or slice_width_kms, not both "
                    f"(got n_lambda={n_lambda}, slice_width_kms={slice_width_kms})"
                )
            if slice_width_kms <= 0:
                raise ValueError(f"slice_width_kms must be > 0, got {slice_width_kms}")
            dlam_slice = lam_center * slice_width_kms / C_KMS
            n_lambda = int(np.ceil((lam_max - lam_min) / dlam_slice)) + 1
            n_lambda = max(n_lambda, 3)

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
    BoxPixel sinc + coarse-pixel-center readout at the 2D output
    (``kl_pipe.grism._apply_post_dispersion_pixel_response``).

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

    # base pixel coordinates (default float dtype: float64 under x64,
    # float32 under KLPIPE_FP32 -- avoids downcast warnings in fp32 mode)
    rows = jnp.arange(Nrow, dtype=float)
    cols = jnp.arange(Ncol, dtype=float)
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

    # trapezoid endpoint weights: the boundary slices are the edges of the
    # integration window, not interior samples, so they contribute half a
    # dlam each. This removes the rectangle-rule flux excess on the smooth
    # continuum; the emission-line term is unaffected (integrated exactly
    # per bin and zero at the window edges).
    quad_weight = jnp.ones(Nlam).at[0].set(0.5).at[-1].set(0.5)

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

        dispersed = dispersed + shifted * throughput[k] * dlam * quad_weight[k]

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
# -0.006 sigma). Grid padding and mean-transfer-function deconvolution were
# measured and rejected as alternatives (the former does not touch the
# posterior-relevant mode; the latter overcorrects ~2x).


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

    # trapezoid endpoint weights, matching disperse_cube (edge slices count
    # half a dlam); the two dispersal pathways must stay quadrature-identical
    quad_weight = np.ones(Nlam)
    quad_weight[0] = 0.5
    quad_weight[-1] = 0.5

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
        scale = throughput[k] * dlam * quad_weight[k]
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


# ---------------------------------------------------------------------------
# Analytic per-spaxel dispersal (no wavelength grid for emission lines).
#
# The emission line's dispersed footprint has a closed form: the cube's erf
# bin integration in lambda followed by bilinear shift-and-add converges, as
# the wavelength grid densifies, to a Gaussian convolved with the triangle
# kernel of bilinear interpolation (the "tent") along the dispersion axis --
# and that convolution is exact in erf/exp terms. Evaluating it per fine
# spaxel removes the wavelength grid from the line entirely; the flat
# continuum has its own closed form (a throughput-weighted box convolution
# along the trace).
# ---------------------------------------------------------------------------


def _normal_cdf(z: jnp.ndarray) -> jnp.ndarray:
    return 0.5 * (1.0 + jax.scipy.special.erf(z / jnp.sqrt(2.0)))


def _normal_pdf(z: jnp.ndarray) -> jnp.ndarray:
    return jnp.exp(-0.5 * z * z) / jnp.sqrt(2.0 * jnp.pi)


def _normal_cdf_antiderivative(z: jnp.ndarray) -> jnp.ndarray:
    # antiderivative of the standard normal CDF: z * Phi(z) + phi(z)
    return z * _normal_cdf(z) + _normal_pdf(z)


def gaussian_tent_profile(u: jnp.ndarray, sigma: jnp.ndarray) -> jnp.ndarray:
    """Closed form of (normalized Gaussian_sigma convolved with unit tent)(u).

    This is the wavelength-continuum limit of one spaxel's dispersed line
    profile under the current pipeline conventions: exact erf bin
    integration in lambda plus bilinear (tent) interpolation of the
    per-slice shift. ``u`` is the detector-axis distance from the center
    of the spaxel's dispersed footprint in fine pixels; ``sigma`` the
    line width in fine pixels. Both broadcast. Dimensionless; integrates
    to 1 over u.
    """
    return sigma * (
        _normal_cdf_antiderivative((u + 1.0) / sigma)
        - 2.0 * _normal_cdf_antiderivative(u / sigma)
        + _normal_cdf_antiderivative((u - 1.0) / sigma)
    )


def disperse_line_analytic(
    I_line: jnp.ndarray,
    xi: jnp.ndarray,
    sigma_s: jnp.ndarray,
    halfwidth: int,
    weight: jnp.ndarray = None,
) -> jnp.ndarray:
    """Disperse one emission line in closed form, one spaxel at a time.

    Each source spaxel (r, j) contributes its line flux
    ``I_line[r, j] * weight[r, j]`` to the dispersed image, spread along
    the dispersion axis (detector +x) within its own row as
    ``gaussian_tent_profile`` centered ``xi[r, j]`` fine pixels from the
    source column -- the line's dispersed footprint. Flux carried beyond
    the stamp edge is dropped, matching ``disperse_cube``'s
    mode='constant' pull semantics.

    Parameters
    ----------
    I_line : jnp.ndarray
        Line surface brightness on the (fine) spatial grid, shape
        (Nrow, Ncol).
    xi : jnp.ndarray
        Footprint-center offset per spaxel in fine pixels along +x:
        ``(lambda_obs - lambda_ref) / dispersion * oversample``.
    sigma_s : jnp.ndarray
        Line width per spaxel in fine pixels:
        ``sigma_lambda / dispersion * oversample``.
    halfwidth : int
        Static half-width, in fine pixels, of the window each spaxel's
        flux is spread over. Must cover ``max|xi| + ~4 max(sigma_s)``;
        flux beyond the window is dropped (erfc-small when sized from
        priors or concrete parameter values).
    weight : jnp.ndarray, optional
        Per-spaxel multiplier (throughput evaluated at each spaxel's
        observed wavelength). None = 1.

    Returns
    -------
    jnp.ndarray
        Dispersed line image, same shape as ``I_line`` (dimensionless
        profile times SB -- same units as ``I_line``).
    """
    if halfwidth < 1:
        raise ValueError(f"halfwidth must be >= 1, got {halfwidth}")
    amp = I_line if weight is None else I_line * weight
    n = I_line.shape[1]
    out = jnp.zeros_like(I_line)
    # consecutive taps share Psi evaluations: the profile at tap w is
    # sigma * (Psi_{w+1} - 2 Psi_w + Psi_{w-1}), so a rolling second
    # difference needs one new Psi (erf + exp) per tap instead of three
    inv_sigma = 1.0 / sigma_s
    amp_sigma = amp * sigma_s
    P_prev = _normal_cdf_antiderivative((-halfwidth - 1 - xi) * inv_sigma)
    P_cur = _normal_cdf_antiderivative((-halfwidth - xi) * inv_sigma)
    for w in range(-halfwidth, halfwidth + 1):
        P_next = _normal_cdf_antiderivative((w + 1 - xi) * inv_sigma)
        term = amp_sigma * (P_next - 2.0 * P_cur + P_prev)
        if w == 0:
            out = out + term
        elif w > 0:
            out = out.at[:, w:].add(term[:, : n - w])
        else:
            out = out.at[:, :w].add(term[:, -w:])
        P_prev, P_cur = P_cur, P_next
    return out


def _tent_running_integral(t: np.ndarray) -> np.ndarray:
    """Integral of the unit tent from -inf to t (numpy, precompute only)."""
    p2 = lambda z: 0.5 * np.square(np.clip(z, 0.0, None))  # noqa: E731
    return p2(t + 1.0) - 2.0 * p2(t) + p2(t - 1.0)


def continuum_trace_kernel(
    grism_pars: GrismPars,
    lambda_grid: jnp.ndarray,
    oversample: int,
    integration_window: Optional[Tuple[float, float]] = None,
) -> tuple:
    """Precompute the exact continuum trace kernel (numpy, static).

    A flat-in-lambda continuum disperses to a (throughput-weighted) box
    convolution along the dispersion axis of the tent-reconstructed image:
    ``D[r, q] = sum_j I_cont[r, j] * kernel[q - j - m_lo]``. The kernel is

        kernel(m) = (dispersion / oversample) *
                    integral_{s_min}^{s_max} T(lambda(s)) tri(m - s) ds

    with s the trace coordinate in fine pixels over the wavelength window
    [lambda_lo, lambda_hi]. Flat throughput uses the exact closed form; a
    sampled throughput is integrated per unit-s segment with 16-point
    Gauss-Legendre (smooth-T assumption; exact to numerical precision for
    physically smooth bandpasses).

    Parameters
    ----------
    lambda_grid : array
        Wavelength samples (nm). When ``integration_window`` is None these
        also define the integration limits (``[lambda_grid[0],
        lambda_grid[-1]]``, the window the slice method integrates with
        trapezoid weights). When ``integration_window`` is given, this
        array serves only as the throughput lookup table; throughput
        outside its range is held at the nearest endpoint value
        (``np.interp`` default), accurate for the gentle throughput
        variation over a widened stamp-scale window.
    integration_window : (float, float), optional
        Explicit ``(lambda_lo, lambda_hi)`` integration limits (nm),
        decoupled from the throughput table. Used to widen the continuum
        trace beyond the line window so it fills the stamp (see
        ``RenderConfig.continuum_fills_stamp``) without resampling the
        throughput array. When None, falls back to the lambda_grid
        endpoints (legacy behavior).

    Returns
    -------
    (np.ndarray, int)
        Kernel values at integer offsets and the first offset ``m_lo``.
    """
    lam = np.asarray(lambda_grid, dtype=float)
    scale = grism_pars.dispersion / oversample  # nm per fine pixel
    if integration_window is None:
        lam_lo, lam_hi = lam[0], lam[-1]
    else:
        lam_lo, lam_hi = integration_window
    s_min = (lam_lo - grism_pars.lambda_ref) / grism_pars.dispersion * oversample
    s_max = (lam_hi - grism_pars.lambda_ref) / grism_pars.dispersion * oversample
    m_lo = int(np.floor(s_min)) - 1
    m_hi = int(np.ceil(s_max)) + 1
    m = np.arange(m_lo, m_hi + 1, dtype=float)

    if grism_pars.throughput is None:
        kern = scale * (
            _tent_running_integral(m - s_min) - _tent_running_integral(m - s_max)
        )
        return kern, m_lo

    # throughput-weighted: integrate per unit-s segment (tent kinks sit on
    # integer s) with fixed-order Gauss-Legendre
    T_samples = np.asarray(grism_pars.throughput, dtype=float)
    nodes, gl_weights = np.polynomial.legendre.leggauss(16)
    breaks = np.unique(
        np.concatenate([[s_min, s_max], np.arange(np.ceil(s_min), np.floor(s_max) + 1)])
    )
    kern = np.zeros_like(m)
    for a, b in zip(breaks[:-1], breaks[1:]):
        mid, half = 0.5 * (a + b), 0.5 * (b - a)
        s = mid + half * nodes
        lam_s = grism_pars.lambda_ref + s * grism_pars.dispersion / oversample
        T_s = np.interp(lam_s, lam, T_samples)
        tri = np.clip(1.0 - np.abs(m[:, None] - s[None, :]), 0.0, None)
        kern += half * (tri * (T_s * gl_weights)[None, :]).sum(axis=1)
    return scale * kern, m_lo


def disperse_continuum_analytic(
    I_cont: jnp.ndarray, kernel: np.ndarray, m_lo: int
) -> jnp.ndarray:
    """Disperse a flat-in-lambda continuum with a precomputed trace kernel.

    ``D[r, q] = sum_j I_cont[r, j] * kernel[q - j - m_lo]`` per row --
    exact closed form of the wavelength-continuum limit (no slices).
    ``I_cont`` is a spectral-density surface brightness [SB/nm]; the
    kernel carries the nm-per-fine-pixel scale so the output is SB.
    """
    n = I_cont.shape[1]
    L = len(kernel)
    kern = jnp.asarray(kernel)
    # D[q] = full[q - m_lo] with full the length n + L - 1 convolution;
    # slice bounds are static python ints, padded where the trace window
    # extends past the stamp
    start = -m_lo
    pad_left = max(0, -start)
    pad_right = max(0, (start + n) - (n + L - 1))

    def row_conv(row):
        full = jnp.convolve(row, kern, mode='full')
        full = jnp.pad(full, (pad_left, pad_right))
        return full[start + pad_left : start + pad_left + n]

    return jax.vmap(row_conv)(I_cont)
