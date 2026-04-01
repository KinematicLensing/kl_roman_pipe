import numpy as np
import jax.numpy as jnp
import jax
from scipy.fft import next_fast_len
from scipy.special import kve as scipy_kve

from kl_pipe.model import IntensityModel
from kl_pipe.transformation import obs2cen, cen2source, source2gal


# default number of Gauss-Legendre quadrature points for LOS integration
_DEFAULT_N_QUAD = 200

# best-fit Spergel nu for Sersic n=4 (de Vaucouleurs)
_DEVAUCOULEURS_NU = -0.6


# ==============================================================================
# Spergel nu <-> Sersic n mapping
# Pre-computed by scripts/compute_nu_n_mapping.py (flux-weighted L2 matching).
# Two tables: face-on (galsim.Spergel vs galsim.Sersic) and inclined
# (our InclinedSpergelModel vs galsim.InclinedSersic, averaged over cosi).
# Exact at n=1/nu=0.5. Valid for n in [0.65, 4.0] (saturates at nu~4 for n<0.65).
# ==============================================================================

_N_GRID = np.array(
    [
        0.300000,
        0.350000,
        0.400000,
        0.450000,
        0.500000,
        0.550000,
        0.600000,
        0.650000,
        0.700000,
        0.750000,
        0.800000,
        0.850000,
        0.900000,
        0.950000,
        1.000000,
        1.100000,
        1.200000,
        1.300000,
        1.400000,
        1.500000,
        1.600000,
        1.700000,
        1.800000,
        1.900000,
        2.000000,
        2.200000,
        2.400000,
        2.600000,
        2.800000,
        3.000000,
        3.200000,
        3.400000,
        3.600000,
        3.800000,
        4.000000,
    ]
)

_NU_TABLE_FACEON = np.array(
    [
        4.000000,
        4.000000,
        4.000000,
        3.999999,
        4.000000,
        4.000000,
        3.999985,
        2.565989,
        1.839106,
        1.400138,
        1.104737,
        0.891520,
        0.729871,
        0.602743,
        0.500000,
        0.343574,
        0.229593,
        0.142302,
        0.072840,
        0.015778,
        -0.032252,
        -0.073571,
        -0.109774,
        -0.141940,
        -0.170920,
        -0.221439,
        -0.264560,
        -0.302262,
        -0.335768,
        -0.365888,
        -0.393272,
        -0.418261,
        -0.441244,
        -0.462466,
        -0.482161,
    ]
)

_NU_TABLE_INCLINED = np.array(
    [
        3.999974,
        3.999980,
        3.999949,
        3.999972,
        3.999984,
        3.999981,
        3.999952,
        3.999987,
        3.087383,
        2.098241,
        1.509082,
        1.122171,
        0.850920,
        0.651659,
        0.500000,
        0.285832,
        0.142495,
        0.039635,
        -0.037857,
        -0.097893,
        -0.144339,
        -0.180206,
        -0.209292,
        -0.234021,
        -0.255682,
        -0.296949,
        -0.332018,
        -0.357537,
        -0.381879,
        -0.404837,
        -0.423129,
        -0.439844,
        -0.456175,
        -0.470317,
        -0.482003,
    ]
)


def sersic_to_spergel(n_sersic, inclined=False):
    """Best-fit Spergel nu for a given Sersic n.

    Interpolates a pre-computed lookup table from flux-weighted L2
    profile matching. Exact at n=1 (nu=0.5). Valid for n in [0.65, 4.0].

    Parameters
    ----------
    n_sersic : float or array
        Sersic index.
    inclined : bool, optional
        If True, use the inclined table (our 3D model vs GalSim
        InclinedSersic, averaged over cosi). Default False (face-on).

    Returns
    -------
    float or array
        Best-fit Spergel nu.
    """
    table = _NU_TABLE_INCLINED if inclined else _NU_TABLE_FACEON
    return np.interp(n_sersic, _N_GRID, table)


def spergel_to_sersic(nu, inclined=False):
    """Best-fit Sersic n for a given Spergel nu.

    Inverse of ``sersic_to_spergel``. Valid for nu in [-0.48, 2.6]
    (face-on) or [-0.16, 1.4] (inclined).

    Parameters
    ----------
    nu : float or array
        Spergel index.
    inclined : bool, optional
        If True, use the inclined table. Default False (face-on).

    Returns
    -------
    float or array
        Best-fit Sersic n.
    """
    table = _NU_TABLE_INCLINED if inclined else _NU_TABLE_FACEON
    # tables are monotonically decreasing for n >= ~0.65; reverse for interp
    mask = table < 3.99  # exclude saturated region
    n_valid = _N_GRID[mask]
    nu_valid = table[mask]
    return np.interp(nu, nu_valid[::-1], n_valid[::-1])


# ==============================================================================
# Shared k-space rendering
# ==============================================================================


def _kspace_render_core(
    ft_image_fn,
    Nrow,
    Ncol,
    pixel_scale,
    pad_factor=2,
    oversample=1,
    psf_kernel_fft=None,
):
    """Shared IFFT plumbing for k-space intensity model rendering.

    Builds the padded k-grid, calls ``ft_image_fn(KX, KY)`` to get the
    model-specific complex FT, optionally fuses a pre-computed PSF in
    k-space, then IFFTs and crops to the output grid.

    Each model builds its own closure for ``ft_image_fn`` that handles
    parameter lookup, geometric transforms (shear, rotation, phase), and
    the analytic FT of its radial + vertical profile.

    Anti-aliasing: the IFFT is computed on a padded grid (pad_factor × N)
    to suppress periodic boundary wrap-around, then cropped to (Nrow, Ncol).

    When ``psf_kernel_fft`` is provided, the PSF is multiplied in k-space
    BEFORE the IFFT crop, so edge pixels see PSF-scattered light from
    source regions beyond the image boundary.

    When ``oversample > 1``, the k-grid extends to N × Nyquist, reducing
    cusp aliasing. The IFFT is computed at fine resolution and subsampled
    back to (Nrow, Ncol). The half-pixel phase correction (inside the
    model's ft_image_fn) uses coarse grid centering so subsampled positions
    align with the standard centered grid.

    Parameters
    ----------
    ft_image_fn : callable
        (KX, KY) -> complex I_hat array. Each model builds its own closure
        that handles all physics (parameter lookup, geometric transforms,
        analytic FT, normalization).
    Nrow, Ncol : int
        Output (coarse) grid dimensions.
    pixel_scale : float
        Output pixel scale (arcsec/pixel).
    pad_factor : int
        FFT padding factor. Default 2.
    oversample : int
        Oversampling factor for sub-pixel anti-aliasing. Default 1.
    psf_kernel_fft : jnp.ndarray, optional
        Pre-computed PSF FFT for fused convolution. Shape must match
        the padded grid.

    Returns
    -------
    jnp.ndarray, shape (Nrow, Ncol)
        Rendered image at coarse resolution.
    """
    eff_Nrow = Nrow * oversample
    eff_Ncol = Ncol * oversample
    eff_ps = pixel_scale / oversample

    if psf_kernel_fft is not None:
        pad_row, pad_col = psf_kernel_fft.shape
    elif pad_factor > 1:
        pad_row = next_fast_len(pad_factor * eff_Nrow)
        pad_col = next_fast_len(pad_factor * eff_Ncol)
    else:
        pad_row = eff_Nrow
        pad_col = eff_Ncol

    # k-grid: ky conjugate to rows (vertical), kx conjugate to cols (horizontal)
    ky = 2.0 * jnp.pi * jnp.fft.fftfreq(pad_row, d=eff_ps)
    kx = 2.0 * jnp.pi * jnp.fft.fftfreq(pad_col, d=eff_ps)
    KY, KX = jnp.meshgrid(ky, kx, indexing='ij')

    I_hat = ft_image_fn(KX, KY)

    if psf_kernel_fft is not None:
        I_hat = I_hat * psf_kernel_fft

    # IFFT on padded grid, then extract center eff_Nrow×eff_Ncol
    # roll amount must align with subsampling grid: (Nrow//2)*oversample
    # ensures DC lands on a subsampled pixel for correct centering
    full = jnp.fft.ifft2(I_hat).real
    roll_row = (Nrow // 2) * oversample
    roll_col = (Ncol // 2) * oversample
    full = jnp.roll(full, (roll_row, roll_col), axis=(0, 1))
    image = full[:eff_Nrow, :eff_Ncol] / eff_ps**2

    if oversample > 1:
        image = image.reshape(Nrow, oversample, Ncol, oversample).mean(axis=(1, 3))

    return image


def _inclined_sech2_ft(
    KX,
    KY,
    radial_ft_fn,
    flux,
    x0,
    y0,
    g1,
    g2,
    theta_int,
    cosi,
    rscale,
    h_over_r,
    pixel_scale,
    Nrow,
    Ncol,
):
    """FT of an inclined model with pluggable radial profile and sech² vertical.

    Computes the physical/geometric transforms shared by all inclined models
    with sech²(z/h_z) vertical structure:

    1. Centroid phase shift
    2. Area-preserving shear of k-vectors
    3. Rotation by position angle
    4. Scaling by scale length → dimensionless k
    5. Radial FT via ``radial_ft_fn(k_sq)`` (model-specific)
    6. Vertical FT: ``u/sinh(u)`` where ``u = (π/2)·h/r·ky·sin(i)``

    Parameters
    ----------
    KX, KY : jnp.ndarray
        k-space grids from ``_kspace_render_core``.
    radial_ft_fn : callable
        k_sq -> ft_radial array. Model-specific radial profile FT.
        Exponential: ``1/(1+k²)^{3/2}``.
        Spergel: ``1/(1+k²)^{1+nu}``.
    flux : float
        Total integrated flux.
    x0, y0 : float
        Centroid position (arcsec).
    g1, g2 : float
        Shear components.
    theta_int : float
        Position angle (radians).
    cosi : float
        Cosine of inclination.
    rscale : float
        Scale length (arcsec).
    h_over_r : float
        Vertical scale height / radial scale length.
    pixel_scale : float
        Coarse pixel scale for half-pixel phase correction.
    Nrow, Ncol : int
        Coarse grid dimensions for half-pixel phase correction.

    Returns
    -------
    jnp.ndarray
        Complex FT array, same shape as KX/KY.
    """
    sini = jnp.sqrt(jnp.maximum(1.0 - cosi**2, 0.0))

    # centroid phase: pair kx with x0 (horizontal), ky with y0 (vertical)
    #   half-pixel correction based on OUTPUT grid centering
    hx = 0.5 * pixel_scale * (1 - Ncol % 2)
    hy = 0.5 * pixel_scale * (1 - Nrow % 2)
    phase = jnp.exp(-1j * (KX * (x0 - hx) + KY * (y0 - hy)))

    # shear: area-preserving M = (1/sqrt(1-|g|²)) * [[1+g1, g2], [g2, 1-g1]]
    #   (1+g1) multiplies kx (horizontal), (1-g1) multiplies ky (vertical)
    norm_shear = 1.0 / jnp.sqrt(1.0 - (g1**2 + g2**2))
    kx_s = norm_shear * ((1.0 + g1) * KX + g2 * KY)
    ky_s = norm_shear * (g2 * KX + (1.0 - g1) * KY)

    # rotation: R(-theta_int) on (kx, ky)
    c = jnp.cos(-theta_int)
    s = jnp.sin(-theta_int)
    kx_gal = c * kx_s - s * ky_s
    ky_gal = s * kx_s + c * ky_s

    # scale to dimensionless k
    kx_scaled = kx_gal * rscale
    ky_scaled = ky_gal * rscale

    # radial FT (model-specific via callback)
    # cosi compresses rows=vertical in k-space (gal2disk)
    k_sq = kx_scaled**2 + (ky_scaled * cosi) ** 2
    ft_radial = radial_ft_fn(k_sq)

    # vertical FT: u/sinh(u), u = (π/2)·h_over_r·ky_scaled·sini
    # safe-where pattern: substitute finite dummy in non-selected branch
    # so JAX autodiff never sees 0/sinh(0) = 0/0 = NaN
    u = (jnp.pi / 2.0) * h_over_r * ky_scaled * sini
    u_safe = jnp.where(jnp.abs(u) < 1e-4, jnp.ones_like(u), u)
    ft_vertical = jnp.where(
        jnp.abs(u) < 1e-4,
        1.0 - u**2 / 6.0,
        u_safe / jnp.sinh(u_safe),
    )

    return flux * ft_radial * ft_vertical * phase


def _kspace_render_image(
    model,
    theta,
    image_pars=None,
    plane='obs',
    X=None,
    Y=None,
    oversample=1,
    *,
    obs=None,
    **kwargs,
):
    """Shared render_image dispatcher for k-space intensity models.

    Handles all obs-based PSF paths (fused k-space, fallback real-space),
    obs without PSF, and legacy image_pars/X,Y calling conventions.

    Used by InclinedSpergelModel and InclinedDeVaucouleursModel.
    InclinedExponentialModel keeps its own render_image (with aliasing
    warning specific to the exponential FT decay rate).

    Calling conventions:
    - render_image(theta, obs=obs) -- with PSF from obs
    - render_image(theta, image_pars=image_pars) -- no PSF

    Parameters
    ----------
    model : IntensityModel
        Model instance with ``_render_kspace`` method.
    theta : jnp.ndarray
        Parameter array.
    image_pars : ImagePars, optional
        Image parameters defining grid geometry.
    plane : str
        Coordinate plane (unused — k-space renders in obs plane).
    X, Y : jnp.ndarray, optional
        Pre-computed grids (for legacy calling convention).
    oversample : int
        Cusp anti-aliasing factor for no-PSF path. Default 1.
    obs : ImageObs, optional
        Observation object with PSF and oversampling config.

    Returns
    -------
    jnp.ndarray
        Rendered image, shape (Nrow, Ncol).
    """
    if obs is not None:
        pixel_scale = obs.image_pars.pixel_scale
        Nrow = obs.image_pars.Nrow
        Ncol = obs.image_pars.Ncol

        if obs.kspace_psf_fft is not None:
            # fused k-space path: render + convolve in one FFT pass
            N = max(obs.oversample, 1)
            image = model._render_kspace(
                theta,
                Nrow * N,
                Ncol * N,
                pixel_scale / N,
                psf_kernel_fft=obs.kspace_psf_fft,
            )
            if N > 1:
                image = image.reshape(Nrow, N, Ncol, N).mean(axis=(1, 3))
            return image

        if obs.psf_data is not None:
            # fallback real-space path
            from kl_pipe.psf import convolve_fft

            if obs.oversample > 1:
                N = obs.oversample
                image = model._render_kspace(theta, Nrow * N, Ncol * N, pixel_scale / N)
            else:
                image = model._render_kspace(theta, Nrow, Ncol, pixel_scale)
            return convolve_fft(image, obs.psf_data)

        # no PSF on obs
        return model._render_kspace(
            theta, Nrow, Ncol, pixel_scale, oversample=oversample
        )

    # legacy/convenience path (no obs)
    if image_pars is None and (X is None or Y is None):
        raise ValueError("Provide obs, image_pars, or (X, Y)")

    if image_pars is not None:
        pixel_scale = image_pars.pixel_scale
        Nrow = image_pars.Nrow
        Ncol = image_pars.Ncol
    else:
        Nrow, Ncol = X.shape
        pixel_scale = jnp.abs(X[0, 1] - X[0, 0])

    return model._render_kspace(theta, Nrow, Ncol, pixel_scale, oversample=oversample)


# ==============================================================================
# Spergel profile helpers (real-space, non-differentiable via scipy callback)
# ==============================================================================


def _bessel_kve_jax(nu, x):
    """K_nu(x) * exp(x), JIT-safe via pure_callback wrapping scipy.

    Returns the exponentially-scaled modified Bessel function of the
    second kind. Wraps ``scipy.special.kve`` through ``jax.pure_callback``
    so it can be called inside JIT-compiled functions.

    NOTE: not auto-differentiable. The k-space inference path (render_image)
    never calls this — it uses the analytic FT ``(1+k²)^{-(1+nu)}`` which
    is fully differentiable. If autodiff through real-space K_nu is ever
    needed, replace with Gauss-Laguerre quadrature of
    ``K_nu(x) = integral_0^inf exp(-x*cosh(t))*cosh(nu*t) dt``.

    Parameters
    ----------
    nu : float or jnp.ndarray (scalar)
        Order of the Bessel function.
    x : jnp.ndarray
        Argument array (must be positive).

    Returns
    -------
    jnp.ndarray
        kve(nu, x) = K_nu(x) * exp(x), same shape as x.
    """
    result_shape = jax.ShapeDtypeStruct(x.shape, x.dtype)

    def _scipy_kve(nu_val, x_np):
        return scipy_kve(float(nu_val), x_np).astype(x_np.dtype)

    return jax.pure_callback(_scipy_kve, result_shape, nu, x)


def _spergel_radial(r, nu, c):
    """Evaluate (r/c)^nu * K_nu(r/c), numerically stable via kve.

    The Spergel radial profile is proportional to x^nu * K_nu(x) where
    x = r/c. Computed in log-space to avoid overflow::

        log[(r/c)^nu * K_nu(r/c)]
        = nu*log(r/c) + log(kve(nu, r/c)) - r/c

    since kve(nu, x) = K_nu(x) * exp(x).

    Parameters
    ----------
    r : jnp.ndarray
        Radius array.
    nu : float or jnp.ndarray (scalar)
        Spergel index.
    c : float or jnp.ndarray (scalar)
        Scale length.

    Returns
    -------
    jnp.ndarray
        Profile values, same shape as r.
    """
    x = r / c
    x_safe = jnp.maximum(x, 1e-30)  # avoid log(0)
    log_profile = nu * jnp.log(x_safe) + jnp.log(_bessel_kve_jax(nu, x_safe)) - x_safe
    return jnp.exp(log_profile)


def _spergel_norm_2d(rscale, nu):
    """Face-on Spergel flux normalization factor.

    The total flux of the Spergel profile integrates to::

        F = 2*pi * c^2 * 2^nu * Gamma(nu+1) * I_0

    so this returns the ``2*pi * c^2 * 2^nu * Gamma(nu+1)`` factor.
    For nu=0.5 this reduces to ``2*pi * c^2 * sqrt(pi/2)``, recovering
    the exponential normalization ``F = 2*pi * r_s^2 * I_0`` after the
    ``sqrt(pi/2)`` factor from ``K_{0.5}`` cancels.

    Parameters
    ----------
    rscale : float
        Scale length c (arcsec).
    nu : float
        Spergel index.

    Returns
    -------
    float
        Normalization factor.
    """
    return (
        2.0
        * jnp.pi
        * rscale**2
        * jnp.power(2.0, nu)
        * jnp.exp(jax.scipy.special.gammaln(nu + 1.0))
    )


def _spergel_evaluate_faceon(flux, rscale, nu, x, y):
    """Face-on Spergel surface brightness in disk plane.

    Computes I(r) = I_0 * (r/c)^nu * K_nu(r/c) where
    I_0 = flux / (2*pi * c^2 * 2^nu * Gamma(nu+1)).

    Parameters
    ----------
    flux : float
        Total integrated flux.
    rscale : float
        Scale length c (arcsec).
    nu : float
        Spergel index.
    x, y : jnp.ndarray
        Disk-plane coordinates.

    Returns
    -------
    jnp.ndarray
        Surface brightness, same shape as x/y.
    """
    r = jnp.sqrt(x**2 + y**2)
    I0 = flux / _spergel_norm_2d(rscale, nu)
    return I0 * _spergel_radial(r, nu, rscale)


def _spergel_los_integrate(
    flux,
    rscale,
    nu,
    h_over_r,
    cosi,
    sini,
    xp,
    yp,
    gl_nodes,
    gl_weights,
):
    """3D LOS Gauss-Legendre integration with Spergel radial + sech² vertical.

    Integrates rho(R, z) = rho0 * (R/c)^nu * K_nu(R/c) * sech²(z/h_z)
    along the line of sight at each pixel in the galaxy frame.

    Volume density normalization::

        rho0 = flux / (4*pi * h_z * c^2 * 2^nu * Gamma(nu+1))

    For nu=0.5 this reduces to ``flux / (4*pi * h_z * r_s^2)``, matching
    the exponential model.

    Parameters
    ----------
    flux : float
        Total integrated flux.
    rscale : float
        Scale length c (arcsec).
    nu : float
        Spergel index.
    h_over_r : float
        Vertical scale height / radial scale length.
    cosi, sini : float
        Cosine and sine of inclination.
    xp, yp : jnp.ndarray
        Coordinates in galaxy frame (NOT disk frame).
    gl_nodes, gl_weights : jnp.ndarray
        Gauss-Legendre quadrature nodes and weights on [-1, 1].

    Returns
    -------
    jnp.ndarray
        Surface brightness, same shape as xp/yp.
    """
    h_z = h_over_r * rscale
    # rho0 = flux / (4*pi * h_z * c^2 * 2^nu * Gamma(nu+1))
    rho0 = flux / (2.0 * h_z * _spergel_norm_2d(rscale, nu))

    # per-pixel GL centering: sech²(z/h_z) peak at ell_center = y_gal*sini/cosi
    # integration half-width delta = 5*h_z/cosi captures >99.99% of sech²
    delta = 5.0 * h_z / jnp.maximum(cosi, 0.1)
    ell_center = yp * sini / jnp.maximum(cosi, 0.1)

    ell = ell_center[..., None] + delta * gl_nodes
    w = delta * gl_weights

    # disk coords at each quadrature point
    y_disk = yp[..., None] * cosi + ell * sini
    z_val = ell * cosi - yp[..., None] * sini
    x_disk = xp[..., None]

    R = jnp.sqrt(x_disk**2 + y_disk**2)

    radial = _spergel_radial(R, nu, rscale)
    z_norm = z_val / h_z
    cosh_z = jnp.cosh(jnp.clip(z_norm, -20.0, 20.0))
    vertical = 1.0 / (cosh_z**2)

    integrand = rho0 * radial * vertical
    return jnp.sum(integrand * w, axis=-1)


# ==============================================================================
# InclinedExponentialModel
# ==============================================================================


class InclinedExponentialModel(IntensityModel):
    """
    3D inclined exponential disk model with sech² vertical profile.

    Matches GalSim's InclinedExponential: radial exponential disk with
    vertical sech²(z/h_z) profile, integrated along the line of sight.

    Two evaluation paths:
    - ``render_image``: k-space FFT (exact analytic FT, no aliasing)
    - ``__call__``: real-space Gauss-Legendre quadrature (N-pt LOS integration)

    Parameters
    ----------
    cosi : float
        Cosine of inclination (0=edge-on, 1=face-on)
    theta_int : float
        Position angle (radians)
    g1, g2 : float
        Shear components
    flux : float
        Total integrated flux (conserved quantity)
    int_rscale : float
        Exponential scale length (arcsec)
    int_h_over_r : float
        Ratio of vertical scale height to radial scale length.
        h_z = int_h_over_r * int_rscale. GalSim default is 0.1.
    int_x0, int_y0 : float
        Centroid position (arcsec)
    """

    PARAMETER_NAMES = (
        'cosi',
        'theta_int',
        'g1',
        'g2',
        'flux',
        'int_rscale',
        'int_h_over_r',
        'int_x0',
        'int_y0',
    )

    # anti-aliasing pad factor for k-space FFT rendering;
    # 2x squashes periodic boundary wrap-around (e.g. 0.7% → 0.005%)
    _kspace_pad_factor = 2

    def __init__(self, meta_pars=None, n_quad=None):
        super().__init__(meta_pars)
        n = n_quad if n_quad is not None else _DEFAULT_N_QUAD
        self._n_quad = n
        nodes, weights = np.polynomial.legendre.leggauss(n)
        self._gl_nodes = jnp.array(nodes)
        self._gl_weights = jnp.array(weights)

    @property
    def name(self) -> str:
        return 'inclined_exp'

    def render_unconvolved(self, theta, image_pars, oversample=5):
        """Render intensity image WITHOUT PSF, using k-space FT.

        For use by SpectralModel.build_cube() — fast, anti-aliased, no PSF.
        Calls _render_kspace without psf_kernel_fft.
        """
        return self._render_kspace(
            theta,
            image_pars.Nrow,
            image_pars.Ncol,
            image_pars.pixel_scale,
            oversample=oversample,
        )

    def evaluate_in_disk_plane(
        self,
        theta: jnp.ndarray,
        x: jnp.ndarray,
        y: jnp.ndarray,
        z: jnp.ndarray = None,
    ) -> jnp.ndarray:
        """
        Evaluate exponential profile in disk plane (thin-disk projection).

        This is the face-on radial profile only; the full 3D LOS integration
        is done in ``__call__`` and ``render_image``. This method is retained
        for the base class interface and velocity flux weighting.

        Parameters
        ----------
        theta : jnp.ndarray
            Model parameters.
        x, y : jnp.ndarray
            Coordinates in disk plane.
        z : jnp.ndarray, optional
            Currently unused for this intensity model.
        """

        flux = self.get_param('flux', theta)
        rscale = self.get_param('int_rscale', theta)

        # compute radius in disk plane
        r_disk = jnp.sqrt(x**2 + y**2)

        # convert flux to central surface brightness
        #   for exponential: F = 2π * I0 * r_scale²
        I0_disk = flux / (2.0 * jnp.pi * rscale**2)

        # evaluate exponential profile in disk plane
        intensity_disk = I0_disk * jnp.exp(-r_disk / rscale)

        return intensity_disk

    def __call__(
        self,
        theta: jnp.ndarray,
        plane: str,
        x: jnp.ndarray,
        y: jnp.ndarray,
        z: jnp.ndarray = None,
    ) -> jnp.ndarray:
        """
        Evaluate 3D inclined exponential via LOS Gauss-Legendre quadrature.

        Integrates rho(R, z) = rho0 * exp(-R/r_s) * sech²(z/h_z) along the
        line of sight at each pixel in the galaxy frame (obs → cen → source → gal,
        but NOT deprojected to disk).
        """
        x0 = self.get_param('int_x0', theta)
        y0 = self.get_param('int_y0', theta)
        g1 = self.get_param('g1', theta)
        g2 = self.get_param('g2', theta)
        theta_int = self.get_param('theta_int', theta)
        cosi = self.get_param('cosi', theta)
        flux = self.get_param('flux', theta)
        rscale = self.get_param('int_rscale', theta)
        h_over_r = self.get_param('int_h_over_r', theta)

        sini = jnp.sqrt(jnp.maximum(1.0 - cosi**2, 0.0))
        h_z = h_over_r * rscale

        # transform to galaxy frame (NOT disk — we integrate LOS ourselves)
        xp, yp = x, y
        if plane == 'obs':
            xp, yp = obs2cen(x0, y0, xp, yp)
        if plane in ('obs', 'cen'):
            xp, yp = cen2source(g1, g2, xp, yp)
        if plane in ('obs', 'cen', 'source'):
            xp, yp = source2gal(theta_int, xp, yp)

        if plane == 'disk':
            # face-on: no LOS integration needed, return thin-disk SB
            r_disk = jnp.sqrt(x**2 + y**2)
            I0_disk = flux / (2.0 * jnp.pi * rscale**2)
            return I0_disk * jnp.exp(-r_disk / rscale)

        # volume density normalization: flux = integral over all space
        # rho0 = flux / (4 * pi * h_z * r_s^2)
        rho0 = flux / (4.0 * jnp.pi * h_z * rscale**2)

        # Per-pixel GL centering: sech²(z/h_z) peak is at ell where
        # z = ell*cosi - y_gal*sini = 0, i.e. ell_center = y_gal*sini/cosi.
        # Integration half-width delta = 5*h_z/cosi captures >99.99% of sech².
        delta = 5.0 * h_z / jnp.maximum(cosi, 0.1)
        ell_center = yp * sini / jnp.maximum(cosi, 0.1)  # (...,)

        # Gauss-Legendre on [ell_center - delta, ell_center + delta]
        ell = ell_center[..., None] + delta * self._gl_nodes  # (..., N_QUAD)
        w = delta * self._gl_weights  # (N_QUAD,)

        # at each quadrature point, compute disk coords
        # y_disk = y_gal * cosi + l * sini
        # z = l * cosi - y_gal * sini
        # x_disk = x_gal (unchanged)
        y_disk = yp[..., None] * cosi + ell * sini  # (..., N_QUAD)
        z_val = ell * cosi - yp[..., None] * sini  # (..., N_QUAD)
        x_disk = xp[..., None]  # (..., N_QUAD)

        R = jnp.sqrt(x_disk**2 + y_disk**2)  # (..., N_QUAD)

        # rho = rho0 * exp(-R/r_s) * sech²(z/h_z)
        radial = jnp.exp(-R / rscale)
        z_norm = z_val / h_z
        # sech²(x) = 1/cosh²(x); clip to avoid overflow
        cosh_z = jnp.cosh(jnp.clip(z_norm, -20.0, 20.0))
        vertical = 1.0 / (cosh_z**2)

        integrand = rho0 * radial * vertical  # (..., N_QUAD)

        # weighted sum over quadrature points
        intensity = jnp.sum(integrand * w, axis=-1)

        return intensity

    def _render_kspace(
        self,
        theta: jnp.ndarray,
        Nrow: int,
        Ncol: int,
        pixel_scale: float,
        pad_factor: int = None,
        oversample: int = 1,
        psf_kernel_fft: jnp.ndarray = None,
    ) -> jnp.ndarray:
        """
        K-space FFT rendering via shared core.

        Builds a closure computing the analytic FT of the 3D inclined
        exponential (radial: ``(1+k²)^{-3/2}``, vertical: ``u/sinh(u)``
        via ``_inclined_sech2_ft``), then delegates IFFT plumbing to
        ``_kspace_render_core``.

        Matches GalSim's SBInclinedExponential kValueHelper exactly.
        No point-sampling aliasing; the thin-disk limit (h_over_r -> 0)
        falls out naturally as ft_vertical -> 1.

        Axis convention
        ---------------
        ky = fftfreq(Nrow) is conjugate to rows (axis 0 = Y, vertical).
        kx = fftfreq(Ncol) is conjugate to cols (axis 1 = X, horizontal).
        gal2disk compresses Y (rows), so cosi acts on ky in k-space.

        Parameters
        ----------
        theta : jnp.ndarray
            Parameter array.
        Nrow, Ncol : int
            Output grid dimensions.
        pixel_scale : float
            Output pixel scale (arcsec/pixel).
        pad_factor : int, optional
            IFFT grid padding factor. Defaults to ``self._kspace_pad_factor``.
            1 = no padding; 2 = 2x grid (squashes boundary flux in log-space).
        oversample : int, optional
            Oversampling factor for cusp anti-aliasing. Pushes Nyquist to
            N × π/pixel_scale, reducing aliasing by ~N³. Default 1.
        psf_kernel_fft : jnp.ndarray, optional
            Pre-computed PSF kernel FFT on the same padded grid. When provided,
            ``I_hat * psf_kernel_fft`` is computed before the IFFT, fusing
            rendering and PSF convolution. Shape must match the padded grid.

        Returns
        -------
        jnp.ndarray
            Rendered image at specified resolution, shape (Nrow, Ncol).
        """
        if pad_factor is None:
            pad_factor = self._kspace_pad_factor

        x0 = self.get_param('int_x0', theta)
        y0 = self.get_param('int_y0', theta)
        g1 = self.get_param('g1', theta)
        g2 = self.get_param('g2', theta)
        pa = self.get_param('theta_int', theta)
        cosi = self.get_param('cosi', theta)
        flux = self.get_param('flux', theta)
        rscale = self.get_param('int_rscale', theta)
        h_over_r = self.get_param('int_h_over_r', theta)

        def ft_image(KX, KY):
            # exponential radial FT: (1 + k²)^{-3/2}
            return _inclined_sech2_ft(
                KX,
                KY,
                lambda k_sq: 1.0 / (1.0 + k_sq) ** 1.5,
                flux,
                x0,
                y0,
                g1,
                g2,
                pa,
                cosi,
                rscale,
                h_over_r,
                pixel_scale,
                Nrow,
                Ncol,
            )

        return _kspace_render_core(
            ft_image,
            Nrow,
            Ncol,
            pixel_scale,
            pad_factor,
            oversample,
            psf_kernel_fft,
        )

    def render_image(
        self,
        theta: jnp.ndarray,
        image_pars=None,
        plane: str = 'obs',
        X: jnp.ndarray = None,
        Y: jnp.ndarray = None,
        oversample: int = 1,
        *,
        obs=None,
        **kwargs,
    ) -> jnp.ndarray:
        """
        Render via k-space FFT, with optional PSF convolution.

        When obs has oversampling, renders at fine scale so convolve_fft
        can bin down to coarse scale.

        Calling conventions:
        - render_image(theta, obs=obs) -- with PSF from obs
        - render_image(theta, image_pars=image_pars) -- no PSF

        Parameters
        ----------
        obs : ImageObs, optional
            Observation object. When obs.kspace_psf_fft is set, uses fused
            k-space path. When obs.psf_data is set, uses fallback real-space.
        oversample : int, optional
            Cusp anti-aliasing factor for the non-PSF path. The exponential
            profile has a cusp at R=0 whose FT decays as k⁻³; power above
            Nyquist aliases into the image. Oversampling pushes Nyquist to
            N × π/pixel_scale, reducing aliasing by ~N³.
            Ignored when PSF is configured via obs (PSF convolution suppresses
            high-k aliasing naturally). Default 1.
        """
        if obs is not None:
            pixel_scale = obs.image_pars.pixel_scale
            Nrow = obs.image_pars.Nrow
            Ncol = obs.image_pars.Ncol

            if obs.kspace_psf_fft is not None:
                # fused k-space path: render + convolve in one FFT pass
                N = max(obs.oversample, 1)
                image = self._render_kspace(
                    theta,
                    Nrow * N,
                    Ncol * N,
                    pixel_scale / N,
                    psf_kernel_fft=obs.kspace_psf_fft,
                )
                if N > 1:
                    image = image.reshape(Nrow, N, Ncol, N).mean(axis=(1, 3))
                return image

            if obs.psf_data is not None:
                # fallback real-space path
                from kl_pipe.psf import convolve_fft

                if obs.oversample > 1:
                    N = obs.oversample
                    image = self._render_kspace(
                        theta, Nrow * N, Ncol * N, pixel_scale / N
                    )
                else:
                    image = self._render_kspace(theta, Nrow, Ncol, pixel_scale)
                return convolve_fft(image, obs.psf_data)

            # no PSF on obs
            return self._render_kspace(
                theta, Nrow, Ncol, pixel_scale, oversample=oversample
            )

        # legacy/convenience path (no obs)
        if image_pars is None and (X is None or Y is None):
            raise ValueError("Provide obs, image_pars, or (X, Y)")

        if image_pars is not None:
            pixel_scale = image_pars.pixel_scale
            Nrow = image_pars.Nrow
            Ncol = image_pars.Ncol
        else:
            Nrow, Ncol = X.shape
            pixel_scale = jnp.abs(X[0, 1] - X[0, 0])

        # warn if cusp aliasing likely exceeds 1% even with current oversample
        try:
            rscale_val = float(self.get_param('int_rscale', theta))
            ps_val = float(pixel_scale)
            k_ny_eff = oversample * np.pi / ps_val
            alias_frac = 1.0 / (1.0 + (k_ny_eff * rscale_val) ** 2) ** 1.5
            if alias_frac > 0.01:
                import warnings

                warnings.warn(
                    f"render_image: estimated cusp aliasing {alias_frac:.1%} of peak "
                    f"(r_s/ps={rscale_val / ps_val:.1f}, oversample={oversample}). "
                    f"Increase oversample or use a finer pixel scale for sub-1% "
                    f"accuracy without PSF convolution.",
                    stacklevel=2,
                )
        except (TypeError, ValueError):
            pass  # inside JIT trace, skip warning

        return self._render_kspace(
            theta, Nrow, Ncol, pixel_scale, oversample=oversample
        )


# ==============================================================================
# InclinedSpergelModel
# ==============================================================================


class InclinedSpergelModel(IntensityModel):
    """
    3D inclined Spergel model with sech² vertical profile.

    The Spergel profile (Spergel 2010) generalizes the exponential via an
    analytic FT ``(1+k²)^{-(1+nu)}`` in k-space. Native parameter is nu
    (Spergel index); nu=0.5 recovers the exponential exactly.

    Two evaluation paths:
    - ``render_image``: k-space FFT (exact analytic FT, fully differentiable).
      Use this for gradient-based inference.
    - ``__call__``: real-space Gauss-Legendre quadrature (uses scipy K_nu
      callback via ``jax.pure_callback``, NOT auto-differentiable).

    Face-on radial profile: ``I(r) = I_0 * (r/c)^nu * K_nu(r/c)``
    where ``I_0 = flux / (2*pi * c^2 * 2^nu * Gamma(nu+1))``.

    K-space radial FT: ``(1 + k_x'^2 + (k_y'*cos(i))^2)^{-(1+nu)}``
    where primed k-vectors are in the galaxy frame after shear + rotation.

    Parameters
    ----------
    cosi : float
        Cosine of inclination (0=edge-on, 1=face-on)
    theta_int : float
        Position angle (radians)
    g1, g2 : float
        Shear components
    flux : float
        Total integrated flux
    int_rscale : float
        Spergel scale length c (arcsec). For nu=0.5, equals exponential r_s.
    int_h_over_r : float
        Vertical scale height / radial scale length.
        h_z = int_h_over_r * int_rscale.
    nu : float
        Spergel index. nu=0.5 is exponential (Sersic n=1),
        nu=-0.6 approximates de Vaucouleurs (Sersic n=4).
        Must satisfy nu > -1 for the profile to be physical.
    int_x0, int_y0 : float
        Centroid position (arcsec)
    """

    PARAMETER_NAMES = (
        'cosi',
        'theta_int',
        'g1',
        'g2',
        'flux',
        'int_rscale',
        'int_h_over_r',
        'nu',
        'int_x0',
        'int_y0',
    )

    _kspace_pad_factor = 2

    def __init__(self, meta_pars=None, n_quad=None):
        super().__init__(meta_pars)
        n = n_quad if n_quad is not None else _DEFAULT_N_QUAD
        self._n_quad = n
        nodes, weights = np.polynomial.legendre.leggauss(n)
        self._gl_nodes = jnp.array(nodes)
        self._gl_weights = jnp.array(weights)

    @property
    def name(self) -> str:
        return 'inclined_spergel'

    def render_unconvolved(self, theta, image_pars, oversample=5):
        """Render intensity image WITHOUT PSF, using k-space FT.

        For use by SpectralModel.build_cube() — fast, anti-aliased, no PSF.
        """
        return self._render_kspace(
            theta,
            image_pars.Nrow,
            image_pars.Ncol,
            image_pars.pixel_scale,
            oversample=oversample,
        )

    def evaluate_in_disk_plane(
        self,
        theta: jnp.ndarray,
        x: jnp.ndarray,
        y: jnp.ndarray,
        z: jnp.ndarray = None,
    ) -> jnp.ndarray:
        """
        Evaluate face-on Spergel profile in disk plane.

        Computes I(r) = I_0 * (r/c)^nu * K_nu(r/c). Uses scipy callback
        for K_nu — not auto-differentiable. For velocity flux weighting
        in gradient-based inference, prefer pre-computed flux maps via
        render_image (k-space path).

        Parameters
        ----------
        theta : jnp.ndarray
            Model parameters.
        x, y : jnp.ndarray
            Coordinates in disk plane.
        z : jnp.ndarray, optional
            Currently unused.
        """
        flux = self.get_param('flux', theta)
        rscale = self.get_param('int_rscale', theta)
        nu = self.get_param('nu', theta)
        return _spergel_evaluate_faceon(flux, rscale, nu, x, y)

    def __call__(
        self,
        theta: jnp.ndarray,
        plane: str,
        x: jnp.ndarray,
        y: jnp.ndarray,
        z: jnp.ndarray = None,
    ) -> jnp.ndarray:
        """
        Evaluate 3D inclined Spergel via LOS Gauss-Legendre quadrature.

        Integrates rho(R, z) = rho0 * (R/c)^nu * K_nu(R/c) * sech²(z/h_z)
        along the line of sight at each pixel in the galaxy frame.

        NOT auto-differentiable (uses scipy K_nu via pure_callback);
        use render_image for gradient-based inference.
        """
        x0 = self.get_param('int_x0', theta)
        y0 = self.get_param('int_y0', theta)
        g1 = self.get_param('g1', theta)
        g2 = self.get_param('g2', theta)
        pa = self.get_param('theta_int', theta)
        cosi = self.get_param('cosi', theta)
        flux = self.get_param('flux', theta)
        rscale = self.get_param('int_rscale', theta)
        h_over_r = self.get_param('int_h_over_r', theta)
        nu = self.get_param('nu', theta)

        if plane == 'disk':
            return _spergel_evaluate_faceon(flux, rscale, nu, x, y)

        sini = jnp.sqrt(jnp.maximum(1.0 - cosi**2, 0.0))

        # transform to galaxy frame (NOT disk — we integrate LOS ourselves)
        xp, yp = x, y
        if plane == 'obs':
            xp, yp = obs2cen(x0, y0, xp, yp)
        if plane in ('obs', 'cen'):
            xp, yp = cen2source(g1, g2, xp, yp)
        if plane in ('obs', 'cen', 'source'):
            xp, yp = source2gal(pa, xp, yp)

        return _spergel_los_integrate(
            flux,
            rscale,
            nu,
            h_over_r,
            cosi,
            sini,
            xp,
            yp,
            self._gl_nodes,
            self._gl_weights,
        )

    def _render_kspace(
        self,
        theta: jnp.ndarray,
        Nrow: int,
        Ncol: int,
        pixel_scale: float,
        pad_factor: int = None,
        oversample: int = 1,
        psf_kernel_fft: jnp.ndarray = None,
    ) -> jnp.ndarray:
        """
        K-space FFT rendering: radial FT = ``(1+k²)^{-(1+nu)}``.

        Builds a closure via ``_inclined_sech2_ft`` with the Spergel
        radial FT, then delegates IFFT plumbing to ``_kspace_render_core``.

        Parameters
        ----------
        theta : jnp.ndarray
            Parameter array (10 elements).
        Nrow, Ncol : int
            Output grid dimensions.
        pixel_scale : float
            Output pixel scale (arcsec/pixel).
        pad_factor : int, optional
            IFFT padding factor. Defaults to ``self._kspace_pad_factor``.
        oversample : int, optional
            Oversampling factor. Default 1.
        psf_kernel_fft : jnp.ndarray, optional
            Pre-computed PSF FFT for fused convolution.

        Returns
        -------
        jnp.ndarray, shape (Nrow, Ncol)
        """
        if pad_factor is None:
            pad_factor = self._kspace_pad_factor

        x0 = self.get_param('int_x0', theta)
        y0 = self.get_param('int_y0', theta)
        g1 = self.get_param('g1', theta)
        g2 = self.get_param('g2', theta)
        pa = self.get_param('theta_int', theta)
        cosi = self.get_param('cosi', theta)
        flux = self.get_param('flux', theta)
        rscale = self.get_param('int_rscale', theta)
        h_over_r = self.get_param('int_h_over_r', theta)
        nu = self.get_param('nu', theta)

        def ft_image(KX, KY):
            # spergel radial FT: (1 + k²)^{-(1+nu)}, captures nu from closure
            return _inclined_sech2_ft(
                KX,
                KY,
                lambda k_sq: 1.0 / (1.0 + k_sq) ** (1.0 + nu),
                flux,
                x0,
                y0,
                g1,
                g2,
                pa,
                cosi,
                rscale,
                h_over_r,
                pixel_scale,
                Nrow,
                Ncol,
            )

        return _kspace_render_core(
            ft_image,
            Nrow,
            Ncol,
            pixel_scale,
            pad_factor,
            oversample,
            psf_kernel_fft,
        )

    def render_image(
        self,
        theta: jnp.ndarray,
        image_pars=None,
        plane: str = 'obs',
        X: jnp.ndarray = None,
        Y: jnp.ndarray = None,
        oversample: int = 1,
        *,
        obs=None,
        **kwargs,
    ) -> jnp.ndarray:
        """
        Render via k-space FFT, with optional PSF convolution.

        When obs has oversampling, renders at fine scale so convolve_fft
        can bin down to coarse scale.

        Calling conventions:
        - render_image(theta, obs=obs) -- with PSF from obs
        - render_image(theta, image_pars=image_pars) -- no PSF

        Parameters
        ----------
        obs : ImageObs, optional
            Observation object. When obs.kspace_psf_fft is set, uses fused
            k-space path. When obs.psf_data is set, uses fallback real-space.
        oversample : int, optional
            Cusp anti-aliasing factor for the non-PSF path. Default 1.
        """
        return _kspace_render_image(
            self, theta, image_pars, plane, X, Y, oversample, obs=obs, **kwargs
        )


# ==============================================================================
# InclinedDeVaucouleursModel
# ==============================================================================


class InclinedDeVaucouleursModel(IntensityModel):
    """
    3D inclined de Vaucouleurs model via Spergel profile with fixed nu.

    Thin wrapper around the Spergel profile functions with ``nu`` fixed
    at ``_DEVAUCOULEURS_NU`` (best-fit for Sersic n=4). Same
    PARAMETER_NAMES as InclinedExponentialModel (no ``nu`` parameter).

    The Spergel approximation to de Vaucouleurs is not exact — see the
    GalSim regression tests for quantification of the approximation error.

    Parameters
    ----------
    cosi : float
        Cosine of inclination (0=edge-on, 1=face-on)
    theta_int : float
        Position angle (radians)
    g1, g2 : float
        Shear components
    flux : float
        Total integrated flux
    int_rscale : float
        Spergel scale length c (arcsec).
    int_h_over_r : float
        Vertical scale height / radial scale length.
    int_x0, int_y0 : float
        Centroid position (arcsec)
    """

    PARAMETER_NAMES = (
        'cosi',
        'theta_int',
        'g1',
        'g2',
        'flux',
        'int_rscale',
        'int_h_over_r',
        'int_x0',
        'int_y0',
    )

    _kspace_pad_factor = 2
    _fixed_nu = _DEVAUCOULEURS_NU

    def __init__(self, meta_pars=None, n_quad=None):
        super().__init__(meta_pars)
        n = n_quad if n_quad is not None else _DEFAULT_N_QUAD
        self._n_quad = n
        nodes, weights = np.polynomial.legendre.leggauss(n)
        self._gl_nodes = jnp.array(nodes)
        self._gl_weights = jnp.array(weights)

    @property
    def name(self) -> str:
        return 'de_vaucouleurs'

    def render_unconvolved(self, theta, image_pars, oversample=5):
        """Render intensity image WITHOUT PSF, using k-space FT."""
        return self._render_kspace(
            theta,
            image_pars.Nrow,
            image_pars.Ncol,
            image_pars.pixel_scale,
            oversample=oversample,
        )

    def evaluate_in_disk_plane(
        self,
        theta: jnp.ndarray,
        x: jnp.ndarray,
        y: jnp.ndarray,
        z: jnp.ndarray = None,
    ) -> jnp.ndarray:
        """
        Evaluate face-on de Vaucouleurs profile in disk plane.

        Delegates to Spergel profile with fixed ``nu = _DEVAUCOULEURS_NU``.
        Uses scipy callback for K_nu — not auto-differentiable.

        Parameters
        ----------
        theta : jnp.ndarray
            Model parameters.
        x, y : jnp.ndarray
            Coordinates in disk plane.
        z : jnp.ndarray, optional
            Currently unused.
        """
        flux = self.get_param('flux', theta)
        rscale = self.get_param('int_rscale', theta)
        return _spergel_evaluate_faceon(flux, rscale, self._fixed_nu, x, y)

    def __call__(
        self,
        theta: jnp.ndarray,
        plane: str,
        x: jnp.ndarray,
        y: jnp.ndarray,
        z: jnp.ndarray = None,
    ) -> jnp.ndarray:
        """
        Evaluate 3D inclined de Vaucouleurs via LOS Gauss-Legendre quadrature.

        Delegates to Spergel LOS integration with fixed ``nu``.
        NOT auto-differentiable; use render_image for inference.
        """
        x0 = self.get_param('int_x0', theta)
        y0 = self.get_param('int_y0', theta)
        g1 = self.get_param('g1', theta)
        g2 = self.get_param('g2', theta)
        pa = self.get_param('theta_int', theta)
        cosi = self.get_param('cosi', theta)
        flux = self.get_param('flux', theta)
        rscale = self.get_param('int_rscale', theta)
        h_over_r = self.get_param('int_h_over_r', theta)

        if plane == 'disk':
            return _spergel_evaluate_faceon(flux, rscale, self._fixed_nu, x, y)

        sini = jnp.sqrt(jnp.maximum(1.0 - cosi**2, 0.0))

        # transform to galaxy frame (NOT disk — we integrate LOS ourselves)
        xp, yp = x, y
        if plane == 'obs':
            xp, yp = obs2cen(x0, y0, xp, yp)
        if plane in ('obs', 'cen'):
            xp, yp = cen2source(g1, g2, xp, yp)
        if plane in ('obs', 'cen', 'source'):
            xp, yp = source2gal(pa, xp, yp)

        return _spergel_los_integrate(
            flux,
            rscale,
            self._fixed_nu,
            h_over_r,
            cosi,
            sini,
            xp,
            yp,
            self._gl_nodes,
            self._gl_weights,
        )

    def _render_kspace(
        self,
        theta: jnp.ndarray,
        Nrow: int,
        Ncol: int,
        pixel_scale: float,
        pad_factor: int = None,
        oversample: int = 1,
        psf_kernel_fft: jnp.ndarray = None,
    ) -> jnp.ndarray:
        """
        K-space FFT rendering with fixed nu for de Vaucouleurs.

        Parameters
        ----------
        theta : jnp.ndarray
            Parameter array (9 elements, no nu).
        Nrow, Ncol : int
            Output grid dimensions.
        pixel_scale : float
            Output pixel scale (arcsec/pixel).
        pad_factor : int, optional
            IFFT padding factor. Defaults to ``self._kspace_pad_factor``.
        oversample : int, optional
            Oversampling factor. Default 1.
        psf_kernel_fft : jnp.ndarray, optional
            Pre-computed PSF FFT for fused convolution.

        Returns
        -------
        jnp.ndarray, shape (Nrow, Ncol)
        """
        if pad_factor is None:
            pad_factor = self._kspace_pad_factor

        x0 = self.get_param('int_x0', theta)
        y0 = self.get_param('int_y0', theta)
        g1 = self.get_param('g1', theta)
        g2 = self.get_param('g2', theta)
        pa = self.get_param('theta_int', theta)
        cosi = self.get_param('cosi', theta)
        flux = self.get_param('flux', theta)
        rscale = self.get_param('int_rscale', theta)
        h_over_r = self.get_param('int_h_over_r', theta)
        nu = self._fixed_nu

        def ft_image(KX, KY):
            # de vaucouleurs radial FT: (1 + k²)^{-(1+nu)} with fixed nu=-0.6
            return _inclined_sech2_ft(
                KX,
                KY,
                lambda k_sq: 1.0 / (1.0 + k_sq) ** (1.0 + nu),
                flux,
                x0,
                y0,
                g1,
                g2,
                pa,
                cosi,
                rscale,
                h_over_r,
                pixel_scale,
                Nrow,
                Ncol,
            )

        return _kspace_render_core(
            ft_image,
            Nrow,
            Ncol,
            pixel_scale,
            pad_factor,
            oversample,
            psf_kernel_fft,
        )

    def render_image(
        self,
        theta: jnp.ndarray,
        image_pars=None,
        plane: str = 'obs',
        X: jnp.ndarray = None,
        Y: jnp.ndarray = None,
        oversample: int = 1,
        *,
        obs=None,
        **kwargs,
    ) -> jnp.ndarray:
        """
        Render via k-space FFT, with optional PSF convolution.

        When obs has oversampling, renders at fine scale so convolve_fft
        can bin down to coarse scale.

        Calling conventions:
        - render_image(theta, obs=obs) -- with PSF from obs
        - render_image(theta, image_pars=image_pars) -- no PSF

        Parameters
        ----------
        obs : ImageObs, optional
            Observation object. When obs.kspace_psf_fft is set, uses fused
            k-space path. When obs.psf_data is set, uses fallback real-space.
        oversample : int, optional
            Cusp anti-aliasing factor for the non-PSF path. Default 1.
        """
        return _kspace_render_image(
            self, theta, image_pars, plane, X, Y, oversample, obs=obs, **kwargs
        )


# ==============================================================================
# CompositeIntensityModel (stub)
# ==============================================================================


class CompositeIntensityModel(IntensityModel):
    """Composite intensity model summing two inclined components (e.g. disk + bulge).

    Each component has its own radial profile, scale, and shape parameter.
    Shared geometric params (cosi, theta_int, g1, g2, centroid) apply to both.
    Component-specific params use prefixes ('disk_', 'bulge_') to avoid collision.

    K-space composition: each component computes its own ``ft_image_fn``
    closure via ``_inclined_sech2_ft`` (with its own rscale, h_over_r, and
    radial_ft_fn). The composite sums these in k-space and passes the total
    to ``_kspace_render_core`` for a single IFFT pass. This is exact
    (linearity of FT) and efficient (one FFT, not two).

    Real-space composition: sum component __call__ results at each pixel.

    NOT YET IMPLEMENTED — stub establishing interface and design for future work.

    Design notes for implementation:
    - ``_render_kspace``: build ``ft_a(KX,KY) + ft_b(KX,KY)``, pass sum to
      ``_kspace_render_core`` (one IFFT)
    - ``__call__``: sum ``component_a(theta_a, ...) + component_b(theta_b, ...)``
    - Parameter slicing: same pattern as ``KLModel.get_intensity_pars``
    - Each component can be any IntensityModel (Exponential, Spergel, etc.)
    """

    # placeholder; dynamically built in __init__ from component PARAMETER_NAMES
    PARAMETER_NAMES = ()

    def __init__(self, component_a, component_b, shared_pars=None, meta_pars=None):
        raise NotImplementedError(
            "CompositeIntensityModel is a stub for future implementation"
        )

    @property
    def name(self) -> str:
        return 'composite'

    def evaluate_in_disk_plane(self, theta, x, y, z=None):
        raise NotImplementedError

    def __call__(self, theta, plane, x, y, z=None):
        raise NotImplementedError


# ==============================================================================
# Factory
# ==============================================================================


INTENSITY_MODEL_TYPES = {
    'default': InclinedExponentialModel,
    'inclined_exp': InclinedExponentialModel,
    'inclined_spergel': InclinedSpergelModel,
    'spergel': InclinedSpergelModel,
    'de_vaucouleurs': InclinedDeVaucouleursModel,
}


def get_intensity_model_types():
    """
    Get dictionary of registered intensity model types.

    Returns
    -------
    dict
        Mapping from model name strings to intensity model classes.
    """
    return INTENSITY_MODEL_TYPES


def build_intensity_model(
    name: str,
    meta_pars: dict = None,
) -> IntensityModel:
    """
    Factory function for constructing intensity models by name.

    Parameters
    ----------
    name : str
        Name of the model to construct (case-insensitive).
    meta_pars : dict, optional
        Fixed metadata for the model.

    Returns
    -------
    IntensityModel
        Instantiated intensity model.

    Raises
    ------
    ValueError
        If the specified model name is not registered.
    """

    name = name.lower()

    if name not in INTENSITY_MODEL_TYPES:
        raise ValueError(f'{name} is not a registered intensity model!')

    return INTENSITY_MODEL_TYPES[name](meta_pars=meta_pars)
