import numpy as np
import jax.numpy as jnp
import jax
from scipy.fft import next_fast_len
from scipy.special import kve as scipy_kve

from kl_pipe.model import IntensityModel
from kl_pipe.transformation import obs2cen, cen2source, source2gal


# default number of Gauss-Legendre quadrature points for LOS integration
_DEFAULT_N_QUAD = 200

# community convention Spergel nu for Sersic n=4 (de Vaucouleurs);
# flux-weighted L2 matching gives -0.48 — see scripts/compute_nu_n_mapping.py
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
    nu_lo, nu_hi = float(nu_valid[-1]), float(nu_valid[0])
    nu_arr = np.asarray(nu)
    if np.any(nu_arr < nu_lo) or np.any(nu_arr > nu_hi):
        raise ValueError(
            f"nu={nu} outside valid range [{nu_lo:.2f}, {nu_hi:.2f}] "
            f"for {'inclined' if inclined else 'face-on'} table"
        )
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


def _inclined_sech2_los_integrate(
    rho0,
    radial_fn,
    h_z,
    cosi,
    sini,
    xp,
    yp,
    gl_nodes,
    gl_weights,
):
    """General LOS Gauss-Legendre integration: radial_fn(R) × sech²(z/h_z).

    Integrates rho(R, z) = rho0 * radial_fn(R) * sech²(z/h_z) along
    the line of sight at each pixel in the galaxy frame. The radial
    profile and normalization are model-specific; the GL plumbing is
    shared across all inclined models with sech² vertical structure.

    Parameters
    ----------
    rho0 : float
        Volume density normalization (model-specific).
    radial_fn : callable
        R -> radial profile array. Model-specific radial density.
        Exponential: ``lambda R: exp(-R/r_s)``.
        Spergel: ``lambda R: _spergel_radial(R, nu, c)``.
    h_z : float
        Vertical scale height (arcsec).
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

    radial = radial_fn(R)
    z_norm = z_val / h_z
    cosh_z = jnp.cosh(jnp.clip(z_norm, -20.0, 20.0))
    vertical = 1.0 / (cosh_z**2)

    integrand = rho0 * radial * vertical
    return jnp.sum(integrand * w, axis=-1)


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
    """3D LOS integration with Spergel radial + sech² vertical.

    Thin wrapper around ``_inclined_sech2_los_integrate`` with Spergel-specific
    normalization and radial profile.
    """
    h_z = h_over_r * rscale
    rho0 = flux / (2.0 * h_z * _spergel_norm_2d(rscale, nu))
    return _inclined_sech2_los_integrate(
        rho0,
        lambda R: _spergel_radial(R, nu, rscale),
        h_z,
        cosi,
        sini,
        xp,
        yp,
        gl_nodes,
        gl_weights,
    )


# ==============================================================================
# Sersic profile helpers
# ==============================================================================


def _sersic_bn(n):
    """Sersic b_n via Ciotti & Bertin (1999) asymptotic expansion.

    Solves the half-light condition: gamma(2n, b_n) = Gamma(2n)/2.
    Accurate to < 1e-4 for n > 0.36. Pure arithmetic — JIT-safe.

    Parameters
    ----------
    n : float or jnp.ndarray
        Sersic index.

    Returns
    -------
    float or jnp.ndarray
        b_n coefficient.
    """
    return (
        2.0 * n
        - 1.0 / 3.0
        + 4.0 / (405.0 * n)
        + 46.0 / (25515.0 * n**2)
        + 131.0 / (1148175.0 * n**3)
        - 2194697.0 / (30690717750.0 * n**4)
    )


# Miller & Pasha (2025, arxiv:2508.20266) emulator constants
_MP_A0 = 2.245374
_MP_A1 = 0.029371526
_MP_A2 = 2.1431181
_MP_A3 = -3.7275262
_MP_A4 = 0.091609545
_MP_A5 = 0.32785136


def _sersic_ft_emulator(k, n):
    """Sersic radial FT via Miller & Pasha (2025) symbolic-regression emulator.

    Returns the normalized Hankel transform of the Sersic profile at
    dimensionless wavenumber ``k = k_physical * R_e`` (unit flux, unit R_e).
    Valid for 0.5 <= n <= 6.0.

    F_r(k, n) = 1 / (1 + exp(G(k, n)))

    where G is built from elementary functions (exp, log, sqrt). Fully
    differentiable w.r.t. both k and n via JAX autodiff.

    Parameters
    ----------
    k : jnp.ndarray
        Dimensionless wavenumber (k_physical * R_e). Must be >= 0.
    n : float or jnp.ndarray
        Sersic index. Valid range: [0.5, 6.0].

    Returns
    -------
    jnp.ndarray
        Normalized FT values in [0, 1]. F_r(0) = 1 (unit total flux).
    """
    # guard k=0 (log(0) = -inf) — DC component is exactly 1
    k_safe = jnp.maximum(k, 1e-30)

    # sub-expressions
    H = _MP_A0 * jnp.sqrt(
        n + _MP_A1 * (k_safe - _MP_A2) * jnp.exp(jnp.exp(jnp.sqrt(n) - n**3))
    )
    J = jnp.exp(_MP_A3 * k_safe - jnp.exp(n - n**2))
    G = (1.0 / n) * ((H + J) * (jnp.log(k_safe) - _MP_A4) - _MP_A5)

    F_r = 1.0 / (1.0 + jnp.exp(G))

    # exact DC: F_r(0) = 1
    return jnp.where(k < 1e-30, 1.0, F_r)


def _sersic_norm_2d(Re, n):
    """2D flux normalization factor for the Sersic profile.

    Returns the factor N such that I_0 = flux / N gives a profile
    ``I(r) = I_0 * exp(-b_n * (r/R_e)^{1/n})`` with total flux = flux.

    N = 2*pi * n * R_e^2 * Gamma(2n) / b_n^{2n}

    Parameters
    ----------
    Re : float
        Half-light radius (arcsec).
    n : float
        Sersic index.

    Returns
    -------
    float
        Normalization factor (arcsec^2).
    """
    bn = _sersic_bn(n)
    log_norm = (
        jnp.log(2.0 * jnp.pi)
        + jnp.log(n)
        + 2.0 * jnp.log(Re)
        + jax.scipy.special.gammaln(2.0 * n)
        - 2.0 * n * jnp.log(bn)
    )
    return jnp.exp(log_norm)


def _sersic_evaluate_faceon(flux, Re, n, x, y):
    """Face-on Sersic surface brightness in the disk plane.

    I(r) = I_0 * exp(-b_n * (r / R_e)^{1/n})

    where I_0 = flux / (2*pi * n * R_e^2 * Gamma(2n) / b_n^{2n}).

    Parameters
    ----------
    flux : float
        Total integrated flux.
    Re : float
        Half-light radius (arcsec).
    n : float
        Sersic index.
    x, y : jnp.ndarray
        Coordinates in disk plane (arcsec).

    Returns
    -------
    jnp.ndarray
        Surface brightness array.
    """
    r = jnp.sqrt(x**2 + y**2)
    bn = _sersic_bn(n)
    I0 = flux / _sersic_norm_2d(Re, n)
    return I0 * jnp.exp(-bn * (r / Re) ** (1.0 / n))


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

        return _inclined_sech2_los_integrate(
            rho0,
            lambda R: jnp.exp(-R / rscale),
            h_z,
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

    .. warning::

       For nu < 0 (concentrated profiles, Sersic n > 1), the Spergel
       radial density diverges as R^{2nu} at R=0. This power-law cusp
       produces fundamentally different inclined morphology from Sersic
       profiles, which have finite central density. At high inclination,
       the cusp causes unphysical broadening along the minor axis — light
       spreads over ~5 pixels where the Sersic equivalent concentrates
       into ~1 pixel. This is a mathematical limitation of the Spergel
       functional form, not a rendering artifact.

       **nu=0.5 (exponential / Sersic n=1) is exact and works perfectly
       at all inclinations.** For concentrated bulge profiles (n >= 2),
       use with caution and only face-on, or consider alternative
       approaches (numerical Sersic FT, composite models).

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
        if not isinstance(theta, jax.core.Tracer) and float(nu) <= -1.0:
            raise ValueError(
                f"nu={float(nu)} <= -1 is unphysical (Spergel profile has "
                f"infinite spatial extent). Valid range: nu > -1."
            )

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
        # validate nu when theta is concrete (not inside JIT trace)
        if not isinstance(theta, jax.core.Tracer):
            nu = float(theta[self.PARAMETER_NAMES.index('nu')])
            if nu <= -1.0:
                raise ValueError(
                    f"nu={nu} <= -1 is unphysical (Spergel profile has "
                    f"infinite spatial extent). Valid range: nu > -1."
                )
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
    at ``_DEVAUCOULEURS_NU`` (community convention for Sersic n=4). Same
    PARAMETER_NAMES as InclinedExponentialModel (no ``nu`` parameter).

    .. warning::

       The Spergel approximation to de Vaucouleurs breaks down at high
       inclination due to the divergent cusp (see InclinedSpergelModel
       warning). Use face-on only, or consider alternative approaches.

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
# InclinedSersicModel
# ==============================================================================


class InclinedSersicModel(IntensityModel):
    """
    3D inclined Sersic model with sech² vertical profile.

    Uses the Miller & Pasha (2025) symbolic-regression emulator for the
    radial Fourier transform, giving a fully differentiable k-space
    rendering path. The Sersic profile ``exp(-b_n (r/R_e)^{1/n})`` is
    finite at R=0 (no cusp), so this model works correctly at all
    inclinations — unlike the Spergel approximation which diverges for
    n >= 2.

    .. note::

       For n=1 (exponential), ``InclinedExponentialModel`` uses an
       exact analytic FT and is preferable. This model uses an emulator
       approximation at all n including n=1.

    Two evaluation paths:
    - ``render_image``: k-space FFT with emulator radial FT. Fully
      differentiable — use for gradient-based inference.
    - ``__call__``: real-space Gauss-Legendre LOS quadrature with
      exact ``exp(-b_n (r/R_e)^{1/n})`` radial profile.

    Parameters
    ----------
    cosi : float
        Cosine of inclination (0=edge-on, 1=face-on).
    theta_int : float
        Position angle (radians).
    g1, g2 : float
        Shear components.
    flux : float
        Total integrated flux.
    int_hlr : float
        Half-light radius R_e (arcsec).
    int_h_over_hlr : float
        Vertical scale height / half-light radius. h_z = int_h_over_hlr * int_hlr.
    n_sersic : float
        Sersic index. Valid range: [0.5, 6.0].
    int_x0, int_y0 : float
        Centroid position (arcsec).
    """

    PARAMETER_NAMES = (
        'cosi',
        'theta_int',
        'g1',
        'g2',
        'flux',
        'int_hlr',
        'int_h_over_hlr',
        'n_sersic',
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
        return 'inclined_sersic'

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
        Evaluate face-on Sersic profile in disk plane.

        Computes I(r) = I_0 * exp(-b_n * (r/R_e)^{1/n}).

        Parameters
        ----------
        theta : jnp.ndarray
            Model parameters.
        x, y : jnp.ndarray
            Coordinates in disk plane (arcsec).
        z : jnp.ndarray, optional
            Currently unused.
        """
        flux = self.get_param('flux', theta)
        hlr = self.get_param('int_hlr', theta)
        n = self.get_param('n_sersic', theta)
        return _sersic_evaluate_faceon(flux, hlr, n, x, y)

    def __call__(
        self,
        theta: jnp.ndarray,
        plane: str,
        x: jnp.ndarray,
        y: jnp.ndarray,
        z: jnp.ndarray = None,
    ) -> jnp.ndarray:
        """
        Evaluate 3D inclined Sersic via LOS Gauss-Legendre quadrature.

        Integrates rho(R, z) = rho0 * exp(-b_n*(R/R_e)^{1/n}) * sech²(z/h_z)
        along the line of sight at each pixel in the galaxy frame.
        """
        x0 = self.get_param('int_x0', theta)
        y0 = self.get_param('int_y0', theta)
        g1 = self.get_param('g1', theta)
        g2 = self.get_param('g2', theta)
        pa = self.get_param('theta_int', theta)
        cosi = self.get_param('cosi', theta)
        flux = self.get_param('flux', theta)
        hlr = self.get_param('int_hlr', theta)
        h_over_hlr = self.get_param('int_h_over_hlr', theta)
        n = self.get_param('n_sersic', theta)

        if not isinstance(theta, jax.core.Tracer):
            n_val = float(n)
            if n_val < 0.5 or n_val > 6.0:
                raise ValueError(
                    f"n_sersic={n_val} outside valid range [0.5, 6.0] "
                    f"for the Sersic FT emulator."
                )

        if plane == 'disk':
            return _sersic_evaluate_faceon(flux, hlr, n, x, y)

        sini = jnp.sqrt(jnp.maximum(1.0 - cosi**2, 0.0))
        h_z = h_over_hlr * hlr

        # transform to galaxy frame (NOT disk — we integrate LOS ourselves)
        xp, yp = x, y
        if plane == 'obs':
            xp, yp = obs2cen(x0, y0, xp, yp)
        if plane in ('obs', 'cen'):
            xp, yp = cen2source(g1, g2, xp, yp)
        if plane in ('obs', 'cen', 'source'):
            xp, yp = source2gal(pa, xp, yp)

        # volume density normalization: rho0 = flux / (2 * h_z * norm_2d)
        rho0 = flux / (2.0 * h_z * _sersic_norm_2d(hlr, n))
        bn = _sersic_bn(n)

        return _inclined_sech2_los_integrate(
            rho0,
            lambda R: jnp.exp(-bn * (R / hlr) ** (1.0 / n)),
            h_z,
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
        K-space FFT rendering with Sersic radial FT via emulator.

        Builds a closure via ``_inclined_sech2_ft`` with the Miller & Pasha
        emulator for the radial FT, then delegates IFFT plumbing to
        ``_kspace_render_core``.

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
        hlr = self.get_param('int_hlr', theta)
        h_over_hlr = self.get_param('int_h_over_hlr', theta)
        n = self.get_param('n_sersic', theta)

        def ft_image(KX, KY):
            # sersic radial FT via emulator; k_sq is dimensionless (k*R_e)²
            # safe-where: sqrt(0) has gradient inf — substitute dummy=1
            # in the DC branch so autodiff never sees sqrt(0)
            def _safe_sersic_ft(k_sq):
                is_dc = k_sq < 1e-20
                k = jnp.sqrt(jnp.where(is_dc, jnp.ones_like(k_sq), k_sq))
                return jnp.where(is_dc, 1.0, _sersic_ft_emulator(k, n))

            return _inclined_sech2_ft(
                KX,
                KY,
                _safe_sersic_ft,
                flux,
                x0,
                y0,
                g1,
                g2,
                pa,
                cosi,
                hlr,
                h_over_hlr,
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

        Calling conventions:
        - render_image(theta, obs=obs) -- with PSF from obs
        - render_image(theta, image_pars=image_pars) -- no PSF
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
    'inclined_sersic': InclinedSersicModel,
    'sersic': InclinedSersicModel,
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
