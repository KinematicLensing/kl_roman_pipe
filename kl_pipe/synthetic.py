"""
Synthetic data generation for testing and validation.

Provides simple, independent implementations of galaxy models that do NOT use
the main kl_pipe modeling code, to enable proper regression testing and
parameter recovery validation.

The generators implement transformations (inclination, position angle, shear)
directly without relying on kl_pipe.transformation, ensuring independence for
testing purposes.

Examples
--------
Generate synthetic velocity data:

>>> from kl_pipe.synthetic import SyntheticVelocity
>>> import numpy as np
>>>
>>> # Define true parameters
>>> true_params = {
...     'v0': 10.0, 'vcirc': 200.0, 'vel_rscale': 5.0,
...     'cosi': 0.8, 'theta_int': 0.785,
...     'g1': 0.0, 'g2': 0.0, 'vel_x0': 0.0, 'vel_y0': 0.0
... }
>>>
>>> # Create synthetic observation
>>> synth_vel = SyntheticVelocity(true_params, model_type='arctan', seed=42)
>>>
>>> # Generate data on a grid
>>> image_pars = ImagePars(shape=(64, 64), pixel_scale=0.3125, indexing='ij')
>>> data_noisy = synth_vel.generate(image_pars, snr=50)
>>>
>>> # Access results
>>> print(synth_vel.data_true)  # Noiseless data
>>> print(synth_vel.variance)   # Noise variance used

Generate synthetic intensity data:

>>> from kl_pipe.synthetic import SyntheticIntensity
>>>
>>> true_params = {
...     'flux': 1.0, 'int_rscale': 3.0, 'n_sersic': 1.0,
...     'cosi': 0.8, 'theta_int': 0.785,
...     'g1': 0.0, 'g2': 0.0, 'int_x0': 0.0, 'int_y0': 0.0
... }
>>>
>>> synth_int = SyntheticIntensity(true_params, model_type='sersic', seed=42)
>>> data_noisy = synth_int.generate(X, Y, snr=100)
"""

import numpy as np
import jax.numpy as jnp
import galsim as gs
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Optional
from scipy.special import gamma

from kl_pipe.parameters import ImagePars
from kl_pipe.utils import build_map_grid_from_image_pars

# Required parameters for each model type

REQUIRED_PARAMS = {
    'arctan': {
        'v0',
        'vcirc',
        'vel_rscale',
        'cosi',
        'theta_int',
        'g1',
        'g2',
        # TODO: we could add support for optional parameters later
        # 'vel_x0',
        # 'vel_y0',
    },
    'sersic': {
        'flux',
        'int_rscale',
        'n_sersic',
        'cosi',
        'theta_int',
        'g1',
        'g2',
        # TODO: we could add support for optional parameters later
        # 'int_x0',
        # 'int_y0',
    },
    'exponential': {
        'flux',
        'int_rscale',
        'int_h_over_r',
        'cosi',
        'theta_int',
        'g1',
        'g2',
        # TODO: we could add support for optional parameters later
        # 'int_x0',
        # 'int_y0',
    },
    'spergel': {
        'flux',
        'int_rscale',
        'nu',
        'int_h_over_r',
        'cosi',
        'theta_int',
        'g1',
        'g2',
    },
}


# ==============================================================================
# Velocity field generators
# ==============================================================================


def generate_arctan_velocity_2d(
    image_pars: ImagePars,
    v0: float,
    vcirc: float,
    vel_rscale: float,
    cosi: float,
    theta_int: float,
    g1: float = 0.0,
    g2: float = 0.0,
    vel_x0: float = 0.0,
    vel_y0: float = 0.0,
    psf=None,
    intensity_for_psf=None,
) -> np.ndarray:
    """
    Generate arctan rotation curve velocity field.

    Parameters
    ----------
    image_pars : ImagePars
        Image parameters defining the coordinate grids.
    v0 : float
        Systemic velocity in km/s.
    vcirc : float
        Asymptotic circular velocity in km/s.
    vel_rscale : float
        Scale radius for rotation curve, same units as X, Y.
    cosi: float
        Cosine of inclination angle (0=face-on, 1=edge-on).
    theta_int : float
        Intrinsic position angle in radians.
    g1, g2 : float, optional
        Shear components. Default is 0.0 (no shear).
    vel_x0, vel_y0 : float, optional
        Centroid offsets, same units as X, Y. Default is 0.0.

    Returns
    -------
    ndarray
        Line-of-sight velocity map in km/s, same shape as X and Y.
    """

    # Build coordinate grids from image parameters
    X, Y = build_map_grid_from_image_pars(image_pars, unit='arcsec', centered=True)

    sini = np.sqrt(1.0 - cosi**2)

    # Step 1: recenter
    X_c = X - vel_x0
    Y_c = Y - vel_y0

    # Step 2: area-preserving shear (image→source), matches GalSim .shear()
    norm = 1.0 / np.sqrt(1.0 - (g1**2 + g2**2))
    X_shear = norm * ((1.0 - g1) * X_c - g2 * Y_c)
    Y_shear = norm * (-g2 * X_c + (1.0 + g1) * Y_c)

    # Step 3: rotate
    cos_pa = np.cos(-theta_int)
    sin_pa = np.sin(-theta_int)
    X_rot = cos_pa * X_shear - sin_pa * Y_shear
    Y_rot = sin_pa * X_shear + cos_pa * Y_shear

    # Step 4: deproject (gal2disk)
    X_disk = X_rot
    Y_disk = Y_rot / cosi if cosi > 0 else Y_rot

    # Compute radius in disk plane
    r_disk = np.sqrt(X_disk**2 + Y_disk**2)

    # Evaluate arctan rotation curve
    v_circ = (2.0 / np.pi) * vcirc * np.arctan(r_disk / vel_rscale)

    # Project to line-of-sight
    phi = np.arctan2(Y_disk, X_disk)
    v_los = sini * np.cos(phi) * v_circ

    v_map = v0 + v_los

    if psf is not None:
        if intensity_for_psf is None:
            raise ValueError("intensity_for_psf required for velocity PSF")
        from kl_pipe.psf import gsobj_to_kernel, convolve_flux_weighted_numpy

        kernel, padded_shape = gsobj_to_kernel(psf, image_pars=image_pars)
        v_map = convolve_flux_weighted_numpy(
            v_map, intensity_for_psf, kernel, padded_shape
        )

    return v_map


# TODO: when we're ready to test more complex velocity models
def generate_arctan_velocity_3d():
    pass


# ==============================================================================
# Intensity profile generators
# ==============================================================================


def _generate_inclined_kspace_scipy(
    radial_ft_fn,
    image_pars,
    flux,
    int_rscale,
    cosi,
    theta_int,
    g1,
    g2,
    int_x0,
    int_y0,
    int_h_over_r,
    psf=None,
    oversample=1,
):
    """Shared numpy k-space rendering with pluggable radial FT.

    Independent numpy implementation matching JAX ``_kspace_render_core`` +
    ``_inclined_sech2_ft``. Used by both Sersic (exponential) and Spergel
    synthetic backends.

    Parameters
    ----------
    radial_ft_fn : callable
        k_sq -> radial FT array.
        Exponential: ``lambda k_sq: 1/(1+k_sq)**1.5``
        Spergel: ``lambda k_sq: 1/(1+k_sq)**(1+nu)``
    """
    from scipy.fft import next_fast_len as _next_fast_len

    sini = np.sqrt(1.0 - cosi**2)
    Nrow, Ncol = image_pars.Nrow, image_pars.Ncol
    ps = image_pars.pixel_scale

    eff_Nrow = Nrow * oversample
    eff_Ncol = Ncol * oversample
    eff_ps = ps / oversample

    pad = 2
    if psf is not None:
        # square padding matching model's fused k-space path
        pad_sq = _next_fast_len(pad * max(eff_Nrow, eff_Ncol))
        pr = pc = pad_sq
    else:
        pr = int(np.ceil(pad * eff_Nrow / 2) * 2)
        pc = int(np.ceil(pad * eff_Ncol / 2) * 2)

    ky = 2 * np.pi * np.fft.fftfreq(pr, d=eff_ps)
    kx = 2 * np.pi * np.fft.fftfreq(pc, d=eff_ps)
    KY, KX = np.meshgrid(ky, kx, indexing='ij')

    # centroid phase: half-pixel correction uses COARSE grid centering
    hx = 0.5 * ps * (1 - Ncol % 2)
    hy = 0.5 * ps * (1 - Nrow % 2)
    phase = np.exp(-1j * (KX * (int_x0 - hx) + KY * (int_y0 - hy)))

    # shear: (1+g1) multiplies kx (horizontal), (1-g1) multiplies ky (vertical)
    norm_s = 1.0 / np.sqrt(1.0 - (g1**2 + g2**2))
    kx_s = norm_s * ((1 + g1) * KX + g2 * KY)
    ky_s = norm_s * (g2 * KX + (1 - g1) * KY)

    # rotation by -theta_int
    c, s = np.cos(-theta_int), np.sin(-theta_int)
    kx_gal = c * kx_s - s * ky_s
    ky_gal = s * kx_s + c * ky_s

    # analytic FT
    kx_sc = kx_gal * int_rscale
    ky_sc = ky_gal * int_rscale
    k_sq = kx_sc**2 + (ky_sc * cosi) ** 2
    ft_radial = radial_ft_fn(k_sq)

    u = (np.pi / 2) * int_h_over_r * ky_sc * sini
    u_safe = np.where(np.abs(u) < 1e-4, np.ones_like(u), u)
    ft_vertical = np.where(np.abs(u) < 1e-4, 1.0 - u**2 / 6.0, u_safe / np.sinh(u_safe))

    I_hat = flux * ft_radial * ft_vertical * phase

    if psf is not None:
        # fuse PSF on the profile's k-grid before IFFT
        # real-space kernel (independent from model's drawKImage path)
        kern_size = psf.getGoodImageSize(eff_ps)
        kern_img = psf.drawImage(
            nx=kern_size, ny=kern_size, scale=eff_ps, method='no_pixel'
        )
        kernel = kern_img.array.astype(np.float64)
        kernel /= kernel.sum()

        # pad to profile's FFT grid and multiply in k-space
        kernel_padded = np.zeros((pr, pc))
        kr, kc = kernel.shape
        kernel_padded[:kr, :kc] = kernel
        kernel_padded = np.roll(kernel_padded, (-(kr // 2), -(kc // 2)), axis=(0, 1))
        I_hat = I_hat * np.fft.fft2(kernel_padded)

    full = np.fft.ifft2(I_hat).real
    roll_row = (Nrow // 2) * oversample
    roll_col = (Ncol // 2) * oversample
    full = np.roll(full, (roll_row, roll_col), axis=(0, 1))
    intensity = full[:eff_Nrow, :eff_Ncol] / eff_ps**2

    if oversample > 1:
        intensity = intensity.reshape(Nrow, oversample, Ncol, oversample).mean(
            axis=(1, 3)
        )

    return intensity


def generate_sersic_intensity_2d(
    image_pars: ImagePars,
    flux: float,
    int_rscale: float,
    n_sersic: float,
    cosi: float,
    theta_int: float,
    g1: float = 0.0,
    g2: float = 0.0,
    int_x0: float = 0.0,
    int_y0: float = 0.0,
    int_h_over_r: float = 0.0,
    backend: str = 'scipy',
    psf=None,
    oversample: int = 1,
) -> np.ndarray:
    """
    Generate Sersic intensity profile.

    Parameters
    ----------
    image_pars : ImagePars
        Image parameters defining the coordinate grids.
    flux : float
        Total flux.
    int_rscale : float
        Scale radius.
    n_sersic : float
        Sersic index (n=1 for exponential, n=4 for de Vaucouleurs).
    cosi : float
        Cosine of inclination angle.
    theta_int : float
        Intrinsic position angle in radians.
    g1, g2 : float, optional
        Shear components.
    int_x0, int_y0 : float, optional
        Centroid offsets.
    int_h_over_r : float, optional
        Scale height / scale radius ratio for 3D disk. Default 0.0 (thin disk).
    backend : str, optional
        Backend for computation ('scipy' or 'galsim'). Default is 'scipy'.
    psf : galsim.GSObject, optional
        PSF to convolve with. Default is None (no PSF).

    Returns
    -------
    ndarray
        Intensity map.
    """

    if backend == 'galsim':
        return _generate_sersic_galsim(
            image_pars,
            flux,
            int_rscale,
            n_sersic,
            cosi,
            theta_int,
            g1,
            g2,
            int_x0,
            int_y0,
            int_h_over_r=int_h_over_r,
            psf=psf,
        )
    else:
        return _generate_sersic_scipy(
            image_pars,
            flux,
            int_rscale,
            n_sersic,
            cosi,
            theta_int,
            g1,
            g2,
            int_x0,
            int_y0,
            int_h_over_r=int_h_over_r,
            psf=psf,
            oversample=oversample,
        )


def _build_sersic_ft_galsim(n_sersic, int_rscale):
    """Build a Sersic radial FT function using GalSim's numerical kValue.

    Uses GalSim's Ogata-based Hankel transform (precomputed + spline cache)
    as the radial FT source. Independent of the Miller & Pasha emulator
    used in the JAX model code.

    The scipy synthetic path uses this to test the numpy k-space rendering
    pipeline independently from the JAX rendering + emulator path.

    Parameters
    ----------
    n_sersic : float
        Sersic index.
    int_rscale : float
        Sersic scale radius r_s (arcsec), where I(r) = I_0 * exp(-(r/r_s)^{1/n}).
        Related to half-light radius by R_e = b_n^n * r_s.

    Returns
    -------
    callable
        k_sq -> FT array (normalized: FT(0) = 1).
        k_sq is dimensionless: (k_physical * int_rscale)^2.
    """
    # convert r_s to R_e for GalSim
    n = n_sersic
    bn = 2.0 * n - 1.0 / 3.0 + 4.0 / (405.0 * n) + 46.0 / (25515.0 * n**2)
    Re = int_rscale * bn**n

    # GalSim Sersic with this R_e (flux=1 gives normalized FT)
    gs_prof = gs.Sersic(n=n_sersic, half_light_radius=Re, flux=1.0)

    def radial_ft_fn(k_sq):
        # k_sq = (k_physical * int_rscale)^2, so k_physical = sqrt(k_sq)/int_rscale
        k_phys = np.sqrt(np.maximum(k_sq, 0.0)) / int_rscale
        # the FT is radially symmetric — evaluate on unique |k| values
        # and interpolate back to the full grid
        k_flat = k_phys.ravel()
        k_unique = np.unique(np.round(k_flat, decimals=10))
        ft_unique = np.ones(len(k_unique))
        for i, kv in enumerate(k_unique):
            if kv > 1e-15:
                ft_unique[i] = gs_prof.kValue(kv, 0.0).real
        # map back via sorted lookup
        from scipy.interpolate import interp1d

        interp = interp1d(
            k_unique,
            ft_unique,
            kind='linear',
            fill_value=0.0,
            bounds_error=False,
        )
        return interp(k_flat).reshape(k_sq.shape)

    return radial_ft_fn


def _generate_sersic_scipy(
    image_pars: ImagePars,
    flux: float,
    int_rscale: float,
    n_sersic: float,
    cosi: float,
    theta_int: float,
    g1: float,
    g2: float,
    int_x0: float,
    int_y0: float,
    int_h_over_r: float = 0.0,
    psf=None,
    oversample: int = 1,
) -> np.ndarray:
    """Generate Sersic profile using scipy.

    When int_h_over_r > 0 and n_sersic == 1.0, uses 3D LOS integration
    through a sech² vertical profile (matching GalSim InclinedExponential).
    """

    # Build coordinate grids from ImagePars
    X, Y = build_map_grid_from_image_pars(image_pars, unit='arcsec', centered=True)

    sini = np.sqrt(1.0 - cosi**2)

    # Step 1: Recenter (obs -> cen)
    X_c = X - int_x0
    Y_c = Y - int_y0

    # Step 2: area-preserving shear (cen -> source), matches GalSim .shear()
    norm = 1.0 / np.sqrt(1.0 - (g1**2 + g2**2))
    X_shear = norm * ((1.0 - g1) * X_c - g2 * Y_c)
    Y_shear = norm * (-g2 * X_c + (1.0 + g1) * Y_c)

    # Step 3: Rotate by position angle (source -> gal)
    cos_pa = np.cos(-theta_int)
    sin_pa = np.sin(-theta_int)
    X_gal = cos_pa * X_shear - sin_pa * Y_shear
    Y_gal = sin_pa * X_shear + cos_pa * Y_shear

    # 3D k-space rendering via shared core
    if int_h_over_r > 0:
        if n_sersic == 1.0:
            # exact exponential FT
            radial_ft = lambda k_sq: 1.0 / (1.0 + k_sq) ** 1.5
        else:
            # numerical Sersic FT via GalSim kValue (independent of emulator)
            radial_ft = _build_sersic_ft_galsim(n_sersic, int_rscale)

        return _generate_inclined_kspace_scipy(
            radial_ft,
            image_pars,
            flux,
            int_rscale,
            cosi,
            theta_int,
            g1,
            g2,
            int_x0,
            int_y0,
            int_h_over_r,
            psf=psf,
            oversample=oversample,
        )

    # thin-disk path (original)
    # Step 4: Deproject inclination (gal -> disk) - divide by cosi
    X_disk = X_gal
    Y_disk = Y_gal / cosi if cosi > 0 else Y_gal

    # Compute radius in disk plane
    r_disk = np.sqrt(X_disk**2 + Y_disk**2)

    # Convert flux to central surface brightness
    if n_sersic == 1.0:
        I0_disk = flux / (2.0 * np.pi * int_rscale**2)
    else:
        norm_factor = int_rscale**2 * 2.0 * np.pi * n_sersic * gamma(2.0 * n_sersic)
        I0_disk = flux / norm_factor

    intensity_disk = I0_disk * np.exp(-np.power(r_disk / int_rscale, 1.0 / n_sersic))
    intensity_obs = intensity_disk / cosi if cosi > 0 else intensity_disk

    if psf is not None:
        from kl_pipe.psf import gsobj_to_kernel, convolve_fft_numpy

        kernel, padded_shape = gsobj_to_kernel(psf, image_pars=image_pars)
        intensity_obs = convolve_fft_numpy(intensity_obs, kernel, padded_shape)

    return intensity_obs


def _generate_sersic_galsim(
    image_pars: ImagePars,
    flux: float,
    int_rscale: float,
    n_sersic: float,
    cosi: float,
    theta_int: float,
    g1: float,
    g2: float,
    int_x0: float,
    int_y0: float,
    int_h_over_r: float = 0.0,
    gsparams: gs.GSParams = None,
    psf=None,
    method: str = 'auto',
) -> np.ndarray:
    """
    Generate Sersic profile using GalSim backend.

    This uses GalSim's native InclinedExponential and InclinedSersic classes,
    which properly handle surface brightness projection effects.

    Parameters
    ----------
    image_pars : ImagePars
        Image parameters defining grid geometry, pixel scale, WCS.
    flux : float
        Total integrated flux.
    int_rscale : float
        Scale radius in arcsec.
    n_sersic : float
        Sersic index.
    cosi : float
        Cosine of inclination angle.
    theta_int : float
        Position angle in radians (measured E of N).
    g1, g2 : float
        Reduced shear components.
    int_x0, int_y0 : float
        Centroid offsets in arcsec.
    gsparams : galsim.GSParams, optional
        GalSim parameters for profile generation.

    Returns
    -------
    ndarray
        Surface brightness map matching image_pars shape.
    """

    inclination = gs.Angle(np.arccos(cosi), gs.radians)

    # scale_h_over_r: use provided value, or GalSim default (0.1)
    h_over_r = int_h_over_r if int_h_over_r > 0 else 0.1

    # Create the inclined profile
    if n_sersic == 1.0:
        # Use InclinedExponential for speed
        profile = gs.InclinedExponential(
            inclination=inclination,
            scale_radius=int_rscale,
            scale_h_over_r=h_over_r,
            flux=flux,
            gsparams=gsparams,
        )
    else:
        # General Sersic profile
        profile = gs.InclinedSersic(
            n=n_sersic,
            inclination=inclination,
            scale_radius=int_rscale,
            scale_h_over_r=h_over_r,
            flux=flux,
            gsparams=gsparams,
        )

    # Apply position angle rotation
    # GalSim rotation is CCW from +x axis, same as our theta_int convention
    profile = profile.rotate(theta_int * gs.radians)

    # Apply shear and centroid offset
    mu = 1.0 / (1.0 - (g1**2 + g2**2))  # magnification factor
    profile = profile.lens(g1=g1, g2=g2, mu=mu)
    profile = profile.shift(int_x0, int_y0)

    # Convolve with PSF if provided (GalSim native convolution)
    if psf is not None:
        profile = gs.Convolve(profile, psf)

    # standard convention: X=horizontal=cols, Y=vertical=rows
    # GalSim drawImage(nx, ny) expects nx=Ncol, ny=Nrow
    nx, ny = image_pars.Nx, image_pars.Ny
    pixel_scale = image_pars.pixel_scale

    image = profile.drawImage(nx=nx, ny=ny, scale=pixel_scale, method=method)

    # GalSim array shape = (ny, nx) = (Nrow, Ncol) — matches our grid directly
    intensity = image.array

    return intensity


# ==============================================================================
# Spergel intensity profile generators
# ==============================================================================


def generate_spergel_intensity_2d(
    image_pars: ImagePars,
    flux: float,
    int_rscale: float,
    nu: float,
    cosi: float,
    theta_int: float,
    g1: float = 0.0,
    g2: float = 0.0,
    int_x0: float = 0.0,
    int_y0: float = 0.0,
    int_h_over_r: float = 0.1,
    backend: str = 'scipy',
    psf=None,
    oversample: int = 1,
) -> np.ndarray:
    """Generate Spergel intensity profile.

    Parameters
    ----------
    image_pars : ImagePars
        Image parameters defining the coordinate grids.
    flux : float
        Total flux.
    int_rscale : float
        Spergel scale length c (arcsec).
    nu : float
        Spergel index. nu=0.5 is exponential, nu=-0.6 ~ de Vaucouleurs.
    cosi : float
        Cosine of inclination angle.
    theta_int : float
        Position angle in radians.
    g1, g2 : float, optional
        Shear components.
    int_x0, int_y0 : float, optional
        Centroid offsets.
    int_h_over_r : float, optional
        Scale height / scale radius. Default 0.1.
    backend : str, optional
        'scipy' (any inclination) or 'galsim' (face-on only).
    psf : galsim.GSObject, optional
        PSF to convolve with.
    oversample : int, optional
        Anti-aliasing oversampling factor. Default 1.

    Returns
    -------
    ndarray
        Intensity map.
    """
    if backend == 'galsim':
        return _generate_spergel_galsim(
            image_pars,
            flux,
            int_rscale,
            nu,
            cosi,
            theta_int,
            g1,
            g2,
            int_x0,
            int_y0,
            psf=psf,
        )
    else:
        return _generate_spergel_scipy(
            image_pars,
            flux,
            int_rscale,
            nu,
            cosi,
            theta_int,
            g1,
            g2,
            int_x0,
            int_y0,
            int_h_over_r,
            psf=psf,
            oversample=oversample,
        )


def _generate_spergel_scipy(
    image_pars,
    flux,
    int_rscale,
    nu,
    cosi,
    theta_int,
    g1,
    g2,
    int_x0,
    int_y0,
    int_h_over_r=0.1,
    psf=None,
    oversample=1,
):
    """Generate Spergel profile via shared numpy k-space core.

    Radial FT: ``(1 + k²)^{-(1+nu)}``. Works at all inclinations.
    """
    return _generate_inclined_kspace_scipy(
        lambda k_sq: 1.0 / (1.0 + k_sq) ** (1.0 + nu),
        image_pars,
        flux,
        int_rscale,
        cosi,
        theta_int,
        g1,
        g2,
        int_x0,
        int_y0,
        int_h_over_r,
        psf=psf,
        oversample=oversample,
    )


def _generate_spergel_galsim(
    image_pars,
    flux,
    int_rscale,
    nu,
    cosi,
    theta_int,
    g1,
    g2,
    int_x0,
    int_y0,
    gsparams=None,
    psf=None,
    method='no_pixel',
):
    """Generate face-on Spergel profile via galsim.Spergel.

    GalSim has no InclinedSpergel, so this only supports face-on (cosi >= 0.99).
    For inclined Spergel, use the scipy backend.

    Raises
    ------
    ValueError
        If cosi < 0.99 (inclined).
    """
    if cosi < 0.99:
        raise ValueError(
            f"GalSim Spergel backend only supports face-on (cosi >= 0.99), "
            f"got cosi={cosi}. Use backend='scipy' for inclined profiles."
        )

    profile = gs.Spergel(
        nu=nu,
        scale_radius=int_rscale,
        flux=flux,
        gsparams=gsparams,
    )

    # apply position angle, area-preserving shear, centroid offset
    profile = profile.rotate(theta_int * gs.radians)
    mu = 1.0 / (1.0 - (g1**2 + g2**2))
    profile = profile.lens(g1=g1, g2=g2, mu=mu)
    profile = profile.shift(int_x0, int_y0)

    if psf is not None:
        profile = gs.Convolve(profile, psf)

    nx, ny = image_pars.Nx, image_pars.Ny
    image = profile.drawImage(nx=nx, ny=ny, scale=image_pars.pixel_scale, method=method)

    return image.array


# ==============================================================================
# Noise generation
# ==============================================================================


def add_noise(
    image: np.ndarray,
    target_snr: float,
    seed: Optional[int] = None,
    include_poisson: bool = True,
    poisson_scale: float = 1.0,
    return_variance: bool = True,
) -> Tuple[np.ndarray, float] | np.ndarray:
    """
    Add realistic noise to achieve target total signal-to-noise ratio.

    Can include both Poisson (shot) noise and Gaussian (read) noise.

    The total S/N is defined as:
        S/N = sqrt(sum(signal^2)) / sqrt(sum(noise^2))

    Parameters
    ----------
    image : ndarray
        Input noiseless image.
    target_snr : float
        Target total signal-to-noise ratio.
    seed : int, optional
        Random seed for reproducibility.
    include_poisson : bool, optional
        If True, include Poisson (shot) noise. If False, only Gaussian noise.
        Default is True.
    poisson_scale : float, optional
        Scale factor to convert image values to photon counts for Poisson noise.
        Higher values = more shot noise. Default is 1.0.
    return_variance : bool, optional
        If True, return both noisy image and variance used.
        If False, return only noisy image. Default is True.

    Returns
    -------
    noisy_image : ndarray
        Image with noise added.
    variance : float, optional
        Effective variance of the noise (returned only if return_variance=True).

    Notes
    -----
    When include_poisson=True:
        - Poisson noise is added: sqrt(counts) per pixel
        - Gaussian read noise is added to reach target SNR

    When include_poisson=False:
        - Only Gaussian noise with constant variance
        - Matches the original add_gaussian_noise behavior

    Examples
    --------
    >>> image = np.random.rand(64, 64) * 100  # counts
    >>>
    >>> # With Poisson + Gaussian noise (more realistic)
    >>> noisy, var = add_noise(image, target_snr=50, include_poisson=True)
    >>>
    >>> # Only Gaussian noise (simpler, for testing)
    >>> noisy, var = add_noise(image, target_snr=50, include_poisson=False)
    """

    rng = np.random.default_rng(seed)

    if include_poisson:
        # Convert image to photon counts
        counts = np.abs(image) * poisson_scale

        # Add Poisson noise (shot noise)
        # For negative values, we'll handle them as if they're background-subtracted
        noisy_counts = np.where(
            counts > 0,
            rng.poisson(counts),
            counts + rng.normal(0, np.sqrt(np.abs(counts)), counts.shape),
        )

        # Convert back to original units
        noisy_image = noisy_counts / poisson_scale

        # Compute variance from Poisson noise
        poisson_var = np.mean(np.abs(image) / poisson_scale)

        # Calculate how much Gaussian noise to add to reach target SNR
        total_signal = np.sqrt(np.sum(image**2))
        target_noise_power = total_signal / target_snr

        # Current noise power from Poisson
        current_noise_power = np.sqrt(np.sum((noisy_image - image) ** 2))

        # Additional Gaussian noise needed
        if target_noise_power > current_noise_power:
            additional_noise_power = np.sqrt(
                target_noise_power**2 - current_noise_power**2
            )
            n_pixels = image.size
            sigma_gaussian = additional_noise_power / np.sqrt(n_pixels)

            gaussian_noise = rng.normal(0, sigma_gaussian, image.shape)
            noisy_image += gaussian_noise

            # Effective variance (combination of Poisson and Gaussian)
            variance = poisson_var + sigma_gaussian**2
        else:
            # Poisson noise alone is sufficient
            variance = poisson_var

    else:
        total_signal = np.sqrt(np.sum(image**2))
        target_noise_power = total_signal / target_snr

        # Per-pixel noise stddev (constant variance)
        n_pixels = image.size
        sigma_per_pixel = target_noise_power / np.sqrt(n_pixels)

        # Add Gaussian noise
        noise = rng.normal(0, sigma_per_pixel, image.shape)
        noisy_image = image + noise

        variance = sigma_per_pixel**2

    if return_variance:
        return noisy_image, variance
    else:
        return noisy_image


# Backward compatibility alias
add_gaussian_noise = lambda *args, **kwargs: add_noise(
    *args, include_poisson=False, **kwargs
)


# ==============================================================================
# Synthetic observation classes
# ==============================================================================


class SyntheticObservation(ABC):
    """
    Base class for synthetic observations.

    Provides structure for generating synthetic data with known true parameters,
    adding noise, and storing results for testing and validation.

    Parameters
    ----------
    true_params : dict
        Dictionary of true model parameters.
    seed : int, optional
        Random seed for reproducibility.

    Attributes
    ----------
    true_params : dict
        True parameters used to generate data.
    X, Y : ndarray or None
        Coordinate grids from the most recent call to generate().
        These are updated each time generate() is called with new grids.
    data_true : ndarray or None
        Noiseless synthetic data from most recent generation.
    data_noisy : ndarray or None
        Noisy synthetic data from most recent generation.
    variance : float or None
        Noise variance from most recent generation.
    """

    def __init__(self, true_params: Dict[str, float], seed: Optional[int] = None):
        self.true_params = true_params
        self.seed = seed

        # storage for last generated data
        self.data_true = None
        self.data_noisy = None
        self.variance = None

    @abstractmethod
    def generate(
        self, X: np.ndarray, Y: np.ndarray, snr: float, seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate synthetic data on specified grid with noise.

        Parameters
        ----------
        X, Y : ndarray
            Coordinate grids.
        snr : float
            Target signal-to-noise ratio (total S/N).
        seed : int, optional
            Random seed for noise generation. If None, uses self.seed.

        Returns
        -------
        ndarray
            Noisy synthetic data.
        """
        pass


class SyntheticVelocity:
    """
    Synthetic velocity field observations.
    """

    def __init__(
        self,
        true_params: Dict[str, float],
        model_type: str = 'arctan',
        seed: Optional[int] = None,
        psf=None,
        intensity_for_psf=None,
    ):
        self.true_params = true_params
        self.model_type = model_type
        self.seed = seed
        self.psf = psf
        self.intensity_for_psf = intensity_for_psf

        # Storage for last generated data
        self.data_true = None
        self.data_noisy = None
        self.variance = None

        # Validate parameters
        self._validate_params()

    def _validate_params(self):
        """Validate that required parameters are present."""
        if self.model_type not in REQUIRED_PARAMS:
            raise ValueError(
                f"Unknown model_type '{self.model_type}'. "
                f"Available: {list(REQUIRED_PARAMS.keys())}"
            )

        required = REQUIRED_PARAMS[self.model_type]
        provided = set(self.true_params.keys())
        missing = required - provided

        if missing:
            raise ValueError(
                f"Missing required parameters for {self.model_type}: {missing}"
            )

    def generate(
        self,
        image_pars: ImagePars,
        snr: float,
        seed: Optional[int] = None,
        include_poisson: bool = True,
    ) -> np.ndarray:
        """
        Generate synthetic velocity data.

        Parameters
        ----------
        image_pars : ImagePars
            Image parameters defining the coordinate grids.
        snr : float
            Target signal-to-noise ratio (total S/N).
        seed : int, optional
            Random seed for noise generation. If None, uses self.seed.
        include_poisson : bool, optional
            Whether to include Poisson (shot) noise. Default is True.

        Returns
        -------
        ndarray
            Noisy velocity map in km/s.
        """

        if seed is None:
            seed = self.seed

        # Generate true velocity field
        if self.model_type == 'arctan':
            self.data_true = generate_arctan_velocity_2d(
                image_pars,
                **self.true_params,
                psf=self.psf,
                intensity_for_psf=self.intensity_for_psf,
            )
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

        # Add noise
        self.data_noisy, self.variance = add_noise(
            self.data_true,
            snr,
            seed=seed,
            include_poisson=include_poisson,
            return_variance=True,
        )

        return self.data_noisy


class SyntheticIntensity:
    """
    Synthetic intensity/surface brightness observations.
    """

    def __init__(
        self,
        true_params: Dict[str, float],
        model_type: str = 'sersic',
        seed: Optional[int] = None,
        psf=None,
    ):
        self.true_params = true_params
        self.model_type = model_type
        self.seed = seed
        self.psf = psf

        # Storage for last generated data
        self.data_true = None
        self.data_noisy = None
        self.variance = None

        # Validate parameters
        self._validate_params()

    def _validate_params(self):
        """Validate that required parameters are present."""
        if self.model_type not in REQUIRED_PARAMS:
            raise ValueError(
                f"Unknown model_type '{self.model_type}'. "
                f"Available: {list(REQUIRED_PARAMS.keys())}"
            )

        required = REQUIRED_PARAMS[self.model_type]
        provided = set(self.true_params.keys())
        missing = required - provided

        if missing:
            raise ValueError(
                f"Missing required parameters for {self.model_type}: {missing}"
            )

    def generate(
        self,
        image_pars: ImagePars,
        snr: float,
        seed: Optional[int] = None,
        include_poisson: bool = True,
        sersic_backend: str = 'scipy',
        oversample: int = 1,
    ) -> np.ndarray:
        """
        Generate synthetic intensity data.

        Parameters
        ----------
        image_pars : ImagePars
            Image parameters defining the coordinate grids.
        snr : float
            Target signal-to-noise ratio (total S/N).
        seed : int, optional
            Random seed for noise generation. If None, uses self.seed.
        include_poisson : bool, optional
            Whether to include Poisson (shot) noise. Default is True.
        sersic_backend : str, optional
            Backend for Sersic profile generation ('scipy' or 'galsim'). Default is
            'scipy'.
        oversample : int, optional
            Oversampling factor for pixel integration. Default 1.

        Returns
        -------
        ndarray
            Noisy intensity map.
        """

        if seed is None:
            seed = self.seed

        # Generate true intensity field
        if self.model_type == 'sersic':
            self.data_true = generate_sersic_intensity_2d(
                image_pars,
                backend=sersic_backend,
                psf=self.psf,
                oversample=oversample,
                **self.true_params,
            )
        elif self.model_type == 'exponential':
            # Exponential is Sersic with n=1
            params = self.true_params.copy()
            params['n_sersic'] = 1.0
            self.data_true = generate_sersic_intensity_2d(
                image_pars,
                backend=sersic_backend,
                psf=self.psf,
                oversample=oversample,
                **params,
            )
        elif self.model_type == 'spergel':
            self.data_true = generate_spergel_intensity_2d(
                image_pars,
                backend=sersic_backend,
                psf=self.psf,
                oversample=oversample,
                **self.true_params,
            )
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

        # Add noise
        self.data_noisy, self.variance = add_noise(
            self.data_true,
            snr,
            seed=seed,
            include_poisson=include_poisson,
            return_variance=True,
        )

        return self.data_noisy


# ==============================================================================
# Numpy reference datacube / grism generators
# ==============================================================================


def generate_datacube_3d(
    image_pars,
    vel_pars,
    int_pars,
    spectral_pars,
    lambda_grid,
    psf=None,
    spatial_oversample=1,
    spectral_oversample=1,
):
    """Independent numpy datacube generator for validation.

    Uses generate_arctan_velocity_2d() and generate_sersic_intensity_2d()
    internally. Completely independent from SpectralModel/JAX code paths.

    Parameters
    ----------
    image_pars : ImagePars
        Spatial grid.
    vel_pars : dict
        {v0, vcirc, vel_rscale, cosi, theta_int, g1, g2}
    int_pars : dict
        {flux, int_rscale, n_sersic, cosi, theta_int, g1, g2}
    spectral_pars : dict
        {z, vel_dispersion, lines: [{lambda_rest, flux, cont}, ...]}
    lambda_grid : ndarray
        Wavelength array nm, shape (Nlambda,)
    psf : galsim.GSObject, optional
        PSF to convolve each wavelength slice.

    Returns
    -------
    ndarray, shape (Nrow, Ncol, Nlambda)
    """
    C_KMS = 299792.458

    z = spectral_pars['z']
    vel_disp = spectral_pars['vel_dispersion']
    lines = spectral_pars['lines']

    # velocity map (LOS, includes v0)
    v_map = generate_arctan_velocity_2d(image_pars, **vel_pars)
    v0 = vel_pars['v0']
    v_rotation = v_map - v0

    # broadband intensity (for continuum)
    int_pars_full = dict(int_pars)
    if 'n_sersic' not in int_pars_full:
        int_pars_full['n_sersic'] = 1.0
    I_broadband = generate_sersic_intensity_2d(
        image_pars, **int_pars_full, oversample=spatial_oversample
    )

    Nrow, Ncol = image_pars.Nrow, image_pars.Ncol
    Nlam = len(lambda_grid)
    osf = spectral_oversample

    # fine wavelength grid (mirrors JAX SpectralModel.build_cube)
    if osf > 1 and Nlam >= 2:
        dl = lambda_grid[1] - lambda_grid[0]
        half = dl / 2.0
        fine_offsets = np.linspace(-half + half / osf, half - half / osf, osf)
        lambda_fine = (lambda_grid[:, None] + fine_offsets[None, :]).reshape(-1)
    else:
        lambda_fine = lambda_grid
        osf = 1

    n_fine = len(lambda_fine)
    cube_fine = np.zeros((Nrow, Ncol, n_fine))

    R_func = spectral_pars.get('R_func', lambda lam: 461.0 * lam / 1000.0)

    for line_info in lines:
        lam_rest = line_info['lambda_rest']
        line_flux = line_info['flux']
        cont = line_info.get('cont', 0.0)

        # per-line intensity: scale broadband by line flux ratio
        # (simplified: uses broadband morphology with different total flux)
        line_int_pars = dict(int_pars_full)
        if 'line_int_pars' in line_info:
            line_int_pars.update(line_info['line_int_pars'])
        line_int_pars['flux'] = line_flux
        I_line = generate_sersic_intensity_2d(
            image_pars, **line_int_pars, oversample=spatial_oversample
        )

        # Doppler-shifted observed wavelength per pixel
        lam_obs = lam_rest * (1.0 + z) * (1.0 + v_rotation / C_KMS)

        # effective sigma
        R_at_line = R_func(lam_rest * (1.0 + z))
        sigma_inst = C_KMS / (2.355 * R_at_line)
        sigma_eff = np.sqrt(vel_disp**2 + sigma_inst**2)
        sigma_lambda = lam_obs * sigma_eff / C_KMS

        # normalized Gaussian on fine grid
        dlam = lambda_fine[None, None, :] - lam_obs[:, :, None]
        sig = sigma_lambda[:, :, None]
        gauss = (1.0 / (sig * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * (dlam / sig) ** 2)

        cube_fine += I_line[:, :, None] * gauss
        cube_fine += I_broadband[:, :, None] * cont

    # bin fine -> coarse
    if osf > 1:
        cube = cube_fine.reshape(Nrow, Ncol, Nlam, osf).mean(axis=-1)
    else:
        cube = cube_fine

    if psf is not None:
        from kl_pipe.psf import gsobj_to_kernel, convolve_fft_numpy

        kernel, padded_shape = gsobj_to_kernel(psf, image_pars=image_pars)
        for k in range(Nlam):
            cube[:, :, k] = convolve_fft_numpy(cube[:, :, k], kernel, padded_shape)

    return cube


def generate_grism_2d(
    image_pars,
    vel_pars,
    int_pars,
    spectral_pars,
    grism_pars,
    lambda_grid=None,
    psf=None,
):
    """Independent numpy grism image generator for validation.

    Builds datacube via generate_datacube_3d(), then disperses with
    scipy.ndimage.map_coordinates.

    Parameters
    ----------
    image_pars : ImagePars
        Spatial grid.
    vel_pars, int_pars, spectral_pars : dict
        As for generate_datacube_3d.
    grism_pars : dict
        {dispersion, lambda_ref, dispersion_angle}
    lambda_grid : ndarray, optional
        If None, built from grism_pars.
    psf : galsim.GSObject, optional

    Returns
    -------
    ndarray, shape (Nrow, Ncol)
    """
    from scipy.ndimage import map_coordinates as scipy_map_coords

    disp = grism_pars['dispersion']
    lam_ref = grism_pars['lambda_ref']
    angle = grism_pars['dispersion_angle']

    if lambda_grid is None:
        # build lambda grid from grism pars
        z = spectral_pars['z']
        lines = spectral_pars['lines']
        lam_obs = [l['lambda_rest'] * (1.0 + z) for l in lines]
        lam_center = 0.5 * (min(lam_obs) + max(lam_obs))
        C_KMS = 299792.458
        vel_window = grism_pars.get('velocity_window_kms', 3000.0)
        dlam_vel = lam_center * vel_window / C_KMS
        lam_min = min(lam_obs) - dlam_vel
        lam_max = max(lam_obs) + dlam_vel
        n_lambda = int(np.ceil((lam_max - lam_min) / disp)) + 1
        lambda_grid = np.linspace(lam_min, lam_max, max(n_lambda, 3))

    cube = generate_datacube_3d(
        image_pars, vel_pars, int_pars, spectral_pars, lambda_grid, psf=psf
    )

    Nrow, Ncol, Nlam = cube.shape
    pixel_offsets = (lambda_grid - lam_ref) / disp

    cos_a = np.cos(angle)
    sin_a = np.sin(angle)

    rows = np.arange(Nrow)
    cols = np.arange(Ncol)
    Y_base, X_base = np.meshgrid(rows, cols, indexing='ij')

    dlam = np.abs(lambda_grid[1] - lambda_grid[0]) if Nlam >= 2 else 1.0

    dispersed = np.zeros((Nrow, Ncol))
    for k in range(Nlam):
        offset_k = pixel_offsets[k]
        dx_k = offset_k * cos_a
        dy_k = offset_k * sin_a
        coords = np.array([Y_base - dy_k, X_base - dx_k])
        shifted = scipy_map_coords(
            cube[:, :, k], coords, order=1, mode='constant', cval=0.0
        )
        dispersed += shifted * dlam

    return dispersed


class SyntheticKLObservation:
    """
    Combined kinematic-lensing synthetic observation.

    Bundles together synthetic velocity and intensity observations for
    joint kinematic-lensing analysis testing.

    Parameters
    ----------
    velocity_obs : SyntheticVelocity
        Synthetic velocity observation.
    intensity_obs : SyntheticIntensity
        Synthetic intensity observation.

    Attributes
    ----------
    velocity : SyntheticVelocity
        Velocity component.
    intensity : SyntheticIntensity
        Intensity component.

    Examples
    --------
    >>> vel_params = {...}
    >>> int_params = {...}
    >>> vel_obs = SyntheticVelocity(vel_params)
    >>> int_obs = SyntheticIntensity(int_params)
    >>> kl_obs = SyntheticKLObservation(vel_obs, int_obs)
    >>>
    >>> # Generate both on same or different grids
    >>> image_pars_vel = ImagePars(shape=(32, 32), pixel_scale=0.3, indexing='ij')
    >>> image_pars_int = ImagePars(shape=(64, 64), pixel_scale=0.1, indexing='ij')
    >>> kl_obs.velocity.generate(image_pars_vel, snr=50)
    >>> kl_obs.intensity.generate(image_pars_int, snr=100)
    """

    def __init__(
        self,
        velocity_obs: SyntheticVelocity,
        intensity_obs: SyntheticIntensity,
    ):
        self.velocity = velocity_obs
        self.intensity = intensity_obs
