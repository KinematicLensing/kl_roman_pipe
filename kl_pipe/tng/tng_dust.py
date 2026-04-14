import numpy as np
from numba import njit, prange
from scipy.interpolate import interp1d

"""This module contains the core numba-accelerated kernels for dust attenuation in the TNG pipeline. The main functions are:
1. numba_bilinear_3d: A 3D bilinear interpolation kernel for getting SEDs from the grid based on particle metallicity and age.
2. numba_bilinear_2d: A 2D bilinear interpolation kernel for getting absolute magnitudes from the grid based on particle metallicity and age.
3. sph_project_2d: A 2D SPH projection kernel that takes particle positions, smoothing lengths, and weights (e.g., luminosities) and projects them onto a 2D pixel grid using a cubic spline kernel. 
4. ccm89_av_ratio: A function to compute the extinction curve A(λ)/A(V) using the Cardelli, Clayton & Mathis (1989) extinction law.
5. DG2000_optical_depth: A function to compute the optical depth τ(λ) using the Devriendt & Guiderdoni 2000 model, which depends on hydrogen column density, metallicity, and redshift.
6. CSK1994_scatter: A function to compute the optical depth τ(λ) corrected for dust scattering effects using the Calzetti, Seaton & Krügel (1994) model, which includes wavelength-dependent albedo and anisotropy factors."""


# -------------------------------------------------------------------------
# 1. THE 3D SED KERNEL. Using numba to unroll the for loops and do the interpolation purely in the CPU cache for maximum speed.
# -------------------------------------------------------------------------
@njit(parallel=True, fastmath=True)
def numba_bilinear_3d(
    met_chunk: np.ndarray,
    age_chunk: np.ndarray,
    met_grid: np.ndarray,
    age_grid: np.ndarray,
    band_grid: np.ndarray,
) -> np.ndarray:
    """Bilinear interpolation in 2D (metallicity, age) for each particle to get SED for a specific band. Returns array of shape (N_particles, N_wave) with interpolated SEDs.

    The input metallicity is the mass metallicity ratio from the TNG simulation. Not solar metallicity. The input age is in Gyr.
    Parameters
    ----------
    met_chunk : np.ndarray
        Metallicity values for each particle
    age_chunk : np.ndarray
        Age values for each particle
    met_grid : np.ndarray
        Grid of metallicity values
    age_grid : np.ndarray
        Grid of age values
    band_grid : np.ndarray
        Grid of SED values for a specific band, shape (N_met, N_age, N_wave)

    Returns
    -------
    np.ndarray
        Interpolated SEDs for each particle
    """

    N_particles = len(met_chunk)
    N_wave = band_grid.shape[2]

    # Pre-allocate the ONLY array that gets written to RAM
    out_seds = np.empty((N_particles, N_wave), dtype=band_grid.dtype)

    # prange splits the 100,000 particles across your CPU cores!
    for i in prange(N_particles):
        met = met_chunk[i]
        age = age_chunk[i]

        # Numba perfectly supports np.searchsorted
        idx_met = np.searchsorted(met_grid, met) - 1
        idx_age = np.searchsorted(age_grid, age) - 1

        # Numba doesn't like np.clip on scalars, so we write explicit bounds
        if idx_met < 0:
            idx_met = 0
        if idx_met > len(met_grid) - 2:
            idx_met = len(met_grid) - 2
        if idx_age < 0:
            idx_age = 0
        if idx_age > len(age_grid) - 2:
            idx_age = len(age_grid) - 2

        met1 = met_grid[idx_met]
        met2 = met_grid[idx_met + 1]
        age1 = age_grid[idx_age]
        age2 = age_grid[idx_age + 1]

        tx = (met - met1) / (met2 - met1)
        ty = (age - age1) / (age2 - age1)

        if tx < 0.0:
            tx = 0.0
        if tx > 1.0:
            tx = 1.0
        if ty < 0.0:
            ty = 0.0
        if ty > 1.0:
            ty = 1.0

        # Weights for the four corners of the grid cell
        w11 = (1.0 - tx) * (1.0 - ty)
        w21 = tx * (1.0 - ty)
        w12 = (1.0 - tx) * ty
        w22 = tx * ty

        # THE MAGIC: Loop over wavelengths purely inside the CPU cache!
        for j in range(N_wave):
            out_seds[i, j] = (
                band_grid[idx_met, idx_age, j] * w11
                + band_grid[idx_met + 1, idx_age, j] * w21
                + band_grid[idx_met, idx_age + 1, j] * w12
                + band_grid[idx_met + 1, idx_age + 1, j] * w22
            )

    return out_seds


# -------------------------------------------------------------------------
# 2. THE 2D ABSOLUTE MAGNITUDE KERNEL. Similar to the 3D SED kernel but simpler since the grid is only 2D (metallicity, age) and the output is a single magnitude value
# for each particle instead of a full SED. Still uses numba for speed.
# -------------------------------------------------------------------------
@njit(parallel=True, fastmath=True)
def numba_bilinear_2d(
    met_chunk: np.ndarray,
    age_chunk: np.ndarray,
    met_grid: np.ndarray,
    age_grid: np.ndarray,
    band_grid: np.ndarray,
) -> np.ndarray:
    """Bilinear interpolation in 2D (metallicity, age) for each particle to get absolute magnitude for a specific band. Returns array of shape (N_particles,)
    with interpolated magnitudes.

    The input metallicity is the mass metallicity ratio from the TNG simulation. Not solar metallicity. The input age is in Gyr.
    Parameters
    ----------
    met_chunk : np.ndarray
        Metallicity values for each particle
    age_chunk : np.ndarray
        Age values for each particle
    met_grid : np.ndarray
        Grid of metallicity values
    age_grid : np.ndarray
        Grid of age values
    band_grid : np.ndarray
        Grid of absolute magnitudes for a specific band, shape (N_met, N_age)

    Returns
    -------
    np.ndarray
        Interpolated magnitudes for each particle
    """
    N_particles = len(met_chunk)
    out_mags = np.empty(N_particles, dtype=band_grid.dtype)

    for i in prange(N_particles):
        met = met_chunk[i]
        age = age_chunk[i]

        idx_met = np.searchsorted(met_grid, met) - 1
        idx_age = np.searchsorted(age_grid, age) - 1

        if idx_met < 0:
            idx_met = 0
        if idx_met > len(met_grid) - 2:
            idx_met = len(met_grid) - 2
        if idx_age < 0:
            idx_age = 0
        if idx_age > len(age_grid) - 2:
            idx_age = len(age_grid) - 2

        met1 = met_grid[idx_met]
        met2 = met_grid[idx_met + 1]
        age1 = age_grid[idx_age]
        age2 = age_grid[idx_age + 1]

        tx = (met - met1) / (met2 - met1)
        ty = (age - age1) / (age2 - age1)

        if tx < 0.0:
            tx = 0.0
        if tx > 1.0:
            tx = 1.0
        if ty < 0.0:
            ty = 0.0
        if ty > 1.0:
            ty = 1.0

        w11 = (1.0 - tx) * (1.0 - ty)
        w21 = tx * (1.0 - ty)
        w12 = (1.0 - tx) * ty
        w22 = tx * ty

        out_mags[i] = (
            band_grid[idx_met, idx_age] * w11
            + band_grid[idx_met + 1, idx_age] * w21
            + band_grid[idx_met, idx_age + 1] * w12
            + band_grid[idx_met + 1, idx_age + 1] * w22
        )

    return out_mags


@njit(fastmath=True)
def sph_project_2d(
    x: np.ndarray,
    y: np.ndarray,
    h_tng: np.ndarray,
    weights: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
) -> np.ndarray:
    """
    2D SPH projection using cubic spline kernel. This is the core of the gridding process, where we take particle positions, smoothing lengths, and weights
    (e.g., luminosities or masses) and project them onto a 2D pixel grid defined by x_edges and y_edges. Using numba to speed up the nested loops and avoid temporary arrays
    for maximum performance.

    Parameters:
    x : np.ndarray
        X coordinates of particles in kpc
    y : np.ndarray
        Y coordinates of particles in kpc
    h_tng : np.ndarray
        Smoothing lengths of particles in kpc (TNG defines Hsml as the full support radius, so we divide by 2 in the code to get the "h" used in the kernel)
    weights : np.ndarray
        Weights of particles (e.g., luminosities or masses)
    x_edges : np.ndarray
        Edges of the pixel grid in the X direction
    y_edges : np.ndarray
        Edges of the pixel grid in the Y direction

    Returns
    -------
    np.ndarray
        2D grid with projected values

    """

    nx = len(x_edges) - 1
    ny = len(y_edges) - 1
    grid = np.zeros((nx, ny))

    dx = x_edges[1] - x_edges[0]
    dy = y_edges[1] - y_edges[0]

    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])

    for i in range(len(x)):
        xi, yi = x[i], y[i]
        # TNG defines Hsml as the full support radius. Our math expects half of that.
        hi = h_tng[i] / 2.0
        wi = weights[i]

        if hi <= 0:
            continue

        # Handle Undersampling: If the particle is smaller than a pixel, dump it as a point mass
        if hi < 0.5 * min(dx, dy):
            ix = np.searchsorted(x_edges, xi) - 1
            iy = np.searchsorted(y_edges, yi) - 1

            if 0 <= ix < nx and 0 <= iy < ny:
                # Convert mass to surface density
                grid[ix, iy] += wi / (dx * dy)
            continue

        # Determine affected pixel range
        x_min = xi - 2 * hi
        x_max = xi + 2 * hi
        y_min = yi - 2 * hi
        y_max = yi + 2 * hi

        ix_min = np.searchsorted(x_centers, x_min, side='left')
        ix_max = np.searchsorted(x_centers, x_max, side='right')
        iy_min = np.searchsorted(y_centers, y_min, side='left')
        iy_max = np.searchsorted(y_centers, y_max, side='right')

        ix_min = max(ix_min, 0)
        ix_max = min(ix_max, nx)
        iy_min = max(iy_min, 0)
        iy_max = min(iy_max, ny)

        # The 2D cubic spline normalization factor
        norm = 10.0 / (7.0 * np.pi * hi**2)

        # Pure scalar nested loops (Zero temporary arrays!)
        for ix in range(ix_min, ix_max):
            dx_pos = x_centers[ix] - xi

            for iy in range(iy_min, iy_max):
                dy_pos = y_centers[iy] - yi

                r = np.sqrt(dx_pos**2 + dy_pos**2)
                q = r / hi

                if q < 1.0:
                    w_val = norm * (1.0 - 1.5 * q**2 + 0.75 * q**3)
                    grid[ix, iy] += wi * w_val
                elif q < 2.0:
                    w_val = norm * (0.25 * (2.0 - q) ** 3)
                    grid[ix, iy] += wi * w_val

    return grid


def ccm89_av_ratio(wavelength_microns: np.ndarray, Rv: float = 3.1):
    """
    Compute A(λ)/A(V) using the Cardelli, Clayton & Mathis (1989) extinction law.

    Parameters
    ----------
    wavelength_microns : numpy array
        Wavelength(s) in microns.
    Rv : float, optional
        Total-to-selective extinction ratio (default = 3.1 for the diffuse ISM).

    Returns
    -------
    A_lambda_over_Av : np.ndarray
        Extinction curve A(λ)/A(V).
    """
    x = 1.0 / np.array(wavelength_microns, ndmin=1)  # inverse microns
    a = np.zeros_like(x)
    b = np.zeros_like(x)

    if np.any((x < 0.3) | (x > 10)):
        print(
            'Warning: Wavelength out of range for CCM89 extinction law (0.1 - 3.3 microns^-1).'
        )

    # --- Infrared (0.3 ≤ x < 1.1 μm⁻¹) ---
    mask_ir = (x >= 0.3) & (x < 1.1)
    a[mask_ir] = 0.574 * x[mask_ir] ** 1.61
    b[mask_ir] = -0.527 * x[mask_ir] ** 1.61

    # --- Optical / NIR (1.1 ≤ x < 3.3 μm⁻¹) ---
    mask_opt = (x >= 1.1) & (x < 3.3)
    y = x[mask_opt] - 1.82
    a[mask_opt] = (
        1
        + 0.17699 * y
        - 0.50447 * y**2
        - 0.02427 * y**3
        + 0.72085 * y**4
        + 0.01979 * y**5
        - 0.77530 * y**6
        + 0.32999 * y**7
    )
    b[mask_opt] = (
        1.41338 * y
        + 2.28305 * y**2
        + 1.07233 * y**3
        - 5.38434 * y**4
        - 0.62251 * y**5
        + 5.30260 * y**6
        - 2.09002 * y**7
    )

    # --- Ultraviolet (3.3 ≤ x ≤ 8.0 μm⁻¹) ---
    mask_uv = (x >= 3.3) & (x <= 8.0)
    Fa = np.zeros_like(x[mask_uv])
    Fb = np.zeros_like(x[mask_uv])
    mask_uv_high = x[mask_uv] >= 5.9
    if np.any(mask_uv_high):
        xx = x[mask_uv][mask_uv_high] - 5.9
        Fa[mask_uv_high] = -0.04473 * xx**2 - 0.009779 * xx**3
        Fb[mask_uv_high] = 0.2130 * xx**2 + 0.1207 * xx**3

    a[mask_uv] = (
        1.752 - 0.316 * x[mask_uv] - 0.104 / ((x[mask_uv] - 4.67) ** 2 + 0.341) + Fa
    )
    b[mask_uv] = (
        -3.090 + 1.825 * x[mask_uv] + 1.206 / ((x[mask_uv] - 4.62) ** 2 + 0.263) + Fb
    )

    # --- Far-UV (8.0 < x ≤ 10 μm⁻¹) ---
    mask_fuv = (x > 8.0) & (x <= 10)
    a[mask_fuv] = (
        -1.073
        - 0.628 * (x[mask_fuv] - 8.0)
        + 0.137 * (x[mask_fuv] - 8.0) ** 2
        - 0.070 * (x[mask_fuv] - 8.0) ** 3
    )
    b[mask_fuv] = (
        13.670
        + 4.257 * (x[mask_fuv] - 8.0)
        - 0.420 * (x[mask_fuv] - 8.0) ** 2
        + 0.374 * (x[mask_fuv] - 8.0) ** 3
    )

    # Final extinction law
    A_lambda_over_Av = a + b / Rv

    return A_lambda_over_Av if np.ndim(wavelength_microns) else A_lambda_over_Av.item()


def DG2000_optical_depth(
    wavelength_microns: np.ndarray,
    N_H_cm2: np.ndarray,
    metallicity: np.ndarray,
    redshift: float,
    Z_sun: float = 0.02,
    beta: float = -0.5,
    Rv: float = 3.1,
) -> np.ndarray:
    """
    Compute the optical depth τ(λ) using the Devriendt & Guiderdoni 2000.

    Parameters
    ----------
    wavelength_microns : numpy array
        Wavelength(s) in microns.
    N_H_cm2 : numpy array
        Hydrogen column density in cm⁻².
    metallicity : numpy array
        Gas metallicity (mass fraction) of the galaxy.
    Z_sun : float, optional
        Solar metallicity (default = 0.02).
    beta : float, optional
        Scaling exponent for redshift dependence.
    Rv : float, optional
        Total-to-selective extinction ratio for the CCM89 law (default = 3.1).

    Returns
    -------
    lambda_tau_no_scatter : numpy array
        Optical depth τ(λ) without the scattering correction.
    """

    A_Lambda_over_Av = np.atleast_1d(ccm89_av_ratio(wavelength_microns, Rv=Rv))
    wavelength_angstroms = (
        np.array(wavelength_microns, ndmin=1) * 1e4
    )  # Convert microns to angstroms
    lambda_tau_no_scatter = np.zeros_like(wavelength_angstroms)

    # Power law index s for metallicity dependence from Guiderdoni & Rocca-Volmerange (1987)
    s = np.zeros_like(wavelength_angstroms)
    mask1 = wavelength_angstroms <= 2000
    s[mask1] = 1.35
    mask2 = wavelength_angstroms > 2000
    s[mask2] = 1.6

    # Calculating the optical depth τ(λ) without scattering correction
    lambda_tau_no_scatter = (
        A_Lambda_over_Av[None, None, :]
        * N_H_cm2[:, :, None]
        / (2.1e21)
        * (1.0 + redshift) ** beta
        * (metallicity[:, :, None] / Z_sun) ** s[None, None, :]
    )

    return lambda_tau_no_scatter


def CSK1994_scatter(
    wavelength_microns: np.ndarray, lambda_tau_no_scatter: np.ndarray
) -> np.ndarray:
    """
    Compute the optical depth τ(λ) using the Calzetti, Seaton & Krügel (1994) model after dust scattering.

    Parameters
    ----------
    wavelength_microns : float or array-like
        Wavelength(s) in microns.
    lambda_tau_no_scatter : numpy array
        Optical depth τ(λ) without scattering correction, as computed by DG2000_optical_depth().

    Returns
    -------
    lambda_tau_scatter : numpy array
        Optical depth τ(λ) corrected for dust scattering.
    """

    wavelength_angstroms = (
        np.array(wavelength_microns, ndmin=1) * 1e4
    )  # Convert microns to angstroms
    omega_Lambda = np.zeros_like(wavelength_angstroms)
    h_Lambda = np.zeros_like(wavelength_angstroms)

    # Calculating the albedo ω(λ) using the Calzetti et al. (1994) empirical fit
    omega_Lambda = np.zeros_like(wavelength_angstroms)
    mask1 = (wavelength_angstroms >= 1000) & (wavelength_angstroms <= 3460)
    y = np.log10(wavelength_angstroms[mask1])
    omega_Lambda[mask1] = 0.43 + 0.366 * (1.0 - np.exp(-((y - 3.0) ** 2) / 0.2))
    mask2 = (wavelength_angstroms > 3460) & (wavelength_angstroms <= 7000)
    y = np.log10(wavelength_angstroms[mask2])
    omega_Lambda[mask2] = -0.48 * y + 2.41

    # Table of omega_lambda from Natta & Panagia 1984 for wavelength between 0.7 to 4.48 microns

    if np.any((wavelength_angstroms > 7000) & (wavelength_angstroms <= 44800)):
        print(
            'Warning: Wavelength out of range for CSK1994 scattering model (0.1 - 0.7 microns). Using Natta & Panagia 1984 table values for longer wavelengths.'
        )
        omega_table_wavelengths = np.array(
            [0.7, 0.9, 1.25, 1.65, 2.2, 3.6, 4.48]
        )  # microns

        omega_table_values = np.array([0.56, 0.50, 0.37, 0.28, 0.22, 0.054, 0.0])

        omega_interp = interp1d(
            omega_table_wavelengths,
            omega_table_values,
            kind='cubic',
            bounds_error=True,
        )

        mask4 = (wavelength_angstroms > 7000) & (wavelength_angstroms <= 44800)
        omega_Lambda[mask4] = omega_interp(
            wavelength_angstroms[mask4] / 1e4
        )  # Convert back to microns for interpolation
    elif np.any((wavelength_angstroms > 44800) | (wavelength_angstroms < 1000)):
        print(
            'Warning: No valid model for wavelength > 4.48 microns or < 0.1 microns in CSK1994 scattering model. Setting albedo to zero.'
        )

    # Calculating the weighting factor h(λ) that accounts for the anistropy in scattering
    mask3 = (wavelength_angstroms >= 1200) & (wavelength_angstroms <= 7000)
    y = np.log10(wavelength_angstroms[mask3])
    h_Lambda[mask3] = 1.0 - 0.561 * np.exp(-(np.abs(y - 3.3112) ** 2.2) / 0.17)

    if np.any((wavelength_angstroms > 7000) & (wavelength_angstroms <= 18000)):
        print(
            'Warning: Applying extrapolation for h(λ) beyond 0.7 microns. Bruzual et al 1988 has a table for g(λ) up to 1.8 microns, the functional form seems to hold approximately.'
        )

        mask5 = (wavelength_angstroms > 7000) & (wavelength_angstroms <= 18000)
        y = np.log10(wavelength_angstroms[mask5])
        h_Lambda[mask5] = 1.0 - 0.561 * np.exp(-(np.abs(y - 3.3112) ** 2.2) / 0.17)
    elif np.any(wavelength_angstroms > 18000):
        print(
            'Warning: No valid model for h(λ) beyond 1.8 microns in CSK1994 scattering model. Setting h(λ) to one.'
        )
        mask6 = wavelength_angstroms > 18000
        h_Lambda[mask6] = 1.0
    elif np.any(wavelength_angstroms < 1200):
        print(
            'Warning: No valid model for h(λ) below 0.12 microns in CSK1994 scattering model. Setting h(λ) to zero.'
        )

    # Correcting the optical depth for scattering effects
    correction_factor = h_Lambda * np.sqrt(1.0 - omega_Lambda) + (1.0 - h_Lambda) * (
        1.0 - omega_Lambda
    )
    lambda_tau_scatter = correction_factor[None, None, :] * lambda_tau_no_scatter

    return lambda_tau_scatter
