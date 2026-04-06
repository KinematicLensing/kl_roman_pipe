"""
Generate mock observation data vectors from TNG50 galaxies.

This module provides functionality to convert TNG50 particle-level data
into pixelized 2D data vectors (images, velocity maps, etc.) with realistic
noise and observational effects.

## Coordinate Systems

**TNG Native Frame:**
- Origin: Simulation box coordinates (comoving kpc/h)
- Units: Comoving kpc/h (need conversion via Hubble parameter)
- Orientation: As simulated (intrinsic Inclination_star, Position_Angle_star)

**Observer Frame (after rendering):**
- Origin: Centered on galaxy (subhalo position or luminosity peak)
- Units: arcseconds (angular separation on sky)
- Orientation: Either native TNG or user-specified (theta_int, cosi)

## Coordinate Transformations

The pipeline uses transform.py's multi-plane system:
1. **obs plane**: Observed image (x0, y0 offset applied)
2. **cen plane**: Centered (no offset)
3. **source plane**: Unlensed (g1, g2 shear removed)
4. **gal plane**: Major/minor axis aligned (theta_int PA removed)
5. **disk plane**: Face-on view (inclination removed, cosi factor)

For TNG galaxies:
- Native orientation → transform_to_disk_plane → face-on
- Face-on → apply new params → custom observation

## Physical Units

**Stellar Data:**
- Luminosities: ~10^36-10^38 erg/s (band-dependent)
- Coordinates: Comoving kpc/h
- Velocities: km/s (peculiar velocities)

**Gas Data:**
- Masses: 10^4-10^6 Msun (typical resolution)
- Coordinates: Comoving kpc/h
- Velocities: km/s (peculiar velocities)

**Rendered Maps:**
- Intensity: Original luminosity units (not normalized)
- Velocity: km/s (line-of-sight component)
- Pixel scale: arcseconds per pixel

## Particle Types

- **Intensity maps**: Use stellar particles (PartType4) with photometric luminosities
- **Velocity maps**: Use gas particles (PartType0) with mass weighting
  - Gas represents ionized ISM (observable via Halpha, [OII], etc.)
  - Stellar kinematics would use PartType4 but less commonly observed

## TNG Inclination Convention

TNG50 outputs inclination in range [0, 180°]:
- 0° = face-on (viewing from +z)
- 90° = edge-on
- >90° = viewing from "below" (-z side)

We convert >90° to equivalent <90° view by:
- inc_new = 180° - inc_old
- PA_new = PA_old + 180°

This avoids negative cos(i) in projection math.

## Key Features

- Proper 3D rotation using angular momentum vectors (SubhaloSpin for stellar, computed L for gas)
- Separate rotation matrices for stellar and gas (preserves physical misalignments)
- Coordinate conversion from comoving kpc/h to arcsec with optional redshift scaling
- Cloud-in-Cell (CIC) gridding for smooth maps
- LOS velocity projection: v_LOS = v_y*sin(i) + v_z*cos(i) (matches arXiv:2201.00739)
- Shared noise utilities from noise.py

## Known Limitations

1. Sparse gas: Velocity maps may have empty pixels where no gas particles fall
2. SNR calibration: Less accurate for very large flux values (>10^9)
3. Absolute calibration: Luminosity units preserved but may need external validation
4. Gaussian noise: Poisson noise available but is not working well yet

## References

- TNG50 documentation: https://www.tng-project.org/
- Pillepich et al. (2019): "First results from the TNG50 simulation"
- Xu et al. (2022): arXiv:2201.00739 (kinematic lensing formalism)
"""

from typing import Dict, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
from numba import njit, prange
from scipy.integrate import simpson
import h5py
from scipy.interpolate import interp1d
from pathlib import Path

from ..parameters import ImagePars
from ..utils import build_map_grid_from_image_pars
from ..noise import add_noise


# TNG50 cosmology (Planck 2013)
TNG_COSMOLOGY = FlatLambdaCDM(
    H0=67.74 * u.km / u.s / u.Mpc, Om0=0.3089, Tcmb0=2.725 * u.K
)

# TNG50 snapshot 99 redshift (all galaxies in our dataset)
# TNG50_SNAPSHOT_99_REDSHIFT = 0.0108
TNG50_SNAPSHOT_99_REDSHIFT = 0.0

# Default data directory. This is needed to load the auxillary data files for the BC03 models, which are used for converting between
# luminosity and magnitude, and for distance conversions. The BC03 models are stored in a preprocessed HDF5 file that contains the
# SED and absolute magnitude grids for different bands, as well as the age and metallicity grids.
DEFAULT_DATA_DIR = Path(__file__).parent.parent.parent / "data" / "tng50"


def convert_tng_to_arcsec(
    coords_kpc: np.ndarray,
    target_redshift: Optional[float] = None,
) -> np.ndarray:
    """
    Convert TNG comoving coordinates to angular separation in arcsec.

    TNG coordinates are comoving kpc/h. This converts to physical kpc,
    then to arcsec using the angular diameter distance.

    Optionally rescale to a different redshift for realistic sub-arcsec observations.
    For TNG galaxies at snapshot 99, they are evolved to redshift zero. We need to
    use target_redshift to place them at higher redshift for Roman-like observations.

    Parameters
    ----------
    coords_kpc : np.ndarray
        Coordinates in comoving kpc/h (TNG native units)
    target_redshift : float, optional
        If provided, scale angular size to this redshift.
        Good values: 0.5-1.0 for Roman-like sub-arcsec resolution.
        If None, use native TNG redshift (~0.011).
    native_redshift : float, default=TNG50_SNAPSHOT_99_REDSHIFT
        Native redshift of TNG50 galaxies (snapshot 99)

    Returns
    -------
    coords_arcsec : np.ndarray
        Angular coordinates in arcsec
    """
    # Step 1: Load in the physical coordinates in kpc.
    coords_physical_kpc = coords_kpc
    # The output catalogue already has physical coordinates in kpc, so we skip the comoving to physical conversion step. The distance_mpc is also the physical distance,
    # so it is consistent with the physical coordinates.

    # Step 2: Convert physical kpc to angular separation at native redshift
    # theta = d_phys / D_A, where D_A is angular diameter distance
    arcsec_per_radian = 180.0 * 3600.0 / np.pi  # 180 deg/rad * 3600 arcsec/deg / pi

    # Step 3: Optionally rescale to target redshift using proper cosmology
    if target_redshift is not None:
        # Use astropy cosmology for accurate angular diameter distance scaling
        # Angular size scales as theta ∝ d_phys / D_A(z)

        # Convert angular diameter distance to target redshift in kpc since the galaxy size is in kpc.
        d_a_target = TNG_COSMOLOGY.angular_diameter_distance(target_redshift).to_value(
            u.kpc
        )

        # TNG galaxies at snapshot 99 are at redshift zero, we know the intrinsic size of the galaxy in kpc, so just need to divide it by the distance to the galaxy to
        # calculate the angular size in radians, then convert to arcseconds.

        coords_arcsec = coords_physical_kpc * arcsec_per_radian / d_a_target

    else:
        raise ValueError(
            "Target redshift must be provided and different from native redshift for angular size scaling if using the TNG50 sample at snapshot 99."
        )

    return coords_arcsec


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

        ix_min = np.searchsorted(x_centers, x_min, side="left")
        ix_max = np.searchsorted(x_centers, x_max, side="right")
        iy_min = np.searchsorted(y_centers, y_min, side="left")
        iy_max = np.searchsorted(y_centers, y_max, side="right")

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


@dataclass
class TNGRenderConfig:
    """
    Configuration for rendering TNG galaxies.

    Parameters
    ----------
    image_pars : ImagePars
        Image parameters (shape, pixel scale, etc.)
    band : str, default='r'
        Photometric band for intensity rendering ('g', 'r', 'i', 'u', 'z')
    use_dusted : bool, default=True
        Whether to use dust-attenuated luminosities
    center_on_peak : bool, default=True
        Whether to center on luminosity peak (False uses subhalo center)
    use_native_orientation : bool, default=True
        If True, use TNG's native inclination/PA. If False, apply transformations
        specified in pars (must undo native orientation first)
    pars : Optional[Dict], default=None
        Parameter dict for NEW orientation (if use_native_orientation=False).
        Expected keys: 'theta_int' (PA in rad), 'cosi', 'x0', 'y0', 'g1', 'g2'
        These define the desired final orientation after undoing TNG native orientation.
    use_cic_gridding : bool, default=True
        Use Cloud-in-Cell interpolation for smoother maps (vs nearest-grid-point)
    target_redshift : float, optional
        If provided, scale galaxy to this redshift for angular size.
        TNG galaxies are at z~0.01, appearing ~21 arcmin. Use target_redshift=0.5-1.0
        for Roman-like sub-arcsec observations. If None, use native z~0.011.
    preserve_gas_stellar_offset : bool, default=True
        If True (default), gas disk keeps its intrinsic misalignment relative to
        stellar disk. The user's (cosi, theta_int) refers to the STELLAR disk
        orientation, and gas will be tilted/rotated by its natural offset
        (~30-40° typical in TNG). This is the physically realistic behavior.
        If False, both gas and stellar are forced to the exact same orientation
        (useful for synthetic tests where perfect alignment is desired).
    apply_cosmological_dimming : bool, default=False
        If True, apply cosmological surface brightness dimming (Tolman effect)
        to intensity and SFR maps: I_obs = I_rest * (1+z)^-4. This accounts for
        the combined effect of photon energy redshift (1+z)^-1, photon rate
        redshift (1+z)^-1, angular size (1+z)^-2, giving total (1+z)^-4.
        Default False for backward compatibility and "truth" mock generation.
        Set True for mission-planning or realistic observation simulations.
    """

    image_pars: ImagePars
    band: str = "r"
    use_dusted: bool = True
    center_on_peak: bool = True
    use_native_orientation: bool = True
    pars: Optional[Dict] = None
    use_cic_gridding: bool = True
    target_redshift: Optional[float] = None
    preserve_gas_stellar_offset: bool = True
    apply_cosmological_dimming: bool = False
    psf: Optional[object] = None  # galsim.GSObject for PSF convolution

    def __post_init__(self):
        """Validate configuration parameters."""
        # Validate shear parameters if custom orientation is used
        if not self.use_native_orientation and self.pars is not None:
            g1 = self.pars.get("g1", 0.0)
            g2 = self.pars.get("g2", 0.0)
            gamma = np.sqrt(g1**2 + g2**2)
            weak_lensing_limit = 1.0
            if gamma >= weak_lensing_limit:
                raise ValueError(
                    f"Shear too large: |g|={gamma:.3f} >= {weak_lensing_limit}. "
                    f"Weak lensing requires |g| < {weak_lensing_limit}."
                )


class TNGDataVectorGenerator:
    """
    Generate 2D data vectors from TNG50 particle data.

    This class handles the projection of 3D particle data onto 2D pixel grids,
    applying geometric transformations, and adding realistic noise.

    Important: TNG galaxies come with intrinsic orientation (inclination, PA).
    The generator can either:
    1. Render at native orientation (use_native_orientation=True)
    2. Transform to new orientation (use_native_orientation=False, provide pars)
    """

    def __init__(self, galaxy_data: Dict[str, Dict], data_dir: Optional[Path] = None):
        """
        Initialize generator for a specific galaxy.

        Parameters
        ----------
        galaxy_data : dict
            Dictionary with keys 'gas', 'stellar', 'subhalo' containing
            the TNG data for one galaxy (from TNG50MockData.get_galaxy())
        """
        self.galaxy_data = galaxy_data
        self.stellar = galaxy_data.get("stellar")
        self.gas = galaxy_data.get("gas")
        self.subhalo = galaxy_data.get("subhalo")

        # Setting some constant parameters for the BC03 models. These are used for converting between luminosity and magnitude, and for distance conversions.
        self.L_sun = 3.826e33  # erg/s. The default value in galaxev
        self.csol = 2.99792458e10  # speed of light in cm/s
        self.H_mass = 1.6735575e-24  # g
        self.Msun_to_g = 1.98847e33  # solar mass in grams
        self.kpc_to_cm = 3.085677581491367e21  # kpc to cm
        self.chunk_size = 100000  # Number of particles to process in each chunk for interpolation. Reduce if you have memory issues, increase if you want faster processing (but more memory usage).
        self.Mpc = 3.085677581491367e24  # cm in a Mpc, used for distance conversions

        # AB absolute manitudes in different SDSS bands
        self.M_abs_sun = {
            "u": 6.39,
            "g": 5.11,
            "r": 4.65,
            "i": 4.53,
            "z": 4.50,
        }

        if self.stellar is None:
            raise ValueError("Stellar data required for data vector generation")

        if self.subhalo is None:
            raise ValueError(
                "Subhalo data required for coordinate conversion and orientation"
            )

        if data_dir is None:
            data_dir = DEFAULT_DATA_DIR
        output_path = data_dir / "SDSS_hr_stelib_stellar_photometrics.hdf5"
        filtdir = data_dir
        self.load_BC03_models(output_path, filtdir)

        # Store key properties
        self.distance_mpc = float(self.subhalo["DistanceMpc"])
        self.native_redshift = TNG50_SNAPSHOT_99_REDSHIFT  # Snapshot 99
        self.native_inclination_deg = float(self.subhalo["Inclination_star"])
        self.native_pa_deg = float(self.subhalo["Position_Angle_star"])

        # Store gas orientation for offset preservation
        self.native_gas_inclination_deg = float(self.subhalo["Inclination_gas"])
        self.native_gas_pa_deg = float(self.subhalo["Position_Angle_gas"])

        # Compute 3D rotation matrices to transform from TNG simulation frame
        # to face-on disk frame. We compute SEPARATE matrices for stellar and gas
        # because they can have different angular momentum directions!
        #
        # The disk frame has:
        #   - Z aligned with angular momentum (disk normal)
        #   - XY is the disk plane (face-on view)
        #
        # Stellar uses SubhaloSpin; Gas angular momentum computed from particles.
        self._compute_disk_rotation_matrices()

        # Handle TNG inclination convention for the 2D parameters
        # (used for applying custom orientations after transforming to disk plane)
        self.flipped_from_below = self.native_inclination_deg > 90
        if self.flipped_from_below:
            self.native_inclination_deg = 180 - self.native_inclination_deg
            self.native_pa_deg = (self.native_pa_deg + 180) % 360

        self.native_cosi = np.cos(np.radians(self.native_inclination_deg))
        self.native_pa_rad = np.radians(self.native_pa_deg)

    def load_BC03_models(
        self, output_path: Path, filtdir: Path, bands: set = {"g", "r", "i", "u", "z"}
    ):
        """Load BC03 SED and absolute magnitude grids from preprocessed HDF5 file. They are both interpolated in the same way (bilinear
        in log(age) and metallicity) to get the SED and absolute magnitude for each particle based on its age and metallicity. The SED grid
        has shape (N_metallicity, N_age, N_wave) and the absolute magnitude grid has shape (N_metallicity, N_age) for each band. The grids
        are stored as attributes of the class for later use in interpolation. Currently only supports SDSS g, r, i, u, z bands, but can be
        extended to other filters by adding more datasets to the HDF5 file and loading them here.

        Parameters
        ----------
        output_path : Path
            Path to the preprocessed HDF5 file containing BC03 models.
        filtname : Path
            Path to the filter file.
        bands : set, optional
            Set of bands to load, by default {"g", "r", "i", "u", "z"}.
        """

        self.absolute_mag_grid = {}
        with h5py.File(output_path, "r") as f:
            self.BC03_age_grid_logGr = f["LogAgeInGyr_bins"][:]  # log(age in Gyr) grid
            self.BC03_metallicity_grid = f["Metallicity_bins"][:]
            BC03_wave_ang = f["Wavelengths"][:]
            BC03_SED_grid = f["SEDs"][:].astype(
                np.float64
            )  # shape (N_metallicity, N_age, N_wave)
            self.BC03_N_age = f["N_LogAgeInGyr"][()]
            self.BC03_N_metallicity = f["N_Metallicity"][()]

            for band in bands:
                # Absolute magnitude grid for this band has shape (N_metallicity, N_age)
                self.absolute_mag_grid[band] = f["Magnitude_" + band][:]

        self.wavelength_SED_within_filter = {}
        self.SED_within_filter_grid = {}
        self.numerator_weights = {}
        self.denominator_scalar = {}
        for band in bands:
            # Currently limited to the SDSS filters, but could be extended in the future.
            filtname = filtdir / f"{band}_SDSS.res"
            f = open(filtname, "r")
            filt_wave, filt_t = np.loadtxt(f, unpack=True)
            f.close()

            filt_spline = interp1d(
                filt_wave, filt_t, kind="linear", bounds_error=False, fill_value=0.0
            )

            wmin_filt, wmax_filt = filt_wave[0], filt_wave[-1]

            cond_filt_rest = (BC03_wave_ang >= wmin_filt) & (BC03_wave_ang <= wmax_filt)

            response_rest = filt_spline(BC03_wave_ang[cond_filt_rest])
            nu_filter_rest = (
                self.csol * 1e8 / BC03_wave_ang[cond_filt_rest]
            )  # Convert from angstrom to cm for the frequency calculation

            response_rest = np.flipud(response_rest)
            nu_filter_rest = np.flipud(nu_filter_rest)

            self.wavelength_SED_within_filter[band] = BC03_wave_ang[
                cond_filt_rest
            ]  # Wavelengths in angstroms

            luminosity_density_erg = (
                BC03_SED_grid[:, :, cond_filt_rest] * self.L_sun * 1e8
            )  # Convert from L_sun/angstrom to erg/s/angstrom using the solar luminosity in erg/s and the conversion from angstrom to cm
            luminosity_density_nu = (
                luminosity_density_erg
                * (self.wavelength_SED_within_filter[band] * 1e-8) ** 2
                / self.csol
            )[
                ..., ::-1
            ]  # erg/s/Hz and swap the last axis to be in increasing frequency order

            self.SED_within_filter_grid[band] = luminosity_density_nu

            # 1. Trick SciPy into calculating the exact integration weights ONCE for the whole frequency range which is fixed.
            simpson_w = simpson(np.eye(len(nu_filter_rest)), x=nu_filter_rest, axis=-1)

            self.numerator_weights[band] = (response_rest / nu_filter_rest) * simpson_w

            self.denominator_scalar[band] = simpson(
                response_rest / nu_filter_rest, x=nu_filter_rest
            )

        # Using galaxev models to estimate the absolute magnitude from stellar mass and age

    def _estimate_magnitude_luminosity(
        self,
        coords_2d_stellar_rotate: np.ndarray,
        coords_2d_gas_rotate: np.ndarray,
        coords_2d_gas_center: np.ndarray,
        band: str,
    ):
        """Estimate the absolute magnitude and luminosity for a galaxy using galaxev models
        Parameters:
        -----------
        coords_2d_stellar_rotate : np.ndarray
            Coordinates of the stellar particles after applying the geometric transformations (shape: N_particles x 2) after rotating to the desired orientation.
            This is needed to calculate the dust attenuation for each particle based on its position in the galaxy and the gas distribution.

        coords_2d_gas_rotate : np.ndarray
            Rotated 2D coordinates of the gas particles. Shape (N_particles, 2).

        coords_2d_gas_center : np.ndarray
            Rotated 2D coordinates of the center of the gas distribution. Shape (2,).
        band : str
            Photometric band for which to estimate the magnitude and luminosity (e.g., 'g', 'r', 'i', 'u', 'z').
        """

        # Get stellar particles for this galaxy
        stellar_particles = self.stellar

        initial_masses = stellar_particles["GFM_InitialMass"]
        stellar_ages = stellar_particles["Stellar_age"]
        metallicities_org = stellar_particles["GFM_Metallicity"]
        ages = np.log10(
            stellar_ages
        )  # log10(Gyr). The time when the stars formed. This is consistent with the age grid from galaxev.

        # Calculate dust attenuation based on Xu et al 2017 model https://arxiv.org/abs/1610.07605
        N_H_cm2, x_edges, y_edges, metallicity_SPH = self.hydrogen_column_density(
            coords_2d_gas_rotate, coords_2d_gas_center
        )  # cm^-2

        # masses = stellar_particles["Masses"]

        # Check if there is any dust in the galaxy (i.e., any pixels with N_H > 0). If not, we can skip the dust calculation for all particles which saves a lot of time.
        has_dust = np.any(N_H_cm2 > 0)

        # The lower and upper bounds of the metallicity grid in the galaxev models. We will clip the metallicities of the stellar
        # particles to be within these bounds to avoid issues with interpolation. We will also print a warning if any particles are
        # out of bounds and how many.
        galaxev_metalicity_limit = [1e-04, 0.1]
        n_out_of_bounds = np.where(
            (metallicities_org < galaxev_metalicity_limit[0])
            | (metallicities_org > galaxev_metalicity_limit[1])
        )[0]

        if len(n_out_of_bounds) > 0:
            print(
                f"Warning: {len(n_out_of_bounds)} stellar particles out of {len(metallicities_org)} have metallicities out of the galaxev model bounds. They will be set to the nearest bound."
            )
            metallicities = np.clip(
                metallicities_org,
                np.nextafter(galaxev_metalicity_limit[0], np.inf),
                np.nextafter(galaxev_metalicity_limit[1], -np.inf),
            )  # The nextafter is to avoid hitting the exact bound which may cause issues with the interpolator
        else:
            metallicities = metallicities_org

        # -----------------------------------------------------------------------------
        # 1. BAND-INDEPENDENT SPATIAL DUST GRID (Calculated exactly once)
        # -----------------------------------------------------------------------------

        # Project coordinates and pre-calculate spatial indices for ALL particles
        x_indices_full = np.digitize(coords_2d_stellar_rotate[:, 0], x_edges) - 1
        y_indices_full = np.digitize(coords_2d_stellar_rotate[:, 1], y_edges) - 1

        # Only apply dust attenuation to particles that fall within the bounds of the N_H grid. This is a simple mask that we can apply to the final dust attenuation values
        # to set them to zero for particles outside the grid, which avoids any issues with indexing or NaNs.
        valid_mask_full = (
            (x_indices_full >= 0)
            & (x_indices_full < N_H_cm2.shape[0])
            & (y_indices_full >= 0)
            & (y_indices_full < N_H_cm2.shape[1])
        )

        def dust_scatter(
            wavelength_microns: np.ndarray,
            has_dust: bool | np.bool_,
            N_H_cm2: np.ndarray,
            metallicity_SPH: Optional[np.ndarray] = None,
        ) -> Optional[np.ndarray]:
            """Calculate the dust optical depth and scattering for a given wavelength grid. This function is called for each band,
            but the underlying N_H_cm2 grid is the same for all bands, so we only calculate it once per band.

            Parameters:
            -----------
            wavelength_microns : np.ndarray
                Wavelength grid in microns for the SED within the filter.
            has_dust : bool | np.bool_
                Whether there is any dust in the galaxy (i.e., any pixels with N_H > 0).
            N_H_cm2 : np.ndarray
                2D grid of hydrogen column density in cm^-2, shape (N_x, N_y).
            metallicity_SPH : np.ndarray, optional
                2D grid of metallicity from SPH projection, shape (N_x, N_y). If None, will use mass-averaged metallicity from galaxy data.

            Returns:
            --------
            np.ndarray or None
                3D grid of dust optical depth for the given wavelength grid, shape (N_x, N_y, N_wave), or None if no dust.
            """
            if has_dust:
                if metallicity_SPH is None:
                    # Using the mass averaged metallcity if the pixel metallicity is not specified.
                    gas_metallicity = self.subhalo[
                        "SubhaloGasMetallicity"
                    ] * np.ones_like(
                        N_H_cm2
                    )  # This is the mass-averaged metallicity of the gas in the galaxy, which is a single value for the whole galaxy. We will use this as a fallback if the
                    # pixel-level metallicity is not available. This is what's done in the Xu+17 paper.
                else:
                    gas_metallicity = metallicity_SPH  # Model C in this paper https://arxiv.org/abs/1707.03395. However, to minimize computation time, we use Xu+17's method
                    # to interpolate on a grid basis instead of per star basis. This means that all stars that fall within the same pixel will have the same dust attenuation,
                    # which is a good approximation and much faster to compute.
                if np.isnan(gas_metallicity).any():
                    gas_metallicity = np.zeros_like(gas_metallicity)

                # The galaxies are at snapshot 99, which corresponds to a redshift of 0.0. The dust extinction calculation will use this redshift not the target redshift,
                # because the dust is associated with the galaxy itself and should be calculated in the galaxy's rest frame.
                # The target redshift is only used for scaling the angular size of the galaxy, not for the dust calculation.
                tau_lambda_a = self.DG2000_optical_depth(
                    wavelength_microns,
                    N_H_cm2,
                    gas_metallicity,
                    self.subhalo["Redshift"],
                )
                tau_lambda = self.CSK1994_scatter(wavelength_microns, tau_lambda_a)
            else:
                tau_lambda = None

            return tau_lambda

        N_particles = len(initial_masses)
        # Fetch user-defined chunk size from config, default to 100,000
        chunk_size = self.chunk_size

        tau_lambda_all = {}
        W_dust_map_all = {}

        # Pre-allocate the master 1D output arrays (These use negligible memory)

        self.stellar["Absolute_Magnitude_rotate_" + band] = np.zeros(N_particles)
        self.stellar["Raw_Luminosity_rotate_" + band] = np.zeros(N_particles)
        self.stellar["Dusted_Luminosity_rotate_" + band] = np.zeros(N_particles)
        self.stellar["Dusted_Absolute_Magnitude_rotate_" + band] = np.zeros(N_particles)

        tau_lambda_all[band] = dust_scatter(
            self.wavelength_SED_within_filter[band]
            * 1e-4,  # Convert from angstrom to microns
            has_dust,
            N_H_cm2,
            metallicity_SPH=metallicity_SPH,
        )

        if has_dust and tau_lambda_all[band] is not None:
            # 1. Calculate A(x,y) = (1 - e^-tau) / tau globally for the unique spatial cells!
            # Shape: (N_x, N_y, 1851)
            A_map = np.ones_like(tau_lambda_all[band])
            valid_tau = tau_lambda_all[band] != 0

            # -expm1(-tau) is mathematically identical to (1 - e^-tau)
            A_map[valid_tau] = (
                -np.expm1(-tau_lambda_all[band][valid_tau])
                / tau_lambda_all[band][valid_tau]
            )

            # 2. Flip the wavelength axis to perfectly match the L_nu grid
            A_map_flipped = A_map[:, :, ::-1]

            # 3. Bake the Simpson weights directly into the spatial grid!
            W_dust_map_all[band] = A_map_flipped * self.numerator_weights[band]
        else:
            W_dust_map_all[band] = None

        # -----------------------------------------------------------------------------
        # 2. PARTICLE CHUNKING LOOP (Strict memory footprint constraint)
        # -----------------------------------------------------------------------------
        for start_idx in range(0, N_particles, chunk_size):
            end_idx = min(start_idx + chunk_size, N_particles)

            # Slice inputs for the current chunk
            met_chunk = metallicities[start_idx:end_idx]
            age_chunk = ages[start_idx:end_idx]
            mass_chunk = initial_masses[start_idx:end_idx]

            print(
                f"Processing particles {start_idx} to {end_idx} for galaxy ID {self.subhalo['SubhaloID']}"
            )

            lum_chunk = (
                numba_bilinear_3d(
                    met_chunk,
                    age_chunk,
                    self.BC03_metallicity_grid,
                    self.BC03_age_grid_logGr,
                    self.SED_within_filter_grid[band],
                )
                * mass_chunk[
                    :, None
                ]  # The BC03 SED grid is in units of luminosity per unit mass, so we multiply by the initial mass of each particle to get the
                # total luminosity for that particle. The [:, None] adds a new axis so that the multiplication broadcasts correctly to give a shape of
                # (N_particles_in_chunk, N_wave)
            )

            # Calulate the raw luminosity by integrating the SED over the filter response using the pre-computed weights. This is a simple dot product along the
            # wavelength axis, which is very fast and memory efficient.
            raw_lum_nu = (
                np.dot(lum_chunk, self.numerator_weights[band])
                / self.denominator_scalar[band]
            )

            dusted_lum_nu = raw_lum_nu.copy()
            if has_dust and W_dust_map_all[band] is not None:
                mask_chunk = valid_mask_full[start_idx:end_idx]
                valid_idx = np.where(mask_chunk)[0]
                if len(valid_idx) > 0:
                    # Get the x, y coordinates for only the particles inside the map
                    x_valid = x_indices_full[start_idx:end_idx][valid_idx]
                    y_valid = y_indices_full[start_idx:end_idx][valid_idx]

                    # Fetch the exact integration weights for these specific particles
                    # Shape: (N_valid_particles, 1851)
                    dust_weights = W_dust_map_all[band][x_valid, y_valid]

                    # np.einsum instantly multiplies and sums along the wavelength axis (axis 1)
                    # This operates in C and allocates ZERO intermediate arrays!
                    dusted_integrals = np.einsum(
                        "ij,ij->i", lum_chunk[valid_idx], dust_weights
                    )

                    dusted_lum_nu[valid_idx] = (
                        dusted_integrals / self.denominator_scalar[band]
                    )

            # Convert L_nu (erg/s/Hz) to Absolute Magnitude using the AB magnitude system definition
            # Wrap the log10 calculations to silence the zero-flux warnings
            with np.errstate(divide="ignore"):
                raw_lum_AB_mag = (
                    -2.5
                    * np.log10(
                        raw_lum_nu / (4.0 * np.pi * (10.0 * self.Mpc / 1e6) ** 2)
                    )
                    - 48.60
                )
                dusted_lum_AB_mag = (
                    -2.5
                    * np.log10(
                        dusted_lum_nu / (4.0 * np.pi * (10.0 * self.Mpc / 1e6) ** 2)
                    )
                    - 48.60
                )

            raw_lum = self.L_sun * 10.0 ** (
                -0.4 * (raw_lum_AB_mag - self.M_abs_sun[band])
            )
            dusted_lum = self.L_sun * 10.0 ** (
                -0.4 * (dusted_lum_AB_mag - self.M_abs_sun[band])
            )

            # Store directly into the pre-allocated master 1D arrays
            # self.stellar_data[galaxy_id]["AB_apparent_magnitude_" + band][
            #     start_idx:end_idx
            # ] = app_mag
            self.stellar["Absolute_Magnitude_rotate_" + band][
                start_idx:end_idx
            ] = raw_lum_AB_mag
            self.stellar["Raw_Luminosity_rotate_" + band][start_idx:end_idx] = raw_lum
            self.stellar["Dusted_Luminosity_rotate_" + band][
                start_idx:end_idx
            ] = dusted_lum
            self.stellar["Dusted_Absolute_Magnitude_rotate_" + band][
                start_idx:end_idx
            ] = dusted_lum_AB_mag

        # -----------------------------------------------------------------------------
        # 4. GALAXY TOTALS
        # -----------------------------------------------------------------------------

        total_raw = np.nansum(self.stellar["Raw_Luminosity_rotate_" + band])
        total_dusted = np.nansum(self.stellar["Dusted_Luminosity_rotate_" + band])

        self.subhalo["Raw_Luminosity_rotate_" + band] = total_raw
        self.subhalo["Dusted_Luminosity_rotate_" + band] = total_dusted

        self.subhalo["Absolute_Magnitude_rotate_" + band] = -2.5 * np.log10(
            np.nansum(
                10.0 ** (-0.4 * self.stellar["Absolute_Magnitude_rotate_" + band])
            )
        )

        self.subhalo["Dusted_Absolute_Magnitude_rotate_" + band] = -2.5 * np.log10(
            np.nansum(
                10.0
                ** (-0.4 * self.stellar["Dusted_Absolute_Magnitude_rotate_" + band])
            )
        )

        print(f"Total raw luminosity in band {band}: {total_raw} erg/s")
        print(f"Total dusted luminosity in band {band}: {total_dusted} erg/s")
        print(
            f"Dusted absolute magnitude in band {band}: {self.subhalo['Dusted_Absolute_Magnitude_rotate_' + band]} mag"
        )
        print(
            f"Raw absolute magnitude in band {band}: {self.subhalo['Absolute_Magnitude_rotate_' + band]} mag"
        )

    def hydrogen_column_density(
        self,
        coords_rotate_2d_gas: np.ndarray,
        coords_rotate_2d_gas_center: np.ndarray,
        max_r: float = 3.0,
        Ngrid: int = 100,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate hydrogen column density N_H (cm⁻²) within a given number of grid cells that cover the galaxy.
        The line-of-sight axis can be specified. The setting for max_r and Ngrid are taken from https://arxiv.org/abs/1610.07605.

        Parameters
        ----------
        coords_rotate_2d_gas : np.ndarray
            Rotated 2D coordinates of the gas particles. Shape (N_particles, 2).
        coords_rotate_2d_gas_center : np.ndarray
            Rotated 2D coordinates of the center of the gas distribution. Shape (2,).
        max_r : float, default=3.0
            Maximum radius in units of the stellar half-mass radius to define the grid size. The default is taken from Xu et al. 2017, which uses 3 times the stellar
            half-mass radius to ensure that we capture the majority of the gas and dust in the galaxy while avoiding excessive empty space that would increase
            computation time.
        Ngrid : int, default=100
            Number of grid cells along each axis (total pixels will be Ngrid x Ngrid). The default is taken from Xu et al. 2017, which uses a 100x100 grid to balance
            resolution and computational efficiency for their sample of TNG galaxies.
        Returns
        -------
        N_H_cm2 : np.ndarray
            Hydrogen column density in cm⁻².
        x_edges : np.ndarray
            Edges of the grid cells along the x-axis.
        y_edges : np.ndarray
            Edges of the grid cells along the y-axis.
        metallicity_sph : np.ndarray
            Smoothed metallicity distribution using SPH kernel.
        """
        # Getting the stellar particles within the same galaxy
        gas_particles = self.gas

        galaxy_data_single = self.subhalo

        R_eff = galaxy_data_single["SubhaloHalfmassRadStars"]  # kpc
        galaxy_pos = coords_rotate_2d_gas_center  # kpc

        # --- define grid boundaries ---
        grid_size = max_r * R_eff  # kpc

        x_edges = np.linspace(
            galaxy_pos[0] - grid_size, galaxy_pos[0] + grid_size, Ngrid + 1
        )
        y_edges = np.linspace(
            galaxy_pos[1] - grid_size, galaxy_pos[1] + grid_size, Ngrid + 1
        )

        if gas_particles["Coordinates"] is None:
            print(f"This galaxy has no gas particles.")
            N_H_cm2 = np.zeros((Ngrid, Ngrid))
            metallicity_sph = np.zeros((Ngrid, Ngrid))
        else:
            gas_coords = coords_rotate_2d_gas  # kpc
            gas_masses = gas_particles["Masses"]  # Msun
            GFM_Metals = gas_particles["GFM_Metals"][
                :, 0
            ]  # total hydrogen mass fraction, hydrogen is the first element
            NeutroHydrogenAbundance = gas_particles[
                "NeutralHydrogenAbundance"
            ]  # fraction of neutral hydrogen
            GFM_Metallicity = gas_particles[
                "GFM_Metallicity"
            ]  # total metallicity, which is the mass fraction of all elements heavier than helium
            # --- project coordinates based on line-of-sight axis ---

            # The neutral hydrogen mass, also the weight.
            weights = gas_masses * GFM_Metals * NeutroHydrogenAbundance
            # Using the SPH kernel to smooth the hydrogen mass distribution
            gas_mass_sph = sph_project_2d(
                gas_coords[:, 0],
                gas_coords[:, 1],
                gas_particles["SubfindHsml"],
                weights,
                x_edges,
                y_edges,
            )

            metallicity_sph = sph_project_2d(
                gas_coords[:, 0],
                gas_coords[:, 1],
                gas_particles["SubfindHsml"],
                GFM_Metallicity * weights,
                x_edges,
                y_edges,
            )

            metallicity_sph = np.divide(
                metallicity_sph,
                gas_mass_sph,
                out=np.zeros_like(metallicity_sph),
                where=gas_mass_sph != 0,
            )

            dx = np.diff(x_edges)[0]
            dy = np.diff(y_edges)[0]

            # Mask to find which particles are actually inside our image frame
            in_bounds = (
                (gas_coords[:, 0] >= x_edges[0])
                & (gas_coords[:, 0] <= x_edges[-1])
                & (gas_coords[:, 1] >= y_edges[0])
                & (gas_coords[:, 1] <= y_edges[-1])
            )

            # Only sum the input mass that belongs in the box!
            total_input = np.sum(weights[in_bounds])
            total_projected = np.sum(gas_mass_sph) * dx * dy  # Msun

            # SPH does not necessarily conserve mass in each pixel, but it should be close to the total input mass when summed over the whole grid.
            # We can check this and print a warning if it's not the case, which may indicate an issue with the SPH projection or the choice of grid parameters.
            if not np.isclose(total_input, total_projected, rtol=5e-2):
                print(
                    f"Warning: Mass conserved roughly. Input: {total_input}, Output: {total_projected}"
                )

            # --- compute per-pixel hydrogen mass in grams ---
            hydrogen_mass_g = gas_mass_sph * self.Msun_to_g  # g per pixel

            # --- pixel area in cm² ---
            dx = (x_edges[-1] - x_edges[0]) / Ngrid  # kpc
            dy = (y_edges[-1] - y_edges[0]) / Ngrid  # kpc
            area_cm2 = np.float64(dx * dy) * np.float64(self.kpc_to_cm) ** 2  # cm²

            # --- compute number column density ---
            N_H_cm2 = hydrogen_mass_g / (area_cm2 * self.H_mass)  # cm⁻²

        return N_H_cm2, x_edges, y_edges, metallicity_sph

    def ccm89_av_ratio(self, wavelength_microns: np.ndarray, Rv: float = 3.1):
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
                "Warning: Wavelength out of range for CCM89 extinction law (0.1 - 3.3 microns^-1)."
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
            -3.090
            + 1.825 * x[mask_uv]
            + 1.206 / ((x[mask_uv] - 4.62) ** 2 + 0.263)
            + Fb
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

        return (
            A_lambda_over_Av if np.ndim(wavelength_microns) else A_lambda_over_Av.item()
        )

    def DG2000_optical_depth(
        self,
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

        A_Lambda_over_Av = self.ccm89_av_ratio(wavelength_microns, Rv=Rv)
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
        self, wavelength_microns: np.ndarray, lambda_tau_no_scatter: np.ndarray
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
                "Warning: Wavelength out of range for CSK1994 scattering model (0.1 - 0.7 microns). Using Natta & Panagia 1984 table values for longer wavelengths."
            )
            omega_table_wavelengths = np.array(
                [0.7, 0.9, 1.25, 1.65, 2.2, 3.6, 4.48]
            )  # microns

            omega_table_values = np.array([0.56, 0.50, 0.37, 0.28, 0.22, 0.054, 0.0])

            omega_interp = interp1d(
                omega_table_wavelengths,
                omega_table_values,
                kind="cubic",
                bounds_error=True,
                fill_value=None,
            )

            mask4 = (wavelength_angstroms > 7000) & (wavelength_angstroms <= 44800)
            omega_Lambda[mask4] = omega_interp(
                wavelength_angstroms[mask4] / 1e4
            )  # Convert back to microns for interpolation
        elif np.any((wavelength_angstroms > 44800) | (wavelength_angstroms < 1000)):
            print(
                "Warning: No valid model for wavelength > 4.48 microns or < 0.1 microns in CSK1994 scattering model. Setting albedo to zero."
            )

        # Calculating the weighting factor h(λ) that accounts for the anistropy in scattering
        mask3 = (wavelength_angstroms >= 1200) & (wavelength_angstroms <= 7000)
        y = np.log10(wavelength_angstroms[mask3])
        h_Lambda[mask3] = 1.0 - 0.561 * np.exp(-(np.abs(y - 3.3112) ** 2.2) / 0.17)

        if np.any((wavelength_angstroms > 7000) & (wavelength_angstroms <= 18000)):
            print(
                "Warning: Applying extrapolation for h(λ) beyond 0.7 microns. Bruzual et al 1988 has a table for g(λ) up to 1.8 microns, the functional form seems to hold approximately."
            )

            mask5 = (wavelength_angstroms > 7000) & (wavelength_angstroms <= 18000)
            y = np.log10(wavelength_angstroms[mask5])
            h_Lambda[mask5] = 1.0 - 0.561 * np.exp(-(np.abs(y - 3.3112) ** 2.2) / 0.17)
        elif np.any(wavelength_angstroms > 18000):
            print(
                "Warning: No valid model for h(λ) beyond 1.8 microns in CSK1994 scattering model. Setting h(λ) to one."
            )
            mask6 = wavelength_angstroms > 18000
            h_Lambda[mask6] = 1.0
        elif np.any(wavelength_angstroms < 1200):
            print(
                "Warning: No valid model for h(λ) below 0.12 microns in CSK1994 scattering model. Setting h(λ) to zero."
            )

        # Correcting the optical depth for scattering effects
        correction_factor = h_Lambda * np.sqrt(1.0 - omega_Lambda) + (
            1.0 - h_Lambda
        ) * (1.0 - omega_Lambda)
        lambda_tau_scatter = correction_factor[None, None, :] * lambda_tau_no_scatter

        return lambda_tau_scatter

    def _compute_disk_rotation_matrices(self):
        """
        Compute rotation matrices to transform from TNG simulation frame to disk frames.

        Computes SEPARATE rotation matrices for stellar and gas because they can have
        significantly different angular momentum directions (37° difference is common!).

        - Stellar: Computed from stellar particle positions, velocities, and luminosities
        - Gas: Computed from gas particle positions, velocities, and masses

        Each matrix aligns the respective angular momentum with the +Z axis.

        Also computes and stores diagnostic information comparing our kinematic inclinations
        with the TNG catalog's morphological inclinations.
        """
        # === Stellar rotation matrix from stellar particles ===
        # Use luminosity-weighted angular momentum for stellar particles
        # This is more physical than SubhaloSpin (which includes all particle types)
        coords_stellar = self.stellar["Coordinates"]
        vel_stellar = self.stellar["Velocities"]

        # Use r-band luminosity as weights (or masses if not available)
        if "Dusted_Luminosity_r" in self.stellar:
            weights_stellar = self.stellar["Dusted_Luminosity_r"]
        elif "Masses" in self.stellar:
            weights_stellar = self.stellar["Masses"]
        else:
            weights_stellar = np.ones(len(coords_stellar))

        # Center on luminosity-weighted centroid
        center_stellar = np.average(coords_stellar, axis=0, weights=weights_stellar)
        coords_stellar_cen = coords_stellar - center_stellar

        # Subtract luminosity-weighted mean velocity
        vel_stellar_mean = np.average(vel_stellar, axis=0, weights=weights_stellar)
        vel_stellar_cen = vel_stellar - vel_stellar_mean

        # Compute luminosity-weighted angular momentum: L = Σ w_i * (r_i × v_i)
        L_stellar_vec = np.sum(
            weights_stellar[:, None] * np.cross(coords_stellar_cen, vel_stellar_cen),
            axis=0,
        )
        L_stellar = L_stellar_vec / np.linalg.norm(L_stellar_vec)
        self._R_to_disk_stellar = self._rodrigues_rotation(L_stellar)

        # Store stellar L direction for diagnostics
        self._L_stellar = L_stellar

        # Compute kinematic inclination from L_stellar (angle from +Z axis)
        # This is the "true" kinematic inclination
        self._kinematic_inc_stellar_deg = np.rad2deg(np.arccos(np.abs(L_stellar[2])))

        # === Gas rotation matrix from particle angular momentum ===
        if self.gas is not None and len(self.gas.get("Coordinates", [])) > 0:
            coords_gas = self.gas["Coordinates"]
            vel_gas = self.gas["Velocities"]
            masses_gas = self.gas["Masses"]

            # Center on mass-weighted centroid
            center_gas = np.average(coords_gas, axis=0, weights=masses_gas)
            coords_gas_cen = coords_gas - center_gas

            # Subtract mass-weighted mean velocity
            vel_gas_mean = np.average(vel_gas, axis=0, weights=masses_gas)
            vel_gas_cen = vel_gas - vel_gas_mean

            # Mass-weighted angular momentum: L = Σ m_i * (r_i × v_i)
            L_gas_vec = np.sum(
                masses_gas[:, None] * np.cross(coords_gas_cen, vel_gas_cen), axis=0
            )
            L_gas = L_gas_vec / np.linalg.norm(L_gas_vec)

            self._R_to_disk_gas = self._rodrigues_rotation(L_gas)
            self._L_gas = L_gas

            # Store angle between stellar and gas L for diagnostics
            self._gas_stellar_L_angle_deg = np.rad2deg(
                np.arccos(np.clip(np.dot(L_stellar, L_gas), -1, 1))
            )

            # Compute kinematic inclination for gas
            self._kinematic_inc_gas_deg = np.rad2deg(np.arccos(np.abs(L_gas[2])))
        else:
            # No gas data - use stellar rotation
            self._R_to_disk_gas = self._R_to_disk_stellar.copy()
            self._L_gas = L_stellar.copy()
            self._gas_stellar_L_angle_deg = 0.0
            self._kinematic_inc_gas_deg = self._kinematic_inc_stellar_deg

        # For backwards compatibility, _R_to_disk is the stellar one
        self._R_to_disk = self._R_to_disk_stellar

        # Diagnostic: compare catalog morphological inclination with our kinematic inclination
        self._catalog_vs_kinematic_offset_deg = abs(
            self.native_inclination_deg - self._kinematic_inc_stellar_deg
        )

    def _rodrigues_rotation(self, L: np.ndarray) -> np.ndarray:
        """
        Compute rotation matrix to align vector L with +Z axis using Rodrigues formula.

        Parameters
        ----------
        L : np.ndarray
            Unit vector to align with Z, shape (3,)

        Returns
        -------
        np.ndarray
            3x3 rotation matrix
        """
        z_axis = np.array([0.0, 0.0, 1.0])
        cos_angle = np.dot(L, z_axis)

        # Handle edge cases where L is already aligned with Z
        # Tolerance: 1 - cos(0.01°) ≈ 0.0001, so use 0.9999 to catch angles < 0.01°
        alignment_tolerance = 1.0 - np.cos(
            np.radians(0.01)
        )  # Near-perfect alignment threshold
        if np.abs(cos_angle) > (1.0 - alignment_tolerance):
            if cos_angle < 0:
                # L points in -Z, flip Z
                return np.diag([1.0, 1.0, -1.0])
            else:
                # Already aligned
                return np.eye(3)

        # Rodrigues formula: rotate around axis = L × Z by angle = arccos(L · Z)
        angle = np.arccos(np.clip(cos_angle, -1, 1))
        axis = np.cross(L, z_axis)
        axis = axis / np.linalg.norm(axis)

        # Skew-symmetric cross-product matrix
        K = np.array(
            [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
        )

        # R = I + sin(θ)K + (1-cos(θ))K²
        return np.eye(3) + np.sin(angle) * K + (1 - cos_angle) * (K @ K)

    def _get_luminosity_key(
        self, band: str, use_dusted: bool, rotate: bool = False
    ) -> str:
        """Get the appropriate luminosity key for the specified band."""
        prefix = "Dusted_Luminosity" if use_dusted else "Raw_Luminosity"
        if rotate and use_dusted:
            prefix += "_rotate"
        key = f"{prefix}_{band}"
        # Use the dusted luminosity after the rotation since the dust attenuation depends on the line-of-sight, so the luminosity values will change after the rotation.
        # The raw luminosity is unaffected by the rotation since it is intrinsic to the stellar population, so we can use the same raw luminosity values before and
        # after the rotation.
        # Validate key exists
        if key not in self.stellar:
            available = [k for k in self.stellar.keys() if "Luminosity" in k]
            raise KeyError(
                f"Luminosity key '{key}' not found in stellar data. "
                f"Available bands: {available}"
            )

        return key

    def _get_reference_center(
        self, center_on_peak: bool, band: str = "r", use_dusted: bool = True
    ) -> np.ndarray:
        """
        Get reference center for coordinate system (shared by intensity and velocity).

        This ensures intensity and velocity maps show the same patch of sky,
        making any physical offsets between stellar/gas distributions visible.

        Parameters
        ----------
        center_on_peak : bool
            If True, use stellar luminosity peak; if False, use subhalo position
        band : str
            Photometric band for luminosity (if center_on_peak=True)
        use_dusted : bool
            Use dust-attenuated luminosity (if center_on_peak=True)

        Returns
        -------
        np.ndarray
            Reference center in TNG comoving coordinates, shape (3,)
        """
        if center_on_peak:
            # Use stellar luminosity-weighted centroid as reference
            lum_key = self._get_luminosity_key(band, use_dusted)
            luminosities = self.stellar[lum_key]
            stellar_coords = self.stellar["Coordinates"]
            center = np.average(stellar_coords, axis=0, weights=luminosities)
        else:
            # Use subhalo position
            center = np.array(
                [
                    self.subhalo["SubhaloPosX"],
                    self.subhalo["SubhaloPosY"],
                    self.subhalo["SubhaloPosZ"],
                ]
            )
        return center

    def _center_coordinates(self, coords: np.ndarray, center: np.ndarray) -> np.ndarray:
        """
        Center coordinates on a given reference point.

        Parameters
        ----------
        coords : np.ndarray
            Particle coordinates, shape (n_particles, 3) in TNG units (ckpc/h)
        center : np.ndarray
            Reference center, shape (3,) in same units as coords

        Returns
        -------
        np.ndarray
            Centered coordinates in same units as input
        """
        return coords - center

    def _undo_native_orientation(
        self, coords: np.ndarray, velocities: np.ndarray, particle_type: str = "stellar"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform from TNG's native obs frame to face-on disk plane.

        This performs a proper 3D rotation to undo the TNG intrinsic inclination
        and PA to get a face-on view where the disk lies in the XY plane.

        Uses the pre-computed rotation matrix derived from the angular momentum
        vector to transform particles into the disk frame where Z is aligned
        with angular momentum.

        IMPORTANT: Stellar and gas have DIFFERENT angular momentum directions
        (can differ by 30-40°!), so we use different rotation matrices for each.

        Parameters
        ----------
        coords : np.ndarray
            Centered coordinates in TNG simulation frame, shape (n_particles, 3)
        velocities : np.ndarray
            Particle velocities in TNG simulation frame, shape (n_particles, 3)
        particle_type : str, default='stellar'
            Which particle type's angular momentum to use: 'stellar' or 'gas'

        Returns
        -------
        coords_disk : np.ndarray
            Coordinates in face-on disk plane, shape (n_particles, 3)
        velocities_disk : np.ndarray
            Velocities in face-on disk plane, shape (n_particles, 3)
        """
        # Select the appropriate rotation matrix
        if particle_type == "gas":
            R = self._R_to_disk_gas
        else:
            R = self._R_to_disk_stellar

        # Apply the rotation matrix: aligns disk normal (angular momentum) with +Z
        coords_disk = (R @ coords.T).T
        velocities_disk = (R @ velocities.T).T

        return coords_disk, velocities_disk

    def _apply_new_orientation(
        self, coords_disk: np.ndarray, velocities_disk: np.ndarray, pars: Dict
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Transform from face-on disk plane to new obs frame with desired orientation.

        Uses proper 3D rotations for inclination to preserve realistic galaxy structure
        and thickness at all viewing angles. This is critical for realistic edge-on views.

        Physical procedure:
        1. Start with face-on disk frame (disk in xy-plane, disk normal along z)
        2. Apply 3D rotation around x-axis by inclination angle (tilts the disk)
        3. Project tilted 3D structure onto observer's x-y plane
        4. Apply PA rotation in the sky plane
        5. Apply weak lensing shear and position offsets

        Parameters
        ----------
        coords_disk : np.ndarray
            Coordinates in face-on disk plane, shape (n_particles, 3)
        velocities_disk : np.ndarray
            Velocities in face-on disk plane, shape (n_particles, 3)
        pars : dict
            Desired orientation parameters: theta_int, cosi, x0, y0, g1, g2

        Returns
        -------
        coords_2d : np.ndarray
            2D projected coordinates in new obs frame, shape (n_particles, 2)
        vel_los : np.ndarray
            Line-of-sight velocities in new obs frame, shape (n_particles,)
        coords_3d_obs : np.ndarray
            3D coordinates in new obs frame (for diagnostics), shape (n_particles, 3)
        """
        # Extract parameters with defaults
        cosi = pars.get("cosi", 1.0)
        theta_int = pars.get("theta_int", 0.0)
        x0 = pars.get("x0", 0.0)
        y0 = pars.get("y0", 0.0)
        g1 = pars.get("g1", 0.0)
        g2 = pars.get("g2", 0.0)

        # Validate shear parameters
        gamma = np.sqrt(g1**2 + g2**2)
        weak_lensing_limit = 1.0
        if gamma >= weak_lensing_limit:
            raise ValueError(
                f"Shear too large: |g|={gamma:.3f} >= {weak_lensing_limit}. "
                f"Weak lensing requires |g| < {weak_lensing_limit}."
            )

        # ===================================================================
        # Step 1: Apply 3D rotation for inclination
        # ===================================================================
        # Rotate particle distribution around x-axis by angle = arccos(cosi)
        # This tilts the disk from face-on (angle=0) to edge-on (angle=90°)
        # Preserves realistic 3D thickness and structure at all viewing angles
        #
        # Rotation matrix around x-axis:
        #   R_x(θ) = [[1,    0,         0     ],
        #             [0, cos(θ), -sin(θ)],
        #             [0, sin(θ),  cos(θ)]]

        angle = np.arccos(np.clip(cosi, -1, 1))  # Clip for numerical stability
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)

        # Build rotation matrix
        R_incl = np.array(
            [[1.0, 0.0, 0.0], [0.0, cos_angle, -sin_angle], [0.0, sin_angle, cos_angle]]
        )

        # Apply 3D rotation to coordinates and velocities
        # This properly transforms the full 3D particle distribution
        coords_inclined = (R_incl @ coords_disk.T).T
        velocities_inclined = (R_incl @ velocities_disk.T).T

        # ===================================================================
        # Step 2: Project onto observer's sky plane and apply PA rotation
        # ===================================================================
        # After inclination tilt:
        #   - x is horizontal on sky (along major axis before PA rotation)
        #   - y is vertical on sky (foreshortened by inclination)
        #   - z is depth along line-of-sight

        # Extract sky-plane coordinates and depth
        x_gal = coords_inclined[:, 0]
        y_gal = coords_inclined[:, 1]
        z_los = coords_inclined[:, 2]  # Depth (used for diagnostics)

        # Apply PA rotation in the sky plane (rotate around observer's LOS)
        cos_pa = np.cos(theta_int)
        sin_pa = np.sin(theta_int)
        x_source = x_gal * cos_pa - y_gal * sin_pa
        y_source = x_gal * sin_pa + y_gal * cos_pa

        # ===================================================================
        # Step 3: Apply weak lensing shear
        # ===================================================================
        if g1 != 0 or g2 != 0:
            # source→cen = A^{-1} = norm * [[1+g1, g2], [g2, 1-g1]]
            norm = 1.0 / (1.0 - (g1**2 + g2**2))
            x_cen = norm * ((1.0 + g1) * x_source + g2 * y_source)
            y_cen = norm * (g2 * x_source + (1.0 - g1) * y_source)
        else:
            x_cen = x_source
            y_cen = y_source

        # ===================================================================
        # Step 4: Apply position offsets
        # ===================================================================
        x_obs = x_cen + x0
        y_obs = y_cen + y0

        coords_2d = np.column_stack([x_obs, y_obs])

        # ===================================================================
        # Step 5: Compute line-of-sight velocity
        # ===================================================================
        # After the 3D inclination rotation, the observer's LOS is along the
        # z-axis of the rotated frame. So LOS velocity is simply the z-component
        # of the rotated velocity vector.
        #
        # This is equivalent to the formula:
        #   vel_los = v_y * sin(i) + v_z * cos(i)
        # where v_y, v_z are in the original disk frame.
        vel_los = velocities_inclined[:, 2]

        # Store 3D coords for diagnostics (includes depth information)
        coords_3d_obs = np.column_stack([x_obs, y_obs, z_los])

        return coords_2d, vel_los, coords_3d_obs

    def _grid_particles_cic(
        self,
        coords_arcsec: np.ndarray,
        values: np.ndarray,
        weights: np.ndarray,
        image_pars: ImagePars,
        mode: str = "weighted_average",
    ) -> np.ndarray:
        """
        Grid particles using Cloud-in-Cell (CIC) interpolation.

        CIC distributes each particle to the 4 nearest grid points
        with bilinear weights, producing smoother maps than NGP.

        Parameters
        ----------
        coords_arcsec : np.ndarray
            2D particle coordinates in arcsec, shape (n_particles, 2)
        values : np.ndarray
            Values to grid (e.g., luminosities or velocities), shape (n_particles,)
        weights : np.ndarray
            Weights for gridding (e.g., luminosities), shape (n_particles,)
        image_pars : ImagePars
            Image parameters defining pixel grid
        mode : str, default='weighted_average'
            'sum': Sum weighted values (for intensity)
            'weighted_average': Weighted average of values (for velocity)

        Returns
        -------
        np.ndarray
            Gridded 2D map, shape determined by image_pars
        """
        # Get grid properties
        X, Y = build_map_grid_from_image_pars(image_pars, unit="arcsec", centered=True)
        pixel_size = image_pars.pixel_scale
        Ny, Nx = (
            image_pars.shape
            if image_pars.indexing == "ij"
            else (image_pars.shape[1], image_pars.shape[0])
        )

        # Convert to pixel coordinates (continuous)
        x_min, y_min = X.min(), Y.min()
        x_pix = (coords_arcsec[:, 0] - x_min) / pixel_size
        y_pix = (coords_arcsec[:, 1] - y_min) / pixel_size

        # Initialize arrays
        weighted_sum = np.zeros((Ny, Nx))
        weight_sum = np.zeros((Ny, Nx)) if mode == "weighted_average" else None

        # Cloud-in-Cell: distribute to 4 nearest pixels
        x_floor = np.floor(x_pix).astype(int)
        y_floor = np.floor(y_pix).astype(int)

        # Fractional distances
        dx = x_pix - x_floor
        dy = y_pix - y_floor

        # Bilinear weights for 4 corners
        w00 = (1 - dx) * (1 - dy)
        w10 = dx * (1 - dy)
        w01 = (1 - dx) * dy
        w11 = dx * dy

        # Vectorized distribution using np.add.at
        # This is MUCH faster than Python loop
        for dx_i, dy_i, w_corner in [
            (0, 0, w00),
            (1, 0, w10),
            (0, 1, w01),
            (1, 1, w11),
        ]:
            xi = x_floor + dx_i
            yi = y_floor + dy_i

            # Bounds mask
            valid = (xi >= 0) & (xi < Nx) & (yi >= 0) & (yi < Ny)

            if valid.any():
                xi_valid = xi[valid]
                yi_valid = yi[valid]
                w_valid = w_corner[valid]
                values_valid = values[valid]
                weights_valid = weights[valid]

                # Add to grid using np.add.at (handles duplicate indices)
                np.add.at(
                    weighted_sum,
                    (yi_valid, xi_valid),
                    values_valid * weights_valid * w_valid,
                )

                if mode == "weighted_average":
                    np.add.at(weight_sum, (yi_valid, xi_valid), weights_valid * w_valid)

        # Compute result based on mode
        if mode == "sum":
            result = weighted_sum
        elif mode == "weighted_average":
            mask = weight_sum > 0
            result = np.zeros((Ny, Nx))
            result[mask] = weighted_sum[mask] / weight_sum[mask]
        else:
            raise ValueError(f"Unknown mode: {mode}")

        return result

    def _grid_particles_ngp(
        self,
        coords_arcsec: np.ndarray,
        values: np.ndarray,
        weights: np.ndarray,
        image_pars: ImagePars,
        mode: str = "weighted_average",
    ) -> np.ndarray:
        """
        Grid particles using nearest-grid-point (NGP) assignment.

        Faster than CIC but produces noisier maps.

        Parameters
        ----------
        coords_arcsec : np.ndarray
            2D particle coordinates in arcsec, shape (n_particles, 2)
        values : np.ndarray
            Values to grid (e.g., luminosities or velocities), shape (n_particles,)
        weights : np.ndarray
            Weights for gridding (e.g., luminosities), shape (n_particles,)
        image_pars : ImagePars
            Image parameters defining pixel grid
        mode : str, default='weighted_average'
            'sum': Sum weighted values (for intensity)
            'weighted_average': Weighted average of values (for velocity)

        Returns
        -------
        np.ndarray
            Gridded 2D map, shape determined by image_pars
        """
        X, Y = build_map_grid_from_image_pars(image_pars, unit="arcsec", centered=True)
        pixel_size = image_pars.pixel_scale
        Ny, Nx = (
            image_pars.shape
            if image_pars.indexing == "ij"
            else (image_pars.shape[1], image_pars.shape[0])
        )

        # Convert to pixel indices
        x_pix = (coords_arcsec[:, 0] - X.min()) / pixel_size
        y_pix = (coords_arcsec[:, 1] - Y.min()) / pixel_size

        # Use histogram2d for NGP
        H_weighted, _, _ = np.histogram2d(
            y_pix,
            x_pix,
            bins=[Ny, Nx],
            range=[[0, Ny], [0, Nx]],
            weights=values * weights,
        )

        if mode == "sum":
            result = H_weighted
        elif mode == "weighted_average":
            H_weights, _, _ = np.histogram2d(
                y_pix, x_pix, bins=[Ny, Nx], range=[[0, Ny], [0, Nx]], weights=weights
            )
            # Weighted average
            mask = H_weights > 0
            result = np.zeros((Ny, Nx))
            result[mask] = H_weighted[mask] / H_weights[mask]
        else:
            raise ValueError(f"Unknown mode: {mode}")

        return result

    def generate_intensity_map(
        self,
        config: TNGRenderConfig,
        snr: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate 2D intensity map from stellar particles.

        Parameters
        ----------
        config : TNGRenderConfig
            Rendering configuration
        snr : float, optional
            Signal-to-noise ratio for noise addition. If None, no noise added.
        seed : int, optional
            Random seed for noise generation

        Returns
        -------
        intensity : np.ndarray
            2D intensity map in luminosity units, shape from image_pars
        variance : np.ndarray
            Variance map, shape from image_pars
        """

        # Get coordinates
        coords = self.stellar["Coordinates"].copy()

        # Get reference center (shared with velocity map for consistent FOV)
        center = self._get_reference_center(
            config.center_on_peak, config.band, config.use_dusted
        )
        coords_centered = self._center_coordinates(coords, center)

        # Handle orientation (stellar)
        if config.use_native_orientation:
            # Simple projection at native TNG orientation
            coords_2d = coords_centered[:, :2]  # Just drop z
        else:
            # Transform stellar to specified orientation
            if config.pars is None:
                raise ValueError(
                    "pars must be provided when use_native_orientation=False"
                )

            # Undo native orientation to get face-on stellar disk
            coords_disk, _ = self._undo_native_orientation(
                coords_centered, np.zeros_like(coords_centered), particle_type="stellar"
            )

            # Apply new stellar orientation from pars
            coords_2d, _, _ = self._apply_new_orientation(
                coords_disk, np.zeros_like(coords_disk), config.pars
            )

            if config.use_dusted:
                # If using the dusted luminosity, we also need to apply the same rotation for the gas particles.
                # This is because the dust attenuation depends on the line-of-sight, so the luminosity values will change after the rotation.
                # We apply the same rotation to the gas particles to ensure that the dust attenuation is consistent with the new orientation of the stellar particles.
                coords_gas = self.gas["Coordinates"].copy()
                center_gas = self._get_reference_center(
                    config.center_on_peak, config.band, config.use_dusted
                )
                coords_gas_centered = self._center_coordinates(coords_gas, center_gas)
                coords_gas_disk, _ = self._undo_native_orientation(
                    coords_gas_centered,
                    np.zeros_like(coords_gas_centered),
                    particle_type="gas",
                )
                coords_gas_2d, _, _ = self._apply_new_orientation(
                    coords_gas_disk, np.zeros_like(coords_gas_disk), config.pars
                )

                # The 2d coordinates have already subtracted the center, so we can just pass in np.array([0.0, 0.0]) for the center in the luminosity estimation.
                self._estimate_magnitude_luminosity(
                    coords_2d, coords_gas_2d, np.array([0.0, 0.0]), config.band
                )

        # Get luminosity weights
        lum_key = self._get_luminosity_key(
            config.band, config.use_dusted, rotate=not config.use_native_orientation
        )
        luminosities = self.stellar[lum_key]

        # Normalize to avoid overflow (will rescale back later)
        lum_scale = luminosities.max()
        luminosities_norm = luminosities / lum_scale

        # Convert to arcsec (with optional redshift scaling)
        coords_arcsec = convert_tng_to_arcsec(
            coords_2d, target_redshift=config.target_redshift
        )

        print(
            config.target_redshift,
            np.min(coords_2d),
            np.max(coords_2d),
            np.min(coords_arcsec),
            np.max(coords_arcsec),
        )

        # Grid luminosities (sum, not weighted average)
        if config.use_cic_gridding:
            intensity = self._grid_particles_cic(
                coords_arcsec,
                luminosities_norm,
                np.ones_like(luminosities_norm),
                config.image_pars,
                mode="sum",
            )
        else:
            intensity = self._grid_particles_ngp(
                coords_arcsec,
                luminosities_norm,
                np.ones_like(luminosities_norm),
                config.image_pars,
                mode="sum",
            )

        # Rescale back to original units
        intensity *= lum_scale

        # Apply cosmological surface brightness dimming if requested
        if config.apply_cosmological_dimming:
            # Tolman dimming: I_obs = I_rest * (1+z)^-4
            # Accounts for: photon energy (1+z)^-1, rate (1+z)^-1, area (1+z)^-2
            z_native = self.native_redshift
            z_target = config.target_redshift if config.target_redshift else z_native
            dimming_factor = ((1.0 + z_native) / (1.0 + z_target)) ** 4
            intensity *= dimming_factor

        # Apply PSF convolution if requested
        if config.psf is not None:
            from ..psf import gsobj_to_kernel, convolve_fft_numpy

            kernel, padded_shape = gsobj_to_kernel(
                config.psf, image_pars=config.image_pars
            )
            intensity = convolve_fft_numpy(intensity, kernel, padded_shape)

        # Add noise if requested
        if snr is not None:
            # For TNG: use Gaussian noise only (flux already in physical units)
            intensity, variance = add_noise(
                intensity, target_snr=snr, include_poisson=False, seed=seed
            )
        else:
            variance = np.zeros_like(intensity)

        return intensity, variance

    def generate_velocity_map(
        self,
        config: TNGRenderConfig,
        snr: Optional[float] = None,
        seed: Optional[int] = None,
        intensity_map: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate 2D line-of-sight velocity map from gas particles.

        Uses gas particle coordinates and velocities for kinematics. Subtracts systemic
        velocity to center the velocity field at v=0 in the galaxy rest frame.

        Parameters
        ----------
        config : TNGRenderConfig
            Rendering configuration
        snr : float, optional
            Signal-to-noise ratio for noise addition. If None, no noise added.
        seed : int, optional
            Random seed for noise generation

        Returns
        -------
        velocity : np.ndarray
            2D velocity map in km/s (relative to systemic), shape from image_pars
        variance : np.ndarray
            Variance map, shape from image_pars
        """
        # Get gas particle data (truth for kinematics)
        if self.gas is None:
            raise ValueError("Gas data required for velocity map generation")

        coords = self.gas["Coordinates"].copy()
        velocities = self.gas["Velocities"].copy()
        masses = self.gas["Masses"].copy()

        # Subtract systemic velocity using mass-weighted mean of inner region
        # Use only inner particles to avoid bias from distant satellites/CGM
        # This centers the velocity field at v=0 in the galaxy rest frame
        center = self._get_reference_center(
            config.center_on_peak, config.band, config.use_dusted
        )
        coords_cen = self._center_coordinates(coords, center)
        radii = np.sqrt(np.sum(coords_cen**2, axis=1))
        inner_mask = radii < np.percentile(radii, 50)  # Inner 50% by radius
        if inner_mask.sum() > 0:
            # Mass-weighted mean of inner particles
            v_systemic = np.average(
                velocities[inner_mask], axis=0, weights=masses[inner_mask]
            )
        else:
            # Fallback to simple median if no inner particles (shouldn't happen)
            v_systemic = np.median(velocities, axis=0)
        velocities -= v_systemic

        # Normalize to avoid overflow
        mass_scale = masses.max()
        masses_norm = masses / mass_scale

        # Reuse centered coordinates from above (same reference as intensity)
        coords_centered = coords_cen

        # Handle orientation (gas with stellar-relative offset preservation)
        if config.use_native_orientation:
            # At native TNG orientation, LOS velocity is z-component
            coords_2d = coords_centered[:, :2]
            vel_los = velocities[:, 2]
        else:
            # Transform gas to the requested orientation
            if config.pars is None:
                raise ValueError(
                    "pars must be provided when use_native_orientation=False"
                )

            # Choose which rotation matrix to use for gas particles:
            # - preserve_gas_stellar_offset=True: use STELLAR rotation matrix
            #   This keeps the physical misalignment between gas and stellar disks.
            #   User's (cosi, theta_int) refers to STELLAR disk orientation.
            # - preserve_gas_stellar_offset=False: use GAS rotation matrix
            #   Both gas and stellar appear at the exact same orientation.
            #   User's (cosi, theta_int) refers to each component independently.
            if config.preserve_gas_stellar_offset:
                # Use stellar rotation for gas -> preserves intrinsic misalignment
                particle_type = "stellar"
            else:
                # Use gas's own rotation -> aligns gas with user's requested orientation
                particle_type = "gas"

            # Undo native orientation to get face-on disk
            coords_disk, velocities_disk = self._undo_native_orientation(
                coords_centered, velocities, particle_type=particle_type
            )

            # Apply requested orientation to disk
            coords_2d, vel_los, _ = self._apply_new_orientation(
                coords_disk, velocities_disk, config.pars
            )

        # Convert to arcsec (with optional redshift scaling)
        coords_arcsec = convert_tng_to_arcsec(
            coords_2d, target_redshift=config.target_redshift
        )

        # Grid velocities (mass-weighted average for gas)
        if config.use_cic_gridding:
            velocity = self._grid_particles_cic(
                coords_arcsec,
                vel_los,
                masses_norm,
                config.image_pars,
                mode="weighted_average",
            )
        else:
            velocity = self._grid_particles_ngp(
                coords_arcsec,
                vel_los,
                masses_norm,
                config.image_pars,
                mode="weighted_average",
            )

        # Apply flux-weighted PSF convolution if requested
        if config.psf is not None:
            from ..psf import gsobj_to_kernel, convolve_flux_weighted_numpy

            if intensity_map is None:
                import warnings

                warnings.warn(
                    "PSF set but no intensity_map provided to generate_velocity_map. "
                    "Generating intensity internally (less efficient).",
                    stacklevel=2,
                )
                intensity_map, _ = self.generate_intensity_map(config, snr=None)

            kernel, padded_shape = gsobj_to_kernel(
                config.psf, image_pars=config.image_pars
            )
            velocity = convolve_flux_weighted_numpy(
                velocity, intensity_map, kernel, padded_shape
            )

        # Add noise if requested
        if snr is not None:
            velocity, variance = add_noise(
                velocity, target_snr=snr, include_poisson=False, seed=seed
            )
        else:
            variance = np.zeros_like(velocity)

        return velocity, variance

    def generate_sfr_map(
        self,
        config: TNGRenderConfig,
        snr: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate 2D star formation rate map from gas particles.

        This serves as a proxy for Hα emission, which traces ionized gas in star-forming regions.
        The relationship is: L_Hα ≈ 1.26e41 * SFR [erg/s per Msun/yr] (Kennicutt 1998)

        Parameters
        ----------
        config : TNGRenderConfig
            Rendering configuration (same as for velocity maps - uses gas particles)
        snr : float, optional
            Signal-to-noise ratio for noise addition. If None, no noise added.
        seed : int, optional
            Random seed for noise generation

        Returns
        -------
        sfr_map : np.ndarray
            2D map of star formation rate surface density, shape from image_pars
        """
        from ..noise import add_noise

        # Get star formation rates (Msun/yr per particle)
        sfr = self.gas["StarFormationRate"].copy()

        # Get coordinates
        coords = self.gas["Coordinates"].copy()

        # Get reference center (use stellar peak for consistency with intensity)
        center = self._get_reference_center(
            config.center_on_peak, config.band, config.use_dusted
        )
        coords_centered = self._center_coordinates(coords, center)

        # Handle orientation (use gas orientation if different from stellar)
        if config.use_native_orientation:
            # Simple projection at native TNG orientation (gas)
            coords_2d = coords_centered[:, :2]  # Just drop z
        else:
            # Transform gas to specified orientation
            if config.pars is None:
                raise ValueError(
                    "pars must be provided when use_native_orientation=False"
                )

            # Undo native GAS orientation to get face-on gas disk
            # Uses the gas-specific angular momentum rotation matrix
            coords_disk, _ = self._undo_native_orientation(
                coords_centered, np.zeros_like(coords_centered), particle_type="gas"
            )

            # Apply requested orientation (same as velocity - uses gas angular momentum)
            coords_2d, _, _ = self._apply_new_orientation(
                coords_disk, np.zeros_like(coords_disk), config.pars
            )

        # Convert to arcsec (with optional redshift scaling)
        coords_arcsec = convert_tng_to_arcsec(
            coords_2d, target_redshift=config.target_redshift
        )

        # Grid SFR (sum to get total SFR per pixel)
        if config.use_cic_gridding:
            sfr_map = self._grid_particles_cic(
                coords_arcsec, sfr, np.ones_like(sfr), config.image_pars, mode="sum"
            )
        else:
            sfr_map = self._grid_particles_ngp(
                coords_arcsec, sfr, np.ones_like(sfr), config.image_pars, mode="sum"
            )

        # Apply cosmological surface brightness dimming if requested
        # SFR maps represent Hα emission (observable flux), so dimming applies
        if config.apply_cosmological_dimming:
            z_native = self.native_redshift
            z_target = config.target_redshift if config.target_redshift else z_native
            dimming_factor = ((1.0 + z_native) / (1.0 + z_target)) ** 4
            sfr_map *= dimming_factor

        # Add noise if requested
        if snr is not None:
            sfr_map, _ = add_noise(
                sfr_map, target_snr=snr, include_poisson=False, seed=seed
            )

        return sfr_map
