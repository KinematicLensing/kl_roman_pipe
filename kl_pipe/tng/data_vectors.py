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
- Units: Arcseconds (angular separation on sky)
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
- Pixel scale: Arcseconds per pixel

## Particle Types

- **Intensity maps**: Use stellar particles (PartType4) with photometric luminosities
- **Velocity maps**: Use gas particles (PartType0) with mass weighting
  - Gas represents ionized ISM (observable via Hα, [OII], etc.)
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
- Separate rotation matrices for stellar and gas (preserves physical misalignment ~30-40°)
- Coordinate conversion from comoving kpc/h to arcsec with optional redshift scaling
- Cloud-in-Cell (CIC) gridding for smooth maps
- LOS velocity projection: v_LOS = v_y*sin(i) + v_z*cos(i) (matches arXiv:2201.00739)
- Shared noise utilities from noise.py

## Known Limitations

1. **Sparse gas**: Velocity maps may have empty pixels where no gas particles fall
2. **SNR calibration**: Less accurate for very large flux values (>10^9)
3. **Absolute calibration**: Luminosity units preserved but may need external validation
4. **No PSF**: Point-spread function convolution not implemented
5. **Gaussian noise**: Poisson noise available but can cause visualization issues

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

from ..parameters import ImagePars
from ..utils import build_map_grid_from_image_pars
from ..noise import add_noise
from ..transformation import transform_to_disk_plane


# TNG50 cosmology (Planck 2013)
TNG_COSMOLOGY = FlatLambdaCDM(
    H0=67.74 * u.km / u.s / u.Mpc, Om0=0.3089, Tcmb0=2.725 * u.K
)


def convert_tng_to_arcsec(
    coords_kpc: np.ndarray,
    distance_mpc: float,
    h: float = 0.6774,
    target_redshift: Optional[float] = None,
    native_redshift: float = 0.011,
) -> np.ndarray:
    """
    Convert TNG comoving coordinates to angular separation in arcsec.

    TNG coordinates are comoving kpc/h. This converts to physical kpc,
    then to arcsec using the angular diameter distance.

    Optionally rescale to a different redshift for realistic sub-arcsec observations.
    TNG50 galaxies are at z~0.01 (~50 Mpc), appearing ~21 arcmin on sky.
    Use target_redshift to place at higher z for Roman-like observations.

    Parameters
    ----------
    coords_kpc : np.ndarray
        Coordinates in comoving kpc/h (TNG native units)
    distance_mpc : float
        Angular diameter distance in Mpc (from subhalo data)
    h : float, default=0.6774
        Hubble parameter h for TNG50
    target_redshift : float, optional
        If provided, scale angular size to this redshift.
        Good values: 0.5-1.0 for Roman-like sub-arcsec resolution.
        If None, use native TNG redshift (~0.011).
    native_redshift : float, default=0.011
        Native redshift of TNG50 galaxies

    Returns
    -------
    coords_arcsec : np.ndarray
        Angular coordinates in arcsec
    """
    # Convert comoving to physical kpc
    coords_physical_kpc = coords_kpc / h

    # Convert to arcsec: theta = d / D_A
    # 1 rad = 206265 arcsec, D_A in Mpc, d in kpc
    # theta [arcsec] = (d [kpc] / D_A [Mpc]) * 206.265
    scale_factor = 206.265 / distance_mpc  # arcsec per kpc
    coords_arcsec = coords_physical_kpc * scale_factor

    # Optionally rescale to target redshift
    # Angular size scales roughly as D_A(z), which grows then decreases
    # Approximation for z<1: D_A(z) ≈ (c/H0) * z / (1+z)
    # For small z: D_A ∝ z, so theta ∝ 1/z
    if target_redshift is not None and target_redshift != native_redshift:
        # Scale angular size inversely with redshift ratio
        # Higher z → smaller angular size (more distant)
        scale_ratio = native_redshift / target_redshift
        coords_arcsec *= scale_ratio

    return coords_arcsec


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
    """

    image_pars: ImagePars
    band: str = 'r'
    use_dusted: bool = True
    center_on_peak: bool = True
    use_native_orientation: bool = True
    pars: Optional[Dict] = None
    use_cic_gridding: bool = True
    target_redshift: Optional[float] = None
    preserve_gas_stellar_offset: bool = True

    def __post_init__(self):
        """Validate configuration parameters."""
        # Validate shear parameters if custom orientation is used
        if not self.use_native_orientation and self.pars is not None:
            g1 = self.pars.get('g1', 0.0)
            g2 = self.pars.get('g2', 0.0)
            gamma = np.sqrt(g1**2 + g2**2)
            if gamma >= 1.0:
                raise ValueError(
                    f"Shear too large: |g|={gamma:.3f} >= 1.0. "
                    f"Weak lensing requires |g| < 1."
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

    def __init__(self, galaxy_data: Dict[str, Dict]):
        """
        Initialize generator for a specific galaxy.

        Parameters
        ----------
        galaxy_data : dict
            Dictionary with keys 'gas', 'stellar', 'subhalo' containing
            the TNG data for one galaxy (from TNG50MockData.get_galaxy())
        """
        self.galaxy_data = galaxy_data
        self.stellar = galaxy_data.get('stellar')
        self.gas = galaxy_data.get('gas')
        self.subhalo = galaxy_data.get('subhalo')

        if self.stellar is None:
            raise ValueError("Stellar data required for data vector generation")

        if self.subhalo is None:
            raise ValueError(
                "Subhalo data required for coordinate conversion and orientation"
            )

        # Store key properties
        self.distance_mpc = float(self.subhalo['DistanceMpc'])
        self.native_inclination_deg = float(self.subhalo['Inclination_star'])
        self.native_pa_deg = float(self.subhalo['Position_Angle_star'])

        # Store gas orientation for offset preservation
        self.native_gas_inclination_deg = float(self.subhalo['Inclination_gas'])
        self.native_gas_pa_deg = float(self.subhalo['Position_Angle_gas'])

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

    def _compute_disk_rotation_matrices(self):
        """
        Compute rotation matrices to transform from TNG simulation frame to disk frames.

        Computes SEPARATE rotation matrices for stellar and gas because they can have
        significantly different angular momentum directions (37° difference is common!).

        - Stellar: Uses SubhaloSpin (angular momentum) from TNG
        - Gas: Computes angular momentum from gas particle coordinates and velocities

        Each matrix aligns the respective angular momentum with the +Z axis.
        """
        # === Stellar rotation matrix from SubhaloSpin ===
        spin = np.array(self.subhalo['SubhaloSpin'])
        L_stellar = spin / np.linalg.norm(spin)
        self._R_to_disk_stellar = self._rodrigues_rotation(L_stellar)

        # === Gas rotation matrix from particle angular momentum ===
        if self.gas is not None and len(self.gas.get('Coordinates', [])) > 0:
            # Compute gas angular momentum: L = Σ m_i * (r_i × v_i)
            # We approximate with equal mass particles (or could use actual masses)
            coords_gas = self.gas['Coordinates']
            vel_gas = self.gas['Velocities']

            # Center coordinates on gas centroid
            center_gas = np.mean(coords_gas, axis=0)
            coords_cen = coords_gas - center_gas

            # Subtract mean velocity
            vel_cen = vel_gas - np.mean(vel_gas, axis=0)

            # Total angular momentum (sum of cross products)
            L_gas_vec = np.sum(np.cross(coords_cen, vel_cen), axis=0)
            L_gas = L_gas_vec / np.linalg.norm(L_gas_vec)

            self._R_to_disk_gas = self._rodrigues_rotation(L_gas)

            # Store angle between stellar and gas L for diagnostics
            self._gas_stellar_L_angle_deg = np.rad2deg(
                np.arccos(np.clip(np.dot(L_stellar, L_gas), -1, 1))
            )
        else:
            # No gas data - use stellar rotation
            self._R_to_disk_gas = self._R_to_disk_stellar.copy()
            self._gas_stellar_L_angle_deg = 0.0

        # For backwards compatibility, _R_to_disk is the stellar one
        self._R_to_disk = self._R_to_disk_stellar

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
        if np.abs(cos_angle) > 0.9999:
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

    def _get_luminosity_key(self, band: str, use_dusted: bool) -> str:
        """Get the appropriate luminosity key for the specified band."""
        prefix = 'Dusted_Luminosity' if use_dusted else 'Raw_Luminosity'
        key = f'{prefix}_{band}'

        # Validate key exists
        if key not in self.stellar:
            available = [k for k in self.stellar.keys() if 'Luminosity' in k]
            raise KeyError(
                f"Luminosity key '{key}' not found in stellar data. "
                f"Available bands: {available}"
            )

        return key

    def _get_reference_center(
        self, center_on_peak: bool, band: str = 'r', use_dusted: bool = True
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
            stellar_coords = self.stellar['Coordinates']
            center = np.average(stellar_coords, axis=0, weights=luminosities)
        else:
            # Use subhalo position
            center = np.array(
                [
                    self.subhalo['SubhaloPosX'],
                    self.subhalo['SubhaloPosY'],
                    self.subhalo['SubhaloPosZ'],
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
        self, coords: np.ndarray, velocities: np.ndarray, particle_type: str = 'stellar'
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
        if particle_type == 'gas':
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

        Uses manual inverse transforms since transform.py only has disk←obs direction.

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
        cosi = pars.get('cosi', 1.0)
        theta_int = pars.get('theta_int', 0.0)
        x0 = pars.get('x0', 0.0)
        y0 = pars.get('y0', 0.0)
        g1 = pars.get('g1', 0.0)
        g2 = pars.get('g2', 0.0)

        # Validate shear parameters
        gamma = np.sqrt(g1**2 + g2**2)
        if gamma >= 1.0:
            raise ValueError(
                f"Shear too large: |g|={gamma:.3f} >= 1.0. Weak lensing requires |g| < 1."
            )

        # Manual inverse transform: disk → gal → source → cen → obs
        # (transform.py only has forward direction, so we implement inverses)

        # disk → gal: Apply inclination (y *= cosi, foreshorten)
        x_gal = coords_disk[:, 0]
        y_gal = coords_disk[:, 1] * cosi
        z_gal = coords_disk[:, 2]  # Not used for 2D projection

        # gal → source: Apply PA rotation (positive rotation)
        cos_pa = np.cos(theta_int)
        sin_pa = np.sin(theta_int)
        x_source = x_gal * cos_pa - y_gal * sin_pa
        y_source = x_gal * sin_pa + y_gal * cos_pa

        # source → cen: Apply shear
        if g1 != 0 or g2 != 0:
            # Shear matrix (inverse of cen2source)
            norm = 1.0 / (1.0 - (g1**2 + g2**2))
            x_cen = norm * ((1.0 - g1) * x_source - g2 * y_source)
            y_cen = norm * (-g2 * x_source + (1.0 + g1) * y_source)
        else:
            x_cen = x_source
            y_cen = y_source

        # cen → obs: Apply offsets
        x_obs = x_cen + x0
        y_obs = y_cen + y0

        coords_2d = np.column_stack([x_obs, y_obs])

        # Compute LOS velocity
        # The disk is tilted by inclination i around the x-axis (before PA rotation).
        # The LOS is along the observer's z-axis.
        #
        # In disk frame: rotation velocity is primarily in xy-plane (tangential)
        # After tilting by i around x-axis:
        #   - v_x stays in sky plane (no LOS contribution)
        #   - v_y gets projected: v_y * sin(i) contributes to LOS
        #   - v_z gets projected: v_z * cos(i) contributes to LOS
        #
        # PA rotation only affects where on the sky things appear, not the LOS direction.
        # So we use the DISK frame velocities, not PA-rotated velocities.
        sini = np.sqrt(1 - cosi**2)
        vel_los = velocities_disk[:, 1] * sini + velocities_disk[:, 2] * cosi

        # Store 3D coords for diagnostics (z is just carried through)
        coords_3d_obs = np.column_stack([x_obs, y_obs, coords_disk[:, 2]])

        return coords_2d, vel_los, coords_3d_obs

    def _grid_particles_cic(
        self,
        coords_arcsec: np.ndarray,
        values: np.ndarray,
        weights: np.ndarray,
        image_pars: ImagePars,
        mode: str = 'weighted_average',
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
        X, Y = build_map_grid_from_image_pars(image_pars, unit='arcsec', centered=True)
        pixel_size = image_pars.pixel_scale
        Ny, Nx = (
            image_pars.shape
            if image_pars.indexing == 'ij'
            else (image_pars.shape[1], image_pars.shape[0])
        )

        # Convert to pixel coordinates (continuous)
        x_min, y_min = X.min(), Y.min()
        x_pix = (coords_arcsec[:, 0] - x_min) / pixel_size
        y_pix = (coords_arcsec[:, 1] - y_min) / pixel_size

        # Initialize arrays
        weighted_sum = np.zeros((Ny, Nx))
        weight_sum = np.zeros((Ny, Nx)) if mode == 'weighted_average' else None

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

                if mode == 'weighted_average':
                    np.add.at(weight_sum, (yi_valid, xi_valid), weights_valid * w_valid)

        # Compute result based on mode
        if mode == 'sum':
            result = weighted_sum
        elif mode == 'weighted_average':
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
        mode: str = 'weighted_average',
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
        X, Y = build_map_grid_from_image_pars(image_pars, unit='arcsec', centered=True)
        pixel_size = image_pars.pixel_scale
        Ny, Nx = (
            image_pars.shape
            if image_pars.indexing == 'ij'
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

        if mode == 'sum':
            result = H_weighted
        elif mode == 'weighted_average':
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
        # Get luminosity weights
        lum_key = self._get_luminosity_key(config.band, config.use_dusted)
        luminosities = self.stellar[lum_key]

        # Normalize to avoid overflow (will rescale back later)
        lum_scale = luminosities.max()
        luminosities_norm = luminosities / lum_scale

        # Get coordinates
        coords = self.stellar['Coordinates'].copy()

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
                coords_centered, np.zeros_like(coords_centered), particle_type='stellar'
            )

            # Apply new stellar orientation from pars
            coords_2d, _, _ = self._apply_new_orientation(
                coords_disk, np.zeros_like(coords_disk), config.pars
            )

        # Convert to arcsec (with optional redshift scaling)
        coords_arcsec = convert_tng_to_arcsec(
            coords_2d, self.distance_mpc, target_redshift=config.target_redshift
        )

        # Grid luminosities (sum, not weighted average)
        if config.use_cic_gridding:
            intensity = self._grid_particles_cic(
                coords_arcsec,
                luminosities_norm,
                np.ones_like(luminosities_norm),
                config.image_pars,
                mode='sum',
            )
        else:
            intensity = self._grid_particles_ngp(
                coords_arcsec,
                luminosities_norm,
                np.ones_like(luminosities_norm),
                config.image_pars,
                mode='sum',
            )

        # Rescale back to original units
        intensity *= lum_scale

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

        coords = self.gas['Coordinates'].copy()
        velocities = self.gas['Velocities'].copy()

        # Subtract systemic velocity (median to be robust to outliers)
        # This centers the velocity field at v=0 in the galaxy rest frame
        v_systemic = np.median(velocities, axis=0)
        velocities -= v_systemic

        # Use gas masses as weights (no luminosity for gas)
        masses = self.gas['Masses'].copy()

        # Normalize to avoid overflow
        mass_scale = masses.max()
        masses_norm = masses / mass_scale

        # Center coordinates using SAME reference as intensity (stellar luminosity peak or subhalo)
        # This ensures intensity and velocity show the same patch of sky
        center = self._get_reference_center(
            config.center_on_peak, config.band, config.use_dusted
        )
        coords_centered = self._center_coordinates(coords, center)

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
            # - preserve_gas_stellar_offset=False (default): use GAS rotation matrix
            #   Both gas and stellar appear at the exact same orientation.
            #   User's (cosi, theta_int) refers to each component independently.
            if config.preserve_gas_stellar_offset:
                # Use stellar rotation for gas -> preserves intrinsic misalignment
                particle_type = 'stellar'
            else:
                # Use gas's own rotation -> aligns gas with user's requested orientation
                particle_type = 'gas'

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
            coords_2d, self.distance_mpc, target_redshift=config.target_redshift
        )

        # Grid velocities (mass-weighted average for gas)
        if config.use_cic_gridding:
            velocity = self._grid_particles_cic(
                coords_arcsec,
                vel_los,
                masses_norm,
                config.image_pars,
                mode='weighted_average',
            )
        else:
            velocity = self._grid_particles_ngp(
                coords_arcsec,
                vel_los,
                masses_norm,
                config.image_pars,
                mode='weighted_average',
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
        sfr = self.gas['StarFormationRate'].copy()

        # Get coordinates
        coords = self.gas['Coordinates'].copy()

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
                coords_centered, np.zeros_like(coords_centered), particle_type='gas'
            )

            # Apply requested orientation (same as velocity - uses gas angular momentum)
            coords_2d, _, _ = self._apply_new_orientation(
                coords_disk, np.zeros_like(coords_disk), config.pars
            )

        # Convert to arcsec (with optional redshift scaling)
        coords_arcsec = convert_tng_to_arcsec(
            coords_2d, self.distance_mpc, target_redshift=config.target_redshift
        )

        # Grid SFR (sum to get total SFR per pixel)
        if config.use_cic_gridding:
            sfr_map = self._grid_particles_cic(
                coords_arcsec, sfr, np.ones_like(sfr), config.image_pars, mode='sum'
            )
        else:
            sfr_map = self._grid_particles_ngp(
                coords_arcsec, sfr, np.ones_like(sfr), config.image_pars, mode='sum'
            )

        # Add noise if requested
        if snr is not None:
            sfr_map, _ = add_noise(
                sfr_map, target_snr=snr, include_poisson=False, seed=seed
            )

        return sfr_map
