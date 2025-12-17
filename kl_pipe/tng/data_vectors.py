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
4. No PSF: Point-spread function convolution not implemented
5. Gaussian noise: Poisson noise available but is not working well yet

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
    # Step 1: Convert comoving to physical coordinates
    # TNG stores coordinates in comoving kpc/h, need to convert to physical kpc
    # Physical distance = (comoving distance / h) * a, where a = 1/(1+z) is scale factor
    coords_comoving_kpc = coords_kpc / h  # Remove h factor
    scale_factor_a = 1.0 / (1.0 + native_redshift)  # Scale factor a = 1/(1+z)
    coords_physical_kpc = coords_comoving_kpc * scale_factor_a

    # Step 2: Convert physical kpc to angular separation at native redshift
    # theta = d_phys / D_A, where D_A is angular diameter distance
    arcsec_per_radian = 180.0 * 3600.0 / np.pi  # 180 deg/rad * 3600 arcsec/deg / pi
    kpc_per_mpc = 1000.0
    d_a_native_mpc = distance_mpc  # Angular diameter distance at native redshift
    scale_factor = arcsec_per_radian / (d_a_native_mpc * kpc_per_mpc)  # arcsec per kpc
    coords_arcsec = coords_physical_kpc * scale_factor

    # Step 3: Optionally rescale to target redshift using proper cosmology
    if target_redshift is not None and target_redshift != native_redshift:
        # Use astropy cosmology for accurate angular diameter distance scaling
        # Angular size scales as theta ∝ d_phys / D_A(z)
        d_a_native = TNG_COSMOLOGY.angular_diameter_distance(native_redshift)
        d_a_target = TNG_COSMOLOGY.angular_diameter_distance(target_redshift)
        # Larger D_A → smaller angular size (more distant)
        scale_ratio = (d_a_native / d_a_target).value  # ratio of D_A values
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

        - Stellar: Computed from stellar particle positions, velocities, and luminosities
        - Gas: Computed from gas particle positions, velocities, and masses

        Each matrix aligns the respective angular momentum with the +Z axis.

        Also computes and stores diagnostic information comparing our kinematic inclinations
        with the TNG catalog's morphological inclinations.
        """
        # === Stellar rotation matrix from stellar particles ===
        # Use luminosity-weighted angular momentum for stellar particles
        # This is more physical than SubhaloSpin (which includes all particle types)
        coords_stellar = self.stellar['Coordinates']
        vel_stellar = self.stellar['Velocities']

        # Use r-band luminosity as weights (or masses if not available)
        if 'Dusted_Luminosity_r' in self.stellar:
            weights_stellar = self.stellar['Dusted_Luminosity_r']
        elif 'Masses' in self.stellar:
            weights_stellar = self.stellar['Masses']
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
            axis=0
        )
        L_stellar = L_stellar_vec / np.linalg.norm(L_stellar_vec)
        self._R_to_disk_stellar = self._rodrigues_rotation(L_stellar)

        # Store stellar L direction for diagnostics
        self._L_stellar = L_stellar

        # Compute kinematic inclination from L_stellar (angle from +Z axis)
        # This is the "true" kinematic inclination
        self._kinematic_inc_stellar_deg = np.rad2deg(np.arccos(np.abs(L_stellar[2])))

        # === Gas rotation matrix from particle angular momentum ===
        if self.gas is not None and len(self.gas.get('Coordinates', [])) > 0:
            coords_gas = self.gas['Coordinates']
            vel_gas = self.gas['Velocities']
            masses_gas = self.gas['Masses']

            # Center on mass-weighted centroid
            center_gas = np.average(coords_gas, axis=0, weights=masses_gas)
            coords_gas_cen = coords_gas - center_gas

            # Subtract mass-weighted mean velocity
            vel_gas_mean = np.average(vel_gas, axis=0, weights=masses_gas)
            vel_gas_cen = vel_gas - vel_gas_mean

            # Mass-weighted angular momentum: L = Σ m_i * (r_i × v_i)
            L_gas_vec = np.sum(
                masses_gas[:, None] * np.cross(coords_gas_cen, vel_gas_cen),
                axis=0
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
        alignment_tolerance = 1.0 - np.cos(np.radians(0.01))  # Near-perfect alignment threshold
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
        cosi = pars.get('cosi', 1.0)
        theta_int = pars.get('theta_int', 0.0)
        x0 = pars.get('x0', 0.0)
        y0 = pars.get('y0', 0.0)
        g1 = pars.get('g1', 0.0)
        g2 = pars.get('g2', 0.0)

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
        R_incl = np.array([
            [1.0, 0.0, 0.0],
            [0.0, cos_angle, -sin_angle],
            [0.0, sin_angle, cos_angle]
        ])

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
            # Shear matrix (inverse of cen2source)
            norm = 1.0 / (1.0 - (g1**2 + g2**2))
            x_cen = norm * ((1.0 - g1) * x_source - g2 * y_source)
            y_cen = norm * (-g2 * x_source + (1.0 + g1) * y_source)
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
        masses = self.gas['Masses'].copy()

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
            v_systemic = np.average(velocities[inner_mask], axis=0, weights=masses[inner_mask])
        else:
            # Fallback to simple median if no inner particles (shouldn't happen)
            v_systemic = np.median(velocities, axis=0)
        velocities -= v_systemic

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
