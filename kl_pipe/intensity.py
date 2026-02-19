import numpy as np
import jax.numpy as jnp
import jax
from scipy.fft import next_fast_len

from kl_pipe.model import IntensityModel
from kl_pipe.transformation import obs2cen, cen2source, source2gal


# default number of Gauss-Legendre quadrature points for LOS integration
_DEFAULT_N_QUAD = 60


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
    ) -> jnp.ndarray:
        """
        Core k-space FFT rendering (analytic FT of 3D inclined exponential).

        Matches GalSim's SBInclinedExponential kValueHelper exactly.
        No point-sampling aliasing; the thin-disk limit (h_over_r -> 0)
        falls out naturally as ft_vertical -> 1.

        Anti-aliasing: the IFFT is computed on a padded grid (pad_factor × N)
        to suppress periodic boundary wrap-around, then cropped to (Nrow, Ncol).
        Analogous to the zero-padding in convolve_fft for linear convolution.

        Axis convention
        ---------------
        krow = fftfreq(Nrow) is conjugate to rows (axis 0 = X, horizontal).
        kcol = fftfreq(Ncol) is conjugate to cols (axis 1 = Y, vertical).
        gal2disk compresses Y (cols), so cosi acts on kcol in k-space.

        Parameters
        ----------
        theta : jnp.ndarray
            Parameter array.
        Nrow, Ncol : int
            Grid dimensions.
        pixel_scale : float
            Pixel scale (arcsec/pixel).
        pad_factor : int, optional
            IFFT grid padding factor. Defaults to ``self._kspace_pad_factor``.
            1 = no padding; 2 = 2x grid (squashes boundary flux in log-space).

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
        theta_int = self.get_param('theta_int', theta)
        cosi = self.get_param('cosi', theta)
        flux = self.get_param('flux', theta)
        rscale = self.get_param('int_rscale', theta)
        h_over_r = self.get_param('int_h_over_r', theta)

        sini = jnp.sqrt(jnp.maximum(1.0 - cosi**2, 0.0))

        # padded FFT grid for anti-aliasing (reduces periodic boundary wrap-around)
        if pad_factor > 1:
            pad_row = next_fast_len(pad_factor * Nrow)
            pad_col = next_fast_len(pad_factor * Ncol)
        else:
            pad_row = Nrow
            pad_col = Ncol

        # 1. k-grid: krow conjugate to rows (X, horizontal), kcol to cols (Y, vertical)
        krow = 2.0 * jnp.pi * jnp.fft.fftfreq(pad_row, d=pixel_scale)
        kcol = 2.0 * jnp.pi * jnp.fft.fftfreq(pad_col, d=pixel_scale)
        KROW, KCOL = jnp.meshgrid(krow, kcol, indexing='ij')

        # 2. centroid phase + half-pixel grid alignment correction
        #    based on OUTPUT grid dimensions (Nrow, Ncol), not padded grid
        hrow = 0.5 * pixel_scale * (1 - Nrow % 2)
        hcol = 0.5 * pixel_scale * (1 - Ncol % 2)
        phase = jnp.exp(-1j * (KROW * (x0 - hrow) + KCOL * (y0 - hcol)))

        # 3. shear: area-preserving M = (1/sqrt(1-|g|^2)) * [[1+g1, g2], [g2, 1-g1]]
        #    matches GalSim .shear() (det=1, flux-preserving, no prefactor on I_hat)
        norm_shear = 1.0 / jnp.sqrt(1.0 - (g1**2 + g2**2))
        krow_s = norm_shear * ((1.0 + g1) * KROW + g2 * KCOL)
        kcol_s = norm_shear * (g2 * KROW + (1.0 - g1) * KCOL)

        # rotation: R(-theta_int) on (krow, kcol)
        c = jnp.cos(-theta_int)
        s = jnp.sin(-theta_int)
        krow_gal = c * krow_s - s * kcol_s
        kcol_gal = s * krow_s + c * kcol_s

        # 4. analytic FT in galaxy frame
        krow_scaled = krow_gal * rscale
        kcol_scaled = kcol_gal * rscale

        # radial FT: (1 + krow² + (kcol*cosi)²)^{-3/2}  [cosi compresses cols]
        k_sq = krow_scaled**2 + (kcol_scaled * cosi) ** 2
        ft_radial = 1.0 / (1.0 + k_sq) ** 1.5

        # vertical FT: u/sinh(u), u = (pi/2)*h_over_r*kcol_scaled*sini
        # safe-where pattern: substitute finite dummy in non-selected branch
        # so JAX autodiff never sees 0/sinh(0) = 0/0 = NaN
        u = (jnp.pi / 2.0) * h_over_r * kcol_scaled * sini
        u_safe = jnp.where(jnp.abs(u) < 1e-4, jnp.ones_like(u), u)
        ft_vertical = jnp.where(
            jnp.abs(u) < 1e-4,
            1.0 - u**2 / 6.0,
            u_safe / jnp.sinh(u_safe),
        )

        I_hat = flux * ft_radial * ft_vertical * phase

        # IFFT on padded grid, then extract center Nrow×Ncol
        # roll DC from index (0,0) to (Nrow//2, Ncol//2) so crop [:Nrow,:Ncol]
        # gives a centered output; for pad_factor=1 this is equivalent to fftshift
        full = jnp.fft.ifft2(I_hat).real
        full = jnp.roll(full, (Nrow // 2, Ncol // 2), axis=(0, 1))
        return full[:Nrow, :Ncol] / pixel_scale**2

    def render_image(
        self,
        theta: jnp.ndarray,
        image_pars=None,
        plane: str = 'obs',
        X: jnp.ndarray = None,
        Y: jnp.ndarray = None,
        **kwargs,
    ) -> jnp.ndarray:
        """
        Render via k-space FFT, with optional PSF convolution.

        When PSF is configured with oversampling, renders at fine scale
        so convolve_fft can bin down to coarse scale.
        """
        if image_pars is None and (X is None or Y is None):
            raise ValueError("Provide image_pars or (X, Y)")

        if image_pars is not None:
            pixel_scale = image_pars.pixel_scale
            Nrow = image_pars.Nrow
            Ncol = image_pars.Ncol
        else:
            Nrow, Ncol = X.shape
            pixel_scale = jnp.abs(X[1, 0] - X[0, 0])

        if self._psf_data is not None:
            from kl_pipe.psf import convolve_fft

            if self._psf_oversample > 1:
                # render at fine scale so convolve_fft can bin down
                N = self._psf_oversample
                image = self._render_kspace(theta, Nrow * N, Ncol * N, pixel_scale / N)
            else:
                image = self._render_kspace(theta, Nrow, Ncol, pixel_scale)
            return convolve_fft(image, self._psf_data)

        return self._render_kspace(theta, Nrow, Ncol, pixel_scale)


INTENSITY_MODEL_TYPES = {
    'default': InclinedExponentialModel,
    'inclined_exp': InclinedExponentialModel,
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
