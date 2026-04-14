import numpy as np
import jax.numpy as jnp
import jax
from scipy.fft import next_fast_len

from kl_pipe.model import IntensityModel
from kl_pipe.transformation import obs2cen, cen2source, source2gal


# default number of Gauss-Legendre quadrature points for LOS integration
_DEFAULT_N_QUAD = 200


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

    def configure_psf(
        self,
        gsobj,
        image_pars=None,
        *,
        image_shape=None,
        pixel_scale=None,
        oversample=5,
        gsparams=None,
        freeze=False,
    ):
        """Configure PSF with fused k-space convolution kernel."""
        super().configure_psf(
            gsobj,
            image_pars=image_pars,
            image_shape=image_shape,
            pixel_scale=pixel_scale,
            oversample=oversample,
            gsparams=gsparams,
            freeze=freeze,
        )

        # compute padded grid dims matching _render_kspace
        if image_pars is not None:
            coarse_Nrow = image_pars.Nrow
            coarse_Ncol = image_pars.Ncol
            ps = image_pars.pixel_scale
        else:
            coarse_Nrow, coarse_Ncol = image_shape
            ps = pixel_scale

        N = max(self._psf_oversample, 1)
        fine_Nrow = coarse_Nrow * N
        fine_Ncol = coarse_Ncol * N
        fine_ps = ps / N

        pad_sq = next_fast_len(self._kspace_pad_factor * max(fine_Nrow, fine_Ncol))

        from kl_pipe.psf import precompute_psf_kspace_fft

        self._psf_kspace_fft = precompute_psf_kspace_fft(
            gsobj, (pad_sq, pad_sq), fine_ps, gsparams=gsparams
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
        Core k-space FFT rendering (analytic FT of 3D inclined exponential).

        Matches GalSim's SBInclinedExponential kValueHelper exactly.
        No point-sampling aliasing; the thin-disk limit (h_over_r -> 0)
        falls out naturally as ft_vertical -> 1.

        Anti-aliasing: the IFFT is computed on a padded grid (pad_factor × N)
        to suppress periodic boundary wrap-around, then cropped to (Nrow, Ncol).
        Analogous to the zero-padding in convolve_fft for linear convolution.

        When ``psf_kernel_fft`` is provided, the PSF is multiplied in k-space
        BEFORE the IFFT crop, so edge pixels see PSF-scattered light from
        source regions beyond the image boundary. This fuses rendering +
        convolution into a single FFT pass and eliminates boundary flux loss.

        When ``oversample > 1``, the k-grid extends to N × Nyquist, reducing
        cusp aliasing by ~N³ (exponential FT decays as k⁻³). The IFFT is
        computed at fine resolution and subsampled back to (Nrow, Ncol).
        The half-pixel phase correction uses the coarse grid centering so
        subsampled positions align with the standard centered grid.

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

        # effective (fine) grid for k-space evaluation
        eff_Nrow = Nrow * oversample
        eff_Ncol = Ncol * oversample
        eff_ps = pixel_scale / oversample

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
        if psf_kernel_fft is not None:
            # fused path: padded grid must match the pre-computed PSF kernel FFT
            pad_row, pad_col = psf_kernel_fft.shape
        elif pad_factor > 1:
            pad_row = next_fast_len(pad_factor * eff_Nrow)
            pad_col = next_fast_len(pad_factor * eff_Ncol)
        else:
            pad_row = eff_Nrow
            pad_col = eff_Ncol

        # 1. k-grid: ky conjugate to rows (vertical), kx conjugate to cols (horizontal)
        ky = 2.0 * jnp.pi * jnp.fft.fftfreq(pad_row, d=eff_ps)
        kx = 2.0 * jnp.pi * jnp.fft.fftfreq(pad_col, d=eff_ps)
        KY, KX = jnp.meshgrid(ky, kx, indexing='ij')

        # 2. centroid phase: pair kx with x0 (horizontal), ky with y0 (vertical)
        #    half-pixel correction based on OUTPUT grid centering
        hx = 0.5 * pixel_scale * (1 - Ncol % 2)
        hy = 0.5 * pixel_scale * (1 - Nrow % 2)
        phase = jnp.exp(-1j * (KX * (x0 - hx) + KY * (y0 - hy)))

        # 3. shear: area-preserving M = (1/sqrt(1-|g|^2)) * [[1+g1, g2], [g2, 1-g1]]
        #    (1+g1) multiplies kx (horizontal), (1-g1) multiplies ky (vertical)
        norm_shear = 1.0 / jnp.sqrt(1.0 - (g1**2 + g2**2))
        kx_s = norm_shear * ((1.0 + g1) * KX + g2 * KY)
        ky_s = norm_shear * (g2 * KX + (1.0 - g1) * KY)

        # rotation: R(-theta_int) on (kx, ky)
        c = jnp.cos(-theta_int)
        s = jnp.sin(-theta_int)
        kx_gal = c * kx_s - s * ky_s
        ky_gal = s * kx_s + c * ky_s

        # 4. analytic FT in galaxy frame
        kx_scaled = kx_gal * rscale
        ky_scaled = ky_gal * rscale

        # radial FT: (1 + kx² + (ky*cosi)²)^{-3/2}  [cosi compresses rows=vertical]
        k_sq = kx_scaled**2 + (ky_scaled * cosi) ** 2
        ft_radial = 1.0 / (1.0 + k_sq) ** 1.5

        # vertical FT: u/sinh(u), u = (pi/2)*h_over_r*ky_scaled*sini
        # safe-where pattern: substitute finite dummy in non-selected branch
        # so JAX autodiff never sees 0/sinh(0) = 0/0 = NaN
        u = (jnp.pi / 2.0) * h_over_r * ky_scaled * sini
        u_safe = jnp.where(jnp.abs(u) < 1e-4, jnp.ones_like(u), u)
        ft_vertical = jnp.where(
            jnp.abs(u) < 1e-4,
            1.0 - u**2 / 6.0,
            u_safe / jnp.sinh(u_safe),
        )

        I_hat = flux * ft_radial * ft_vertical * phase

        # fused PSF convolution: multiply in k-space BEFORE IFFT crop
        if psf_kernel_fft is not None:
            I_hat = I_hat * psf_kernel_fft

        # IFFT on padded grid, then extract center eff_Nrow×eff_Ncol
        full = jnp.fft.ifft2(I_hat).real
        full = jnp.roll(full, (eff_Nrow // 2, eff_Ncol // 2), axis=(0, 1))
        image = full[:eff_Nrow, :eff_Ncol] / eff_ps**2

        if oversample > 1:
            image = image[::oversample, ::oversample]

        return image

    def render_image(
        self,
        theta: jnp.ndarray,
        image_pars=None,
        plane: str = 'obs',
        X: jnp.ndarray = None,
        Y: jnp.ndarray = None,
        oversample: int = 1,
        **kwargs,
    ) -> jnp.ndarray:
        """
        Render via k-space FFT, with optional PSF convolution.

        When PSF is configured with oversampling, renders at fine scale
        so convolve_fft can bin down to coarse scale.

        Parameters
        ----------
        oversample : int, optional
            Cusp anti-aliasing factor for the non-PSF path. The exponential
            profile has a cusp at R=0 whose FT decays as k⁻³; power above
            Nyquist aliases into the image. Oversampling pushes Nyquist to
            N × π/pixel_scale, reducing aliasing by ~N³.
            Ignored when PSF is configured (PSF convolution suppresses
            high-k aliasing naturally). Default 2.
        """
        if image_pars is None and (X is None or Y is None):
            raise ValueError("Provide image_pars or (X, Y)")

        if image_pars is not None:
            pixel_scale = image_pars.pixel_scale
            Nrow = image_pars.Nrow
            Ncol = image_pars.Ncol
        else:
            Nrow, Ncol = X.shape
            pixel_scale = jnp.abs(X[0, 1] - X[0, 0])

        if self._psf_kspace_fft is not None:
            # fused k-space path: render + convolve in one FFT pass
            N = max(self._psf_oversample, 1)
            image = self._render_kspace(
                theta,
                Nrow * N,
                Ncol * N,
                pixel_scale / N,
                psf_kernel_fft=self._psf_kspace_fft,
            )
            if N > 1:
                image = image.reshape(Nrow, N, Ncol, N).mean(axis=(1, 3))
            return image

        if self._psf_data is not None:
            # fallback real-space path (shouldn't happen for this model,
            # but keeps base class contract)
            from kl_pipe.psf import convolve_fft

            if self._psf_oversample > 1:
                N = self._psf_oversample
                image = self._render_kspace(theta, Nrow * N, Ncol * N, pixel_scale / N)
            else:
                image = self._render_kspace(theta, Nrow, Ncol, pixel_scale)
            return convolve_fft(image, self._psf_data)

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
