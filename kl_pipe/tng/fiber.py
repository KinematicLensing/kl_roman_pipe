"""
Fiber-based spectral simulation from transformed TNG image products.

This module converts 2D intensity/velocity maps into a set of 1D spectra for
user-defined circular fiber placements.

Design goals:
- Reuse existing TNG transformation/rendering pipeline in ``data_vectors.py``.
- Keep API small and explicit for observation and emission configuration.
- Provide deterministic, testable behavior with optional Gaussian noise.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import galsim
from scipy.sparse import dia_matrix

from ..parameters import ImagePars
from .data_vectors import TNGDataVectorGenerator, TNGRenderConfig

# Speed of light in km/s
_C_KMS = 299792.458

# =============================================================================
# FiberDataVector: TNG particle-to-fiber spectrum pipeline
# =============================================================================


class FiberDataVector(TNGDataVectorGenerator):
    """Generate fiber spectra directly from TNG particle data via datacubes.

    Inherits TNGDataVectorGenerator to gain access to particle rendering
    (generate_intensity_map, generate_velocity_map, generate_cube).

    Adds PSF handling and fiber observation methods to convert 3D datacubes
    into 1D fiber spectra with realistic instrumental effects.

    All computations use NumPy (not JAX) for compatibility with TNG data pipelines.
    Works with FiberPars from spectral.py to define fiber placement and observation
    configuration on datacubes.
    """

    def __init__(
        self,
        galaxy_data: Dict[str, Dict],
        fiber_pars: Any,
        *,
        gsparams: Optional[Any] = None,
        psf_oversample: int = 5,
    ):
        """Initialize FiberDataVector and precompute fiber observation state.

        Parameters
        ----------
        galaxy_data : Dict[str, Dict]
            TNG galaxy particle data dictionary.
        fiber_pars : FiberPars
            Fiber configuration (from spectral.py) used to initialize
            wavelength grid, aperture mask, PSF data, and resolution matrix.
        gsparams : optional
            Optional GalSim GSParams passed to PSF precomputation.
        psf_oversample : int, default=5
            Oversampling factor for PSF FFT precomputation.
        """
        super().__init__(galaxy_data)
        self.fiber_pars = fiber_pars
        self.psf = None
        self._fiber_psf_data = None
        self.ATMPSF_conv_fiber_mask = None
        self.resolution_mat = None
        self.wave = np.asarray(fiber_pars.lambda_grid)

        self.psf = self._build_PSF_model_fiber(
            fiber_pars.obs_conf,
            lam_mean=fiber_pars.lambda_eff,
        )

        if self.psf is not None:
            self.configure_fiber_psf(
                self.psf,
                fiber_pars.cube_pars,
                oversample=psf_oversample,
                gsparams=gsparams,
            )

        self.precompute_PSF_convolved_fiber_mask(fiber_pars)
        self.get_resolution_matrix_fiber(fiber_pars)

    def get_fiber_mask(self, fiber_pars: Any) -> np.ndarray:
        """Create circular fiber aperture mask on datacube spatial grid."""
        from photutils.geometry import circular_overlap_grid as cog

        mNx, mNy = fiber_pars.spatial_shape[1], fiber_pars.spatial_shape[0]
        mscale = fiber_pars.pix_scale

        if fiber_pars.is_dispersed:
            fiber_cen = [
                fiber_pars.obs_conf['FIBERDX'],
                fiber_pars.obs_conf['FIBERDY'],
            ]
            fiber_rad = fiber_pars.obs_conf['FIBERRAD']
            xmin, xmax = -mNx / 2 * mscale, mNx / 2 * mscale
            ymin, ymax = -mNy / 2 * mscale, mNy / 2 * mscale
            mask = cog(
                xmin - fiber_cen[0],
                xmax - fiber_cen[0],
                ymin - fiber_cen[1],
                ymax - fiber_cen[1],
                mNx,
                mNy,
                fiber_rad,
                1,
                2,
            )
        else:
            mask = np.ones([mNy, mNx])

        return np.array(mask, dtype=np.float64)

    def configure_fiber_psf(
        self,
        gsobj: Any,
        cube_pars: Any,
        oversample: int = 5,
        gsparams: Optional[Any] = None,
    ) -> None:
        """Pre-compute PSF FFT at the cube's spatial grid."""
        from ..psf import precompute_psf_fft

        self._fiber_psf_data = precompute_psf_fft(
            gsobj,
            image_pars=cube_pars.image_pars,
            oversample=oversample,
            gsparams=gsparams,
        )

    def _build_PSF_model_fiber(self, config: Dict, **kwargs) -> Optional[Any]:
        """Generate GalSim PSF model for fiber observations."""
        _type = config.get('PSFTYPE', 'none').lower()

        if _type == 'none':
            return None

        if _type == 'airy':
            lam = kwargs.get('lam', 1000)
            scale_unit = kwargs.get('scale_unit', galsim.arcsec)
            return galsim.Airy(
                lam=lam, diam=config['DIAMETER'] / 100, scale_unit=scale_unit
            )
        elif _type == 'airy_mean':
            scale_unit = kwargs.get('scale_unit', galsim.arcsec)
            lam = kwargs.get("lam_mean", 1000)
            return galsim.Airy(
                lam=lam, diam=config['DIAMETER'] / 100, scale_unit=scale_unit
            )
        elif _type == 'airy_fwhm':
            loverd = config['PSFFWHM'] / 1.028993969962188
            scale_unit = kwargs.get('scale_unit', galsim.arcsec)
            return galsim.Airy(lam_over_diam=loverd, scale_unit=scale_unit)
        elif _type == 'moffat':
            beta = config.get('PSFBETA', 2.5)
            fwhm = config.get('PSFFWHM', 0.5)
            return galsim.Moffat(beta=beta, fwhm=fwhm)
        else:
            raise ValueError(f'{_type} has not been implemented yet!')

    def precompute_PSF_convolved_fiber_mask(self, fiber_pars: Any) -> None:
        """Pre-compute fiber mask convolved with atmospheric PSF."""
        mNx, mNy = fiber_pars.spatial_shape[1], fiber_pars.spatial_shape[0]
        mscale = fiber_pars.pix_scale

        mask = galsim.InterpolatedImage(
            galsim.Image(array=self.get_fiber_mask(fiber_pars)), scale=mscale
        )

        maskC = mask if self.psf is None else galsim.Convolve([mask, self.psf])
        ary = maskC.drawImage(nx=mNx, ny=mNy, scale=mscale).array

        self.ATMPSF_conv_fiber_mask = np.array(ary, dtype=np.float64)

    def get_resolution_matrix_fiber(self, fiber_pars: Any) -> None:
        """Build spectral resolution matrix for fiber spectra."""
        if fiber_pars.is_dispersed:
            diameter_in_pixel = fiber_pars.obs_conf['FIBRBLUR']
            sigma = diameter_in_pixel / 4.0
            x_in_pixel = np.arange(-5, 6, dtype=np.float64)

            kernel = np.exp(-0.5 * (x_in_pixel / sigma) ** 2) / (
                np.sqrt(2 * np.pi) * sigma
            )

            band = np.array([kernel]).repeat(fiber_pars.n_lambda, axis=0).T
            offset = np.arange(kernel.shape[0] // 2, -(kernel.shape[0] // 2) - 1, -1)
            Rmat = dia_matrix(
                (band, offset), shape=(fiber_pars.n_lambda, fiber_pars.n_lambda)
            )
            self.resolution_mat = np.array(Rmat.toarray(), dtype=np.float64)
        else:
            self.resolution_mat = None

    def fiber_observe_cube(
        self,
        cube: np.ndarray,
        fiber_pars: Any = None,
        force_noise_free: bool = True,
        run_mode: str = 'ETC',
    ) -> Union[np.ndarray, Tuple[np.ndarray, Optional[np.ndarray]]]:
        """Extract 1D fiber spectrum from a 3D datacube with realistic noise.

        Integrates datacube over fiber aperture, applying spectral resolution,
        throughput, exposure time scaling, and Gaussian noise modeling.

        Parameters
        ----------
        cube : np.ndarray
            Datacube of shape (Nrow, Ncol, Nlambda) from generate_cube().
        fiber_pars : FiberPars
            Fiber observation parameters with spatial_shape, lambda_grid,
            obs_conf containing DIAMETER, EXPTIME, GAIN, etc.
        force_noise_free : bool, default=True
            If True, return spectrum without noise.
        run_mode : str, default='ETC'
            'ETC' (exposure time calculator) or 'SNR' mode.

        Returns
        -------
        If force_noise_free:
            spec_1D : np.ndarray
                1D spectrum of shape (Nlambda,).
        Else:
            (spec_1D, noise) : tuple
                Spectrum and noise realization.
        """
        fiber_pars = self.fiber_pars if fiber_pars is None else fiber_pars
        # Photometry mode (non-dispersed)
        if not fiber_pars.is_dispersed:
            from ..psf import convolve_fft

            self.ATMPSF_conv_fiber_mask = None
            self.resolution_mat = None
            psfdata = self._fiber_psf_data

            if run_mode == 'ETC':
                cube_bp = np.array(cube) * np.array(fiber_pars._bp_array)
                raw_img = np.sum(cube_bp, axis=2)
                highres_img = np.repeat(
                    np.repeat(raw_img, psfdata.oversample, axis=0),
                    psfdata.oversample,
                    axis=1,
                )
                factor = (
                    np.pi
                    * (fiber_pars.obs_conf['DIAMETER'] / 2.0) ** 2
                    * fiber_pars.obs_conf['EXPTIME']
                    / fiber_pars.obs_conf['GAIN']
                )

                photometric_image = factor * convolve_fft(highres_img, psfdata)
                return photometric_image

        # Spectroscopy mode (dispersed, 1D spectrum)
        else:
            self.wave = fiber_pars.lambda_grid

            spec_1D = np.sum(
                self.ATMPSF_conv_fiber_mask[:, :, np.newaxis] * np.array(cube),
                axis=(0, 1),
            )

            if run_mode == 'ETC':
                spec_1D = spec_1D * fiber_pars._bp_array
                factor = (
                    np.pi
                    * (fiber_pars.obs_conf['DIAMETER'] / 2.0) ** 2
                    * fiber_pars.obs_conf['EXPTIME']
                    / fiber_pars.obs_conf['GAIN']
                )
                spec_1D = spec_1D * factor

            if self.resolution_mat is not None:
                spec_1D = np.dot(self.resolution_mat, spec_1D)

            if force_noise_free:
                return spec_1D, None
            else:
                if run_mode == 'ETC':
                    try:
                        skysb_file = fiber_pars.obs_conf.get("SKYMODEL")
                        if skysb_file:
                            skysb = galsim.LookupTable.from_file(skysb_file, f_log=True)
                            fiber_area = np.pi * (fiber_pars.obs_conf["FIBERRAD"]) ** 2
                            _wave = self.wave * 10
                            _dwave = _wave[1] - _wave[0]
                            _hnu = 1986445857.148928 / _wave
                            skyct = skysb(_wave) * fiber_area * _dwave / _hnu
                            skyct *= (
                                fiber_pars._bp_array
                                * np.pi
                                * (fiber_pars.obs_conf['DIAMETER'] / 2.0) ** 2
                                * fiber_pars.obs_conf['EXPTIME']
                                / fiber_pars.obs_conf['GAIN']
                            )
                            noise_std = np.sqrt(
                                skyct + spec_1D + fiber_pars.obs_conf['RDNOISE'] ** 2
                            )
                        else:
                            raise KeyError("SKYMODEL not in obs_conf")
                    except (KeyError, FileNotFoundError):
                        noise_std = np.sqrt(spec_1D) * fiber_pars.obs_conf.get(
                            'NOISESIG', 0.1
                        )
                else:
                    noise_std = np.ones_like(spec_1D) * fiber_pars.obs_conf.get(
                        'NOISESIG', 0.0
                    )

                noise = np.random.randn(spec_1D.shape[0]) * noise_std

                if fiber_pars.obs_conf.get('ADDNOISE', True):
                    return spec_1D + noise, noise
                else:
                    return spec_1D, noise
