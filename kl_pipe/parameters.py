import numpy as np
from warnings import warn
from astropy.wcs import WCS

import kl_pipe.utils as utils

"""
Defines ImagePars: the container for 2D-image metadata (shape, pixel_scale,
WCS, units, and indexing logic) used throughout the pipeline for grid setup
and coordinate conventions.
"""


class ImagePars(object):
    '''
    A convenience class that holds all of the metadata for a 2D image,
    including size, pixel_scale, WCS, etc.

    NOTE: By default the class assumes that you are passing shape
    information in the numpy convention of (Nrow, Ncol) where:
    - Nrow = shape[0] = number of rows = Ny (vertical pixel count)
    - Ncol = shape[1] = number of cols = Nx (horizontal pixel count)
    and that the pixel_scale is in arcsec/pixel. You can override
    this by setting indexing='xy' instead of 'ij' in the constructor.

    shape: tuple
        A tuple of the form (Naxis1, Naxis2) that defines the size of the image.
        If indexing='ij', this is (Nrow, Ncol) = (Ny, Nx) to match numpy. If
        indexing='xy', this is (Ncol, Nrow) = (Nx, Ny) to match FITS convention.
    indexing: str; default=None
        The indexing convention for the passed image shape: 'ij' (numpy,
        (Nrow, Ncol)) or 'xy' (FITS, (Ncol, Nrow)). Required when pixel_scale
        is given (there is no coordinate system to disambiguate the axis order).
        When a wcs is passed instead, may be omitted and defaults to 'ij'.
    pixel_scale: float; default=None
        The pixel scale, typically in arcseconds/pixel (but other Astropy
        units are allowed). This is used to define the WCS coordinate system if
        you do not provide one. Can only be passed if wcs is None.
    wcs: WCS object; default=None
        An astropy WCS object that defines the coordinate system for the image.
        Can only be passed if pixel_scale is None.
    '''

    def __init__(self, shape, indexing=None, pixel_scale=None, wcs=None):

        if not isinstance(shape, tuple):
            raise TypeError('shape must be a tuple!')
        if len(shape) != 2:
            raise ValueError('shape must be a tuple of length 2!')
        if not all(isinstance(i, int) for i in shape):
            raise TypeError('shape must be a tuple of ints!')
        if shape[0] <= 0:
            raise ValueError('shape[0] must be > 0!')
        if shape[1] <= 0:
            raise ValueError('shape[1] must be > 0!')
        self.shape = shape

        if (pixel_scale is None) and (wcs is None):
            raise ValueError('Either pixel_scale or wcs must be passed!')
        if (pixel_scale is not None) and (wcs is not None):
            raise ValueError('Only one of pixel_scale or wcs can be passed!')

        # indexing is required when there is no WCS to disambiguate the axis
        # order. A WCS fixes the FITS convention, so when one is passed indexing
        # may be omitted; we then interpret the passed shape in numpy 'ij'
        # (Nrow, Ncol) order (pass indexing='xy' if it is FITS-ordered instead).
        if indexing is None:
            if wcs is None:
                raise ValueError(
                    'indexing must be specified ("ij" or "xy") when no WCS is '
                    'passed (no coordinate system to disambiguate the axis order)'
                )
            indexing = 'ij'
        if indexing not in ['ij', 'xy']:
            raise ValueError('indexing must be "ij" or "xy"!')
        self.indexing = indexing

        if pixel_scale is not None:
            if not isinstance(pixel_scale, (int, float)):
                raise TypeError('pixel_scale must be an int or float!')
            self.pixel_scale = pixel_scale
            self.wcs = WCS(naxis=2)
            self.wcs.wcs.cdelt = np.array([pixel_scale, pixel_scale])
            self.wcs.wcs.crpix = np.array([shape[1] / 2, shape[0] / 2])
            self.wcs.wcs.crval = np.array([0, 0])
            self.wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
            self.wcs.wcs.cunit = ['arcsec', 'arcsec']
            self.wcs.wcs.set()
        else:
            # a WCS was passed
            if not isinstance(wcs, WCS):
                raise TypeError('wcs must be an astropy WCS object!')
            if wcs.naxis != 2:
                raise ValueError('Image WCS must have 2 axes!')

            if wcs.pixel_shape is None:
                raise ValueError('WCS object is missing pixel_shape info!')
            naxis1, naxis2 = wcs.pixel_shape

            # make sure the user didn't pass inconsistent shapes
            if indexing == 'ij':
                expected_shape = (naxis2, naxis1)
            elif indexing == 'xy':
                expected_shape = (naxis1, naxis2)

            if shape != expected_shape:
                raise ValueError(
                    f'Shape {shape} does not match the expected shape '
                    f'{expected_shape} for the passed WCS!'
                )

            self.wcs = wcs
            self.pixel_scale = (
                float(np.mean(self._estimate_pixel_scale())) * 3600
            )  # convert to arcsec/pixel

        return

    @property
    def Nx(self):
        '''
        Number of columns in the image associated with a Cartesian axis
        '''

        if self.indexing == 'ij':
            return self.shape[1]
        else:
            # must be xy indexing
            return self.shape[0]

    @property
    def Ny(self):
        '''
        Number of rows in the image associated with a Cartesian axis
        '''

        if self.indexing == 'ij':
            return self.shape[0]
        else:
            # must be xy indexing
            return self.shape[1]

    @property
    def Nrow(self):
        '''
        Number of rows in the image associated with a Cartesian axis
        '''

        if self.indexing == 'ij':
            return self.shape[0]
        else:
            # must be xy indexing
            return self.shape[1]

    @property
    def Ncol(self):
        '''
        Number of columns in the image associated with a Cartesian axis
        '''

        if self.indexing == 'ij':
            return self.shape[1]
        else:
            # must be xy indexing
            return self.shape[0]

    def make_fine_scale(self, oversample):
        """Return new ImagePars at finer pixel scale for oversampled rendering.

        Parameters
        ----------
        oversample : int
            Oversampling factor. Odd integers recommended to avoid
            centroid-shift artifacts (Bernstein & Gruen 2014).
        """
        import warnings

        if oversample % 2 == 0:
            warnings.warn(
                f"oversample={oversample} is even. Odd values recommended to "
                "avoid half-pixel centroid shifts in oversampled rendering "
                "(Bernstein & Gruen 2014). Use 3, 5, 7, or 9.",
                stacklevel=2,
            )
        return ImagePars(
            shape=(self.Nrow * oversample, self.Ncol * oversample),
            pixel_scale=self.pixel_scale / oversample,
            indexing='ij',
        )

    def _estimate_pixel_scale(self):
        # Optionally use astropy.wcs.utils.proj_plane_pixel_scales
        from astropy.wcs.utils import proj_plane_pixel_scales

        return proj_plane_pixel_scales(self.wcs)

    def pixel_to_world(self, *pixel_coords):
        return self.wcs.pixel_to_world(*pixel_coords)

    def world_to_pixel(self, *sky_coords):
        return self.wcs.world_to_pixel(*sky_coords)
