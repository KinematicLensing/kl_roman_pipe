import numpy as np
from copy import deepcopy
from warnings import warn
import astropy.units as units
from astropy.wcs import WCS

import kl_tools.utils as utils

'''
This file defines the structure and conversions between a params dict
and a parameter list (theta) that is used both in MCMC sampling and
in numba functions that won't accept a dict

pars: dict
    A dictionary that holds *at least* the defined params for the model

theta: list or np.array
    A list or numpy array that has the model pararmeters in a fixed order

TODO: To make this future-proof, what we should do is make a function that
      *returns* PARS_ORDER given a velocity model name. This way it is 
      accessible throughout, but allows things to be a bit more flexible

We also provide a separate convenience class called ImagePars that is a 
container for all the metadata, units, and logic for a 2D image such as size,
pixel_scale, WCS, etc.
'''

class Pars(object):
    '''
    Holds all of the parameters for a needed MCMC run, both
    sampled and meta parameters
    '''

    def __init__(self, sampled_pars, meta_pars):
        '''
        sampled_pars: list of str's
            A list of parameter names to be sampled in the MCMC.
            Their order will be used to define pars2theta
        meta_pars: dict
            A dictionary of meta parameters and their values for
            a particular experiment and MCMC run
        '''

        args = {
            'sampled_pars': (sampled_pars, list),
            'meta_pars': (meta_pars, dict)
            }
        utils.check_types(args)

        for name in sampled_pars:
            utils.check_type(name, 'sampled_par_val', str)

        pars_order = dict(zip(sampled_pars, range(len(sampled_pars))))
        self.sampled = SampledPars(pars_order)
        self.meta = MCMCPars(meta_pars)

        return

    def pars2theta(self, pars):
        return self.sampled.pars2theta(pars)

    def theta2pars(self, theta):
        return self.sampled.theta2pars(theta)

    def copy(self):
        return self.__copy__()

    def __copy__(self):
        return Pars(deepcopy(self.sampled), deepcopy(self.meta))

class SampledPars(object):
    '''
    Sets the structure for arbitrary sampled parameters, which
    are stored internally as a fixed list in the samplers
    '''

    def __init__(self, pars_order):
        '''
        pars_order: dict
            A dictionary that defines the par_name: sampler_index
            relationship in the used sampler

            For example:
            pars_order = {
                'g1': 0,
                'g2': 1,
                ...
            }
        '''

        utils.check_type(pars_order, 'pars_order', dict)

        for key, val in pars_order.items():
            if not isinstance(val, int):
                raise TypeError('pars_order must have int values!')

        self.pars_order = pars_order

        self.names = [*self.pars_order]

        # TODO: think about how to set this properly through the interface
        self.wrapped_pars = None

        return

    def theta2pars(self, theta):
        '''
        uses pars_order to convert list of sampled params to dict
        '''

        assert len(theta) == len(self.pars_order)

        pars = {}
        for key, indx in self.pars_order.items():
            pars[key] = theta[indx]

        return pars

    def pars2theta(self, pars):
        '''
        convert dict of parameters to theta list
        '''

        assert len(pars) == len(self.pars_order)

        # initialize w/ junk that will fail if not set correctly
        theta = len(self.pars_order) * ['']

        for name, indx in self.pars_order.items():
            theta[indx] = pars[name]

        return theta

    def set_wrapped_pars(self, wrapped_pars: list) -> None:
        pass

    # TODO: write this method!
    def get_wrapped_pars(self) -> list:
        '''
        TODO: Implement!
        '''
        if self.wrapped_pars is None:
            Npars = len(self.pars_order)
            wrapped_pars = Npars * [False]
        else:
            wrapped_pars = self.wrapped_pars

        return wrapped_pars

    def __repr__(self):
        return str(self.pars_order)

    def __len__(self):
        return len(self.pars_order.keys())

    def copy(self):
        return self.__copy__()

    def __copy__(self):
        return SampledPars(deepcopy(self.pars_order))

class MetaPars(object):
    '''
    Base class that defines structure for the general meta
    parameters needed for an object, e.g. DataCube, likelihood
    MCMC, etc.
    '''

    _req_fields = []

    def __init__(self, pars):
        '''
        pars: dict
            Dictionary of meta pars
        '''

        utils.check_type(pars, 'pars', dict)

        self._check_pars(pars)
        self.pars = pars

        return

    @classmethod
    def _check_pars(cls, pars):
        '''
        Make sure that the general parameter list
        contains a few required entries
        '''

        for key in cls._req_fields:
            if key not in pars:
                return KeyError(f'{key} is a required field ' +\
                                'in the parameter list!')

        return

    def __getitem__(self, key):
        return self.pars[key]

    def __setitem__(self, key, val):
        self.pars[key] = val
        return

    def __delitem__(self, key):
        del self.pars[key]
        return

    def __iter__(self):
        return iter(self.pars)

    def __repr__(self):
        return str(self.pars)

    def copy(self):
        return self.__copy__()

    def __copy__(self):
        return MetaPars(deepcopy(self.pars))

    def keys(self):
        return self.pars.keys()

    def items(self):
        return self.pars.items()

    def values(self):
        return self.pars.values()

class MCMCPars(MetaPars):

    '''
    Class that defines structure for the parameters
    used in MCMC sampling for a given experiment, modeling
    choices, etc.
    '''

    _req_fields = ['intensity', 'priors', 'units']
    _opt_fields = ['run_options', 'velocity', '_likelihood']

    def copy_with_sampled_pars(self, theta_pars):
        '''
        Scans through a MetaPars dict and sets any sampled meta pars
        to the current value in a returned copy

        theta_pars: dict
            A dict of the sampled mcmc params for both the velocity
            map and the tranformation matrices
        '''

        pars = deepcopy(self.pars)

        return MCMCPars(self._set_sampled_pars(theta_pars, pars))

    @classmethod
    def _set_sampled_pars(cls, theta_pars, pars):
        '''
        Helper func for copy_with_sampled_pars()
        Assumes pars is already a copy of self.pars
        '''

        for key, val in pars.items():
            if isinstance(val, str) and (val.lower() == 'sampled'):
                pars[key] = theta_pars[key]

            elif isinstance(val, dict):
                pars[key] = cls._set_sampled_pars(theta_pars, pars[key])

        return pars

class ImagePars(object):
    '''
    A convenience class that holds all of the metadata for a 2D image, 
    including size, pixel_scale, WCS, etc.

    NOTE: By default the class assumes that you are passing shape
    information in the numpy convention of (Nrow, Ncol) = (Ny, Nx)
    and that the pixel_scale is in arcsec/pixel. You can override
    this by setting indexing='xy' instead of 'ij' in the constructor.

    shape: tuple
        A tuple of the form (Naxis1, Naxis2) that defines the size of the image.
        If indexing='ij', this is (Nrow, Ncol) = (Ny, Nx) to match numpy. If
        indexing='xy', this is (Ncol, Nrow) = (Nx, Ny) to match FITS convention.
    indexing: str; default='ij'
        The indexing convention for the passed image shape. Can be 'ij' (numpy) 
        or 'xy' (FITS)
    pixel_scale: float; default=None
        The pixel scale, typically in arcseconds/pixel (but other Astropy 
        units are allowed). This is used to define the WCS coordinate system if 
        you do not provide one. Can only be passed if wcs is None.
    wcs: WCS object; default=None
        An astropy WCS object that defines the coordinate system for the image.
        Can only be passed if pixel_scale is None.
    '''

    def __init__(
            self, shape, indexing, pixel_scale=None, wcs=None
            ):

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
            
        if indexing not in ['ij', 'xy']:
            raise ValueError('indexing must be "ij" or "xy"!')
        self.indexing = indexing

        if (pixel_scale is None) and (wcs is None):
            raise ValueError('Either pixel_scale or wcs must be passed!')
        if (pixel_scale is not None) and (wcs is not None):
            raise ValueError('Only one of pixel_scale or wcs can be passed!')

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
            self.pixel_scale = float(np.mean(
                self._estimate_pixel_scale()
            )) * 3600 # convert to arcsec/pixel

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

    def _estimate_pixel_scale(self):
        # Optionally use astropy.wcs.utils.proj_plane_pixel_scales
        from astropy.wcs.utils import proj_plane_pixel_scales
        return proj_plane_pixel_scales(self.wcs)

    def pixel_to_world(self, *pixel_coords):
        return self.wcs.pixel_to_world(*pixel_coords)

    def world_to_pixel(self, *sky_coords):
        return self.wcs.world_to_pixel(*sky_coords)
