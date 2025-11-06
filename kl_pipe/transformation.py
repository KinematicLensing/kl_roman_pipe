'''
This file contains transformation functions. These
are all static functions so that numba can be used
efficiently.

Definition of each plane:

    disk: Face-on view of the galactic disk, no inclination angle.
          This will be cylindrically symmetric for most models

    gal:  Galaxy major/minor axis frame with inclination angle same as
          source plane. Will now be ~ellipsoidal for sini!=0

    source: View from the lensing source plane, rotated version of gal
            plane with theta = theta_intrinsic

    cen: View from the object-centered observed plane. Sheared version of
         source plane

    obs:  Observed image plane. Offset version of cen plane
'''

from abc import abstractmethod
import numpy as np

SUPPORTED_PLANES = ['disk', 'gal', 'source', 'cen', 'obs']


class TransformableImage(object):
    '''
    This base class defines the transformation properties
    and structure for an image (e.g. velocity map, source
    intensity) that can be rendered in the various planes
    relevant to kinematic lensing: disk, gal, source, and obs

    The unifying feature of these images will be that their
    rendering in a given plane will be handled by the __call__
    attribute, along with the plane name, image coords to eval
    at, and any necessary transformation parameters
    '''

    def __init__(self, transform_pars):
        '''
        transform_pars: dict
            A dictionary that defines the parameters needed
            to evaluate the plane transformations
        '''

        self.transform_pars = transform_pars
        self._planes = ['disk', 'gal', 'source', 'cen', 'obs']

        self._setup_transformations()

        return

    def _setup_transformations(self):
        '''
        TODO: Do we need this?
        '''

        pars = self.transform_pars

        self.cen2source = _transform_cen2source(pars)
        self.source2gal = _transform_source2gal(pars)
        self.gal2disk = _transform_gal2disk(pars)

        # we don't do the following as it is simpler to apply
        # a translation directly instead of building a 3x3 matrix
        # self.obs2cen = _transform_obs2cen(pars)

        return

    def __call__(self, plane, x, y, use_numba=False):
        '''
        plane: str
            The plane to evaluate the image in
        x: np.ndarray
            The x position(s) to evaluate the image at
        y: np.ndarray
            The y position(s) to evaluate the image at
        use_numba: bool
            Whether to use numba-friendly versions of the
            transformation functions
            NOTE: Not yet fully implemented
        '''

        if plane not in self._planes:
            raise ValueError(f'{plane} not a valid image plane!')

        if not isinstance(x, np.ndarray):
            assert not isinstance(y, np.ndarray)
            x = np.array(x)
            y = np.array(y)

        if x.shape != y.shape:
            raise Exception('x and y arrays must be the same shape!')

        return

    def _get_plane_eval_func(self, plane, use_numba=False):
        '''
        plane: str
            Name of the plane to evaluate positions in
        use_numba: bool
            Set to use numba versions of functions
        '''

        if plane == 'obs':
            if use_numba is True:
                func = nb.eval_in_obs_plane
            else:
                func = self._eval_in_obs_plane
        elif plane == 'cen':
            if use_numba is True:
                func = nb.eval_in_cen_plane
            else:
                func = self._eval_in_cen_plane
        elif plane == 'source':
            if use_numba is True:
                func = nb.eval_in_source_plane
            else:
                func = self._eval_in_source_plane
        elif plane == 'gal':
            if use_numba is True:
                func = nb.eval_in_gal_plane
            else:
                func = self._eval_in_gal_plane
        elif plane == 'disk':
            if use_numba is True:
                func = nb.eval_in_disk_plane
            else:
                func = self._eval_in_disk_plane

        return func

    def _eval_map_in_plane(self, plane, x, y, use_numba=False):
        '''
        We use static methods defined in transformation.py
        to speed up these very common function calls

        The input (x,y) position is defined in the plane

        plane: str
            The name of the plane to evaluate the map in the given
            coords
        use_numba: bool
            Set to True to use numba versions of transformations
        '''

        pars = self.transform_pars


        func = self._get_plane_eval_func(plane, use_numba=use_numba)

        return func(pars, x, y)
    @classmethod
    def _eval_in_obs_plane(cls, pars, x, y, **kwargs):
        '''
        pars: dict
            Holds the model & transformation parameters
        x,y: np.ndarray
            The position coordintates in the obs plane

        kwargs holds any additional params that might be needed
        in subclass evaluations, such as using speed instead of
        velocity
        '''

        xp, yp = _obs2cen(pars, x, y)

        return cls._eval_in_cen_plane(pars, xp, yp, **kwargs)

    @classmethod
    def _eval_in_cen_plane(cls, pars, x, y, **kwargs):
        '''
        pars: dict
            Holds the model & transformation parameters
        x,y: np.ndarray
            The position coordintates in the cen plane

        kwargs holds any additional params that might be needed
        in subclass evaluations, such as using speed instead of
        velocity
        '''

        xp, yp = _cen2source(pars, x, y)

        return cls._eval_in_source_plane(pars, xp, yp, **kwargs)

    @classmethod
    def _eval_in_source_plane(cls, pars, x, y, **kwargs):
        '''
        pars: dict
            Holds the model & transformation parameters
        x,y: np.ndarray
            The position coordintates in the source plane

        kwargs holds any additional params that might be needed
        in subclass evaluations, such as using speed instead of
        velocity
        '''

        xp, yp = _source2gal(pars, x, y)

        return cls._eval_in_gal_plane(pars, xp, yp, **kwargs)

    @classmethod
    def _eval_in_gal_plane(cls, pars, x, y, **kwargs):
        '''
        pars: dict
            Holds the model & transformation parameters
        x,y: np.ndarray
            The position coordintates in the gal plane

        kwargs holds any additional params that might be needed
        in subclass evaluations, such as using speed instead of
        velocity
        '''

        xp, yp = _gal2disk(pars, x, y)

        return cls._eval_in_disk_plane(pars, xp, yp, **kwargs)

    @abstractmethod
    def _eval_in_disk_plane(pars, x, y, **kwargs):
        pass

def transform_coords(x, y, plane1, plane2, pars):
    '''
    Transform coords (x,y) defined in plane1 into plane2

    pars: dict holding model information
    '''

    # each plane assigned an index in order from
    # simplest disk plane to most complex obs plane
    planes = ['disk', 'gal', 'source', 'cen', 'obs']
    plane_map = dict(zip(planes, range(len(planes))))

    for plane in [plane1, plane2]:
        if plane not in planes:
            raise ValueError(f'{plane} not a valid plane!')

    # get start & end indices
    start = plane_map[plane1]
    end   = plane_map[plane2]

    if start == end:
        return x, y

    # transforms in direction from disk to obs
    if start < end:
        transforms = [_disk2gal, _gal2source, _source2cen, _cen2obs]
        step = 1

    # transforms in direction from obs to disk
    else:
        transforms = [_gal2disk, _source2gal, _cen2source, _obs2cen]
        step = -1

        # Account for different starting point for inv transforms
        start -= 1
        end -= 1

    # there is no transform starting with the end indx
    for i in range(start, end, step):
        transform = transforms[i]
        x, y = transform(pars, x, y)

    return x, y

def _multiply(transform, x, y):
    '''
    transform: a (2x2) coordinate transformation matrix
    x: np array of x position
    y: np array of y position

    returns: (x', y')
    '''

    # TODO: We can generalize this by just reshaping an arbitrary ndarray
    #       into a 2xNx*Ny*.... array
    # Can't just do a matrix multiplication because we can't assume
    # structure of pos; e.g. it might be a meshgrid array

    assert x.shape == y.shape

    # x,y are vectors
    if len(x.shape) == 1:
        pos = np.array([x, y])

        out = np.matmul(transform, pos)
        xp, yp = out[0], out[1]

    # x,y are matrices
    elif len(x.shape) == 2:
        # gotta do it the slow way
        # out = np.zeros((2, x.shape[0], x.shape[1]))
        xp = np.zeros(x.shape)
        yp = np.zeros(y.shape)

        N1, N2 = x.shape
        for i in range(N1):
            pos = np.array([x[i,:], y[i,:]])
            out = np.matmul(transform, pos)
            xp[i,:] = out[0]
            yp[i,:] = out[1]

    else:
        raise ValueError('Plane transformations are not yet implemented ' +\
                         f'for input positions of shape {x.shape}!')

    return xp, yp

# The following transformation definitions require basic knowledge
# of the source shear, intrinsic orientation, profile inclination
# angle, and centroid offset

def _transform_obs2cen(pars):
    '''
    Lensing transformation from obs to cen plane

    pars is a dict
    (x,y) is position in obs plane

    NOTE: we don't do the usual as we would have to
    setup a 3x3 matrix to do the translation
    '''
    raise NotImplementedError('obs2cen not currently handled by a matrix!')

def _transform_cen2obs(pars):
    '''
    Lensing transformation from cen to obs plane

    pars is a dict
    (x,y) is position in cen plane

    NOTE: we don't do the usual as we would have to
    setup a 3x3 matrix to do the translation
    '''
    raise NotImplementedError('cen2obs not currently handled by a matrix!')

def _transform_cen2source(pars):
    '''
    Lensing transformation from cen to source plane

    pars is a dict
    (x,y) is position in cen plane
    '''

    g1, g2 = pars['g1'], pars['g2']

    # Lensing transformation
    # NOTE: ignoring kappa for now
    transform = np.array([
        [1.-g1, -g2],
        [-g2, 1.+g1]
    ])

    return transform

def _transform_source2cen(pars):
    '''
    Inverse lensing transformation from source to cen plane

    pars is a dict
    (x,y) is position in source plane
    '''

    g1, g2 = pars['g1'], pars['g2']

    # NOTE: ignoring kappa for now
    norm = 1. / (1. - (g1**2 + g2**2))
    transform = norm * np.array([
        [1.+g1, g2],
        [g2, 1.-g1]
    ])

    return transform

def _transform_source2gal(pars):
    '''
    Rotation by intrinsic angle

    pars is a dict
    (x,y) is position in source plane
    '''

    theta_int = pars['theta_int']

    # want to 'subtract' orientation
    theta = -theta_int

    c, s = np.cos(theta), np.sin(theta)

    transform = np.array([
        [c, -s],
        [s,  c]
    ])

    return transform

def _transform_gal2source(pars):
    '''
    Rotation by intrinsic angle

    pars is a dict
    (x,y) is position in gal plane
    '''

    theta_int = pars['theta_int']

    c, s = np.cos(theta_int), np.sin(theta_int)

    transform = np.array([
        [c, -s],
        [s,  c]
    ])

    return transform

def _transform_gal2disk(pars):
    '''
    Account for inclination angle

    pars is a dict
    (x,y) is position in galaxy plane
    '''

    sini = pars['sini']
    cosi = np.sqrt(1-sini**2)

    transform = np.array([
        [1., 0],
        [0, 1. / cosi]
    ])

    return transform

def _transform_disk2gal(pars):
    '''
    Account for inclination angle

    pars is a dict
    (x,y) is position in disk plane
    '''

    sini = pars['sini']
    cosi = np.sqrt(1-sini**2)

    transform = np.array([
        [1., 0],
        [0, cosi]
    ])

    return transform

def _obs2cen(pars, x, y):
    '''
    pars is a dict
    (x,y) is position in obs plane

    returns: (x', y') in cen plane
    '''

    # NOTE: we don't do the usual as we would have to
    # setup a 3x3 matrix to do the translation; we
    # instead use a simpler approach
    # transform = _transform_obs2cen(pars)
    # return _multiply(transform, x, y)

    try:
        x0 = pars['x0']
        y0 = pars['y0']
    except KeyError:
        # no offsets to apply in passed model
        x0 = 0.
        y0 = 0.

    xp = x - x0
    yp = y - y0

    return xp, yp

def _cen2source(pars, x, y):
    '''
    pars is a dict
    (x,y) is position in cenplane

    returns: (x', y') in cen plane
    '''

    transform = _transform_cen2source(pars)

    return _multiply(transform, x, y)

def _source2gal(pars, x, y):
    '''
    pars is a dict
    (x,y) is position in source plane

    returns: (x', y') in gal plane
    '''

    transform = _transform_source2gal(pars)

    return _multiply(transform, x, y)

def _gal2disk(pars, x, y):
    '''
    pars is a dict
    (x,y) is position in gal plane

    returns: (x', y') in disk plane
    '''

    transform = _transform_gal2disk(pars)

    return _multiply(transform, x, y)

def _cen2obs(pars, x, y):
    '''
    pars is a dict
    (x,y) is position in cen plane

    returns: (x', y') in obs plane
    '''

    # NOTE: we don't do the usual as we would have to
    # setup a 3x3 matrix to do the translation; we
    # instead use a simpler approach
    # transform = _transform_obs2cen(pars)
    # return _multiply(transform, x, y)

    try:
        x0 = pars['x0']
        y0 = pars['y0']
    except KeyError:
        # no offsets to apply in passed model
        x0 = 0.
        y0 = 0.

    xp = x + x0
    yp = y + y0

    return xp, yp

def _source2cen(pars, x, y):
    '''
    pars is a dict
    (x,y) is position in source plane

    returns: (x', y') in cen plane
    '''

    transform = _transform_source2cen(pars)

    return _multiply(transform, x, y)

def _gal2source(pars, x, y):
    '''
    pars is a dict
    (x,y) is position in gal plane

    returns: (x', y') in source plane
    '''

    transform = _transform_gal2source(pars)

    return _multiply(transform, x, y)

def _disk2gal(pars, x, y):
    '''
    pars is a dict
    (x,y) is position in disk plane

    returns: (x', y') in gal plane
    '''

    transform = _transform_disk2gal(pars)

    return _multiply(transform, x, y)
