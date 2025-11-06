import numpy as np
from abc import abstractmethod, ABC

from kl_pipe.parameters import ImagePars

# NOTE: Current descope
# class VelocityPars(ABC):
#     '''
#     The parameters needed for a given velocity model.
#     '''

#     def __init__(self, pars: dict | np.ndarray) -> None:

#         if isinstance(pars, dict):
#             # initialize from dictionary
#             # ...
#             pass
#         elif isinstance(pars, np.ndarray):
#             # initialize from numpy array
#             # ...
#             pass
#         else:
#             raise TypeError("pars must be of type dict or np.ndarray")

#         # self._pars = ...
#         # self._theta = ...

#         return

#     @property
#     @abstractmethod
#     def pars(self) -> dict:
#         '''
#         The parameters stored as a dictionary.
#         '''
#         pass

#     @property
#     @abstractmethod
#     def theta(self) -> np.ndarray:
#         '''
#         The parameters stored as a 1D numpy array.
#         '''
#         pass

class VelcocityModel(ABC):
    '''
    The velocity model itself.
    '''

    def __init__(
            self,
            pars: dict = None,
            theta: np.array = None
            ) -> None:

        if (pars is not None) and (theta is not None):
            raise ValueError('Only one of pars or theta should be provided.')

        if (pars is None) and (theta is None):
            raise ValueError('One of pars or theta must be provided.')

        if pars is not None:
            theta = self.pars2theta(pars)

        self.theta = theta

        return

    @property
    @abstractmethod
    def name(self) -> str:
        '''
        The name of the velocity model.
        '''
        pass

    @abstractmethod
    def theta2pars(self, theta: np.array) -> dict:
        '''
        The parameters of the velocity model, stored as a dictionary indexed by 
        parameter names.
        '''
        pass

    @abstractmethod
    def pars2theta(self, pars: dict) -> np.ndarray:
        '''
        The parameters of the velocity model, stored as an ordered 1D numpy array.
        '''
        pass

    def update_parameters(self, new_parameters: np.ndarray | dict) -> None:
        '''
        '''

        if isinstance(new_parameters, np.ndarray):
            self.theta = new_parameters
        elif isinstance(new_parameters, dict):
            # update parameters object
            self.theta = self.pars2theta(new_parameters)
        else:
            raise TypeError(
                'new_parameters must be of type np.ndarray or dict'
                )

        return

    def render(self, render_type: str) -> VelocityMap | VelocityCube:
        '''
        High level function for endering the velocity model.
        Calls the relevant functions depending on the requested `render_type`
        '''

        if render_type == 'map':
            return self.render_map()
        elif render_type == 'cube':
            return self.render_cube()

        pass

    def render_map(self, image_pars: ImagePars) -> VelocityMap:
        '''
        Render the velocity model onto a discretized 2D image.
        '''
        # return VelocityMap(...)
        pass

    def render_cube(self, cube_pars) -> VelocityCube:
        '''
        Render the velocity model onto a discretized 3D spectral cube.
        '''
        # return VelocityCube(...)
        pass

    @abstractmethod
    def __call__(
            self,
            plane: str,
            X: np.ndarray,
            Y: np.ndarray,
            Z: np.ndarray = None,
            ) -> np.ndarray:
        '''
        Evaluate the velocity model at given positions.
        '''
        # return np.ndarray
        pass

class VelocityCube:
    '''
    The rendering of a velocity model onto a 3D spectral cube.
    '''
    pass

class VelocityMap(object):
    '''
    Container for  rendered velocity map and the necessary meta data
    '''

    def __init__(
        self,
        vmap: np.ndarray,
        image_pars: ImagePars,
        truth_pars: VelocityPars = None
        ) -> None:
        # ...
        # self._truth_pars = truth_pars
        pass

def build_velocity_model(
        name: str,
        pars: VelocityPars
        ) -> VelcocityModel:
    '''
    A factory function to build velocity models.
    '''
    pass

# ...