import jax
import jax.numpy as jnp

from kl_pipe.model import KLModel

def log_likelihood(
    theta: jnp.ndarray,
    kl_model: KLModel,
    datavector: jnp.ndarray,
    meta_params: dict
) -> float:
    """
    Compute log-likelihood for kinematic lensing model.
    
    Parameters
    ----------
    theta : jnp.ndarray
        Composite parameter array.
    kl_model : KLModel
        Combined velocity and intensity model.
    datavector : jnp.ndarray
        Observed data vector.
    meta_params : dict
        Fixed metadata including coordinate grids.
        
    Returns
    -------
    float
        Log-likelihood value.
    """

    velocity_map, intensity_map = kl_model(
        theta,
        plane='obs',
        X=meta_params['X'],
        Y=meta_params['Y']
    )
    
    model_prediction = velocity_map * intensity_map
    
    residuals = datavector - model_prediction
    chi2 = jnp.sum(residuals**2)
    
    return -0.5 * chi2