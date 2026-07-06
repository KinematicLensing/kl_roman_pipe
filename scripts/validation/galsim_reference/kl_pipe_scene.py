"""``kl_pipe``-side scene construction + render for the GalSim-chromatic
reference gate, following the ``tests/test_flagship.py`` construction
pattern (simplified: no shear, no continuum, centered galaxy, Halpha line
only -- matches what ``galsim_reference.render`` supports).
"""

from __future__ import annotations

import dataclasses

import numpy as np
import galsim

from kl_pipe.dispersion import build_grism_pars_for_line
from kl_pipe.intensity import InclinedExponentialModel
from kl_pipe.lines import EmissionLine, LINE_LAMBDAS
from kl_pipe.observation import build_grism_obs
from kl_pipe.parameters import ImagePars
from kl_pipe.render import RenderConfig
from kl_pipe.source import SourceModel
from kl_pipe.velocity import CenteredVelocityModel


def true_pars_dotted(z: float, vcirc: float, g1: float = 0.0, g2: float = 0.0) -> dict:
    """Truth parameter dict (dotted ``SourceModel`` keys) for the reference
    scene: inclined exponential disk, arctan rotation curve, Halpha line
    only, no continuum/off-center. Shear defaults to zero.
    """
    return {
        'cosi': 0.6,
        'theta_int': np.pi / 4,
        'g1': g1,
        'g2': g2,
        'vel.v0': 10.0,
        'vel.vcirc': vcirc,
        'vel.rscale': 0.3,
        'Halpha.flux': 100.0,
        'Halpha.rscale': 0.25,
        'Halpha.h_over_r': 0.1,
        'Halpha.x0': 0.0,
        'Halpha.y0': 0.0,
        'Halpha.dispersion': 50.0,
        'z': z,
    }


def build_kl_pipe_scene(
    z: float = 1.0,
    vcirc: float = 200.0,
    pixel_scale: float = 0.11,
    shape: tuple = (32, 32),
    psf_fwhm: float = 0.18,
    dispersion_nm_per_pix: float = 1.1,
    oversample: int = 5,
    n_lambda: int | None = None,
    g1: float = 0.0,
    g2: float = 0.0,
    throughput_fn=None,
):
    """Build the ``SourceModel`` + grism obs for the reference-gated scene.

    Parameters
    ----------
    n_lambda : int, optional
        If given, overrides ``GrismPars.to_cube_pars``'s default coarse
        wavelength-slice count (caller-tuned resolution for the
        velocity-entanglement pathway; see
        ``docs/validation/galsim_reference_gate.md``).
    g1, g2 : float
        Reduced lensing shear applied to the scene (default 0).
    throughput_fn : callable, optional
        Wavelength-dependent throughput T(lambda_nm), vectorized. Sampled
        at the final cube wavelength grid and installed as
        ``GrismPars.throughput``. None = flat 100%.

    Returns
    -------
    dict with keys: 'source', 'pars', 'grism_pars', 'obs_grism',
    'image_pars', 'psf'.
    """
    pars = true_pars_dotted(z, vcirc, g1=g1, g2=g2)
    image_pars = ImagePars(shape=shape, pixel_scale=pixel_scale, indexing='ij')
    psf = galsim.Gaussian(fwhm=psf_fwhm)

    grism_pars = build_grism_pars_for_line(
        LINE_LAMBDAS['Halpha'],
        redshift=z,
        image_pars=image_pars,
        dispersion=dispersion_nm_per_pix,
    )

    halpha_int = InclinedExponentialModel()
    source = SourceModel(
        velocity_model=CenteredVelocityModel(),
        emission_lines={'Halpha': EmissionLine(intensity=halpha_int)},
    )

    obs_grism = build_grism_obs(
        grism_pars,
        z=z,
        psf=psf,
        render_config=RenderConfig(oversample=oversample),
    )

    if n_lambda is not None:
        cube_pars = grism_pars.to_cube_pars(z=z, n_lambda=n_lambda)
        obs_grism = dataclasses.replace(obs_grism, cube_pars=cube_pars)

    if throughput_fn is not None:
        # sample T at the final cube wavelength grid (after any n_lambda
        # override) so the per-slice throughput array lines up with the
        # slices disperse_cube integrates
        lam = np.asarray(obs_grism.cube_pars.lambda_grid)
        throughput = np.asarray(throughput_fn(lam), dtype=float)
        if throughput.shape != lam.shape or np.any(throughput < 0):
            raise ValueError(
                'throughput_fn must map a wavelength array to an equal-shape '
                'array of non-negative values'
            )
        grism_pars = dataclasses.replace(grism_pars, throughput=throughput)
        obs_grism = dataclasses.replace(obs_grism, grism_pars=grism_pars)

    return {
        'source': source,
        'pars': pars,
        'grism_pars': grism_pars,
        'obs_grism': obs_grism,
        'image_pars': image_pars,
        'psf': psf,
    }


def render_kl_pipe_grism(scene: dict) -> np.ndarray:
    """Render the dispersed grism image with kl_pipe's own API."""
    image = scene['source'].render_grism(scene['pars'], scene['obs_grism'])
    return np.asarray(image)
