#!/usr/bin/env python
"""
Render all validation tests using kl_pipe.

Part of the grism cross-code validation (see docs/validation/grism_validation_plan.md).
Loads the 28 test configurations from test_params.yaml, builds a KLModel for each,
and renders noiseless datacube + grism + velocity/intensity maps. Outputs are saved
as .npz files for comparison against geko, kl-tools, and grizli.

Each .npz contains keys: cube, grism, vmap, imap, lambda_grid.

Usage
-----
    # render all tests
    python scripts/validation/render_kl_pipe.py

    # render a single test
    python scripts/validation/render_kl_pipe.py --test rotating_base

    # custom output directory
    python scripts/validation/render_kl_pipe.py --outdir /tmp/val

    # custom config YAML
    python scripts/validation/render_kl_pipe.py --config my_params.yaml

CLI Arguments
-------------
--test TEST_NAME   Render a single test by name (default: all 35).
--outdir DIR       Output directory (default: tests/data/validation/kl_pipe).
--config PATH      Config YAML path (default: scripts/validation/test_params.yaml).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update('jax_enable_x64', True)

from kl_pipe.dispersion import GrismPars
from kl_pipe.intensity import InclinedExponentialModel
from kl_pipe.model import KLModel
from kl_pipe.parameters import ImagePars
from kl_pipe.spectral import SpectralConfig, SpectralModel
from kl_pipe.velocity import CenteredVelocityModel

from utils import get_kl_pipe_params, load_config

_SHARED_PARS = {'cosi', 'theta_int', 'g1', 'g2'}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Render kl_pipe validation outputs')
    parser.add_argument(
        '--test',
        type=str,
        default=None,
        help='Render single test (default: all)',
    )
    parser.add_argument(
        '--outdir',
        type=str,
        default='tests/data/validation/kl_pipe',
        help='Output directory',
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Config YAML path (default: scripts/validation/test_params.yaml)',
    )
    return parser


def render_test(test_name: str, config: dict, outdir: Path) -> None:
    """Render datacube + grism for a single test."""
    kl = get_kl_pipe_params(test_name, config)

    # build models
    vel_model = CenteredVelocityModel()
    int_model = InclinedExponentialModel()

    spec_config = SpectralConfig(
        lines=kl['lines'],
        spectral_oversample=kl['rendering']['spectral_oversample'],
    )
    spec_model = SpectralModel(spec_config, int_model, vel_model)

    kl_model = KLModel(
        vel_model,
        int_model,
        shared_pars=_SHARED_PARS,
        spectral_model=spec_model,
    )

    # merge all pars into a flat dict for pars2theta
    all_pars = {}
    all_pars.update(kl['vel_pars'])
    all_pars.update(kl['int_pars'])
    all_pars.update(kl['spectral_pars'])
    theta = kl_model.pars2theta(all_pars)

    # build spatial grid
    image_pars = ImagePars(**kl['image_pars_kwargs'])

    # build grism pars
    grism_pars = GrismPars(image_pars=image_pars, **kl['grism_pars_kwargs'])

    # build cube pars from grism
    cube_pars = grism_pars.to_cube_pars(
        z=kl['z'],
        velocity_window_kms=kl['rendering']['velocity_window_kms'],
        line_lambdas_rest=kl['line_lambdas_rest'],
    )

    # configure PSF if needed
    if kl['psf_kwargs'] is not None:
        import galsim

        fwhm = kl['psf_kwargs']['fwhm']
        psf_type = kl['psf_kwargs']['type']
        if psf_type != 'gaussian':
            raise ValueError(
                f"Unsupported PSF type '{psf_type}'. "
                "Only 'gaussian' is defined in test_params.yaml."
            )
        gsobj = galsim.Gaussian(fwhm=fwhm)

        kl_model.configure_grism_psf(
            gsobj,
            cube_pars,
            oversample=kl['rendering']['spatial_oversample'],
        )

    # render datacube
    cube = kl_model.render_cube(theta, cube_pars, plane='obs')

    # render grism
    grism = kl_model.render_grism(theta, grism_pars, plane='obs', cube_pars=cube_pars)

    # render velocity and intensity maps
    from kl_pipe.utils import build_map_grid_from_image_pars

    X, Y = build_map_grid_from_image_pars(image_pars)
    theta_vel = kl_model.get_velocity_pars(theta)
    theta_int = kl_model.get_intensity_pars(theta)

    vmap = vel_model(theta_vel, 'obs', X, Y)
    imap = int_model.render_unconvolved(theta_int, image_pars)

    # save — all outputs in kl_pipe-native units:
    #   imap: arcsec^-2 (surface brightness)
    #   vmap: km/s
    #   grism/cube: kl_pipe native (arcsec grid, nm wavelength)
    # Other render scripts must convert to these conventions before saving.
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / f'{test_name}.npz'
    np.savez(
        outpath,
        cube=np.asarray(cube),
        grism=np.asarray(grism),
        vmap=np.asarray(vmap),
        imap=np.asarray(imap),
        lambda_grid=np.asarray(cube_pars.lambda_grid),
    )
    print(f"  saved {outpath}")


def main():
    args = build_parser().parse_args()

    config = load_config(args.config)
    outdir = Path(args.outdir)

    tests = [args.test] if args.test else sorted(config['tests'].keys())

    print(f"Rendering {len(tests)} tests to {outdir}/")
    t0 = time.time()

    for test_name in tests:
        t1 = time.time()
        print(f"[{test_name}]")
        render_test(test_name, config, outdir)
        print(f"  {time.time() - t1:.1f}s")

    print(f"\nDone. {len(tests)} tests in {time.time() - t0:.1f}s")


if __name__ == '__main__':
    main()
