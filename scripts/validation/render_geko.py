#!/usr/bin/env python
"""
Render all validation tests using geko (astro-geko).

Part of the grism cross-code validation (see docs/validation/grism_validation_plan.md).
Loads the 35 test configurations from test_params.yaml, maps parameters to geko's
native format via get_geko_params(), and renders noiseless datacube + grism +
velocity/intensity maps. Outputs are saved as .npz files for comparison against
kl_pipe, kl-tools, and grizli.

Each .npz contains keys: cube, grism, vmap, imap, lambda_grid.

STATUS: Skeleton — geko API calls are placeholders pending parameter mapping
verification (PA convention, re definition, q0 vs int_h_over_r, Ie normalization).
See test_params.yaml and docs/validation/grism_validation_plan.md open questions.

Requires: pip install astro-geko

Usage
-----
    # render all 35 tests
    python scripts/validation/render_geko.py

    # render a single test
    python scripts/validation/render_geko.py --test rotating_base

    # custom output directory
    python scripts/validation/render_geko.py --outdir /tmp/val

    # custom config YAML
    python scripts/validation/render_geko.py --config my_params.yaml

CLI Arguments
-------------
--test TEST_NAME   Render a single test by name (default: all 35).
--outdir DIR       Output directory (default: tests/data/validation/geko).
--config PATH      Config YAML path (default: scripts/validation/test_params.yaml).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from utils import get_geko_params, get_test_params, load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Render geko validation outputs')
    parser.add_argument(
        '--test',
        type=str,
        default=None,
        help='Render single test (default: all)',
    )
    parser.add_argument(
        '--outdir',
        type=str,
        default='tests/data/validation/geko',
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
    """Render datacube + grism for a single test via geko."""
    try:
        import geko
    except ImportError:
        raise ImportError("geko not installed. Install with: pip install astro-geko")

    geko_pars = get_geko_params(test_name, config)
    params = get_test_params(test_name, config)
    obs = params['observation']

    # ------------------------------------------------------------------
    # TODO: Replace with actual geko API once parameter mapping verified.
    # The calls below are placeholders based on expected geko interface.
    # ------------------------------------------------------------------

    # build geko galaxy model
    # galaxy = geko.Galaxy(
    #     Va=geko_pars['Va'],
    #     rt=geko_pars['rt'],
    #     i=geko_pars['i'],
    #     PAmorph=geko_pars['PAmorph'],
    #     PAkin=geko_pars['PAkin'],
    #     re=geko_pars['re'],
    #     Ie=geko_pars['Ie'],
    #     v0=geko_pars['v0'],
    #     sigma0=geko_pars['sigma0'],
    #     q0=geko_pars['q0'],
    #     z=geko_pars['z'],
    #     n_sersic=geko_pars['n_sersic'],
    # )

    # build observation grid
    # Nrow, Ncol = obs['Nrow'], obs['Ncol']
    # pixel_scale = obs['pixel_scale']

    # render datacube
    # cube = galaxy.render_datacube(Nrow=Nrow, Ncol=Ncol, pixel_scale=pixel_scale, ...)
    # lambda_grid = galaxy.lambda_grid

    # render grism
    # grism = galaxy.render_grism(
    #     dispersion=obs['dispersion'],
    #     dispersion_angle=obs['dispersion_angle'],
    #     ...
    # )

    # render maps
    # vmap = galaxy.velocity_map(Nrow=Nrow, Ncol=Ncol, pixel_scale=pixel_scale)
    # imap = galaxy.intensity_map(Nrow=Nrow, Ncol=Ncol, pixel_scale=pixel_scale)

    raise NotImplementedError(
        f"geko rendering for '{test_name}' not yet implemented. "
        "Waiting on parameter mapping verification (see test_params.yaml open questions)."
    )

    # save
    # outdir.mkdir(parents=True, exist_ok=True)
    # outpath = outdir / f'{test_name}.npz'
    # np.savez(
    #     outpath,
    #     cube=np.asarray(cube),
    #     grism=np.asarray(grism),
    #     vmap=np.asarray(vmap),
    #     imap=np.asarray(imap),
    #     lambda_grid=np.asarray(lambda_grid),
    # )
    # print(f"  saved {outpath}")


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
        try:
            render_test(test_name, config, outdir)
            print(f"  {time.time() - t1:.1f}s")
        except NotImplementedError as e:
            print(f"  SKIPPED: {e}")

    print(f"\nDone in {time.time() - t0:.1f}s")


if __name__ == '__main__':
    main()
