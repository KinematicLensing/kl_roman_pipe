"""
Tests for scripts/validation/utils.py against the real test_params.yaml.

No grism_validation marker -- runs under default `make test`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# add scripts/validation to path for utils
_SCRIPTS_DIR = Path(__file__).parent.parent / 'scripts' / 'validation'
sys.path.insert(0, str(_SCRIPTS_DIR))

from utils import (
    compare_datacubes,
    compare_images,
    get_geko_params,
    get_kl_pipe_params,
    get_test_params,
    load_config,
    load_tolerances,
)

EXPECTED_TEST_COUNT = 35


class TestValidationUtils:
    """Tests for validation utility functions."""

    # ---- load_config ----

    def test_load_config_keys(self):
        config = load_config()
        assert 'base_params' in config
        assert 'observation' in config
        assert 'tests' in config
        assert 'tolerances' in config
        assert 'rendering' in config

    def test_load_config_test_count(self):
        config = load_config()
        assert len(config['tests']) == EXPECTED_TEST_COUNT

    # ---- get_test_params ----

    def test_get_test_params_static_base(self):
        p = get_test_params('static_base')
        # base params unchanged
        assert p['vcirc'] == 0.0
        assert p['flux'] == 100.0
        assert p['cosi'] == 0.5
        assert 'observation' in p

    def test_get_test_params_rotating_psf(self):
        p = get_test_params('rotating_psf')
        assert p['vcirc'] == 200.0
        assert p['observation']['psf_fwhm'] == 0.15

    def test_get_test_params_redshift_sweep(self):
        p = get_test_params('redshift_sweep_05')
        assert p['z'] == 0.5

    def test_get_test_params_nonexistent(self):
        with pytest.raises(KeyError, match='nonexistent'):
            get_test_params('nonexistent')

    # ---- get_kl_pipe_params ----

    def test_get_kl_pipe_params_static_base(self):
        kl = get_kl_pipe_params('static_base')
        assert kl['vel_pars']['vcirc'] == 0.0
        assert kl['int_pars']['flux'] == 100.0

    def test_get_kl_pipe_params_rotating_psf(self):
        kl = get_kl_pipe_params('rotating_psf')
        assert kl['psf_kwargs'] is not None
        assert kl['psf_kwargs']['fwhm'] == 0.15

    # ---- get_geko_params ----

    def test_get_geko_params_static_base(self):
        with pytest.warns(UserWarning, match='geko parameter mapping'):
            gp = get_geko_params('static_base')
        # cosi=0.5 -> i=60 deg
        assert abs(gp['i'] - 60.0) < 0.1
        # re = 1.678 * int_rscale = 1.678 * 0.3
        assert abs(gp['re'] - 1.678 * 0.3) < 1e-6

    # ---- load_tolerances ----

    def test_load_tolerances_primary(self):
        tols = load_tolerances('primary')
        assert 'total_flux_rtol' in tols
        assert 'peak_pixel_rtol' in tols
        assert 'rms_residual' in tols
        assert 'spatial_centroid_pix' in tols

    def test_load_tolerances_nonexistent(self):
        with pytest.raises(KeyError, match='nonexistent'):
            load_tolerances('nonexistent')

    # ---- compare_images ----

    def test_compare_images_identical(self):
        img = np.random.RandomState(42).rand(32, 48)
        tols = load_tolerances('primary')
        results = compare_images(img, img, tols)
        for metric, (value, passed) in results.items():
            assert value == 0.0, f"{metric} should be 0 for identical images"
            if passed is not None:
                assert passed, f"{metric} should pass for identical images"

    # ---- compare_datacubes ----

    def test_compare_datacubes_identical(self):
        rng = np.random.RandomState(42)
        cube = rng.rand(32, 48, 20)
        lam = np.linspace(1300, 1330, 20)
        tols = load_tolerances('primary')
        results = compare_datacubes(cube, cube, lam, tols)
        for metric, (value, passed) in results.items():
            assert value == 0.0, f"{metric} should be 0 for identical cubes"
            if passed is not None:
                assert passed, f"{metric} should pass for identical cubes"
