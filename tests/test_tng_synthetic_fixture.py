"""
Exercise kl_pipe.tng end to end on the synthetic TNG fixture.

Guards the loader/generator API and file layout without the private TNG50
data; physics fidelity is covered by the tng50-marked tests on real data.
"""

import numpy as np
import pytest

from fixtures.tng_synthetic import (
    DEFAULT_SUBHALO_IDS,
    REQUIRED_FILES,
    write_synthetic_tng,
)
from kl_pipe.parameters import ImagePars
from kl_pipe.tng import TNG50MockData, TNGDataVectorGenerator, TNGRenderConfig

SHAPE = (40, 32)


@pytest.fixture(scope='module')
def tng_dir(tmp_path_factory):
    return write_synthetic_tng(tmp_path_factory.mktemp('tng_synthetic'))


@pytest.fixture(scope='module')
def image_pars():
    return ImagePars(shape=SHAPE, pixel_scale=0.11, indexing='ij')


def test_fixture_files_and_loader(tng_dir):
    for name in REQUIRED_FILES:
        assert (tng_dir / name).is_file(), name
    data = TNG50MockData(data_dir=tng_dir)
    assert len(data) == len(DEFAULT_SUBHALO_IDS)
    np.testing.assert_array_equal(data.subhalo_ids, DEFAULT_SUBHALO_IDS)
    galaxy = data.get_galaxy(subhalo_id=DEFAULT_SUBHALO_IDS[0])
    assert galaxy['stellar']['Coordinates'].shape[1] == 3
    assert galaxy['gas']['GFM_Metals'].ndim == 2


def test_fixture_deterministic(tng_dir, tmp_path):
    other = write_synthetic_tng(tmp_path / 'again')
    a = TNG50MockData(data_dir=tng_dir).get_galaxy(index=0)
    b = TNG50MockData(data_dir=other).get_galaxy(index=0)
    for kind in ('stellar', 'gas'):
        for key, val in a[kind].items():
            np.testing.assert_array_equal(val, b[kind][key], err_msg=f'{kind}.{key}')


@pytest.mark.parametrize(
    'orientation',
    [
        None,
        dict(cosi=0.4, theta_int=0.7, g1=0.05, g2=-0.03, x0=0.1, y0=-0.05),
    ],
    ids=['native', 'custom'],
)
@pytest.mark.parametrize('subhalo_id', DEFAULT_SUBHALO_IDS)
def test_render_maps(tng_dir, image_pars, subhalo_id, orientation):
    galaxy = TNG50MockData(data_dir=tng_dir).get_galaxy(subhalo_id=subhalo_id)
    gen = TNGDataVectorGenerator(galaxy, data_dir=tng_dir)
    config = TNGRenderConfig(
        image_pars=image_pars,
        use_native_orientation=orientation is None,
        pars=orientation,
        target_redshift=1.0,
    )

    intensity, int_var = gen.generate_intensity_map(config, snr=50, seed=0)
    velocity, vel_var = gen.generate_velocity_map(config, snr=50, seed=1)
    sfr = gen.generate_sfr_map(config)

    for name, arr in [
        ('intensity', intensity),
        ('intensity variance', int_var),
        ('velocity', velocity),
        ('velocity variance', vel_var),
        ('sfr', sfr),
    ]:
        assert arr.shape == SHAPE, f'{name} shape {arr.shape}'
        assert np.all(np.isfinite(arr)), f'{name} has non-finite pixels'
    clean, _ = gen.generate_intensity_map(config)
    assert clean.sum() > 0
    assert sfr.sum() > 0
    assert np.abs(velocity).max() > 0
