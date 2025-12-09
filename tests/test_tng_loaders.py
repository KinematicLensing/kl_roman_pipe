"""
Tests for TNG50 mock data loading.
"""

import pytest
import numpy as np
from pathlib import Path
from kl_pipe.tng import (
    load_gas_data,
    load_stellar_data,
    load_subhalo_data,
    TNG50MockData,
    get_available_keys,
)


@pytest.mark.tng50
def test_load_gas_data():
    """Test loading gas mock data."""
    data = load_gas_data()
    assert isinstance(data, np.ndarray)
    assert len(data) > 0
    assert isinstance(data[0], dict)


@pytest.mark.tng50
def test_load_stellar_data():
    """Test loading stellar mock data."""
    data = load_stellar_data()
    assert isinstance(data, np.ndarray)
    assert len(data) > 0
    assert isinstance(data[0], dict)


@pytest.mark.tng50
def test_load_subhalo_data():
    """Test loading subhalo mock data."""
    data = load_subhalo_data()
    assert isinstance(data, np.ndarray)
    assert len(data) > 0
    assert isinstance(data[0], dict)


@pytest.mark.tng50
def test_tng50_mock_data_all():
    """Test loading all data via TNG50MockData class."""
    mock_data = TNG50MockData()

    assert mock_data.gas is not None
    assert mock_data.stellar is not None
    assert mock_data.subhalo is not None

    assert isinstance(mock_data.gas, np.ndarray)
    assert isinstance(mock_data.stellar, np.ndarray)
    assert isinstance(mock_data.subhalo, np.ndarray)
    
    # Test new features
    assert mock_data.n_galaxies == len(mock_data.gas)
    assert mock_data.subhalo_ids is not None
    assert len(mock_data) == mock_data.n_galaxies


@pytest.mark.tng50
def test_tng50_mock_data_selective():
    """Test selective loading of data."""
    mock_data = TNG50MockData(load_gas=True, load_stellar=False, load_subhalo=False)

    assert mock_data.gas is not None
    assert mock_data.stellar is None
    assert mock_data.subhalo is None


@pytest.mark.tng50
def test_get_available_keys():
    """Test getting available data keys."""
    keys = get_available_keys()

    assert 'gas' in keys
    assert 'stellar' in keys
    assert 'subhalo' in keys

    # All should be lists (or None if files not found)
    assert isinstance(keys['gas'], (list, type(None)))
    assert isinstance(keys['stellar'], (list, type(None)))
    assert isinstance(keys['subhalo'], (list, type(None)))


def test_missing_files_raise_error():
    """Test that missing files raise FileNotFoundError."""
    fake_dir = Path("/nonexistent/directory")

    with pytest.raises(FileNotFoundError, match="Run 'make download-cyverse-data'"):
        load_gas_data(fake_dir)

    with pytest.raises(FileNotFoundError, match="Run 'make download-cyverse-data'"):
        load_stellar_data(fake_dir)

    with pytest.raises(FileNotFoundError, match="Run 'make download-cyverse-data'"):
        load_subhalo_data(fake_dir)
