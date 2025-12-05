"""
Load and access TNG50 mock observation data.

This module handles loading the TNG50 mock data files downloaded from CyVerse
and provides a convenient interface for accessing the data.
"""

from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np


# Default data directory
DEFAULT_DATA_DIR = Path(__file__).parent.parent.parent / "data" / "tng50"


class TNG50MockData:
    """
    Container for TNG50 mock observation data.
    
    Attributes
    ----------
    gas : dict or None
        Gas data from gas_data_analysis.npz
    stellar : dict or None
        Stellar data from stellar_data_analysis.npz
    subhalo : dict or None
        Subhalo data from subhalo_data_analysis.npz
    data_dir : Path
        Directory containing the TNG50 data files
    """
    
    def __init__(
        self,
        data_dir: Optional[Path] = None,
        load_gas: bool = True,
        load_stellar: bool = True,
        load_subhalo: bool = True,
    ):
        """
        Initialize TNG50 mock data loader.
        
        Parameters
        ----------
        data_dir : Path, optional
            Directory containing TNG50 data files. If None, uses default location.
        load_gas : bool, default=True
            Whether to load gas data
        load_stellar : bool, default=True
            Whether to load stellar data
        load_subhalo : bool, default=True
            Whether to load subhalo data
        """
        self.data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
        
        self.gas = load_gas_data(self.data_dir) if load_gas else None
        self.stellar = load_stellar_data(self.data_dir) if load_stellar else None
        self.subhalo = load_subhalo_data(self.data_dir) if load_subhalo else None
    
    def __repr__(self) -> str:
        loaded = []
        if self.gas is not None:
            loaded.append("gas")
        if self.stellar is not None:
            loaded.append("stellar")
        if self.subhalo is not None:
            loaded.append("subhalo")
        
        return f"TNG50MockData(loaded={loaded}, data_dir='{self.data_dir}')"


def load_gas_data(data_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load TNG50 gas mock data.
    
    Parameters
    ----------
    data_dir : Path, optional
        Directory containing the data file. If None, uses default location.
    
    Returns
    -------
    dict
        Dictionary containing gas data arrays
    
    Raises
    ------
    FileNotFoundError
        If gas_data_analysis.npz is not found
    """
    data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    filepath = data_dir / "gas_data_analysis.npz"
    
    if not filepath.exists():
        raise FileNotFoundError(
            f"Gas data file not found: {filepath}\n"
            "Run 'make download-cyverse-data' to download TNG50 mock data."
        )
    
    with np.load(filepath, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def load_stellar_data(data_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load TNG50 stellar mock data.
    
    Parameters
    ----------
    data_dir : Path, optional
        Directory containing the data file. If None, uses default location.
    
    Returns
    -------
    dict
        Dictionary containing stellar data arrays
    
    Raises
    ------
    FileNotFoundError
        If stellar_data_analysis.npz is not found
    """
    data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    filepath = data_dir / "stellar_data_analysis.npz"
    
    if not filepath.exists():
        raise FileNotFoundError(
            f"Stellar data file not found: {filepath}\n"
            "Run 'make download-cyverse-data' to download TNG50 mock data."
        )
    
    with np.load(filepath, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def load_subhalo_data(data_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load TNG50 subhalo mock data.
    
    Parameters
    ----------
    data_dir : Path, optional
        Directory containing the data file. If None, uses default location.
    
    Returns
    -------
    dict
        Dictionary containing subhalo data arrays
    
    Raises
    ------
    FileNotFoundError
        If subhalo_data_analysis.npz is not found
    """
    data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    filepath = data_dir / "subhalo_data_analysis.npz"
    
    if not filepath.exists():
        raise FileNotFoundError(
            f"Subhalo data file not found: {filepath}\n"
            "Run 'make download-cyverse-data' to download TNG50 mock data."
        )
    
    with np.load(filepath, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def get_available_keys(data_dir: Optional[Path] = None) -> Dict[str, list]:
    """
    Get available data keys from all TNG50 mock data files.
    
    Parameters
    ----------
    data_dir : Path, optional
        Directory containing the data files. If None, uses default location.
    
    Returns
    -------
    dict
        Dictionary with keys 'gas', 'stellar', 'subhalo' containing lists of
        available data keys in each file
    """
    data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    
    available = {}
    
    try:
        gas_data = load_gas_data(data_dir)
        available['gas'] = list(gas_data.keys())
    except FileNotFoundError:
        available['gas'] = None
    
    try:
        stellar_data = load_stellar_data(data_dir)
        available['stellar'] = list(stellar_data.keys())
    except FileNotFoundError:
        available['stellar'] = None
    
    try:
        subhalo_data = load_subhalo_data(data_dir)
        available['subhalo'] = list(subhalo_data.keys())
    except FileNotFoundError:
        available['subhalo'] = None
    
    return available
