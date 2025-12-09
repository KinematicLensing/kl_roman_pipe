"""
Load and access TNG50 mock observation data.

This module handles loading the TNG50 mock data files downloaded from CyVerse
and provides a convenient interface for accessing the data.

Data Structure
--------------
TNG50 data files contain particle-level information for simulated galaxies:
- **Gas data**: 8 keys (Coordinates, Velocities, Masses, Temperature, etc.)
- **Stellar data**: 21 keys (Coordinates, Velocities, Masses, multi-band luminosities)
- **Subhalo data**: 102 keys (galaxy-level properties like PA, inclination, SubhaloID)

Each file stores data as a 1D numpy array where each element is a dictionary
containing all particle/property data for one galaxy.
"""

from pathlib import Path
from typing import Optional, Dict, Any, List, Union
import numpy as np


# Default data directory
DEFAULT_DATA_DIR = Path(__file__).parent.parent.parent / "data" / "tng50"


class TNG50MockData:
    """
    Container for TNG50 mock observation data with convenient galaxy access.

    Attributes
    ----------
    gas : np.ndarray or None
        Array of gas data dicts from gas_data_analysis.npz
    stellar : np.ndarray or None
        Array of stellar data dicts from stellar_data_analysis.npz
    subhalo : np.ndarray or None
        Array of subhalo data dicts from subhalo_data_analysis.npz
    data_dir : Path
        Directory containing the TNG50 data files
    n_galaxies : int
        Number of galaxies in the dataset
    subhalo_ids : np.ndarray or None
        Array of SubhaloID values for each galaxy (if subhalo data loaded)
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

        # Determine number of galaxies
        for data in [self.gas, self.stellar, self.subhalo]:
            if data is not None:
                self.n_galaxies = len(data)
                break
        else:
            self.n_galaxies = 0

        # Extract SubhaloIDs if available
        if self.subhalo is not None:
            self.subhalo_ids = np.array(
                [gal['SubhaloID'] for gal in self.subhalo], dtype=np.int64
            )
        else:
            self.subhalo_ids = None

    def get_galaxy(
        self, index: Optional[int] = None, subhalo_id: Optional[int] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get data for a specific galaxy by index or SubhaloID.

        Parameters
        ----------
        index : int, optional
            Array index of the galaxy (0 to n_galaxies-1)
        subhalo_id : int, optional
            SubhaloID of the galaxy

        Returns
        -------
        dict
            Dictionary with keys 'gas', 'stellar', 'subhalo' containing the
            respective data dicts for the galaxy (None if that data type not loaded)

        Raises
        ------
        ValueError
            If neither index nor subhalo_id provided, or if subhalo_id not found
        IndexError
            If index out of range
        """
        if index is None and subhalo_id is None:
            raise ValueError("Must provide either index or subhalo_id")

        if subhalo_id is not None:
            if self.subhalo_ids is None:
                raise ValueError("Cannot search by subhalo_id: subhalo data not loaded")
            matches = np.where(self.subhalo_ids == subhalo_id)[0]
            if len(matches) == 0:
                raise ValueError(f"SubhaloID {subhalo_id} not found in dataset")
            index = matches[0]

        if index < 0 or index >= self.n_galaxies:
            raise IndexError(f"Index {index} out of range [0, {self.n_galaxies})")

        return {
            'gas': self.gas[index] if self.gas is not None else None,
            'stellar': self.stellar[index] if self.stellar is not None else None,
            'subhalo': self.subhalo[index] if self.subhalo is not None else None,
        }

    def get_available_keys(self, data_type: str = 'all') -> Dict[str, List[str]]:
        """
        Get list of available data keys for each loaded dataset.

        Parameters
        ----------
        data_type : str, default='all'
            Which data type to query: 'gas', 'stellar', 'subhalo', or 'all'

        Returns
        -------
        dict
            Dictionary mapping data type names to lists of available keys
        """
        result = {}

        if data_type in ['gas', 'all'] and self.gas is not None:
            result['gas'] = sorted(self.gas[0].keys())

        if data_type in ['stellar', 'all'] and self.stellar is not None:
            result['stellar'] = sorted(self.stellar[0].keys())

        if data_type in ['subhalo', 'all'] and self.subhalo is not None:
            result['subhalo'] = sorted(self.subhalo[0].keys())

        return result

    def __repr__(self) -> str:
        loaded = []
        if self.gas is not None:
            loaded.append("gas")
        if self.stellar is not None:
            loaded.append("stellar")
        if self.subhalo is not None:
            loaded.append("subhalo")

        return (
            f"TNG50MockData(n_galaxies={self.n_galaxies}, "
            f"loaded={loaded}, data_dir='{self.data_dir}')"
        )

    def __len__(self) -> int:
        """Return number of galaxies."""
        return self.n_galaxies

    def __getitem__(self, index: int) -> Dict[str, Dict[str, Any]]:
        """Get galaxy by index using bracket notation."""
        return self.get_galaxy(index=index)


def load_gas_data(data_dir: Optional[Path] = None) -> np.ndarray:
    """
    Load TNG50 gas mock data.

    Parameters
    ----------
    data_dir : Path, optional
        Directory containing the data file. If None, uses default location.

    Returns
    -------
    np.ndarray
        1D array of length n_galaxies, where each element is a dict containing
        gas particle data with keys: 'Coordinates', 'Velocities', 'Masses',
        'Temperature', 'StarFormationRate', 'GFM_Metallicity', 'GFM_Metals',
        'NeutralHydrogenAbundance'

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
        return data['arr_0']


def load_stellar_data(data_dir: Optional[Path] = None) -> np.ndarray:
    """
    Load TNG50 stellar mock data.

    Parameters
    ----------
    data_dir : Path, optional
        Directory containing the data file. If None, uses default location.

    Returns
    -------
    np.ndarray
        1D array of length n_galaxies, where each element is a dict containing
        stellar particle data with keys including: 'Coordinates', 'Velocities',
        'Masses', 'Dusted_Luminosity_*', 'Raw_Luminosity_*', 'AB_apparent_magnitude_*'
        for bands g, r, i, u, z

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
        return data['arr_0']


def load_subhalo_data(data_dir: Optional[Path] = None) -> np.ndarray:
    """
    Load TNG50 subhalo mock data.

    Parameters
    ----------
    data_dir : Path, optional
        Directory containing the data file. If None, uses default location.

    Returns
    -------
    np.ndarray
        1D array of length n_galaxies, where each element is a dict containing
        subhalo/galaxy-level data with 102 keys including: 'SubhaloID',
        'Position_Angle_star', 'Inclination_star', 'StellarMass', 'SubhaloSFR',
        and many derived properties

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
        return data['arr_0']


def get_available_keys(data_dir: Optional[Path] = None) -> Dict[str, Union[List[str], None]]:
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
        available data keys in each file, or None if file not found
    """
    data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR

    available = {}

    try:
        gas_data = load_gas_data(data_dir)
        available['gas'] = sorted(gas_data[0].keys()) if len(gas_data) > 0 else []
    except FileNotFoundError:
        available['gas'] = None

    try:
        stellar_data = load_stellar_data(data_dir)
        available['stellar'] = (
            sorted(stellar_data[0].keys()) if len(stellar_data) > 0 else []
        )
    except FileNotFoundError:
        available['stellar'] = None

    try:
        subhalo_data = load_subhalo_data(data_dir)
        available['subhalo'] = (
            sorted(subhalo_data[0].keys()) if len(subhalo_data) > 0 else []
        )
    except FileNotFoundError:
        available['subhalo'] = None

    return available
