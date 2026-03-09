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
DEFAULT_DATA_DIR = "/ocean/projects/phy250048p/shared/tng/data/tng50"
DEFAULT_SPLIT_DIR = Path(DEFAULT_DATA_DIR) / "gold_split"


def _resolve_split_modality_dir(split_data_dir: Optional[Path], modality: str) -> Path:
    """Resolve modality directory under split data root."""
    root = Path(split_data_dir) if split_data_dir else DEFAULT_SPLIT_DIR
    return root / modality


def _load_split_manifest_index(modality_dir: Path) -> Dict[str, Dict[Any, str]]:
    """Load manifest.csv mapping for split files."""
    manifest_path = modality_dir / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Split manifest not found: {manifest_path}\n"
            "Run the split utility to generate per-galaxy files first."
        )

    mapping_index: Dict[int, str] = {}
    mapping_subhalo: Dict[int, str] = {}

    # Use numpy text parsing to avoid adding extra dependencies.
    data = np.genfromtxt(
        manifest_path,
        delimiter=",",
        names=True,
        dtype=None,
        encoding="utf-8",
    )

    if np.size(data) == 0:
        return {"index": mapping_index, "subhalo_id": mapping_subhalo}

    rows = np.atleast_1d(data)
    for row in rows:
        idx = int(row["index"])
        fname = str(row["filename"])
        mapping_index[idx] = fname

        subhalo_val = row["subhalo_id"]
        if subhalo_val not in ("", "None"):
            try:
                mapping_subhalo[int(subhalo_val)] = fname
            except (TypeError, ValueError):
                pass

    return {"index": mapping_index, "subhalo_id": mapping_subhalo}


def _load_single_split_galaxy(modality_dir: Path, filename: str) -> Dict[str, Any]:
    """Load a single per-galaxy NPZ file into a dictionary."""
    path = modality_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Split galaxy file not found: {path}")

    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def _load_split_galaxy_data(
    modality: str,
    split_data_dir: Optional[Path] = None,
    index: Optional[int] = None,
    subhalo_id: Optional[int] = None,
) -> Dict[str, Any]:
    """Load one galaxy dictionary from split data."""
    modality_dir = _resolve_split_modality_dir(split_data_dir, modality)
    if not modality_dir.exists():
        raise FileNotFoundError(
            f"Split modality directory not found: {modality_dir}\n"
            "Run the split utility or pass split_data_dir to the correct location."
        )

    mapping = _load_split_manifest_index(modality_dir)

    if index is None and subhalo_id is None:
        raise ValueError("Must provide either index or subhalo_id")

    if index is not None and subhalo_id is not None:
        raise ValueError("Provide only one of index or subhalo_id")

    if index is not None:
        if index not in mapping["index"]:
            raise IndexError(
                f"Index {index} not found in split manifest for '{modality}'"
            )
        return _load_single_split_galaxy(modality_dir, mapping["index"][index])

    if subhalo_id not in mapping["subhalo_id"]:
        raise ValueError(
            f"SubhaloID {subhalo_id} not found in split manifest for '{modality}'"
        )
    return _load_single_split_galaxy(modality_dir, mapping["subhalo_id"][subhalo_id])


class TNG50Galaxy:
    """Container for loading a single galaxy from split gold sample files."""

    def __init__(
        self,
        index: Optional[int] = None,
        subhalo_id: Optional[int] = None,
        split_data_dir: Optional[Path] = None,
        load_gas: bool = True,
        load_stellar: bool = True,
        load_subhalo: bool = True,
    ):
        """Initialize a single-galaxy loader backed by split files.

        Parameters
        ----------
        index : int, optional
            Galaxy index in split manifests.
        subhalo_id : int, optional
            SubhaloID of the target galaxy.
        split_data_dir : Path, optional
            Root directory containing split folders (gas/stellar/subhalo).
        load_gas : bool, default=True
            Whether to load gas data for this galaxy.
        load_stellar : bool, default=True
            Whether to load stellar data for this galaxy.
        load_subhalo : bool, default=True
            Whether to load subhalo data for this galaxy.
        """
        if index is None and subhalo_id is None:
            raise ValueError("Must provide either index or subhalo_id")
        if index is not None and subhalo_id is not None:
            raise ValueError("Provide only one of index or subhalo_id")

        self.index = index
        self.subhalo_id = subhalo_id
        self.split_data_dir = (
            Path(split_data_dir) if split_data_dir else DEFAULT_SPLIT_DIR
        )

        self.gas = self.load_gas_data() if load_gas else None
        self.stellar = self.load_stellar_data() if load_stellar else None
        self.subhalo = self.load_subhalo_data() if load_subhalo else None

    def load_gas_data(self) -> Dict[str, Any]:
        """Load gas data for this single galaxy from split files."""
        return _load_split_galaxy_data(
            modality="gas",
            split_data_dir=self.split_data_dir,
            index=self.index,
            subhalo_id=self.subhalo_id,
        )

    def load_stellar_data(self) -> Dict[str, Any]:
        """Load stellar data for this single galaxy from split files."""
        return _load_split_galaxy_data(
            modality="stellar",
            split_data_dir=self.split_data_dir,
            index=self.index,
            subhalo_id=self.subhalo_id,
        )

    def load_subhalo_data(self) -> Dict[str, Any]:
        """Load subhalo data for this single galaxy from split files."""
        return _load_split_galaxy_data(
            modality="subhalo",
            split_data_dir=self.split_data_dir,
            index=self.index,
            subhalo_id=self.subhalo_id,
        )

    def get_galaxy(self) -> Dict[str, Optional[Dict[str, Any]]]:
        """Return loaded single-galaxy data across modalities."""
        return {
            "gas": self.gas,
            "stellar": self.stellar,
            "subhalo": self.subhalo,
        }

    def get_available_keys(self, data_type: str = "all") -> Dict[str, List[str]]:
        """Get available keys for loaded single-galaxy modalities."""
        result = {}

        if data_type in ["gas", "all"] and self.gas is not None:
            result["gas"] = sorted(self.gas.keys())

        if data_type in ["stellar", "all"] and self.stellar is not None:
            result["stellar"] = sorted(self.stellar.keys())

        if data_type in ["subhalo", "all"] and self.subhalo is not None:
            result["subhalo"] = sorted(self.subhalo.keys())

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
            "TNG50Galaxy("
            f"index={self.index}, subhalo_id={self.subhalo_id}, "
            f"loaded={loaded}, split_data_dir='{self.split_data_dir}'"
            ")"
        )


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
        use_gold: bool = False,
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
        use_gold : bool, default=False
            Whether to load data from gold samples (if True, loads *_gold.npz files)
        """
        self.data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR

        self.gas = load_gas_data(self.data_dir, use_gold=use_gold) if load_gas else None
        self.stellar = (
            load_stellar_data(self.data_dir, use_gold=use_gold)
            if load_stellar
            else None
        )
        self.subhalo = (
            load_subhalo_data(self.data_dir, use_gold=use_gold)
            if load_subhalo
            else None
        )

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


def load_gas_data(
    data_dir: Optional[Path] = None,
    use_gold: bool = False,
) -> np.ndarray:
    """
    Load TNG50 gas mock data.

    Parameters
    ----------
    data_dir : Path, optional
        Directory containing the data file. If None, uses default location.
    use_gold : bool, default=False
        Whether to use gas data from gold samples.
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
    if use_gold:
        filepath = data_dir / "gas_data_analysis_gold.npz"
    else:
        filepath = data_dir / "gas_data_analysis.npz"

    if not filepath.exists():
        raise FileNotFoundError(
            f"Gas data file not found: {filepath}\n"
            "Run 'make download-cyverse-data' to download TNG50 mock data."
        )

    with np.load(filepath, allow_pickle=True) as data:
        return data['arr_0']


def load_stellar_data(
    data_dir: Optional[Path] = None,
    use_gold: bool = False,
) -> np.ndarray:
    """
    Load TNG50 stellar mock data.

    Parameters
    ----------
    data_dir : Path, optional
        Directory containing the data file. If None, uses default location.
    use_gold : bool, default=False
        Whether to use stellar data from gold samples.
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
    if use_gold:
        filepath = data_dir / "stellar_data_analysis_gold.npz"
    else:
        filepath = data_dir / "stellar_data_analysis.npz"

    if not filepath.exists():
        raise FileNotFoundError(
            f"Stellar data file not found: {filepath}\n"
            "Run 'make download-cyverse-data' to download TNG50 mock data."
        )

    with np.load(filepath, allow_pickle=True) as data:
        return data['arr_0']


def load_subhalo_data(
    data_dir: Optional[Path] = None,
    use_gold: bool = False,
) -> np.ndarray:
    """
    Load TNG50 subhalo mock data.

    Parameters
    ----------
    data_dir : Path, optional
        Directory containing the data file. If None, uses default location.
    use_gold : bool, default=False
        Whether to use subhalo data from gold samples.
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
    if use_gold:
        filepath = data_dir / "subhalo_data_analysis_gold.npz"
    else:
        filepath = data_dir / "subhalo_data_analysis.npz"

    if not filepath.exists():
        raise FileNotFoundError(
            f"Subhalo data file not found: {filepath}\n"
            "Run 'make download-cyverse-data' to download TNG50 mock data."
        )

    with np.load(filepath, allow_pickle=True) as data:
        return data['arr_0']


def get_available_keys(
    data_dir: Optional[Path] = None,
) -> Dict[str, Union[List[str], None]]:
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
