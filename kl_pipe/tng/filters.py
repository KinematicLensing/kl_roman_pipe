"""
Composable filters for TNG particle and image data.

Filters select subsets of particles or image pixels before/after rendering.
Two hierarchies:

- ParticleFilter: applied to particle arrays in the face-on disk frame, before
  gridding. Subclasses implement _compute_mask(**kwargs) -> bool array (N,).
- ImageFilter: applied to rendered 2D maps, after gridding but before PSF
  convolution and noise. Subclasses implement _compute_mask(**kwargs) -> bool
  array (Ny, Nx).

Usage
-----
Pass filter instances via TNGRenderConfig:

    config = TNGRenderConfig(
        image_pars=...,
        use_native_orientation=False,
        pars=pars,
        stellar_filters=[RotationalSupportFilter3D(threshold=0.7)],
        gas_filters=[FoFFilter3D(linking_length=5.0)],
        image_filters=[LargestConnectedFilter2D()],
    )

Combination rules
-----------------
- Single filter: always valid.
- Multiple filters in the same list: AND-combined. The pair must appear in
  VALID_COMBINATIONS or config construction raises ValueError.
- invert=True: selects the complement (e.g. bulge stars instead of disk stars).
  Inverted and non-inverted instances of the same class are logically contradictory;
  the runtime empty-mask guard catches this.
"""

from abc import ABC, abstractmethod
from typing import FrozenSet

import numpy as np
from scipy.ndimage import label as ndimage_label
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

# ---------------------------------------------------------------------------
# Valid multi-filter combinations (allowlist by FILTER_NAME pairs/sets)
# ---------------------------------------------------------------------------

VALID_COMBINATIONS: set = {
    frozenset({"rotational_support_3d", "fof_3d"}),
}
"""
Allowlist of valid multi-filter name sets.

Single-filter use is always valid. For multiple filters in the same list the
frozenset of their FILTER_NAME values must appear here. Extend this set as new
compatible combinations are validated experimentally.
"""


# ---------------------------------------------------------------------------
# Base classes
# ---------------------------------------------------------------------------


class ParticleFilter(ABC):
    """
    Abstract base for disk-frame particle filters.

    Subclasses must define:
        FILTER_NAME: str          unique identifier used in VALID_COMBINATIONS
        REQUIRED_KEYS: frozenset  keys the generator must supply in kwargs
        REQUIRES_DISK_FRAME: bool True means incompatible with native orientation

    Subclasses implement _compute_mask(**kwargs) -> np.ndarray[bool] of shape (N,).
    The base __call__ applies optional inversion.
    """

    FILTER_NAME: str
    REQUIRED_KEYS: FrozenSet[str]
    REQUIRES_DISK_FRAME: bool = False

    def __init__(self, invert: bool = False):
        self.invert = invert

    def __call__(self, **kwargs) -> np.ndarray:
        mask = self._compute_mask(**kwargs)
        return ~mask if self.invert else mask

    @abstractmethod
    def _compute_mask(self, **kwargs) -> np.ndarray:
        """Return raw (non-inverted) boolean mask of shape (N,)."""


class ImageFilter(ABC):
    """
    Abstract base for post-grid image filters.

    Subclasses must define:
        FILTER_NAME: str
        REQUIRED_KEYS: frozenset  must include 'image'

    Subclasses implement _compute_mask(**kwargs) -> np.ndarray[bool] of shape (Ny, Nx).
    """

    FILTER_NAME: str
    REQUIRED_KEYS: FrozenSet[str]

    def __init__(self, invert: bool = False):
        self.invert = invert

    def __call__(self, **kwargs) -> np.ndarray:
        mask = self._compute_mask(**kwargs)
        return ~mask if self.invert else mask

    @abstractmethod
    def _compute_mask(self, **kwargs) -> np.ndarray:
        """Return raw (non-inverted) boolean mask of shape (Ny, Nx)."""


# ---------------------------------------------------------------------------
# Concrete filters
# ---------------------------------------------------------------------------


class RotationalSupportFilter3D(ParticleFilter):
    """
    Select rotationally dominated particles in the face-on disk frame.

    Uses the ratio of azimuthal speed to total speed:
        v_phi = (x * vy - y * vx) / r_cyl
        rotational_support = |v_phi| / |v|

    Particles with rotational_support > threshold are selected (disk stars).
    With invert=True, selects pressure-supported / bulge particles instead.

    Parameters
    ----------
    threshold : float, default=0.5
        Minimum rotational support fraction to pass the filter.
        0.5: warm disk + pseudo-bulge; 0.7: thin disk; 0.9: cold circular only.
    invert : bool, default=False
        If True, select bulge/halo particles (rotational_support <= threshold).
    """

    FILTER_NAME = "rotational_support_3d"
    REQUIRED_KEYS = frozenset({"positions", "velocities"})
    REQUIRES_DISK_FRAME = True

    def __init__(self, threshold: float = 0.5, invert: bool = False):
        super().__init__(invert=invert)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {threshold}")
        self.threshold = threshold

    def _compute_mask(self, **kwargs) -> np.ndarray:
        pos = kwargs["positions"]
        vel = kwargs["velocities"]

        r_cyl = np.sqrt(pos[:, 0] ** 2 + pos[:, 1] ** 2)

        with np.errstate(divide="ignore", invalid="ignore"):
            v_phi = (pos[:, 0] * vel[:, 1] - pos[:, 1] * vel[:, 0]) / r_cyl
        v_phi = np.nan_to_num(v_phi, nan=0.0, posinf=0.0, neginf=0.0)

        v_tot = np.linalg.norm(vel, axis=1)

        with np.errstate(divide="ignore", invalid="ignore"):
            rotational_support = np.abs(v_phi) / v_tot
        rotational_support = np.nan_to_num(rotational_support, nan=0.0)

        return rotational_support > self.threshold


class FoFFilter3D(ParticleFilter):
    """
    Select the largest spatially connected group of particles via Friends-of-Friends.

    Links particles within linking_length of each other (in 3D disk-frame
    coordinates) using scipy.spatial.cKDTree. Returns a mask for the largest
    connected component found.

    With invert=True, selects particles NOT in the largest group (satellites).

    Parameters
    ----------
    linking_length : float
        Maximum separation between linked particles, in the same units as the
        disk-frame position array (comoving kpc/h before arcsec conversion).
    invert : bool, default=False
        If True, select particles outside the largest group.
    """

    FILTER_NAME = "fof_3d"
    REQUIRED_KEYS = frozenset({"positions"})
    REQUIRES_DISK_FRAME = True

    _MAX_PAIRS = 100_000_000

    def __init__(self, linking_length: float, invert: bool = False):
        super().__init__(invert=invert)
        if linking_length < 0:
            raise ValueError(f"linking_length must be >= 0, got {linking_length}")
        self.linking_length = linking_length

    def _compute_mask(self, **kwargs) -> np.ndarray:
        positions = kwargs["positions"]
        n = len(positions)

        tree = cKDTree(positions)
        # output_type='ndarray' returns int array (M, 2) instead of a Python set of
        # tuples — avoids O(M * 72 bytes) object overhead
        pairs = tree.query_pairs(self.linking_length, output_type="ndarray")

        m = len(pairs)
        if m > self._MAX_PAIRS:
            gb = m * 9 / 1e9  # upper-triangle bool CSR: ~9 bytes per edge
            raise ValueError(
                f"FoFFilter3D: linking_length={self.linking_length} produces {m:,} pairs "
                f"({gb:.1f} GB) with {n:,} particles — reduce linking_length to avoid OOM."
            )

        if m == 0:
            mask = np.zeros(n, dtype=bool)
            if n > 0:
                mask[0] = True
            return mask

        data = np.ones(m, dtype=bool)
        graph = csr_matrix((data, (pairs[:, 0], pairs[:, 1])), shape=(n, n))
        _, labels = connected_components(graph, directed=False)

        component_sizes = np.bincount(labels)
        largest_label = int(np.argmax(component_sizes))
        return labels == largest_label


class LargestConnectedFilter2D(ImageFilter):
    """
    Keep only the largest spatially connected region in a 2D image.

    Thresholds the image, labels connected components via scipy.ndimage.label,
    then returns a mask for the largest component by pixel count.

    With invert=True, selects all regions EXCEPT the largest (satellites/companions).

    Parameters
    ----------
    threshold : float, default=0.0
        Pixels with value > threshold are considered non-zero for connectivity.
    invert : bool, default=False
        If True, return mask selecting non-dominant regions.
    """

    FILTER_NAME = "largest_connected_2d"
    REQUIRED_KEYS = frozenset({"image"})

    def __init__(self, threshold: float = 0.0, invert: bool = False):
        super().__init__(invert=invert)
        self.threshold = threshold

    def _compute_mask(self, **kwargs) -> np.ndarray:
        image = kwargs["image"]
        binary = image > self.threshold
        # 8-connectivity: diagonally adjacent pixels are considered connected,
        # preventing thin diagonal features from appearing as separate components.
        labeled, n_features = ndimage_label(
            binary, structure=np.ones((3, 3), dtype=int)
        )

        if n_features == 0:
            return np.zeros(image.shape, dtype=bool)

        if n_features == 1:
            return labeled == 1

        uniq_labels, npix = np.unique(labeled, return_counts=True)
        valid = uniq_labels != 0  # exclude background label
        largest_label = uniq_labels[valid][np.argmax(npix[valid])]
        return labeled == largest_label
