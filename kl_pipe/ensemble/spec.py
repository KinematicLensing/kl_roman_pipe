"""
Ensemble spec + observation-config registry.

Two YAML-backed, strictly-validated config layers:

- ``ObservationConfig``: the structural instrument setup (bands, grism rolls,
  PSFs, stamp geometry), shared by every fit in a campaign. Lives in a
  small registry (``configs/observation/<id>.yaml``) and is referenced by id
  from the ensemble spec. The expander snapshots the referenced file verbatim
  into the run's provenance directory with a content hash; the runner loads
  the snapshot, never the live registry.
- ``EnsembleSpec``: one campaign (one run name): the galaxy population
  (stratified grid + drawn nuisance truths + fixed constants + shear/ring
  scheme), model render settings, observation reference + SNR knobs, fit
  settings, dispatch settings, and output retention.

Unknown YAML keys raise. Every enum-like field is validated at construction.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import yaml


def _require_keys(d: dict, required: Tuple[str, ...], context: str) -> None:
    missing = [k for k in required if k not in d]
    if missing:
        raise ValueError(f"{context}: missing required keys {missing}")


def _reject_unknown(d: dict, allowed: Tuple[str, ...], context: str) -> None:
    unknown = [k for k in d if k not in allowed]
    if unknown:
        raise ValueError(
            f"{context}: unknown keys {unknown}; allowed keys are {list(allowed)}"
        )


# =============================================================================
# Observation-config registry
# =============================================================================

_PSF_TYPES = ('gaussian', 'roman_wfi')
_ROMAN_WFI_DEFAULT_SCA = 10
_ROMAN_WFI_DEFAULT_PUPIL_BIN = 4
_GALSIM_DEFAULT_FOLDING_THRESHOLD = 5e-3


@dataclass(frozen=True)
class FoldingThresholdTier:
    """One redshift tier of the fit-kernel folding_threshold schedule.

    ``z_max`` is the inclusive upper redshift bound this tier covers
    (``None`` = open, covering all remaining z); ``ft`` is the GSParams
    folding_threshold applied to fit kernels for scenes in the tier. Tiers
    are stored sorted by ``z_max`` ascending with the open (None) tier last;
    validation lives in ``_validate_folding_tiers``.
    """

    z_max: Optional[float]
    ft: float


def _validate_folding_tiers(
    tiers: Tuple['FoldingThresholdTier', ...], context: str
) -> None:
    """Validate a fit-kernel folding_threshold tier schedule.

    Enforces: non-empty; every ``ft`` a float in (0, 1); the final tier open
    (``z_max is None``) and no earlier tier open; bounded ``z_max`` values
    positive floats and strictly increasing. Together these guarantee the
    schedule covers every non-negative z exactly once.
    """
    if not tiers:
        raise ValueError(f"{context}: folding_threshold_tiers must be non-empty")
    for i, tier in enumerate(tiers):
        if not isinstance(tier.ft, float) or not (0.0 < tier.ft < 1.0):
            raise ValueError(
                f"{context}: tier {i} ft must be a float in (0, 1), got " f"{tier.ft!r}"
            )
        is_last = i == len(tiers) - 1
        if is_last:
            if tier.z_max is not None:
                raise ValueError(
                    f"{context}: final tier must have z_max: null (open upper "
                    f"bound covering all remaining z), got {tier.z_max!r}"
                )
        else:
            if tier.z_max is None:
                raise ValueError(
                    f"{context}: only the final tier may have z_max: null; "
                    f"tier {i} of {len(tiers)} has null z_max"
                )
            if not isinstance(tier.z_max, float) or tier.z_max <= 0:
                raise ValueError(
                    f"{context}: tier {i} z_max must be a positive float, got "
                    f"{tier.z_max!r}"
                )
    bounded = [t.z_max for t in tiers[:-1]]
    if any(b2 <= b1 for b1, b2 in zip(bounded, bounded[1:])):
        raise ValueError(
            f"{context}: tier z_max values must be strictly increasing, got "
            f"{bounded}"
        )


@dataclass(frozen=True)
class PSFSpec:
    """One channel's PSF specification (a broadband band or the grism).

    ``gaussian`` carries ``fwhm_arcsec``; ``roman_wfi`` (realistic WFI PSF
    via ``galsim.roman.getPSF``, monochromatic) carries ``sca`` and
    ``pupil_bin``, plus optional GSParams folding thresholds (None keeps
    GalSim's default 5e-3). A looser threshold shrinks the rendered kernel
    stamp -- and with it the padded convolution FFT -- at the cost of
    truncating more far-wing flux.

    The FIT-model kernels (the per-eval convolution cost) take their
    folding_threshold from EITHER ``folding_threshold`` (a single value for
    all z) OR ``folding_threshold_tiers`` (a redshift schedule); the two are
    mutually exclusive. ``mock_folding_threshold`` applies to the TRUTH
    (mock-data) render kernels, paid once per fit. Leaving
    ``mock_folding_threshold`` at None while loosening the fit threshold
    gives the realistic asymmetry: mock data is rendered at higher kernel
    fidelity than the inference model. The mock kernels must be at least as
    accurate as the fit kernels at every z (raises otherwise; with tiers the
    binding tier is the most accurate = smallest ft). Parameter-level bias
    from the split was gated at folding_threshold=0.01 (all MAP shifts
    <= 0.07 sigma vs a matched-kernel control at line SNR 100) for z <= 1.2;
    the rigor audit found the shape/shear/continuum wing-truncation bias
    grows with kernel size and exceeds that envelope by z=1.9 (cosi -0.093,
    g1 -0.037, continuum -0.066), motivating the tiered schedule (looser ft
    at low z, tighter at high z). 0.02 fails the gate at all z. Cross-type
    fields must be None (raises otherwise).
    """

    psf_type: str
    fwhm_arcsec: Optional[float] = None  # gaussian only, arcsec
    sca: Optional[int] = None  # roman_wfi only, 1-18
    pupil_bin: Optional[int] = None  # roman_wfi only
    folding_threshold: Optional[float] = None  # roman_wfi only, fit kernels (scalar)
    mock_folding_threshold: Optional[float] = None  # roman_wfi only, truth kernels
    # roman_wfi only, fit kernels (z schedule); mutually exclusive with
    # folding_threshold
    folding_threshold_tiers: Optional[Tuple[FoldingThresholdTier, ...]] = None

    def __post_init__(self):
        if self.psf_type == 'gaussian':
            if self.fwhm_arcsec is None or self.fwhm_arcsec <= 0:
                raise ValueError(
                    f"gaussian psf needs a positive fwhm_arcsec, got "
                    f"{self.fwhm_arcsec!r}"
                )
            if self.sca is not None or self.pupil_bin is not None:
                raise ValueError("sca/pupil_bin are roman_wfi-only psf fields")
            if (
                self.folding_threshold is not None
                or self.mock_folding_threshold is not None
                or self.folding_threshold_tiers is not None
            ):
                raise ValueError(
                    "folding_threshold/mock_folding_threshold/"
                    "folding_threshold_tiers are roman_wfi-only psf fields"
                )
        elif self.psf_type == 'roman_wfi':
            if self.fwhm_arcsec is not None:
                raise ValueError("fwhm_arcsec is a gaussian-only psf field")
            if not isinstance(self.sca, int) or not (1 <= self.sca <= 18):
                raise ValueError(
                    f"roman_wfi sca must be an int in [1, 18], got {self.sca!r}"
                )
            if not isinstance(self.pupil_bin, int) or self.pupil_bin < 1:
                raise ValueError(
                    f"roman_wfi pupil_bin must be a positive int, got "
                    f"{self.pupil_bin!r}"
                )
            if self.folding_threshold is not None and (
                not isinstance(self.folding_threshold, float)
                or not (0.0 < self.folding_threshold < 1.0)
            ):
                raise ValueError(
                    f"roman_wfi folding_threshold must be a float in (0, 1), "
                    f"got {self.folding_threshold!r}"
                )
            if self.mock_folding_threshold is not None and (
                not isinstance(self.mock_folding_threshold, float)
                or not (0.0 < self.mock_folding_threshold < 1.0)
            ):
                raise ValueError(
                    f"roman_wfi mock_folding_threshold must be a float in "
                    f"(0, 1), got {self.mock_folding_threshold!r}"
                )
            if (
                self.folding_threshold is not None
                and self.folding_threshold_tiers is not None
            ):
                raise ValueError(
                    "folding_threshold (scalar) and folding_threshold_tiers "
                    "(z schedule) are mutually exclusive; set at most one"
                )
            if self.folding_threshold_tiers is not None:
                _validate_folding_tiers(
                    self.folding_threshold_tiers, "roman_wfi folding_threshold_tiers"
                )
            # mock kernels must be at least as accurate as fit kernels at every
            # z (None = galsim default 5e-3). With a tier schedule the binding
            # fit tier is the most accurate one (smallest ft).
            mock_eff = (
                self.mock_folding_threshold
                if self.mock_folding_threshold is not None
                else _GALSIM_DEFAULT_FOLDING_THRESHOLD
            )
            if self.folding_threshold_tiers is not None:
                fit_eff_most_accurate = min(t.ft for t in self.folding_threshold_tiers)
            else:
                fit_eff_most_accurate = (
                    self.folding_threshold
                    if self.folding_threshold is not None
                    else _GALSIM_DEFAULT_FOLDING_THRESHOLD
                )
            if mock_eff > fit_eff_most_accurate:
                raise ValueError(
                    f"mock_folding_threshold ({mock_eff}) must be <= the most "
                    f"accurate fit folding_threshold ({fit_eff_most_accurate}): "
                    f"truth-render kernels must be at least as accurate as fit "
                    f"kernels at every z"
                )
        else:
            raise NotImplementedError(
                f"psf type {self.psf_type!r} not supported; supported types: "
                f"{list(_PSF_TYPES)}"
            )

    def resolve_fit_folding_threshold(self, z: float) -> Optional[float]:
        """Fit-kernel folding_threshold at scene redshift ``z``.

        Returns the scalar ``folding_threshold`` (z-independent) when no tier
        schedule is configured, else the ``ft`` of the first tier whose
        ``z_max`` covers ``z`` (tiers sorted ascending; the final open tier
        covers all remaining z). ``None`` means GalSim's default (5e-3).
        """
        if self.folding_threshold_tiers is None:
            return self.folding_threshold
        if (
            not isinstance(z, (int, float))
            or isinstance(z, bool)
            or not math.isfinite(float(z))
        ):
            raise ValueError(
                f"folding_threshold tier resolution needs a finite numeric z, "
                f"got {z!r}"
            )
        for tier in self.folding_threshold_tiers:
            if tier.z_max is None or z <= tier.z_max:
                return tier.ft
        # unreachable when tiers are validated (final tier is open)
        raise ValueError(
            f"no folding_threshold tier covers z={z}; tiers={self.folding_threshold_tiers}"
        )


def _require_yaml_int(block: dict, key: str, default: int, context: str) -> int:
    """Fetch an integer key, rejecting floats/strings instead of coercing."""
    val = block.get(key, default)
    if isinstance(val, bool) or not isinstance(val, int):
        raise ValueError(
            f"{context}: '{key}' must be an integer, got {val!r} "
            f"({type(val).__name__})"
        )
    return val


def _parse_folding_tiers(
    block: dict, context: str
) -> Optional[Tuple[FoldingThresholdTier, ...]]:
    """Parse an optional folding_threshold_tiers list from a psf YAML block.

    Each entry is a ``{z_max, ft}`` mapping (``z_max: null`` for the final
    open tier). Returns None when the key is absent. Structural validation
    (ordering, open-final, ft range) happens in the PSFSpec constructor.
    """
    tiers_raw = block.get('folding_threshold_tiers')
    if tiers_raw is None:
        return None
    if not isinstance(tiers_raw, list) or not tiers_raw:
        raise ValueError(
            f"{context}: folding_threshold_tiers must be a non-empty list of "
            f"{{z_max, ft}} mappings, got {tiers_raw!r}"
        )
    parsed = []
    for i, entry in enumerate(tiers_raw):
        entry_ctx = f"{context}:folding_threshold_tiers[{i}]"
        if not isinstance(entry, dict):
            raise ValueError(
                f"{entry_ctx}: each tier must be a {{z_max, ft}} mapping, got "
                f"{entry!r}"
            )
        _reject_unknown(entry, ('z_max', 'ft'), entry_ctx)
        _require_keys(entry, ('z_max', 'ft'), entry_ctx)
        z_max = entry['z_max']
        parsed.append(
            FoldingThresholdTier(
                z_max=float(z_max) if z_max is not None else None,
                ft=float(entry['ft']),
            )
        )
    return tuple(parsed)


def _parse_roman_wfi_psf(block: dict, context: str) -> PSFSpec:
    _reject_unknown(
        block,
        (
            'type',
            'sca',
            'pupil_bin',
            'folding_threshold',
            'mock_folding_threshold',
            'folding_threshold_tiers',
        ),
        context,
    )
    thresholds = {}
    for key in ('folding_threshold', 'mock_folding_threshold'):
        value = block.get(key)
        thresholds[key] = float(value) if value is not None else None
    return PSFSpec(
        psf_type='roman_wfi',
        sca=_require_yaml_int(block, 'sca', _ROMAN_WFI_DEFAULT_SCA, context),
        pupil_bin=_require_yaml_int(
            block, 'pupil_bin', _ROMAN_WFI_DEFAULT_PUPIL_BIN, context
        ),
        folding_threshold_tiers=_parse_folding_tiers(block, context),
        **thresholds,
    )


def _parse_broadband_psf(
    block: dict, bands: Tuple[str, ...], context: str
) -> Dict[str, PSFSpec]:
    psf_type = block.get('type')
    if psf_type == 'gaussian':
        _reject_unknown(block, ('type', 'fwhm_arcsec'), context)
        _require_keys(block, ('type', 'fwhm_arcsec'), context)
        fwhm = block['fwhm_arcsec']
        if not isinstance(fwhm, dict):
            raise ValueError(
                f"{context}: broadband gaussian fwhm_arcsec must be a "
                f"band -> arcsec mapping, got {fwhm!r}"
            )
        return {
            band: PSFSpec(psf_type='gaussian', fwhm_arcsec=float(value))
            for band, value in fwhm.items()
        }
    if psf_type == 'roman_wfi':
        spec = _parse_roman_wfi_psf(block, context)
        return {band: spec for band in bands}
    raise NotImplementedError(
        f"{context}: psf type {psf_type!r} not supported; supported types: "
        f"{list(_PSF_TYPES)}"
    )


def _parse_grism_psf(block: dict, context: str) -> PSFSpec:
    psf_type = block.get('type')
    if psf_type == 'gaussian':
        _reject_unknown(block, ('type', 'fwhm_arcsec'), context)
        _require_keys(block, ('type', 'fwhm_arcsec'), context)
        fwhm = block['fwhm_arcsec']
        if isinstance(fwhm, dict):
            raise ValueError(
                f"{context}: grism gaussian fwhm_arcsec must be a scalar, "
                f"got {fwhm!r}"
            )
        return PSFSpec(psf_type='gaussian', fwhm_arcsec=float(fwhm))
    if psf_type == 'roman_wfi':
        return _parse_roman_wfi_psf(block, context)
    raise NotImplementedError(
        f"{context}: psf type {psf_type!r} not supported; supported types: "
        f"{list(_PSF_TYPES)}"
    )


@dataclass(frozen=True)
class ObservationConfig:
    """Structural observation setup shared by all fits in a campaign."""

    id: str
    bands: Tuple[str, ...]
    band_psf: Dict[str, PSFSpec]  # per band
    grism_rolls_deg: Tuple[float, ...]
    grism_dispersion_nm_per_pix: float
    grism_psf: PSFSpec
    lines: Tuple[str, ...]
    pixel_scale_arcsec: float
    stamp_broadband_pix: int
    stamp_grism_pix: int
    # noise_model: how mock noise is generated.
    #   'matched_filter' (default) -- uniform per-channel Gaussian variance
    #       normalized so the labeled SNR is exact (current baseline).
    #   'poisson' -- flat per-pixel background anchored to the published
    #       survey depths plus the source's own shot noise (catalog mode
    #       only; the labeled SNR stays the selection/plot axis and the
    #       realized snr_effective columns report the actual depth).
    noise_model: str = 'matched_filter'
    content_hash: str = ''  # sha256 of the source YAML file bytes

    def __post_init__(self):
        if not self.bands:
            raise ValueError("observation config needs at least one band")
        for band in self.bands:
            if band not in self.band_psf:
                raise ValueError(f"band '{band}' has no entry in broadband psf spec")
            if not isinstance(self.band_psf[band], PSFSpec):
                raise ValueError(f"band '{band}' psf entry must be a PSFSpec")
        if not isinstance(self.grism_psf, PSFSpec):
            raise ValueError("grism_psf must be a PSFSpec")
        if not self.grism_rolls_deg:
            raise ValueError("observation config needs at least one grism roll")
        if tuple(self.lines) != ('Halpha',):
            raise NotImplementedError(
                f"v1 supports the single-Halpha line config only, got {self.lines}"
            )
        for name, value in [
            ('grism_dispersion_nm_per_pix', self.grism_dispersion_nm_per_pix),
            ('pixel_scale_arcsec', self.pixel_scale_arcsec),
        ]:
            if value <= 0:
                raise ValueError(f"{name} ({value}) must be positive")
        for name, value in [
            ('stamp_broadband_pix', self.stamp_broadband_pix),
            ('stamp_grism_pix', self.stamp_grism_pix),
        ]:
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} ({value}) must be a positive int")
        if self.noise_model not in _NOISE_MODELS:
            raise ValueError(
                f"noise_model must be one of {_NOISE_MODELS}, got "
                f"{self.noise_model!r}"
            )

    @classmethod
    def from_yaml(cls, path: Path) -> 'ObservationConfig':
        path = Path(path)
        raw_bytes = path.read_bytes()
        raw = yaml.safe_load(raw_bytes)
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: observation config must be a mapping")
        required = (
            'id',
            'bands',
            'grism',
            'lines',
            'psf',
            'pixel_scale_arcsec',
            'stamp',
        )
        # noise_model is optional: absent means the matched_filter baseline,
        # so existing configs keep their meaning
        _reject_unknown(raw, required + ('noise_model',), str(path))
        _require_keys(raw, required, str(path))

        grism = raw['grism']
        _reject_unknown(grism, ('rolls_deg', 'dispersion_nm_per_pix'), f"{path}:grism")
        _require_keys(grism, ('rolls_deg', 'dispersion_nm_per_pix'), f"{path}:grism")

        psf = raw['psf']
        _reject_unknown(psf, ('broadband', 'grism'), f"{path}:psf")
        _require_keys(psf, ('broadband', 'grism'), f"{path}:psf")
        band_psf = _parse_broadband_psf(
            psf['broadband'], tuple(raw['bands']), f"{path}:psf.broadband"
        )
        grism_psf = _parse_grism_psf(psf['grism'], f"{path}:psf.grism")

        stamp = raw['stamp']
        _reject_unknown(stamp, ('broadband_pix', 'grism_pix'), f"{path}:stamp")
        _require_keys(stamp, ('broadband_pix', 'grism_pix'), f"{path}:stamp")

        return cls(
            id=str(raw['id']),
            bands=tuple(raw['bands']),
            band_psf=band_psf,
            grism_rolls_deg=tuple(float(a) for a in grism['rolls_deg']),
            grism_dispersion_nm_per_pix=float(grism['dispersion_nm_per_pix']),
            grism_psf=grism_psf,
            lines=tuple(raw['lines']),
            pixel_scale_arcsec=float(raw['pixel_scale_arcsec']),
            stamp_broadband_pix=int(stamp['broadband_pix']),
            stamp_grism_pix=int(stamp['grism_pix']),
            noise_model=str(raw.get('noise_model', 'matched_filter')),
            content_hash=hashlib.sha256(raw_bytes).hexdigest(),
        )


# =============================================================================
# Ensemble spec
# =============================================================================

_NOISE_MODELS = ('matched_filter', 'poisson')
_DRAW_DISTS = ('uniform', 'lognormal_tf')
_POPULATION_TYPES = ('sampled', 'catalog')
_SHEAR_SCHEMES = ('fixed', 'grid')
_DISPATCH_MODES = ('static', 'dynamic')
_DISPATCH_BACKENDS = ('local', 'slurm')
_SAVE_POLICIES = ('none', 'subset', 'all')
_MEASUREMENTS = ('sigma_eps_vs_cosi', 'sigma_eps_vs_line_snr', 'shear_bias')
_SAMPLED_MEASUREMENTS = ('sigma_eps_vs_cosi', 'sigma_eps_vs_line_snr')
_CATALOG_DEFAULT_KIND = 'flagship2'
_CATALOG_DEFAULT_DATA_DIR = 'data/cosmohub'

# spec draw/fixed keys may be shared top-level params, aliased short names, or
# fully-dotted source-model params; aliases resolve here
_PARAM_ALIASES = {'vcirc': 'vel.vcirc'}
# short fixed keys broadcast to every scene component carrying that suffix
_BROADCAST_FIXED = ('h_over_r', 'x0', 'y0')


@dataclass(frozen=True)
class DrawSpec:
    """One drawn truth distribution from the spec's population.draw block."""

    dist: str
    params: Dict[str, float]

    def __post_init__(self):
        if self.dist not in _DRAW_DISTS:
            raise ValueError(
                f"unknown draw dist '{self.dist}'; supported: {_DRAW_DISTS}"
            )
        if self.dist == 'uniform':
            _require_keys(self.params, ('low', 'high'), 'draw:uniform')
            if self.params['high'] <= self.params['low']:
                raise ValueError(
                    f"uniform draw: high ({self.params['high']}) must be > "
                    f"low ({self.params['low']})"
                )
        elif self.dist == 'lognormal_tf':
            _require_keys(
                self.params, ('center_kms', 'sigma_tf_dex'), 'draw:lognormal_tf'
            )


@dataclass(frozen=True)
class CatalogPopulationSpec:
    """Catalog-backed population block (``population.type: catalog``).

    Galaxies come from input-catalog rows (structural + flux truths; the
    catalog adapter selected by ``catalog.kind`` owns the schema and
    preprocess), with kinematics painted on via scaling relations, an
    isotropic orientation redraw, and per-pair shear draws. All keys are
    required in the YAML unless a default is stated; unknown keys raise.
    """

    # catalog
    catalog_download: str  # basename of <data_dir>/<name>.parquet download
    catalog_data_dir: str  # default 'data/cosmohub'

    # preprocess
    flux_variant: str  # Halpha flux column variant (adapter vocabulary)
    h: float  # catalog cosmology little-h; logM*_phys = col - 2*log10(h)

    # selection
    z_range: Tuple[float, float]
    # matched-filter line SNR floor on the COADDED depth over all grism
    # passes -- the depth the joint multi-roll fit sees, not one pass's
    snr_line_total_min: float
    bulge_fraction_max: Optional[float]  # None = no cut
    bulge_nsersic_range: Optional[Tuple[float, float]]  # None = no cut
    # floor on disk r50 / diffraction PSF FWHM at observed Halpha. Sub-PSF
    # galaxies carry no velocity gradient to fit, and the extended-fiducial
    # flux-limit reference rewards compact sources, so without this the
    # selection admits objects an order of magnitude below the PSF.
    # None = no cut.
    min_r50_over_psf_fwhm: Optional[float]

    # sample
    n_galaxies: int  # subsampled without replacement

    # paint: inverted TFR (logv = logv0 + (logM - logm0)/slope) + sigma0(z);
    # paint_bulge toggles the bulge morphology paint (paint.bulge, default
    # true): false = disk-only twin (single-disk broadband truth + fit)
    tfr_logv0: float
    tfr_logm0: float
    tfr_slope: float
    tfr_scatter_dex: float  # Gaussian scatter in logv
    sigma0_intercept_kms: float
    sigma0_slope_kms: float  # per unit z
    sigma0_scatter_kms: float
    sigma0_min_kms: float  # resample below this floor

    # orientation: isotropic redraw; catalog inclination kept for validation
    cosi_range: Tuple[float, float]
    ring_members: int  # 1, or 2 for a theta / theta + pi/2 ring pair

    # shear: iid per-component Gaussian, shared within a ring pair
    shear_sigma: float
    shear_gmax: float  # redraw until |g| < gmax

    # priors
    logm_obs_scatter_dex: float  # simulated photometric-mass error

    # paint.bulge (default true): false = disk-only twin (no bulge paint,
    # single-disk broadband truth keys, single-disk fit model + priors)
    paint_bulge: bool = True

    # catalog.kind (default 'flagship2'): selects the catalog adapter that
    # owns the raw schema, unique row key, and preprocess
    catalog_kind: str = _CATALOG_DEFAULT_KIND

    # sample.galaxy_ids (optional): restriction to specific population row
    # indices (0-based, into the seeded n_galaxies sample), applied AFTER the
    # population build -- the bank and every per-galaxy draw and noise seed
    # stay identical to the unrestricted run with the same seed. Enables
    # cheap paired subset reruns (e.g. escalation-tier resampling) keyed by
    # galaxy_id across runs.
    galaxy_ids: Optional[Tuple[int, ...]] = None

    def __post_init__(self):
        if not self.catalog_download:
            raise ValueError("catalog.download must be a non-empty name")
        # local import: the registry imports nothing from this module at
        # module level, so the lookup is cycle-free
        from kl_pipe.ensemble.catalogs import get_catalog_adapter

        adapter = get_catalog_adapter(self.catalog_kind)  # unknown kind raises
        if self.flux_variant not in adapter.flux_variants:
            raise ValueError(
                f"preprocess.flux_variant '{self.flux_variant}'; supported "
                f"by catalog '{adapter.kind}': {adapter.flux_variants}"
            )
        if self.h <= 0:
            raise ValueError(f"preprocess.h ({self.h}) must be positive")
        z_lo, z_hi = self.z_range
        if not (0.0 < z_lo < z_hi):
            raise ValueError(
                f"selection.z_range ({z_lo}, {z_hi}) must satisfy 0 < lo < hi"
            )
        if self.snr_line_total_min <= 0:
            raise ValueError(
                f"selection.snr_line_total_min ({self.snr_line_total_min}) "
                f"must be positive"
            )
        if self.min_r50_over_psf_fwhm is not None and self.min_r50_over_psf_fwhm <= 0:
            raise ValueError(
                f"selection.min_r50_over_psf_fwhm "
                f"({self.min_r50_over_psf_fwhm}) must be positive or null"
            )
        if self.bulge_fraction_max is not None and not (
            0.0 < self.bulge_fraction_max <= 1.0
        ):
            raise ValueError(
                f"selection.bulge_fraction_max ({self.bulge_fraction_max}) "
                f"must be in (0, 1] or null"
            )
        if self.bulge_nsersic_range is not None:
            n_lo, n_hi = self.bulge_nsersic_range
            if not (0.0 < n_lo < n_hi):
                raise ValueError(
                    f"selection.bulge_nsersic_range ({n_lo}, {n_hi}) must "
                    f"satisfy 0 < lo < hi or be null"
                )
        if not isinstance(self.n_galaxies, int) or self.n_galaxies < 1:
            raise ValueError(
                f"sample.n_galaxies ({self.n_galaxies!r}) must be a positive int"
            )
        if self.galaxy_ids is not None:
            if len(self.galaxy_ids) == 0:
                raise ValueError("sample.galaxy_ids must be non-empty or absent")
            bad = [i for i in self.galaxy_ids if not isinstance(i, int)]
            if bad:
                raise ValueError(f"sample.galaxy_ids must be ints, got {bad!r}")
            if len(set(self.galaxy_ids)) != len(self.galaxy_ids):
                raise ValueError("sample.galaxy_ids contains duplicates")
            out_of_range = [i for i in self.galaxy_ids if not 0 <= i < self.n_galaxies]
            if out_of_range:
                raise ValueError(
                    f"sample.galaxy_ids {out_of_range} outside "
                    f"[0, n_galaxies={self.n_galaxies})"
                )
        if self.tfr_slope == 0:
            raise ValueError("paint.tfr.slope must be nonzero")
        if self.tfr_scatter_dex < 0:
            raise ValueError(
                f"paint.tfr.scatter_dex ({self.tfr_scatter_dex}) must be >= 0"
            )
        if self.sigma0_scatter_kms < 0:
            raise ValueError(
                f"paint.sigma0.scatter_kms ({self.sigma0_scatter_kms}) must " f"be >= 0"
            )
        if self.sigma0_min_kms <= 0:
            raise ValueError(
                f"paint.sigma0.min_kms ({self.sigma0_min_kms}) must be positive"
            )
        c_lo, c_hi = self.cosi_range
        if not (0.0 <= c_lo < c_hi <= 1.0):
            raise ValueError(
                f"orientation.cosi_range ({c_lo}, {c_hi}) must satisfy "
                f"0 <= lo < hi <= 1"
            )
        if self.ring_members not in (1, 2):
            raise ValueError(
                f"orientation.ring.members ({self.ring_members!r}) must be 1 or 2"
            )
        if self.shear_sigma <= 0:
            raise ValueError(f"shear.sigma ({self.shear_sigma}) must be positive")
        if not (0.0 < self.shear_gmax < 1.0):
            raise ValueError(f"shear.gmax ({self.shear_gmax}) must be in (0, 1)")
        if self.logm_obs_scatter_dex < 0:
            raise ValueError(
                f"priors.logm_obs_scatter_dex ({self.logm_obs_scatter_dex}) "
                f"must be >= 0"
            )


def _parse_pair(value, context: str) -> Tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{context}: must be a [lo, hi] pair, got {value!r}")
    return (float(value[0]), float(value[1]))


def _parse_catalog_population(population: dict, context: str) -> CatalogPopulationSpec:
    """Parse the ``population.type: catalog`` block.

    Sampled-only keys (stratify/n_gal_per_bin/draw/fixed/ring) are rejected
    as unknown; every catalog-mode key is required unless a default is
    stated (catalog.data_dir).
    """
    allowed = (
        'type',
        'catalog',
        'preprocess',
        'selection',
        'sample',
        'paint',
        'orientation',
        'shear',
        'priors',
    )
    _reject_unknown(population, allowed, context)
    _require_keys(population, allowed, context)

    catalog = population['catalog']
    _reject_unknown(catalog, ('kind', 'download', 'data_dir'), f"{context}.catalog")
    _require_keys(catalog, ('download',), f"{context}.catalog")

    pre = population['preprocess']
    _reject_unknown(pre, ('flux_variant', 'h'), f"{context}.preprocess")
    _require_keys(pre, ('flux_variant', 'h'), f"{context}.preprocess")

    sel = population['selection']
    if 'snr_line_min' in sel:
        raise ValueError(
            f"{context}.selection.snr_line_min is the PER-PASS line SNR but "
            f"was applied as though it were the total, so a cut written as "
            f"10 selected a coadded 20. Replace it with snr_line_total_min, "
            f"the SNR coadded over every grism pass (per-pass x sqrt(passes))"
        )
    sel_keys = (
        'z_range',
        'snr_line_total_min',
        'bulge_fraction_max',
        'bulge_nsersic_range',
        'min_r50_over_psf_fwhm',
    )
    _reject_unknown(sel, sel_keys, f"{context}.selection")
    _require_keys(sel, sel_keys, f"{context}.selection")

    sample = population['sample']
    _reject_unknown(
        sample, ('n_galaxies', 'replace', 'galaxy_ids'), f"{context}.sample"
    )
    _require_keys(sample, ('n_galaxies', 'replace'), f"{context}.sample")
    galaxy_ids = sample.get('galaxy_ids')
    if galaxy_ids is not None:
        if not isinstance(galaxy_ids, list):
            raise ValueError(
                f"{context}.sample.galaxy_ids must be a list of ints, got "
                f"{type(galaxy_ids).__name__}"
            )
        galaxy_ids = tuple(galaxy_ids)
    if sample['replace']:
        raise NotImplementedError(
            f"{context}.sample: replace: true (bootstrap resampling) is not "
            f"implemented; set replace: false"
        )

    paint = population['paint']
    _reject_unknown(paint, ('tfr', 'sigma0', 'bulge'), f"{context}.paint")
    _require_keys(paint, ('tfr', 'sigma0'), f"{context}.paint")
    # paint.bulge (optional, default true): false = disk-only twin
    paint_bulge = paint.get('bulge', True)
    if not isinstance(paint_bulge, bool):
        raise ValueError(
            f"{context}.paint.bulge must be a boolean (true = BulgeDisk "
            f"broadband, false = single-disk twin), got {paint_bulge!r}"
        )
    tfr = paint['tfr']
    tfr_keys = ('logv0', 'logm0', 'slope', 'scatter_dex')
    _reject_unknown(tfr, tfr_keys, f"{context}.paint.tfr")
    _require_keys(tfr, tfr_keys, f"{context}.paint.tfr")
    sigma0 = paint['sigma0']
    sigma0_keys = ('intercept_kms', 'slope_kms', 'scatter_kms', 'min_kms')
    _reject_unknown(sigma0, sigma0_keys, f"{context}.paint.sigma0")
    _require_keys(sigma0, sigma0_keys, f"{context}.paint.sigma0")

    orientation = population['orientation']
    _reject_unknown(orientation, ('cosi_range', 'ring'), f"{context}.orientation")
    _require_keys(orientation, ('cosi_range', 'ring'), f"{context}.orientation")
    ring = orientation['ring']
    _reject_unknown(ring, ('members',), f"{context}.orientation.ring")
    _require_keys(ring, ('members',), f"{context}.orientation.ring")

    shear = population['shear']
    _reject_unknown(shear, ('sigma', 'gmax'), f"{context}.shear")
    _require_keys(shear, ('sigma', 'gmax'), f"{context}.shear")

    priors = population['priors']
    _reject_unknown(priors, ('logm_obs_scatter_dex',), f"{context}.priors")
    _require_keys(priors, ('logm_obs_scatter_dex',), f"{context}.priors")

    bulge_fraction_max = sel['bulge_fraction_max']
    bulge_nsersic_range = sel['bulge_nsersic_range']
    return CatalogPopulationSpec(
        catalog_kind=str(catalog.get('kind', _CATALOG_DEFAULT_KIND)),
        catalog_download=str(catalog['download']),
        catalog_data_dir=str(catalog.get('data_dir', _CATALOG_DEFAULT_DATA_DIR)),
        flux_variant=str(pre['flux_variant']),
        h=float(pre['h']),
        z_range=_parse_pair(sel['z_range'], f"{context}.selection.z_range"),
        snr_line_total_min=float(sel['snr_line_total_min']),
        min_r50_over_psf_fwhm=(
            float(sel['min_r50_over_psf_fwhm'])
            if sel['min_r50_over_psf_fwhm'] is not None
            else None
        ),
        bulge_fraction_max=(
            float(bulge_fraction_max) if bulge_fraction_max is not None else None
        ),
        bulge_nsersic_range=(
            _parse_pair(bulge_nsersic_range, f"{context}.selection.bulge_nsersic_range")
            if bulge_nsersic_range is not None
            else None
        ),
        n_galaxies=_require_yaml_int(sample, 'n_galaxies', 0, f"{context}.sample"),
        galaxy_ids=galaxy_ids,
        tfr_logv0=float(tfr['logv0']),
        tfr_logm0=float(tfr['logm0']),
        tfr_slope=float(tfr['slope']),
        tfr_scatter_dex=float(tfr['scatter_dex']),
        sigma0_intercept_kms=float(sigma0['intercept_kms']),
        sigma0_slope_kms=float(sigma0['slope_kms']),
        sigma0_scatter_kms=float(sigma0['scatter_kms']),
        sigma0_min_kms=float(sigma0['min_kms']),
        cosi_range=_parse_pair(
            orientation['cosi_range'], f"{context}.orientation.cosi_range"
        ),
        ring_members=_require_yaml_int(
            ring, 'members', 0, f"{context}.orientation.ring"
        ),
        shear_sigma=float(shear['sigma']),
        shear_gmax=float(shear['gmax']),
        logm_obs_scatter_dex=float(priors['logm_obs_scatter_dex']),
        paint_bulge=paint_bulge,
    )


@dataclass(frozen=True)
class EscalationSpec:
    """Quality-gated escalation retry policy (spec ``fit.escalation`` block).

    When enabled, a fit whose first attempt fails the convergence gate
    (``max_rhat > rhat_max`` OR ``min_ess < ess_min``) is retried exactly
    once with a stronger sampler config: warmup/samples raised to the
    escalation values and the first attempt's warmup-adapted inverse mass
    matrix donated as the retry's initial NUTS metric. The gate is on
    convergence quality, NOT divergences: the silent unconverged class shows
    zero divergences, and a fresh-seed retry alone does not rescue it (both
    measured on a 200-fit production-config census, 2026-07-24), so the
    retry must change the sampler config. Default provenance: rhat 1.05 /
    min ESS 50 reproduce the census quality gate that isolated the
    slow-convergence tier; warmup 800 / samples 1000 are 4x/1x the census
    baseline (200/1000); the same-fit donated metric was measured to cut
    mean tree depth from >255 to ~48 on a census escalation fit.
    """

    enabled: bool = False
    rhat_max: float = 1.05
    ess_min: float = 50.0
    n_warmup: int = 800
    n_samples: int = 1000

    def __post_init__(self):
        if not isinstance(self.enabled, bool):
            raise ValueError(
                f"escalation.enabled must be a boolean, got {self.enabled!r}"
            )
        if not isinstance(self.rhat_max, float) or self.rhat_max <= 1.0:
            raise ValueError(
                f"escalation.rhat_max ({self.rhat_max!r}) must be a float > 1.0"
            )
        if not isinstance(self.ess_min, float) or self.ess_min <= 0:
            raise ValueError(
                f"escalation.ess_min ({self.ess_min!r}) must be a positive float"
            )
        for name, value in [
            ('n_warmup', self.n_warmup),
            ('n_samples', self.n_samples),
        ]:
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(
                    f"escalation.{name} ({value!r}) must be a positive int"
                )


def _parse_escalation(block, context: str) -> EscalationSpec:
    """Parse the optional ``fit.escalation`` block (absent = disabled)."""
    if block is None:
        return EscalationSpec()
    if not isinstance(block, dict):
        raise ValueError(f"{context}: must be a mapping, got {block!r}")
    allowed = ('enabled', 'rhat_max', 'ess_min', 'n_warmup', 'n_samples')
    _reject_unknown(block, allowed, context)
    return EscalationSpec(
        enabled=block.get('enabled', False),
        rhat_max=float(block.get('rhat_max', 1.05)),
        ess_min=float(block.get('ess_min', 50.0)),
        n_warmup=_require_yaml_int(block, 'n_warmup', 800, context),
        n_samples=_require_yaml_int(block, 'n_samples', 1000, context),
    )


@dataclass(frozen=True)
class EnsembleSpec:
    """One ensemble campaign, loaded from ensemble_spec.yaml."""

    run_name: str
    version: int
    description: str
    seed: int
    measurement: str

    # population: exactly one plot axis, either a truth stratification (cosi
    # bins) or a config sweep (line_snr values; galaxies + noise shared
    # across the sweep -- common random numbers). For catalog populations
    # the stratify/draw machinery is unused (catalog_population carries the
    # whole population definition).
    population_type: str  # 'sampled' | 'catalog'
    stratify_param: str  # 'cosi' | 'line_snr'
    stratify_n_bins: int  # cosi axis only; 0 for a config sweep
    stratify_range: Tuple[float, float]  # cosi axis only; (0, 0) for a sweep
    sweep_values: Tuple[float, ...]  # config-sweep axis only; () for cosi
    n_gal_per_bin: int
    m_noise: int
    draw: Dict[str, DrawSpec]
    fixed: Dict[str, float]  # resolved dotted names

    # shear
    shear_scheme: str
    g1: float
    g2: float
    shear_grid: Tuple[float, ...]  # grid scheme only; empty for fixed
    shear_component: str  # 'g1' | 'g2' (grid scheme)

    # ring
    ring_enabled: bool

    # model render
    render_oversample: int

    # observation; line_snr is -1 (sentinel) for catalog populations, whose
    # per-fit line SNR lives in the population table
    observed_config: str
    broadband_snr: float
    line_snr: float

    # fit
    n_warmup: int
    n_samples: int
    n_chains: int
    precondition: str
    # sample the preconditioned path in unconstrained coordinates (removes
    # hard prior-truncation walls from the NUTS potential; see
    # NumpyroSamplerConfig.precondition_unconstrained)
    unconstrained: bool
    # adapt the mass matrix during warmup starting from the Laplace metric
    # instead of freezing it (see NumpyroSamplerConfig.precondition_adapt_mass)
    adapt_mass: bool
    target_accept: float
    n_map_starts: int
    pin_z_to_truth: bool
    # sample the bulge Sersic index instead of pinning it at the painted
    # truth. Adds {band}.bulge_n_sersic with a population prior; expect the
    # bulge decomposition to loosen, since the index is degenerate with
    # bulge_frac and bulge_hlr and pinning it suppressed one leg of that.
    sample_bulge_nsersic: bool

    # dispatch
    backend: str
    mode: str
    workers_per_node: int
    target_task_walltime_min: float
    queue: str
    account: str  # SLURM allocation (-A); empty = omit the directive
    max_fit_walltime_min: float

    # output
    save_chains: str
    save_mocks: str

    # per-component Gaussian width of the shear fit prior; wider = more
    # data-driven sigma_eps (less prior floor). Defaults to 0.2, matching the
    # published Roman KL prior half-width (Xu+ 2023) as an isotropic Gaussian.
    shear_fit_prior_sigma: float = 0.2
    # 'gaussian' (default, N(0, sigma)) or 'uniform' (flat on
    # [-halfwidth, halfwidth]); the flat prior removes prior shrinkage of
    # the per-galaxy shear posterior, so the ensemble estimator needs no
    # width-dependent shrinkage correction
    shear_fit_prior_type: str = 'gaussian'
    shear_fit_prior_halfwidth: float = 0.3

    # analytic-dispersal deposit window for the FIT observations ('global' |
    # 'local'); mock data are always rendered with the global window
    render_line_window_mode: str = 'global'

    # catalog-backed population definition (population.type: catalog only;
    # None for sampled populations)
    catalog_population: Optional[CatalogPopulationSpec] = None

    # quality-gated escalation retry (fit.escalation block; disabled by
    # default -- see EscalationSpec)
    escalation: EscalationSpec = EscalationSpec()

    def __post_init__(self):
        if self.population_type not in _POPULATION_TYPES:
            raise ValueError(
                f"population type '{self.population_type}'; supported: "
                f"{_POPULATION_TYPES}"
            )
        if self.render_line_window_mode not in ('global', 'local'):
            raise ValueError(
                f"model.render.line_window_mode must be 'global' or 'local', got "
                f"{self.render_line_window_mode!r}"
            )
        if not isinstance(self.render_oversample, int) or self.render_oversample <= 0:
            raise ValueError(
                f"render_oversample ({self.render_oversample}) must be a "
                f"positive int"
            )
        if self.measurement not in _MEASUREMENTS:
            raise NotImplementedError(
                f"measurement '{self.measurement}' not supported; "
                f"available: {_MEASUREMENTS}"
            )
        if self.population_type == 'catalog':
            if self.catalog_population is None:
                raise ValueError(
                    "population type 'catalog' requires a catalog_population " "block"
                )
            if self.measurement != 'shear_bias':
                raise ValueError(
                    f"catalog populations require run.measurement: "
                    f"shear_bias, got '{self.measurement}'"
                )
        else:
            if self.catalog_population is not None:
                raise ValueError(
                    "catalog_population is only valid for population type " "'catalog'"
                )
            if self.measurement not in _SAMPLED_MEASUREMENTS:
                raise ValueError(
                    f"measurement '{self.measurement}' requires "
                    f"population.type: catalog; sampled populations support "
                    f"{_SAMPLED_MEASUREMENTS}"
                )
        if self.measurement == 'sigma_eps_vs_cosi':
            if self.stratify_param != 'cosi':
                raise ValueError(
                    f"measurement sigma_eps_vs_cosi stratifies cosi, got "
                    f"'{self.stratify_param}'"
                )
            if self.stratify_n_bins < 1:
                raise ValueError(f"n_bins ({self.stratify_n_bins}) must be >= 1")
            lo, hi = self.stratify_range
            if not (0.0 < lo < hi <= 1.0):
                raise ValueError(
                    f"cosi stratify range ({lo}, {hi}) must satisfy "
                    f"0 < lo < hi <= 1"
                )
        elif self.measurement == 'sigma_eps_vs_line_snr':
            if self.stratify_param != 'line_snr':
                raise ValueError(
                    f"measurement sigma_eps_vs_line_snr sweeps line_snr, "
                    f"got '{self.stratify_param}'"
                )
            if len(self.sweep_values) < 2:
                raise ValueError(
                    f"line_snr sweep needs >= 2 values, got "
                    f"{list(self.sweep_values)}"
                )
            if any(v <= 0 for v in self.sweep_values):
                raise ValueError(
                    f"line_snr sweep values must be positive, got "
                    f"{list(self.sweep_values)}"
                )
            if len(set(self.sweep_values)) != len(self.sweep_values):
                raise ValueError("line_snr sweep values must be unique")
            if 'cosi' not in self.draw:
                raise ValueError(
                    "sigma_eps_vs_line_snr requires cosi in population.draw "
                    "(cosi is a drawn population truth on this axis)"
                )
        if self.n_gal_per_bin < 1 or self.m_noise < 1:
            raise ValueError("n_gal_per_bin and m_noise must be >= 1")
        if self.shear_scheme not in _SHEAR_SCHEMES:
            raise ValueError(
                f"shear scheme '{self.shear_scheme}'; supported: {_SHEAR_SCHEMES}"
            )
        if self.shear_scheme == 'grid':
            if len(self.shear_grid) < 3:
                raise ValueError("shear grid needs >= 3 steps for a bias fit")
            if self.shear_component not in ('g1', 'g2'):
                raise ValueError(
                    f"shear grid component must be 'g1' or 'g2', got "
                    f"'{self.shear_component}'"
                )
        if abs(self.g1) >= 1 or abs(self.g2) >= 1:
            raise ValueError(f"|g| must be < 1, got ({self.g1}, {self.g2})")
        # catalog mode: both SNR scalars are -1 sentinels (per-galaxy values
        # come from the population table), so positivity checks are
        # sampled-only
        if self.population_type != 'catalog':
            if self.broadband_snr <= 0:
                raise ValueError(
                    f"broadband SNR ({self.broadband_snr}) must be positive"
                )
            if self.line_snr <= 0:
                raise ValueError(f"line SNR ({self.line_snr}) must be positive")
        if self.backend not in _DISPATCH_BACKENDS:
            raise ValueError(
                f"dispatch backend '{self.backend}'; supported: "
                f"{_DISPATCH_BACKENDS}"
            )
        if self.mode not in _DISPATCH_MODES:
            raise ValueError(
                f"dispatch mode '{self.mode}'; supported: {_DISPATCH_MODES}"
            )
        if self.workers_per_node < 1:
            raise ValueError(f"workers_per_node ({self.workers_per_node}) must be >= 1")
        for name, value in [
            ('save_chains', self.save_chains),
            ('save_mocks', self.save_mocks),
        ]:
            if value not in _SAVE_POLICIES:
                raise ValueError(f"{name} '{value}'; supported: {_SAVE_POLICIES}")
        if not self.pin_z_to_truth:
            raise NotImplementedError(
                "sampled-z (narrow spec-z prior) is planned but not wired in "
                "v1; set fit.pin_z_to_truth: true"
            )
        if self.sample_bulge_nsersic:
            # the index only exists when the scene has a bulge, so silently
            # accepting the flag would leave the fit unchanged while the spec
            # claims otherwise
            if self.catalog_population is None:
                raise ValueError(
                    "fit.sample_bulge_nsersic requires a catalog population: "
                    "sampled-mode scenes are single-disk and have no bulge "
                    "index to sample"
                )
            if not self.catalog_population.paint_bulge:
                raise ValueError(
                    "fit.sample_bulge_nsersic requires population.paint.bulge: "
                    "true; with the bulge paint disabled the bands are "
                    "single-disk and there is no index to sample"
                )
        if self.escalation.enabled:
            # the retry needs an initial metric from the Laplace path: with
            # fit.adapt_mass true it donates the first attempt's
            # warmup-adapted inverse mass matrix; with adapt_mass false
            # (frozen first-pass metric) the retry re-enables mass
            # adaptation on top of the Laplace preconditioner instead
            if self.precondition != 'laplace':
                raise ValueError(
                    "fit.escalation.enabled requires fit.precondition: "
                    "laplace (the escalation retry starts from the "
                    "Laplace-preconditioned metric)"
                )
            if self.escalation.n_warmup < self.n_warmup:
                raise ValueError(
                    f"fit.escalation.n_warmup ({self.escalation.n_warmup}) "
                    f"must be >= fit.n_warmup ({self.n_warmup}): the retry "
                    f"must not be weaker than the first attempt"
                )
            if self.escalation.n_samples < self.n_samples:
                raise ValueError(
                    f"fit.escalation.n_samples ({self.escalation.n_samples}) "
                    f"must be >= fit.n_samples ({self.n_samples}): the retry "
                    f"must not be weaker than the first attempt"
                )

    @property
    def n_axis_steps(self) -> int:
        """Number of steps along the plot axis (cosi bins or sweep values)."""
        if self.population_type == 'catalog':
            raise ValueError(
                "n_axis_steps is a sampled-population concept; catalog "
                "populations carry no stratification axis"
            )
        if self.sweep_values:
            return len(self.sweep_values)
        return self.stratify_n_bins

    @property
    def n_fits(self) -> int:
        """Total fits this spec expands to."""
        if self.population_type == 'catalog':
            cp = self.catalog_population
            n_gal = len(cp.galaxy_ids) if cp.galaxy_ids is not None else cp.n_galaxies
            return n_gal * cp.ring_members * self.m_noise
        n_shear = len(self.shear_grid) if self.shear_scheme == 'grid' else 1
        n_ring = 2 if self.ring_enabled else 1
        return self.n_axis_steps * self.n_gal_per_bin * self.m_noise * n_shear * n_ring

    @classmethod
    def from_yaml(cls, path: Path) -> 'EnsembleSpec':
        path = Path(path)
        raw = yaml.safe_load(path.read_text())
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: ensemble spec must be a mapping")

        allowed = (
            'run',
            'population',
            'model',
            'observation',
            'fit',
            'dispatch',
            'output',
        )
        _reject_unknown(raw, allowed, str(path))
        # the model block is optional (render defaults apply)
        _require_keys(raw, tuple(k for k in allowed if k != 'model'), str(path))

        run = raw['run']
        _reject_unknown(
            run,
            ('name', 'version', 'description', 'seed', 'measurement', 'noise_reps'),
            f"{path}:run",
        )
        _require_keys(
            run,
            ('name', 'version', 'description', 'seed', 'measurement'),
            f"{path}:run",
        )

        population = raw['population']
        if 'type' not in population:
            raise ValueError(f"{path}:population: missing required keys ['type']")
        population_type = str(population['type'])

        observation = raw['observation']
        _reject_unknown(observation, ('config', 'snr'), f"{path}:observation")
        _require_keys(observation, ('config',), f"{path}:observation")
        # catalog populations derive BOTH channels' per-fit SNR from the
        # population table (matched-filter depth anchors), so the snr block
        # is rejected outright there; sampled populations require it
        if population_type == 'catalog':
            if 'snr' in observation:
                raise ValueError(
                    f"{path}:observation.snr is not valid for catalog "
                    f"populations: per-fit broadband and line SNRs come from "
                    f"the population table's published-depth matched-filter "
                    f"columns; remove the block"
                )
            snr = {}
        else:
            _require_keys(observation, ('snr',), f"{path}:observation")
            snr = observation['snr']
            # observation.snr.line is required as a scalar UNLESS it is the
            # swept axis
            _reject_unknown(snr, ('broadband', 'line'), f"{path}:observation.snr")
            _require_keys(snr, ('broadband',), f"{path}:observation.snr")

        # sampled-mode placeholders (overwritten in the sampled branch)
        catalog_population: Optional[CatalogPopulationSpec] = None
        strat_param = ''
        strat_n_bins = 0
        strat_range = (0.0, 0.0)
        sweep_values: Tuple[float, ...] = ()
        n_gal_per_bin = 1
        draw: Dict[str, DrawSpec] = {}
        fixed: Dict[str, float] = {}
        scheme = 'fixed'
        shear_g1 = 0.0
        shear_g2 = 0.0
        shear_grid: Tuple[float, ...] = ()
        shear_component = ''
        ring_enabled = False

        if population_type == 'catalog':
            catalog_population = _parse_catalog_population(
                population, f"{path}:population"
            )
            ring_enabled = catalog_population.ring_members == 2
        else:
            _reject_unknown(
                population,
                (
                    'type',
                    'stratify',
                    'n_gal_per_bin',
                    'draw',
                    'fixed',
                    'shear',
                    'ring',
                ),
                f"{path}:population",
            )
            _require_keys(
                population,
                ('type', 'stratify', 'n_gal_per_bin', 'draw', 'shear', 'ring'),
                f"{path}:population",
            )
            n_gal_per_bin = int(population['n_gal_per_bin'])

            stratify = population['stratify']
            if len(stratify) != 1:
                raise ValueError(
                    f"{path}:population.stratify must contain exactly one "
                    f"parameter, got {list(stratify)}"
                )
            strat_param, strat_cfg = next(iter(stratify.items()))
            if strat_param == 'line_snr':
                _reject_unknown(
                    strat_cfg, ('values',), f"{path}:population.stratify.{strat_param}"
                )
                _require_keys(
                    strat_cfg, ('values',), f"{path}:population.stratify.{strat_param}"
                )
                sweep_values = tuple(float(v) for v in strat_cfg['values'])
                if 'line' in snr:
                    raise ValueError(
                        f"{path}: observation.snr.line conflicts with the "
                        f"line_snr sweep axis; remove it (per-fit values come "
                        f"from population.stratify.line_snr.values)"
                    )
            else:
                if 'line' not in snr:
                    raise ValueError(
                        f"{path}:observation.snr: missing required keys ['line']"
                    )
                _reject_unknown(
                    strat_cfg,
                    ('n_bins', 'range'),
                    f"{path}:population.stratify.{strat_param}",
                )
                _require_keys(
                    strat_cfg,
                    ('n_bins', 'range'),
                    f"{path}:population.stratify.{strat_param}",
                )
                strat_n_bins = int(strat_cfg['n_bins'])
                strat_range = (
                    float(strat_cfg['range'][0]),
                    float(strat_cfg['range'][1]),
                )

            for name, dcfg in population['draw'].items():
                dcfg = dict(dcfg)
                dist = dcfg.pop('dist', None)
                if dist is None:
                    raise ValueError(f"{path}:population.draw.{name}: missing 'dist'")
                if dist == 'uniform' and 'range' in dcfg:
                    lo, hi = dcfg.pop('range')
                    dcfg.update(low=lo, high=hi)
                resolved = _PARAM_ALIASES.get(name, name)
                draw[resolved] = DrawSpec(dist=dist, params=dcfg)

            fixed = _resolve_fixed_block(population.get('fixed', {}), context=f"{path}")

            shear = population['shear']
            _reject_unknown(
                shear,
                ('scheme', 'g1', 'g2', 'grid', 'component'),
                f"{path}:population.shear",
            )
            scheme = shear.get('scheme', 'fixed')
            shear_g1 = float(shear.get('g1', 0.0))
            shear_g2 = float(shear.get('g2', 0.0))
            if scheme == 'grid':
                _require_keys(shear, ('grid', 'component'), f"{path}:population.shear")
                shear_grid = tuple(float(g) for g in shear['grid'])
                shear_component = shear['component']

            ring = population['ring']
            _reject_unknown(
                ring, ('enabled', 'antithetic_g'), f"{path}:population.ring"
            )
            if ring.get('antithetic_g', False):
                raise NotImplementedError(
                    "antithetic_g (+/-g pairs) is scoped out of v1"
                )
            ring_enabled = bool(ring.get('enabled', False))

        model = raw.get('model', {})
        _reject_unknown(model, ('render',), f"{path}:model")
        render = model.get('render', {})
        _reject_unknown(
            render, ('oversample', 'line_window_mode'), f"{path}:model.render"
        )
        render_oversample = _require_yaml_int(
            render, 'oversample', 3, f"{path}:model.render"
        )

        fit = raw['fit']
        _reject_unknown(
            fit,
            (
                'sampler',
                'n_warmup',
                'n_samples',
                'n_chains',
                'precondition',
                'unconstrained',
                'adapt_mass',
                'target_accept',
                'n_map_starts',
                'pin_z_to_truth',
                'sample_bulge_nsersic',
                'shear_prior_sigma',
                'shear_prior_type',
                'shear_prior_halfwidth',
                'escalation',
            ),
            f"{path}:fit",
        )
        if fit.get('sampler', 'numpyro') != 'numpyro':
            raise NotImplementedError(
                f"v1 supports the numpyro sampler only, got " f"{fit.get('sampler')!r}"
            )
        escalation = _parse_escalation(fit.get('escalation'), f"{path}:fit.escalation")

        dispatch = raw['dispatch']
        _reject_unknown(
            dispatch,
            (
                'backend',
                'mode',
                'workers_per_node',
                'target_task_walltime_min',
                'queue',
                'account',
                'max_fit_walltime_min',
            ),
            f"{path}:dispatch",
        )

        output = raw.get('output', {})
        _reject_unknown(output, ('save_chains', 'save_mocks'), f"{path}:output")

        return cls(
            run_name=str(run['name']),
            version=int(run['version']),
            description=str(run['description']),
            seed=int(run['seed']),
            measurement=str(run['measurement']),
            population_type=population_type,
            stratify_param=strat_param,
            stratify_n_bins=strat_n_bins,
            stratify_range=strat_range,
            sweep_values=sweep_values,
            n_gal_per_bin=n_gal_per_bin,
            m_noise=int(run.get('noise_reps', 1)),
            draw=draw,
            fixed=fixed,
            shear_scheme=scheme,
            g1=shear_g1,
            g2=shear_g2,
            shear_grid=shear_grid,
            shear_component=shear_component,
            shear_fit_prior_sigma=float(fit.get('shear_prior_sigma', 0.2)),
            shear_fit_prior_type=str(fit.get('shear_prior_type', 'gaussian')),
            shear_fit_prior_halfwidth=float(fit.get('shear_prior_halfwidth', 0.3)),
            ring_enabled=ring_enabled,
            catalog_population=catalog_population,
            render_oversample=render_oversample,
            render_line_window_mode=str(render.get('line_window_mode', 'global')),
            observed_config=str(observation['config']),
            # catalog populations carry per-galaxy SNRs (both channels) in
            # the population table; the scalar fields get -1 sentinels so any
            # accidental use fails the positive-SNR checks loudly. Swept
            # axes: per-fit values live in the manifest; the scalar field is
            # unused (set to the first sweep value as a placeholder that
            # keeps validation simple)
            broadband_snr=(
                -1.0 if population_type == 'catalog' else float(snr['broadband'])
            ),
            line_snr=(
                -1.0
                if population_type == 'catalog'
                else float(snr['line']) if 'line' in snr else float(sweep_values[0])
            ),
            n_warmup=int(fit.get('n_warmup', 500)),
            n_samples=int(fit.get('n_samples', 1000)),
            n_chains=int(fit.get('n_chains', 4)),
            precondition=str(fit.get('precondition', 'laplace')),
            unconstrained=bool(fit.get('unconstrained', False)),
            adapt_mass=bool(fit.get('adapt_mass', False)),
            target_accept=float(fit.get('target_accept', 0.8)),
            n_map_starts=int(fit.get('n_map_starts', 4)),
            pin_z_to_truth=bool(fit.get('pin_z_to_truth', True)),
            sample_bulge_nsersic=bool(fit.get('sample_bulge_nsersic', False)),
            escalation=escalation,
            backend=str(dispatch.get('backend', 'local')),
            mode=str(dispatch.get('mode', 'dynamic')),
            workers_per_node=int(dispatch.get('workers_per_node', 1)),
            target_task_walltime_min=float(
                dispatch.get('target_task_walltime_min', 90.0)
            ),
            queue=str(dispatch.get('queue', 'gh-dev')),
            account=str(dispatch.get('account', '')),
            max_fit_walltime_min=float(dispatch.get('max_fit_walltime_min', 30.0)),
            save_chains=str(output.get('save_chains', 'none')),
            save_mocks=str(output.get('save_mocks', 'none')),
        )


def _resolve_fixed_block(fixed_raw: dict, context: str) -> Dict[str, float]:
    """Resolve the spec's population.fixed block to dotted parameter names.

    Short keys in _BROADCAST_FIXED are kept as-is (broadcast to every scene
    component by the scene builder); dotted keys pass through; aliased keys
    resolve; 'flux' is rejected as ambiguous across components.
    """
    fixed: Dict[str, float] = {}
    for name, value in fixed_raw.items():
        if name == 'flux':
            raise ValueError(
                f"{context}:population.fixed: 'flux' is ambiguous across "
                f"components (band flux vs line flux vs continuum "
                f"flux_per_nm); use dotted names like 'F087.flux', or omit "
                f"to keep scene defaults"
            )
        resolved = _PARAM_ALIASES.get(name, name)
        fixed[resolved] = float(value)
    return fixed
