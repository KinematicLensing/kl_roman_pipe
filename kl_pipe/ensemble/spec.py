"""
Ensemble spec + observing-config registry.

Two YAML-backed, strictly-validated config layers:

- ``ObservingConfig``: the structural instrument setup (bands, grism rolls,
  PSFs, stamp/render geometry), shared by every fit in a campaign. Lives in a
  small registry (``configs/observing/<id>.yaml``) and is referenced by id
  from the ensemble spec. The expander snapshots the referenced file verbatim
  into the run's provenance directory with a content hash; the runner loads
  the snapshot, never the live registry.
- ``EnsembleSpec``: one campaign (one run_name): the galaxy bank (stratified
  grid + drawn nuisance truths + fixed constants), shear scheme, SNR knobs,
  fit settings, dispatch settings, and output retention.

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
# Observing-config registry
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
class ObservingConfig:
    """Structural observing setup shared by all fits in a campaign."""

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
    oversample: int
    content_hash: str = ''  # sha256 of the source YAML file bytes

    def __post_init__(self):
        if not self.bands:
            raise ValueError("observing config needs at least one band")
        for band in self.bands:
            if band not in self.band_psf:
                raise ValueError(f"band '{band}' has no entry in broadband psf spec")
            if not isinstance(self.band_psf[band], PSFSpec):
                raise ValueError(f"band '{band}' psf entry must be a PSFSpec")
        if not isinstance(self.grism_psf, PSFSpec):
            raise ValueError("grism_psf must be a PSFSpec")
        if not self.grism_rolls_deg:
            raise ValueError("observing config needs at least one grism roll")
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
            ('oversample', self.oversample),
        ]:
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} ({value}) must be a positive int")

    @classmethod
    def from_yaml(cls, path: Path) -> 'ObservingConfig':
        path = Path(path)
        raw_bytes = path.read_bytes()
        raw = yaml.safe_load(raw_bytes)
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: observing config must be a mapping")
        allowed = (
            'id',
            'bands',
            'grism',
            'lines',
            'psf',
            'pixel_scale_arcsec',
            'stamp',
            'render',
        )
        _reject_unknown(raw, allowed, str(path))
        _require_keys(raw, allowed, str(path))

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

        render = raw['render']
        _reject_unknown(render, ('oversample',), f"{path}:render")
        _require_keys(render, ('oversample',), f"{path}:render")

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
            oversample=int(render['oversample']),
            content_hash=hashlib.sha256(raw_bytes).hexdigest(),
        )


# =============================================================================
# Ensemble spec
# =============================================================================

_DRAW_DISTS = ('uniform', 'lognormal_tf')
_SHEAR_SCHEMES = ('fixed', 'grid')
_DISPATCH_MODES = ('static', 'dynamic')
_DISPATCH_BACKENDS = ('local', 'slurm')
_SAVE_POLICIES = ('none', 'subset', 'all')
_MEASUREMENTS = ('sigma_eps_vs_cosi', 'sigma_eps_vs_line_snr')

# spec draw/fixed keys may be shared top-level params, aliased short names, or
# fully-dotted source-model params; aliases resolve here
_PARAM_ALIASES = {'vcirc': 'vel.vcirc'}
# short fixed keys broadcast to every scene component carrying that suffix
_BROADCAST_FIXED = ('h_over_r', 'x0', 'y0')


@dataclass(frozen=True)
class DrawSpec:
    """One drawn truth distribution from the spec's bank.draw block."""

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
class EnsembleSpec:
    """One ensemble campaign, loaded from ensemble_spec.yaml."""

    run_name: str
    version: int
    description: str
    seed: int
    measurement: str

    # bank: exactly one plot axis, either a truth stratification (cosi bins)
    # or a config sweep (line_snr values; galaxies + noise shared across the
    # sweep -- common random numbers)
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

    # observing
    observed_config: str
    broadband_snr: float
    line_snr: float

    # fit
    n_warmup: int
    n_samples: int
    n_chains: int
    precondition: str
    target_accept: float
    n_map_starts: int
    pin_z_to_truth: bool

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

    def __post_init__(self):
        if self.measurement not in _MEASUREMENTS:
            raise NotImplementedError(
                f"measurement '{self.measurement}' not supported; "
                f"available: {_MEASUREMENTS}"
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
                    "sigma_eps_vs_line_snr requires cosi in bank.draw "
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
        if self.broadband_snr <= 0 or self.line_snr <= 0:
            raise ValueError("SNR values must be positive")
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

    @property
    def n_axis_steps(self) -> int:
        """Number of steps along the plot axis (cosi bins or sweep values)."""
        if self.sweep_values:
            return len(self.sweep_values)
        return self.stratify_n_bins

    @property
    def n_fits(self) -> int:
        """Total fits this spec expands to."""
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
            'run_name',
            'version',
            'description',
            'seed',
            'measurement',
            'bank',
            'shear',
            'ring',
            'observed_config',
            'broadband_snr',
            'line_snr',
            'fit',
            'dispatch',
            'output',
        )
        _reject_unknown(raw, allowed, str(path))
        # line_snr is required as a scalar UNLESS it is the swept axis
        _require_keys(raw, tuple(k for k in allowed if k != 'line_snr'), str(path))

        bank = raw['bank']
        _reject_unknown(
            bank,
            ('stratify', 'n_gal_per_bin', 'm_noise', 'draw', 'fixed'),
            f"{path}:bank",
        )
        _require_keys(bank, ('stratify', 'n_gal_per_bin', 'draw'), f"{path}:bank")
        stratify = bank['stratify']
        if len(stratify) != 1:
            raise ValueError(
                f"{path}:bank.stratify must contain exactly one parameter, "
                f"got {list(stratify)}"
            )
        strat_param, strat_cfg = next(iter(stratify.items()))
        strat_n_bins = 0
        strat_range = (0.0, 0.0)
        sweep_values: Tuple[float, ...] = ()
        if strat_param == 'line_snr':
            _reject_unknown(
                strat_cfg, ('values',), f"{path}:bank.stratify.{strat_param}"
            )
            _require_keys(strat_cfg, ('values',), f"{path}:bank.stratify.{strat_param}")
            sweep_values = tuple(float(v) for v in strat_cfg['values'])
            if 'line_snr' in raw:
                raise ValueError(
                    f"{path}: top-level line_snr conflicts with the "
                    f"line_snr sweep axis; remove it (per-fit values come "
                    f"from bank.stratify.line_snr.values)"
                )
        else:
            if 'line_snr' not in raw:
                raise ValueError(f"{path}: missing required keys ['line_snr']")
            _reject_unknown(
                strat_cfg, ('n_bins', 'range'), f"{path}:bank.stratify.{strat_param}"
            )
            _require_keys(
                strat_cfg, ('n_bins', 'range'), f"{path}:bank.stratify.{strat_param}"
            )
            strat_n_bins = int(strat_cfg['n_bins'])
            strat_range = (
                float(strat_cfg['range'][0]),
                float(strat_cfg['range'][1]),
            )

        draw = {}
        for name, dcfg in bank['draw'].items():
            dcfg = dict(dcfg)
            dist = dcfg.pop('dist', None)
            if dist is None:
                raise ValueError(f"{path}:bank.draw.{name}: missing 'dist'")
            if dist == 'uniform' and 'range' in dcfg:
                lo, hi = dcfg.pop('range')
                dcfg.update(low=lo, high=hi)
            resolved = _PARAM_ALIASES.get(name, name)
            draw[resolved] = DrawSpec(dist=dist, params=dcfg)

        fixed = _resolve_fixed_block(bank.get('fixed', {}), context=f"{path}")

        shear = raw['shear']
        _reject_unknown(
            shear,
            ('scheme', 'g1', 'g2', 'grid', 'component', 'fit_prior_sigma'),
            f"{path}:shear",
        )
        scheme = shear.get('scheme', 'fixed')
        shear_grid: Tuple[float, ...] = ()
        shear_component = ''
        if scheme == 'grid':
            _require_keys(shear, ('grid', 'component'), f"{path}:shear")
            shear_grid = tuple(float(g) for g in shear['grid'])
            shear_component = shear['component']

        ring = raw.get('ring', {})
        _reject_unknown(ring, ('enabled', 'antithetic_g'), f"{path}:ring")
        if ring.get('antithetic_g', False):
            raise NotImplementedError("antithetic_g (+/-g pairs) is scoped out of v1")

        fit = raw['fit']
        _reject_unknown(
            fit,
            (
                'sampler',
                'n_warmup',
                'n_samples',
                'n_chains',
                'precondition',
                'target_accept',
                'n_map_starts',
                'pin_z_to_truth',
            ),
            f"{path}:fit",
        )
        if fit.get('sampler', 'numpyro') != 'numpyro':
            raise NotImplementedError(
                f"v1 supports the numpyro sampler only, got " f"{fit.get('sampler')!r}"
            )

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
            run_name=str(raw['run_name']),
            version=int(raw['version']),
            description=str(raw['description']),
            seed=int(raw['seed']),
            measurement=str(raw['measurement']),
            stratify_param=strat_param,
            stratify_n_bins=strat_n_bins,
            stratify_range=strat_range,
            sweep_values=sweep_values,
            n_gal_per_bin=int(bank['n_gal_per_bin']),
            m_noise=int(bank.get('m_noise', 1)),
            draw=draw,
            fixed=fixed,
            shear_scheme=scheme,
            g1=float(shear.get('g1', 0.0)),
            g2=float(shear.get('g2', 0.0)),
            shear_grid=shear_grid,
            shear_component=shear_component,
            shear_fit_prior_sigma=float(shear.get('fit_prior_sigma', 0.2)),
            ring_enabled=bool(ring.get('enabled', False)),
            observed_config=str(raw['observed_config']),
            broadband_snr=float(raw['broadband_snr']),
            # swept axis: per-fit values live in the manifest; the scalar
            # field is unused (set to the first sweep value as a placeholder
            # that keeps validation simple)
            line_snr=(
                float(raw['line_snr']) if 'line_snr' in raw else float(sweep_values[0])
            ),
            n_warmup=int(fit.get('n_warmup', 500)),
            n_samples=int(fit.get('n_samples', 1000)),
            n_chains=int(fit.get('n_chains', 4)),
            precondition=str(fit.get('precondition', 'laplace')),
            target_accept=float(fit.get('target_accept', 0.8)),
            n_map_starts=int(fit.get('n_map_starts', 4)),
            pin_z_to_truth=bool(fit.get('pin_z_to_truth', True)),
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
    """Resolve the spec's bank.fixed block to dotted parameter names.

    Short keys in _BROADCAST_FIXED are kept as-is (broadcast to every scene
    component by the scene builder); dotted keys pass through; aliased keys
    resolve; 'flux' is rejected as ambiguous across components.
    """
    fixed: Dict[str, float] = {}
    for name, value in fixed_raw.items():
        if name == 'flux':
            raise ValueError(
                f"{context}:bank.fixed: 'flux' is ambiguous across components "
                f"(band flux vs line flux vs continuum flux_per_nm); use "
                f"dotted names like 'F087.flux', or omit to keep scene "
                f"defaults"
            )
        resolved = _PARAM_ALIASES.get(name, name)
        fixed[resolved] = float(value)
    return fixed
