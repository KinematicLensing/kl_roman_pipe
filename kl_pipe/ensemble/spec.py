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


@dataclass(frozen=True)
class PSFSpec:
    """One channel's PSF specification (a broadband band or the grism).

    ``gaussian`` carries ``fwhm_arcsec``; ``roman_wfi`` (realistic WFI PSF
    via ``galsim.roman.getPSF``, monochromatic) carries ``sca`` and
    ``pupil_bin``. Cross-type fields must be None (raises otherwise).
    """

    psf_type: str
    fwhm_arcsec: Optional[float] = None  # gaussian only, arcsec
    sca: Optional[int] = None  # roman_wfi only, 1-18
    pupil_bin: Optional[int] = None  # roman_wfi only

    def __post_init__(self):
        if self.psf_type == 'gaussian':
            if self.fwhm_arcsec is None or self.fwhm_arcsec <= 0:
                raise ValueError(
                    f"gaussian psf needs a positive fwhm_arcsec, got "
                    f"{self.fwhm_arcsec!r}"
                )
            if self.sca is not None or self.pupil_bin is not None:
                raise ValueError("sca/pupil_bin are roman_wfi-only psf fields")
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
        else:
            raise NotImplementedError(
                f"psf type {self.psf_type!r} not supported; supported types: "
                f"{list(_PSF_TYPES)}"
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


def _parse_roman_wfi_psf(block: dict, context: str) -> PSFSpec:
    _reject_unknown(block, ('type', 'sca', 'pupil_bin'), context)
    return PSFSpec(
        psf_type='roman_wfi',
        sca=_require_yaml_int(block, 'sca', _ROMAN_WFI_DEFAULT_SCA, context),
        pupil_bin=_require_yaml_int(
            block, 'pupil_bin', _ROMAN_WFI_DEFAULT_PUPIL_BIN, context
        ),
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

    @property
    def band_psf_fwhm(self) -> Dict[str, float]:
        """Per-band gaussian FWHM (arcsec). Raises for non-gaussian PSFs."""
        for band, psf in self.band_psf.items():
            if psf.psf_type != 'gaussian':
                raise ValueError(
                    f"band_psf_fwhm is defined for gaussian PSFs only; band "
                    f"'{band}' has psf type '{psf.psf_type}' (use band_psf)"
                )
        return {band: psf.fwhm_arcsec for band, psf in self.band_psf.items()}

    @property
    def grism_psf_fwhm(self) -> float:
        """Grism gaussian FWHM (arcsec). Raises for non-gaussian PSFs."""
        if self.grism_psf.psf_type != 'gaussian':
            raise ValueError(
                f"grism_psf_fwhm is defined for gaussian PSFs only; grism "
                f"psf type is '{self.grism_psf.psf_type}' (use grism_psf)"
            )
        return self.grism_psf.fwhm_arcsec

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
