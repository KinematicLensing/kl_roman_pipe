"""
Deterministic ensemble expansion: spec -> manifest.parquet + provenance.

The manifest is the dispatch source of truth: one row per fit, fully-resolved
truth values, immutable once written (status is derived from the filesystem
ledger, never stored in the manifest).

Determinism contract
--------------------
- fit_id hashes the integer index tuple (run_name, version, cosi_bin,
  galaxy_id, ring_member, noise_rep, shear_step) -- never float truth values,
  whose repr is unstable across numpy versions.
- Per-galaxy truths are drawn from numpy SeedSequence streams keyed on
  (spec.seed, cosi_bin, galaxy_id): re-expanding the same spec reproduces
  identical rows.
- Noise seeds follow the CRN rule: one seed per (galaxy, ring_member,
  noise_rep), held constant across the shear grid (contrasts share noise);
  fresh per galaxy and per noise_rep (absolutes use independent noise).

Provenance: the run directory receives a verbatim copy of the spec, a
verbatim snapshot of the referenced observation config plus its content
hash, the git commit, and the expander version. The runner loads the
snapshot, never the live registry.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from kl_pipe.ensemble.population import (
    N_GRISM_PASSES,
    build_population,
    write_population,
)
from kl_pipe.ensemble.scene import scene_truth_defaults
from kl_pipe.ensemble.spec import DrawSpec, EnsembleSpec, ObservationConfig

# bump when the expansion algorithm changes in a way that alters manifests
EXPANDER_VERSION = 1

# seed-stream domain tags: keep galaxy-truth draws and noise seeds in
# non-overlapping SeedSequence key spaces
_GALAXY_STREAM = 1
_NOISE_STREAM = 2

TRUTH_PREFIX = 'truth.'
POP_PREFIX = 'pop.'

# population-table columns carried through to the manifest (prefixed 'pop.')
# for prior construction (prior_vcirc_*), selection/binning diagnostics, and
# future BulgeDisk truth wiring (bulge_*)
_POP_PASSTHROUGH = (
    'halo_id',
    'galaxy_id',
    'snr_line_per_pass',
    'snr_line_total',
    'ew_rest_a',
    'logm',
    'logm_obs',
    'prior_vcirc_mu_kms',
    'prior_vcirc_sigma_dex',
    'sigma0_kms',
    'bulge_fraction',
    'bulge_r50_arcsec',
    'bulge_nsersic',
    'f_line_cgs',
)


def compute_fit_id(
    run_name: str,
    version: int,
    cosi_bin: int,
    galaxy_id: int,
    ring_member: int,
    noise_rep: int,
    shear_step: int,
    sweep_step: int = 0,
) -> str:
    """Stable fit id from the integer index tuple (plus run identity)."""
    key = (
        f'{run_name}|{version}|{cosi_bin}|{galaxy_id}|{ring_member}'
        f'|{noise_rep}|{shear_step}|{sweep_step}'
    )
    return hashlib.sha1(key.encode()).hexdigest()[:16]


def _draw_value(draw: DrawSpec, rng: np.random.Generator) -> float:
    if draw.dist == 'uniform':
        return float(rng.uniform(draw.params['low'], draw.params['high']))
    if draw.dist == 'lognormal_tf':
        sigma_ln = draw.params['sigma_tf_dex'] * np.log(10.0)
        return float(
            np.exp(np.log(draw.params['center_kms']) + sigma_ln * rng.normal())
        )
    raise ValueError(f"unknown draw dist '{draw.dist}'")


def _galaxy_rng(spec_seed: int, cosi_bin: int, galaxy_id: int):
    ss = np.random.SeedSequence([spec_seed, _GALAXY_STREAM, cosi_bin, galaxy_id])
    return np.random.default_rng(ss)


def _noise_seed(
    spec_seed: int,
    cosi_bin: int,
    galaxy_id: int,
    ring_member: int,
    noise_rep: int,
) -> int:
    # NOTE: shear_step deliberately absent -- CRN across the shear grid
    ss = np.random.SeedSequence(
        [spec_seed, _NOISE_STREAM, cosi_bin, galaxy_id, ring_member, noise_rep]
    )
    return int(ss.generate_state(1, dtype=np.uint32)[0])


def _cosi_bin_centers(spec: EnsembleSpec) -> np.ndarray:
    lo, hi = spec.stratify_range
    edges = np.linspace(lo, hi, spec.stratify_n_bins + 1)
    return 0.5 * (edges[:-1] + edges[1:])


def _shear_for_step(spec: EnsembleSpec, step: int) -> Dict[str, float]:
    if spec.shear_scheme == 'fixed':
        return {'g1': spec.g1, 'g2': spec.g2}
    value = spec.shear_grid[step]
    if spec.shear_component == 'g1':
        return {'g1': value, 'g2': 0.0}
    return {'g1': 0.0, 'g2': value}


def build_manifest(
    spec: EnsembleSpec,
    config: ObservationConfig,
    population: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Expand the spec into the per-fit manifest table.

    Sampled populations: axis semantics are a cosi stratification (each bin
    holds its OWN galaxy bank -- independent noise) or a config sweep
    (line_snr: ONE galaxy bank shared across every sweep step with the SAME
    noise seed -- common random numbers).

    Catalog populations: rows cross the population table (one row per
    galaxy, from ``build_population``) with ring members and noise reps;
    ``population`` is required.
    """
    if spec.catalog_population is not None:
        if population is None:
            raise ValueError(
                "catalog-mode manifests require the population table "
                "(build_population output); got population=None"
            )
        rows = _catalog_rows(spec, config, population)
    else:
        if population is not None:
            raise ValueError(
                "a population table is catalog-mode only; sampled specs draw "
                "their galaxies internally"
            )
        rows = _sampled_rows(spec, config)

    manifest = pd.DataFrame(rows)
    if manifest['fit_id'].duplicated().any():
        raise RuntimeError("duplicate fit_id in manifest -- expander bug")
    if len(manifest) != spec.n_fits:
        raise RuntimeError(
            f"manifest has {len(manifest)} rows, spec expects {spec.n_fits} "
            f"-- expander bug"
        )
    _apply_subset_policy(manifest, spec)
    return manifest


def _sampled_rows(spec: EnsembleSpec, config: ObservationConfig) -> List[dict]:
    """Manifest rows for a sampled (stratified/swept) population."""
    base_truth = scene_truth_defaults(config, spec.fixed)

    required_draws = {'theta_int', 'vel.vcirc', 'z'}
    missing = required_draws - set(spec.draw)
    if missing:
        raise ValueError(
            f"spec population.draw must include {sorted(required_draws)}; "
            f"missing {sorted(missing)}"
        )

    n_shear = len(spec.shear_grid) if spec.shear_scheme == 'grid' else 1
    ring_members = (0, 90) if spec.ring_enabled else (0,)

    if spec.sweep_values:
        # config sweep: one galaxy bank (bank bin 0), swept observed knob
        bank_bins = [(0, {})]
        sweep_steps = list(enumerate(spec.sweep_values))
    else:
        # truth stratification: a galaxy bank per cosi bin
        bank_bins = [
            (b, {'cosi': float(c)}) for b, c in enumerate(_cosi_bin_centers(spec))
        ]
        sweep_steps = [(0, None)]

    rows: List[dict] = []
    for cosi_bin, strat_truth in bank_bins:
        for galaxy_id in range(spec.n_gal_per_bin):
            rng = _galaxy_rng(spec.seed, cosi_bin, galaxy_id)
            # fixed draw order = stable across spec-dict insertion order
            drawn = {
                name: _draw_value(spec.draw[name], rng) for name in sorted(spec.draw)
            }
            for ring_member in ring_members:
                theta = drawn['theta_int']
                if ring_member == 90:
                    theta = (theta + np.pi / 2) % np.pi
                for shear_step in range(n_shear):
                    shear = _shear_for_step(spec, shear_step)
                    for sweep_step, sweep_value in sweep_steps:
                        for noise_rep in range(spec.m_noise):
                            truth = dict(base_truth)
                            truth.update(drawn)
                            truth.update(strat_truth)
                            truth.update(
                                {
                                    'theta_int': float(theta),
                                    'g1': shear['g1'],
                                    'g2': shear['g2'],
                                }
                            )
                            id_args = (
                                spec.run_name,
                                spec.version,
                                cosi_bin,
                                galaxy_id,
                            )
                            fit_id = compute_fit_id(
                                *id_args,
                                ring_member,
                                noise_rep,
                                shear_step,
                                sweep_step,
                            )
                            ring_partner = (
                                compute_fit_id(
                                    *id_args,
                                    90 if ring_member == 0 else 0,
                                    noise_rep,
                                    shear_step,
                                    sweep_step,
                                )
                                if spec.ring_enabled
                                else ''
                            )
                            row = {
                                'fit_id': fit_id,
                                'run_name': spec.run_name,
                                'cosi_bin': cosi_bin,
                                'galaxy_id': galaxy_id,
                                'ring_member': ring_member,
                                'ring_partner_id': ring_partner,
                                'noise_rep': noise_rep,
                                'shear_step': shear_step,
                                'sweep_step': sweep_step,
                                # CRN: seed excludes shear_step AND sweep_step
                                'noise_seed': _noise_seed(
                                    spec.seed,
                                    cosi_bin,
                                    galaxy_id,
                                    ring_member,
                                    noise_rep,
                                ),
                                'observation_config_id': config.id,
                                'broadband_snr': spec.broadband_snr,
                                'line_snr': (
                                    float(sweep_value)
                                    if sweep_value is not None
                                    else spec.line_snr
                                ),
                                'save_chains': spec.save_chains == 'all',
                                'save_mocks': spec.save_mocks == 'all',
                            }
                            row.update(
                                {f'{TRUTH_PREFIX}{k}': v for k, v in truth.items()}
                            )
                            rows.append(row)
    return rows


def _catalog_rows(
    spec: EnsembleSpec, config: ObservationConfig, population: pd.DataFrame
) -> List[dict]:
    """Manifest rows for a catalog population: galaxies x ring x noise reps.

    The population table (one row per galaxy) carries the drawn orientation,
    shear, and painted kinematics; this expansion adds the ring partner
    (theta + pi/2, SAME g1/g2 -- the pair-shared shear is drawn per galaxy
    upstream) and the noise reps. fit_id/noise_seed reuse the sampled-mode
    index scheme with cosi_bin = 0 and galaxy_id = the population row index,
    so the CRN seed semantics are unchanged.
    """
    cp = spec.catalog_population
    if len(population) != cp.n_galaxies:
        raise ValueError(
            f"population table has {len(population)} rows, spec expects "
            f"{cp.n_galaxies} galaxies"
        )
    # the selection cuts on the line SNR coadded over N_GRISM_PASSES, while
    # each roll's mock is drawn at one pass's depth; if the config's roll
    # count differs, the depth the fit sees is not the depth selected for
    if len(config.grism_rolls_deg) != N_GRISM_PASSES:
        raise ValueError(
            f"observation config '{config.id}' has "
            f"{len(config.grism_rolls_deg)} grism rolls but the population "
            f"selection assumes N_GRISM_PASSES = {N_GRISM_PASSES} when "
            f"coadding the line SNR; the selected depth would not match the "
            f"fitted depth"
        )
    if cp.galaxy_ids is not None:
        # restrict AFTER the full-bank build so per-galaxy draws and noise
        # seeds are identical to the unrestricted run with the same seed;
        # original row indices are kept so galaxy_id pairs across runs
        missing = sorted(set(cp.galaxy_ids) - set(population.index))
        if missing:
            raise ValueError(
                f"sample.galaxy_ids {missing} not in the population index "
                f"(0..{len(population) - 1})"
            )
        population = population.loc[sorted(cp.galaxy_ids)]
    # catalog specs carry no population.fixed block; scene defaults apply.
    # broadband bands are BulgeDiskModel (per-galaxy bulge from the catalog)
    # unless the spec disables the bulge paint (disk-only twin)
    base_truth = scene_truth_defaults(config, {}, bulge_bands=cp.paint_bulge)
    ring_members = (0, 90) if cp.ring_members == 2 else (0,)
    # painted bulge columns are absent from a no-bulge population; keep
    # bulge_fraction (catalog fact, misspecification diagnostic) either way
    pop_passthrough = tuple(
        c
        for c in _POP_PASSTHROUGH
        if cp.paint_bulge or c not in ('bulge_r50_arcsec', 'bulge_nsersic')
    )

    rows: List[dict] = []
    for pop_index, g in population.iterrows():
        rscale = float(g['rscale_arcsec'])
        z = float(g['z'])
        for ring_member in ring_members:
            theta = float(g['theta_int'])
            if ring_member == 90:
                theta = (theta + np.pi / 2) % np.pi
            for noise_rep in range(spec.m_noise):
                truth = dict(base_truth)
                truth.update(
                    {
                        'cosi': float(g['cosi']),
                        'theta_int': theta,
                        'g1': float(g['g1']),
                        'g2': float(g['g2']),
                        'z': z,
                        'vel.vcirc': float(g['vcirc_kms']),
                        'Halpha.dispersion': float(g['sigma0_kms']),
                    }
                )
                # the catalog disk scale length sets the disk spatial scales
                # (bands, line, continuum, velocity). broadband bands are
                # BulgeDiskModel: the disk scale is disk_rscale, and the bulge
                # component gets the catalog bulge fraction and half-light
                # radius per galaxy (total_flux, disk_h_over_r, bulge_h_over_hlr,
                # x0, y0 keep the scene defaults from base_truth). The Halpha
                # line + continuum stay single-disk (rscale). With the bulge
                # paint disabled, bands are single-disk too: the band flux
                # keeps the scene default, which equals the bulge-mode
                # total_flux, so the twin's total broadband flux matches.
                for band in config.bands:
                    if cp.paint_bulge:
                        truth[f'{band}.disk_rscale'] = rscale
                        truth[f'{band}.bulge_frac'] = float(g['bulge_fraction'])
                        truth[f'{band}.bulge_hlr'] = float(g['bulge_r50_arcsec'])
                        truth[f'{band}.bulge_n_sersic'] = float(g['bulge_nsersic'])
                    else:
                        truth[f'{band}.rscale'] = rscale
                truth['Halpha.rscale'] = rscale
                truth['Halpha.cont.rscale'] = rscale
                truth['vel.rscale'] = rscale
                # continuum amplitude from the catalog rest-frame EW:
                # EW_obs [nm] = ew_rest_a [A] * (1 + z) / 10, and
                # flux_per_nm = line_flux / EW_obs [flux / nm] -- the scene's
                # internal flux units cancel, so only the EW ratio matters.
                # Band flux amplitudes stay scene defaults (internal units;
                # physical flux enters only through SNR); catalog-color
                # scaling of band fluxes is a flagged future upgrade.
                ew_obs_nm = float(g['ew_rest_a']) * (1.0 + z) / 10.0
                truth['Halpha.cont.flux_per_nm'] = truth['Halpha.flux'] / ew_obs_nm

                id_args = (spec.run_name, spec.version, 0, int(pop_index))
                fit_id = compute_fit_id(*id_args, ring_member, noise_rep, 0, 0)
                ring_partner = (
                    compute_fit_id(
                        *id_args, 90 if ring_member == 0 else 0, noise_rep, 0, 0
                    )
                    if cp.ring_members == 2
                    else ''
                )
                row = {
                    'fit_id': fit_id,
                    'run_name': spec.run_name,
                    'cosi_bin': 0,
                    'galaxy_id': int(pop_index),
                    'ring_member': ring_member,
                    'ring_partner_id': ring_partner,
                    'noise_rep': noise_rep,
                    'shear_step': 0,
                    'sweep_step': 0,
                    'noise_seed': _noise_seed(
                        spec.seed, 0, int(pop_index), ring_member, noise_rep
                    ),
                    'observation_config_id': config.id,
                    # broadband depth stays the spec scalar for now; a
                    # per-galaxy imaging depth anchor is a documented future
                    # step of the catalog integration
                    'broadband_snr': spec.broadband_snr,
                    # per-galaxy PER-PASS line SNR from the population's
                    # matched-filter calculation: the mock noise is drawn
                    # once per grism roll, so each roll gets one pass's
                    # depth. The coadded depth the fit sees, and the one the
                    # selection cut is applied to, is snr_line_total.
                    'line_snr': float(g['snr_line_per_pass']),
                    'save_chains': spec.save_chains == 'all',
                    'save_mocks': spec.save_mocks == 'all',
                }
                row.update({f'{TRUTH_PREFIX}{k}': v for k, v in truth.items()})
                row.update({f'{POP_PREFIX}{c}': g[c] for c in pop_passthrough})
                rows.append(row)
    return rows


def _apply_subset_policy(manifest: pd.DataFrame, spec: EnsembleSpec) -> None:
    """Resolve save_chains/save_mocks 'subset' to per-row booleans.

    Subset = the first galaxy of each cosi bin (first ring member, first
    noise rep, first shear step) -- one diagnostic-grade fit per bin.
    """
    for policy, col in [
        (spec.save_chains, 'save_chains'),
        (spec.save_mocks, 'save_mocks'),
    ]:
        if policy != 'subset':
            continue
        mask = (
            (manifest['galaxy_id'] == 0)
            & (manifest['ring_member'] == 0)
            & (manifest['noise_rep'] == 0)
            & (manifest['shear_step'] == 0)
        )
        manifest[col] = mask


def truth_from_row(row: Dict) -> Dict[str, float]:
    """Extract the dotted truth dict from a manifest row (dict or pd.Series)."""
    truth = {}
    for key, value in dict(row).items():
        if key.startswith(TRUTH_PREFIX):
            truth[key[len(TRUTH_PREFIX) :]] = float(value)
    if not truth:
        raise ValueError("row carries no truth.* columns")
    return truth


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).resolve().parent,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return 'unknown'


def expand(
    spec_path: Path,
    registry_dir: Path,
    runs_dir: Path,
    overwrite: bool = False,
) -> Path:
    """
    Expand a spec file into a run directory.

    Creates::

        runs_dir/<run_name>/
            manifest.parquet
            population.parquet              (catalog populations only)
            population_meta.json            (catalog populations only)
            provenance/
                ensemble_spec.yaml          (verbatim copy)
                observation_config.yaml     (verbatim snapshot)
                expansion.json              (hashes, git commit, versions)
            status/{claims,done,failed}/
            results/
            chains/
            mocks/

    Parameters
    ----------
    spec_path : Path
        The human-authored ensemble spec YAML.
    registry_dir : Path
        Directory holding observation configs (``<id>.yaml``).
    runs_dir : Path
        Parent directory for run outputs.
    overwrite : bool
        Refuse to overwrite an existing run directory unless True.

    Returns
    -------
    Path
        The run directory.
    """
    spec_path = Path(spec_path)
    spec = EnsembleSpec.from_yaml(spec_path)
    config_path = Path(registry_dir) / f'{spec.observed_config}.yaml'
    if not config_path.exists():
        raise FileNotFoundError(
            f"observation config '{spec.observed_config}' not found at "
            f"{config_path}"
        )
    config = ObservationConfig.from_yaml(config_path)

    run_dir = Path(runs_dir) / spec.run_name
    if run_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"run directory {run_dir} exists; pass overwrite=True "
                f"(--overwrite) to replace it"
            )
        shutil.rmtree(run_dir)

    for sub in (
        'provenance',
        'status/claims',
        'status/done',
        'status/failed',
        'results',
        'chains',
        'mocks',
    ):
        (run_dir / sub).mkdir(parents=True)

    # catalog populations: build + persist the population table alongside the
    # manifest; its sha256 and stage counts join the expansion record
    population = None
    population_record = {}
    if spec.catalog_population is not None:
        population, pop_meta = build_population(spec)
        pop_parquet, _ = write_population(run_dir, population, pop_meta)
        population_record = {
            'population_sha256': hashlib.sha256(pop_parquet.read_bytes()).hexdigest(),
            'population_stage_counts': {
                'n_raw': pop_meta['n_raw'],
                'n_disk': pop_meta['n_disk'],
                'kills': pop_meta['kills'],
                'n_selected': pop_meta['n_selected'],
                'n_sampled': pop_meta['n_sampled'],
            },
        }

    manifest = build_manifest(spec, config, population=population)
    manifest.to_parquet(run_dir / 'manifest.parquet', index=False)

    shutil.copy2(spec_path, run_dir / 'provenance' / 'ensemble_spec.yaml')
    shutil.copy2(config_path, run_dir / 'provenance' / 'observation_config.yaml')
    expansion_record = {
        'run_name': spec.run_name,
        'spec_version': spec.version,
        'expander_version': EXPANDER_VERSION,
        'n_fits': int(len(manifest)),
        'observation_config_id': config.id,
        'observation_config_hash': config.content_hash,
        'spec_hash': hashlib.sha256(spec_path.read_bytes()).hexdigest(),
        'git_commit': _git_commit(),
        **population_record,
    }
    (run_dir / 'provenance' / 'expansion.json').write_text(
        json.dumps(expansion_record, indent=2) + '\n'
    )
    return run_dir


def load_run(run_dir: Path):
    """Load (spec, config, manifest) from a run directory's provenance."""
    run_dir = Path(run_dir)
    spec = EnsembleSpec.from_yaml(run_dir / 'provenance' / 'ensemble_spec.yaml')
    config = ObservationConfig.from_yaml(
        run_dir / 'provenance' / 'observation_config.yaml'
    )
    manifest = pd.read_parquet(run_dir / 'manifest.parquet')
    return spec, config, manifest
