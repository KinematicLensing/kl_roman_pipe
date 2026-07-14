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
verbatim snapshot of the referenced observing config plus its content hash,
the git commit, and the expander version. The runner loads the snapshot,
never the live registry.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from kl_pipe.ensemble.scene import scene_truth_defaults
from kl_pipe.ensemble.spec import DrawSpec, EnsembleSpec, ObservingConfig

# bump when the expansion algorithm changes in a way that alters manifests
EXPANDER_VERSION = 1

# seed-stream domain tags: keep galaxy-truth draws and noise seeds in
# non-overlapping SeedSequence key spaces
_GALAXY_STREAM = 1
_NOISE_STREAM = 2

TRUTH_PREFIX = 'truth.'


def compute_fit_id(
    run_name: str,
    version: int,
    cosi_bin: int,
    galaxy_id: int,
    ring_member: int,
    noise_rep: int,
    shear_step: int,
) -> str:
    """Stable fit id from the integer index tuple (plus run identity)."""
    key = (
        f'{run_name}|{version}|{cosi_bin}|{galaxy_id}|{ring_member}'
        f'|{noise_rep}|{shear_step}'
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


def build_manifest(spec: EnsembleSpec, config: ObservingConfig) -> pd.DataFrame:
    """Expand the spec into the per-fit manifest table."""
    base_truth = scene_truth_defaults(config, spec.fixed)

    required_draws = {'theta_int', 'vel.vcirc', 'z'}
    missing = required_draws - set(spec.draw)
    if missing:
        raise ValueError(
            f"spec bank.draw must include {sorted(required_draws)}; "
            f"missing {sorted(missing)}"
        )

    n_shear = len(spec.shear_grid) if spec.shear_scheme == 'grid' else 1
    ring_members = (0, 90) if spec.ring_enabled else (0,)
    cosi_centers = _cosi_bin_centers(spec)

    rows: List[dict] = []
    for cosi_bin, cosi in enumerate(cosi_centers):
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
                    for noise_rep in range(spec.m_noise):
                        truth = dict(base_truth)
                        truth.update(drawn)
                        truth.update(
                            {
                                'cosi': float(cosi),
                                'theta_int': float(theta),
                                'g1': shear['g1'],
                                'g2': shear['g2'],
                            }
                        )
                        fit_id = compute_fit_id(
                            spec.run_name,
                            spec.version,
                            cosi_bin,
                            galaxy_id,
                            ring_member,
                            noise_rep,
                            shear_step,
                        )
                        ring_partner = (
                            compute_fit_id(
                                spec.run_name,
                                spec.version,
                                cosi_bin,
                                galaxy_id,
                                90 if ring_member == 0 else 0,
                                noise_rep,
                                shear_step,
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
                            'noise_seed': _noise_seed(
                                spec.seed,
                                cosi_bin,
                                galaxy_id,
                                ring_member,
                                noise_rep,
                            ),
                            'observed_config_id': config.id,
                            'broadband_snr': spec.broadband_snr,
                            'grism_snr': spec.grism_snr,
                            'save_chains': spec.save_chains == 'all',
                            'save_mocks': spec.save_mocks == 'all',
                        }
                        row.update({f'{TRUTH_PREFIX}{k}': v for k, v in truth.items()})
                        rows.append(row)

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
            provenance/
                ensemble_spec.yaml          (verbatim copy)
                observing_config.yaml       (verbatim snapshot)
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
        Directory holding observing configs (``<id>.yaml``).
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
            f"observing config '{spec.observed_config}' not found at " f"{config_path}"
        )
    config = ObservingConfig.from_yaml(config_path)

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

    manifest = build_manifest(spec, config)
    manifest.to_parquet(run_dir / 'manifest.parquet', index=False)

    shutil.copy2(spec_path, run_dir / 'provenance' / 'ensemble_spec.yaml')
    shutil.copy2(config_path, run_dir / 'provenance' / 'observing_config.yaml')
    expansion_record = {
        'run_name': spec.run_name,
        'spec_version': spec.version,
        'expander_version': EXPANDER_VERSION,
        'n_fits': int(len(manifest)),
        'observed_config_id': config.id,
        'observing_config_hash': config.content_hash,
        'spec_hash': hashlib.sha256(spec_path.read_bytes()).hexdigest(),
        'git_commit': _git_commit(),
    }
    (run_dir / 'provenance' / 'expansion.json').write_text(
        json.dumps(expansion_record, indent=2) + '\n'
    )
    return run_dir


def load_run(run_dir: Path):
    """Load (spec, config, manifest) from a run directory's provenance."""
    run_dir = Path(run_dir)
    spec = EnsembleSpec.from_yaml(run_dir / 'provenance' / 'ensemble_spec.yaml')
    config = ObservingConfig.from_yaml(run_dir / 'provenance' / 'observing_config.yaml')
    manifest = pd.read_parquet(run_dir / 'manifest.parquet')
    return spec, config, manifest
