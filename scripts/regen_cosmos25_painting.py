"""Regenerate the COSMOS25 painted emission-line section from the painting
notebook's own code.

Jiachuan Xu delivers the painting as a notebook (IAEstimate.ipynb); the
painted FITS is regenerated locally by executing the notebook's own cells
verbatim, up to and including the cell that writes the mock_emission_lines
file. Later notebook sections (redMaGiC / IA / eBOSS) are never executed.
The only modifications to the executed sources are the input/output path
substitutions in PATCHES below (plus one unicode-minus normalization his
astropy accepted but ours rejects); every patch is asserted to occur an
exact number of times and recorded in the provenance sidecar.

After the write, the notebook's own density-printing cell is executed
verbatim twice -- against the in-memory results and against the written
FITS read back -- and both must reproduce the notebook's printed number
densities to 4 decimals (deep KL 6.8307, deep WL 58.4793, medium KL 1.9276
per arcmin^2). Any mismatch deletes the output and fails hard.

The painting is deterministic (no RNG in the executed cells), so the output
is a pure function of the notebook code and the three v1 input sections,
all sha256-pinned in the sidecar.

Setup: copy the delivered notebook into the (gitignored) private data dir:

    cp ~/Downloads/IAEstimate.ipynb data/cosmos2025/private/IAEstimate.ipynb

Usage:
    python scripts/regen_cosmos25_painting.py
    python scripts/regen_cosmos25_painting.py --notebook <path> --force
"""

import argparse
import hashlib
import io
import json
import platform
import re
import shutil
import sys
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path

import nbformat

REPO_ROOT = Path(__file__).resolve().parent.parent

OUT_NAME = 'COSMOSWeb_mastercatalog_v1_mock_emission_lines.fits'
BACKUP_NAME = (
    'COSMOSWeb_mastercatalog_v1_mock_emission_lines.orig_delivery_20260728.fits'
)

# code cells executed, in order, ending at the FITS-write cell; the density
# cell is executed afterwards for the handshake only
EXEC_CELL_INDICES = (0, 2, 4, 6, 7, 8, 9, 10)
DENSITY_CELL_INDEX = 14

# the notebook's own printed densities [gal/arcmin^2] (saved outputs of the
# density cell, 2026-07-29 notebook revision); the mandatory handshake
DENSITY_TARGETS = {'deep_kl': 6.8307, 'deep_wl': 58.4793, 'medium_kl': 1.9276}
DENSITY_TOL = 5e-5

# his machine's catalog dir, as written in the notebook
_JX_DIR = '/Users/jiachuanxu/Workspace/COSMOS2025'


def _patch_table(data_dir: Path, out_fits: Path) -> list:
    """(cell, old, new, expected occurrence count) source substitutions."""
    return [
        # environment compat: the notebook's grism-depth unit strings carry
        # the UNICODE MINUS SIGN (U+2212); astropy 7.x's FITS parser rejects
        # it while his astropy accepted it. ASCII hyphen parses to the
        # numerically identical unit
        (2, 'erg s−1 cm−2', 'erg s-1 cm-2', 2),
        (
            9,
            f'{_JX_DIR}/COSMOSWeb_mastercatalog_v1_photom_primary.fits',
            str(data_dir / 'COSMOSWeb_mastercatalog_v1_photom_primary.fits'),
            1,
        ),
        (
            9,
            f'{_JX_DIR}/COSMOSWeb_mastercatalog_v1_cigale.fits',
            str(data_dir / 'COSMOSWeb_mastercatalog_v1_cigale.fits'),
            1,
        ),
        (
            9,
            f'{_JX_DIR}/COSMOSWeb_mastercatalog_v1_lephare.fits',
            str(data_dir / 'COSMOSWeb_mastercatalog_v1_lephare.fits'),
            1,
        ),
        (10, f'{_JX_DIR}/{OUT_NAME}', str(out_fits), 1),
    ]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 22), b''):
            h.update(chunk)
    return h.hexdigest()


def extract_sources(nb, patches: list) -> tuple:
    """Verbatim cell sources with the asserted patches applied.

    Returns (sources dict keyed by cell index, applied-patch records).
    """
    patched_cells = {c for c, *_ in patches}
    sources, applied = {}, []
    for i in EXEC_CELL_INDICES + (DENSITY_CELL_INDEX,):
        cell = nb.cells[i]
        if cell.cell_type != 'code':
            raise RuntimeError(
                f"notebook cell {i} is {cell.cell_type}, expected code; the "
                f"notebook layout drifted -- re-pin EXEC_CELL_INDICES"
            )
        src = cell.source
        original_sha = hashlib.sha256(src.encode()).hexdigest()
        for cell_idx, old, new, n_expected in patches:
            if cell_idx != i:
                continue
            n_occ = src.count(old)
            if n_occ != n_expected:
                raise RuntimeError(
                    f"cell {i}: patch target occurs {n_occ} times "
                    f"(expected {n_expected}): {old!r}; the notebook drifted "
                    f"-- re-verify the patch table"
                )
            src = src.replace(old, new)
            applied.append({'cell': i, 'old': old, 'new': new, 'occurrences': n_occ})
        sources[i] = {
            'source': src,
            'original_sha256': original_sha,
            'patched': i in patched_cells,
        }
    # structural gates on the pinned indices
    if OUT_NAME not in sources[EXEC_CELL_INDICES[-1]]['source']:
        raise RuntimeError(
            f"cell {EXEC_CELL_INDICES[-1]} does not write {OUT_NAME}; the "
            f"notebook layout drifted -- re-pin EXEC_CELL_INDICES"
        )
    if 'number density' not in sources[DENSITY_CELL_INDEX]['source']:
        raise RuntimeError(
            f"cell {DENSITY_CELL_INDEX} does not print number densities; "
            f"re-pin DENSITY_CELL_INDEX"
        )
    return sources, applied


def run_density_cell(density_code, namespace: dict, label: str) -> dict:
    """Execute the notebook's density cell and gate its printed numbers."""
    buf = io.StringIO()
    with redirect_stdout(buf):
        exec(density_code, namespace)
    text = buf.getvalue()
    print(f'--- densities ({label}) ---\n{text.strip()}')
    patterns = {
        'deep_kl': r'Deep Tier KL.*?= ([\d.]+)',
        'deep_wl': r'Deep Tier WL.*?= ([\d.]+)',
        'medium_kl': r'Medium Tier.*?= ([\d.]+)',
    }
    got = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match is None:
            raise RuntimeError(f'{label}: density line for {key} not printed')
        got[key] = float(match.group(1))
        if abs(got[key] - DENSITY_TARGETS[key]) >= DENSITY_TOL:
            raise RuntimeError(
                f'{label}: density {key} = {got[key]:.4f} != notebook '
                f'{DENSITY_TARGETS[key]:.4f}; the regeneration does not '
                f'reproduce the delivering notebook'
            )
    return got


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-dir', type=Path, default=REPO_ROOT / 'data/cosmos2025')
    ap.add_argument(
        '--notebook',
        type=Path,
        default=None,
        help='painting notebook (default <data-dir>/private/IAEstimate.ipynb)',
    )
    ap.add_argument(
        '--force', action='store_true', help='replace an existing regenerated file'
    )
    args = ap.parse_args()

    private_dir = args.data_dir / 'private'
    notebook = args.notebook or private_dir / 'IAEstimate.ipynb'
    out_fits = private_dir / OUT_NAME
    backup = private_dir / BACKUP_NAME
    sidecar_path = private_dir / f'{OUT_NAME}.provenance.json'

    if not notebook.exists():
        raise FileNotFoundError(
            f"{notebook} not found; copy the delivered notebook there "
            f"(cp ~/Downloads/IAEstimate.ipynb {notebook})"
        )
    for name in (
        'COSMOSWeb_mastercatalog_v1_photom_primary.fits',
        'COSMOSWeb_mastercatalog_v1_cigale.fits',
        'COSMOSWeb_mastercatalog_v1_lephare.fits',
    ):
        if not (args.data_dir / name).exists():
            raise FileNotFoundError(
                f"{args.data_dir / name} not found; run "
                f"'python scripts/download_cosmos2025.py {args.data_dir} "
                f"--version v1'"
            )

    # one-time backup of the original 2026-07-28 delivered file (and its
    # sidecar) before anything replaces it; never overwrite the backup
    if out_fits.exists():
        if not backup.exists():
            shutil.copy2(out_fits, backup)
            if sidecar_path.exists():
                shutil.copy2(
                    sidecar_path, private_dir / f'{BACKUP_NAME}.provenance.json'
                )
            print(f'backed up original delivery to {backup}')
        elif not args.force:
            raise RuntimeError(
                f"{out_fits} exists and the original delivery is already "
                f"backed up; pass --force to replace the regenerated file"
            )
        out_fits.unlink()  # the notebook's write uses overwrite=False, kept verbatim

    nb = nbformat.read(notebook, as_version=4)
    sources, applied = extract_sources(nb, _patch_table(args.data_dir, out_fits))

    ns = {'__name__': '__cosmos25_regen__'}
    for i in EXEC_CELL_INDICES:
        print(f'executing notebook cell {i} ...', flush=True)
        exec(compile(sources[i]['source'], f'<IAEstimate cell {i}>', 'exec'), ns)
    if not out_fits.exists():
        raise RuntimeError(f'write cell executed but {out_fits} is missing')
    print(f'wrote {out_fits} ({out_fits.stat().st_size / 1e6:.1f} MB)')

    density_code = compile(
        sources[DENSITY_CELL_INDEX]['source'],
        f'<IAEstimate cell {DENSITY_CELL_INDEX}>',
        'exec',
    )
    try:
        got_mem = run_density_cell(density_code, ns, 'in-memory results')
        ns['emission_line_results'] = ns['Table'].read(out_fits)
        got_file = run_density_cell(density_code, ns, 'regenerated FITS read back')
    except Exception:
        out_fits.unlink()  # never leave an unverified painted file in place
        raise

    import astropy
    import numpy

    sidecar = {
        'file': OUT_NAME,
        'sha256': sha256_file(out_fits),
        'generator': 'scripts/regen_cosmos25_painting.py',
        'notebook': notebook.name,
        'notebook_sha256': sha256_file(notebook),
        'executed_cells': list(EXEC_CELL_INDICES),
        'density_cell': DENSITY_CELL_INDEX,
        'cell_source_sha256': {
            str(i): sources[i]['original_sha256'] for i in sorted(sources)
        },
        'patches': applied,
        'input_sha256': {
            name: sha256_file(args.data_dir / name)
            for name in (
                'COSMOSWeb_mastercatalog_v1_photom_primary.fits',
                'COSMOSWeb_mastercatalog_v1_cigale.fits',
                'COSMOSWeb_mastercatalog_v1_lephare.fits',
            )
        },
        'density_handshake': {
            'targets': DENSITY_TARGETS,
            'in_memory': got_mem,
            'from_file': got_file,
        },
        'env': {
            'python': platform.python_version(),
            'numpy': numpy.__version__,
            'astropy': astropy.__version__,
        },
        'ts_run': datetime.now(timezone.utc).isoformat(),
        'description': (
            'Painted emission-line section (F_Ha/F_OII/F_OIII + lambda_obs, '
            'redshift, sfr_young, log_OH), regenerated from the delivered '
            'painting notebook and row-matched to the COSMOS2025 v1 '
            'sections. Deterministic given notebook + inputs. PRIVATE until '
            'the describing paper is published.'
        ),
    }
    sidecar_path.write_text(json.dumps(sidecar, indent=1))
    print(f'wrote {sidecar_path}')
    print('density handshakes passed (in-memory and from file)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
