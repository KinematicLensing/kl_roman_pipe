#!/usr/bin/env python3
"""Split TNG50 gold sample NPZ files into per-galaxy NPZ files.

The source gold files are expected to live in:
    /ocean/projects/phy250048p/shared/tng/data/tng50/*_gold.npz

This script performs a one-time conversion from the monolithic object-array format
(typically stored as arr_0) into many small files, one per galaxy, to enable
memory-safe random access in low-RAM environments.
"""

from __future__ import annotations

import argparse
import csv
import gc
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np


DEFAULT_INPUT_DIR = Path("/ocean/projects/phy250048p/shared/tng/data/tng50")
DEFAULT_OUTPUT_DIR = Path("/ocean/projects/phy250048p/shared/tng/data/tng50/gold_split")

DATA_FILES = {
    "gas": "gas_data_analysis_gold.npz",
    "stellar": "stellar_data_analysis_gold.npz",
    "subhalo": "subhalo_data_analysis_gold.npz",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split TNG50 gold sample files into individual galaxy NPZ files. "
            "Each output file stores one galaxy dictionary as NPZ keys."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Directory with *_gold.npz files (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for split files (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--no-compress",
        action="store_true",
        help="Use np.savez instead of np.savez_compressed for faster writes.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing per-galaxy files if they already exist.",
    )
    return parser.parse_args()


def load_object_array(npz_path: Path) -> np.ndarray:
    with np.load(npz_path, allow_pickle=True) as data:
        if "arr_0" not in data:
            raise KeyError(f"Expected key 'arr_0' in {npz_path}")
        arr = data["arr_0"]

    if arr.ndim != 1:
        raise ValueError(f"Expected 1D array in {npz_path}, got shape {arr.shape}")
    return arr


def as_numpy_value(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    return np.array(value)


def format_filename(index: int, subhalo_id: Optional[int]) -> str:
    if subhalo_id is None:
        return f"gal_{index:06d}.npz"
    return f"gal_{index:06d}_subhalo_{subhalo_id}.npz"


def write_manifest(rows: Iterable[Dict[str, Any]], manifest_path: Path) -> None:
    fieldnames = ["index", "subhalo_id", "filename"]
    with manifest_path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def split_modality(
    modality: str,
    source_path: Path,
    output_dir: Path,
    subhalo_ids: Optional[np.ndarray],
    compress: bool,
    overwrite: bool,
) -> int:
    print(f"\n[{modality}] Loading {source_path}")
    arr = load_object_array(source_path)
    n_gal = len(arr)
    print(f"[{modality}] Found {n_gal} galaxies")

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []

    for i, galaxy in enumerate(arr):
        if not isinstance(galaxy, dict):
            raise TypeError(
                f"{modality}: expected dict per galaxy at index {i}, got {type(galaxy)}"
            )

        subhalo_id: Optional[int] = None
        if subhalo_ids is not None and i < len(subhalo_ids):
            subhalo_id = int(subhalo_ids[i])

        filename = format_filename(i, subhalo_id)
        out_path = output_dir / filename

        if out_path.exists() and not overwrite:
            manifest_rows.append(
                {"index": i, "subhalo_id": subhalo_id, "filename": filename}
            )
            continue

        payload = {k: as_numpy_value(v) for k, v in galaxy.items()}

        if compress:
            np.savez_compressed(out_path, **payload)
        else:
            np.savez(out_path, **payload)

        manifest_rows.append(
            {"index": i, "subhalo_id": subhalo_id, "filename": filename}
        )

        if (i + 1) % 100 == 0 or (i + 1) == n_gal:
            print(f"[{modality}] Wrote {i + 1}/{n_gal}")

    manifest_path = output_dir / "manifest.csv"
    write_manifest(manifest_rows, manifest_path)
    print(f"[{modality}] Wrote manifest: {manifest_path}")

    del arr
    gc.collect()
    return n_gal


def main() -> None:
    args = parse_args()

    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir
    compress = not args.no_compress

    missing = [name for name in DATA_FILES.values() if not (input_dir / name).exists()]
    if missing:
        missing_str = "\n".join(f"  - {input_dir / m}" for m in missing)
        raise FileNotFoundError(f"Missing required gold files:\n{missing_str}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load subhalo IDs once so output files from all modalities can share naming.
    subhalo_arr = load_object_array(input_dir / DATA_FILES["subhalo"])
    subhalo_ids = np.array([int(g["SubhaloID"]) for g in subhalo_arr], dtype=np.int64)
    del subhalo_arr
    gc.collect()

    counts = {}
    for modality, filename in DATA_FILES.items():
        source = input_dir / filename
        modality_out = output_dir / modality
        counts[modality] = split_modality(
            modality=modality,
            source_path=source,
            output_dir=modality_out,
            subhalo_ids=subhalo_ids,
            compress=compress,
            overwrite=args.overwrite,
        )

    index_file = output_dir / "subhalo_ids.npy"
    np.save(index_file, subhalo_ids)

    print("\nDone.")
    print(f"Output root: {output_dir}")
    print(f"Subhalo ID index: {index_file}")
    print("Galaxy counts:")
    for modality, count in counts.items():
        print(f"  - {modality}: {count}")


if __name__ == "__main__":
    main()
