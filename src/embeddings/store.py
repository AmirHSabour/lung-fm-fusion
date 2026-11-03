"""On-disk format for tile embeddings: one HDF5 file per slide per model.

Datasets: ``coords`` (N, 2) int32 tile corners (x, y) on the 0.5 um/px slide, and one
float32 (N, D) array per embedding (``cls``; ``patch_mean`` for Virchow2). File attributes
record how the embeddings were made (model revision, precision, tiling, tile-list checksum,
code version). Files are written under a temporary name and renamed when complete.
"""
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import h5py
import numpy as np


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_version() -> str:
    """Git commit of the repository, with '-dirty' if there are uncommitted changes."""
    try:
        rev = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, check=True).stdout.strip()
        return rev + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def write_embeddings(path: Path, coords: np.ndarray, feats: dict[str, np.ndarray], attrs: dict) -> None:
    """Write one slide's embeddings atomically."""
    tmp = path.with_name(path.name + ".partial")
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(tmp, "w") as f:
        f.create_dataset("coords", data=coords.astype(np.int32))
        for name, arr in feats.items():
            if len(arr) != len(coords):
                raise ValueError(f"{name}: {len(arr)} rows for {len(coords)} tiles")
            f.create_dataset(name, data=arr.astype(np.float32))
        f.attrs.update(attrs)
    tmp.replace(path)


def is_complete(path: Path, n_tiles: int, tiles_sha256: str) -> bool:
    """True if path holds embeddings for exactly this tile list."""
    if not path.is_file():
        return False
    try:
        with h5py.File(path, "r") as f:
            return f.attrs.get("tiles_sha256") == tiles_sha256 and len(f["coords"]) == n_tiles
    except OSError:
        return False


def read_embeddings(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray], dict]:
    """(coords, {name: (N, D) array}, attrs) of one slide."""
    with h5py.File(path, "r") as f:
        feats = {k: f[k][()] for k in f if k != "coords"}
        return f["coords"][()], feats, dict(f.attrs)
