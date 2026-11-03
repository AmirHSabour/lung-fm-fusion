"""Tile-to-slide aggregation (docs/methods.md D17).

Mean pooling: the slide vector is the average of the model card's tile embeddings over
all kept tiles. For Virchow2 the tile embedding is [class token, mean patch token]
(2,560-d); for UNI2-h and GenBio-PathFM it is the class token.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np


def mean_pool(path: Path, fields: list[str]) -> tuple[np.ndarray, int]:
    """(slide vector, number of tiles) for one slide's embedding file."""
    with h5py.File(path, "r") as f:
        parts = [f[k][()].astype(np.float64).mean(axis=0) for k in fields]
        n = len(f["coords"])
    return np.concatenate(parts).astype(np.float32), n
