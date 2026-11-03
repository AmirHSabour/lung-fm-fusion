"""A fixed random subset of tiles per slide, with every model's embedding of the same tiles (v04).

Label-free: the subset of each slide is drawn with a generator seeded by the slide number, so it
is the same for every run and every model. Used by the multiple-instance recipes (src/fusion/mil.py).
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd

FIELDS = {"uni2h": ["cls"], "virchow2": ["cls", "patch_mean"], "genbio_pathfm": ["cls"]}


def tile_subset(wsi_id: str, n_tiles: int, n: int) -> np.ndarray:
    """Sorted indices of `n` tiles of a slide (all tiles if it has fewer)."""
    rng = np.random.default_rng(int(wsi_id.split("-")[1]))
    if n_tiles <= n:
        return np.arange(n_tiles)
    return np.sort(rng.choice(n_tiles, n, replace=False))


def build(wsi_ids: list[str], models: list[str], emb_dir: Path, out_dir: Path, n: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, arrays = [], {m: [] for m in models}
    for w in wsi_ids:
        coords = None
        for m in models:
            with h5py.File(emb_dir / m / f"{w}.h5", "r") as f:
                c = f["coords"][()]
                if coords is None:
                    coords = c
                    idx = tile_subset(w, len(c), n)
                elif not np.array_equal(c, coords):
                    raise RuntimeError(f"{w}: {m} has a different tile list")
                arrays[m].append(np.concatenate([f[k][idx] for k in FIELDS[m]], axis=1).astype(np.float32))
        rows.append({"WSI_ID": w, "n_tiles": len(coords), "n_cached": len(idx)})
    if any(r["n_cached"] != n for r in rows):
        raise RuntimeError("every slide must have at least n tiles")
    for m in models:
        np.save(out_dir / f"{m}.npy", np.stack(arrays[m]))
    pd.DataFrame(rows).to_csv(out_dir / "index.csv", index=False)


def load(out_dir: Path, model: str) -> np.ndarray:
    return np.load(Path(out_dir) / f"{model}.npy", mmap_mode="r")


def index(out_dir: Path) -> dict[str, int]:
    return {w: i for i, w in enumerate(pd.read_csv(Path(out_dir) / "index.csv")["WSI_ID"])}


# ---- v08: spatial context (decisions/v08.md) -------------------------------------------------

def box_means(coords: np.ndarray, X: np.ndarray, stride: int, radius: int, rows: np.ndarray) -> np.ndarray:
    """For tiles `rows`, the mean of X over the tissue tiles within `radius` grid cells (Chebyshev),
    the tile itself included. Exact, via an integral image of the dense grid (float64)."""
    g = (coords - coords.min(axis=0)) // stride                       # grid cell of every tile
    H, W = g.max(axis=0) + 1
    r = radius
    S = np.zeros((H + 2 * r + 1, W + 2 * r + 1, X.shape[1]))         # padded, +1 for the integral image
    C = np.zeros((H + 2 * r + 1, W + 2 * r + 1))
    S[g[:, 0] + r + 1, g[:, 1] + r + 1] = X
    C[g[:, 0] + r + 1, g[:, 1] + r + 1] = 1
    S = S.cumsum(0).cumsum(1)
    C = C.cumsum(0).cumsum(1)
    i, j = g[rows, 0] + r + 1, g[rows, 1] + r + 1                    # padded positions of the queried tiles
    box = lambda A: A[i + r, j + r] - A[i - r - 1, j + r] - A[i + r, j - r - 1] + A[i - r - 1, j - r - 1]
    return (box(S) / box(C)[:, None]).astype(np.float32)


def _context_slide(args) -> tuple[int, dict[str, np.ndarray]]:
    k, w, models, emb_dir, n, radii = args
    out = {}
    for m in models:
        with h5py.File(Path(emb_dir) / m / f"{w}.h5", "r") as f:
            coords = f["coords"][()]
            stride = int(f.attrs["stride_px"])
            X = np.concatenate([f[key][()] for key in FIELDS[m]], axis=1).astype(np.float64)
        idx = tile_subset(w, len(coords), n)
        out[m] = np.concatenate([X[idx].astype(np.float32)] + [box_means(coords, X, stride, r, idx) for r in radii],
                                axis=1)
    return k, out


def build_context(wsi_ids: list[str], models: list[str], emb_dir: Path, out_dir: Path, n: int,
                  radii: tuple[int, ...] = (1, 3), tile_cache: Path | None = None, workers: int = 8) -> None:
    """Tokens [x_t ; mean over radius r_1 ; mean over radius r_2 …] for the tile cache's tiles."""
    from multiprocessing import get_context
    from numpy.lib.format import open_memmap
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if tile_cache is not None and list(pd.read_csv(Path(tile_cache) / "index.csv")["WSI_ID"]) != list(wsi_ids):
        raise RuntimeError("the context cache must have the tile cache's row order")
    arrays = {}
    with get_context("spawn").Pool(workers) as pool:
        jobs = [(k, w, models, str(emb_dir), n, tuple(radii)) for k, w in enumerate(wsi_ids)]
        for k, out in pool.imap_unordered(_context_slide, jobs):
            for m, A in out.items():
                if m not in arrays:
                    arrays[m] = open_memmap(out_dir / f"{m}.npy", mode="w+", dtype=np.float32,
                                            shape=(len(wsi_ids), n, A.shape[1]))
                if A.shape[0] != n:
                    raise RuntimeError(f"{wsi_ids[k]}: fewer than {n} tiles")
                arrays[m][k] = A
    for A in arrays.values():
        A.flush()
    pd.DataFrame({"WSI_ID": wsi_ids}).to_csv(out_dir / "index.csv", index=False)
