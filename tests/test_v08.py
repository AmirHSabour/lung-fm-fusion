"""v08: spatial-context tokens (neighbourhood means on the tile grid) and the ctx family (CPU)."""
from __future__ import annotations

import h5py
import numpy as np
import pandas as pd
import pytest

from src.embeddings.tile_cache import box_means, build, build_context, index, load
from src.fusion.blocks import ID, META, Blocks
from src.fusion.candidates import all_candidates, build_candidate
from src.utils.config import load_config
from tests.test_v04 import _cfg, cache  # noqa: F401  (fixture)


def _brute(coords, X, stride, r, rows):
    g = coords // stride
    out = []
    for t in rows:
        near = np.all(np.abs(g - g[t]) <= r, axis=1)
        out.append(X[near].mean(axis=0))
    return np.array(out)


def test_box_means_match_brute_force_with_holes_edges_and_offsets():
    rng = np.random.default_rng(0)
    cells = [(i, j) for i in range(9) for j in range(7) if rng.random() < 0.6]   # tissue with holes
    coords = (np.array(cells) + [3, 5]) * 224                                    # offset grid
    X = rng.normal(size=(len(cells), 5))
    rows = np.arange(len(cells))
    for r in (0, 1, 3):
        assert np.allclose(box_means(coords, X, 224, r, rows), _brute(coords - coords.min(0), X, 224, r, rows),
                           atol=1e-5)
    assert np.allclose(box_means(coords, X, 224, 0, rows), X, atol=1e-6)        # radius 0 = the tile itself


def test_context_cache_same_rows_and_tiles_as_tile_cache(tmp_path):
    rng = np.random.default_rng(1)
    emb = tmp_path / "emb"
    (emb / "uni2h").mkdir(parents=True)
    wsi = ["WSI-3", "WSI-7"]
    for w in wsi:
        coords = (np.array([(i, j) for i in range(6) for j in range(5)]) * 224).astype(np.int32)
        with h5py.File(emb / "uni2h" / f"{w}.h5", "w") as f:
            f["coords"] = coords
            f["cls"] = rng.normal(size=(len(coords), 4)).astype(np.float32)
            f.attrs["stride_px"] = 224
    build(wsi, ["uni2h"], emb, tmp_path / "tiles", 8)
    build_context(wsi, ["uni2h"], emb, tmp_path / "ctx", 8, (1, 3), tmp_path / "tiles", workers=1)
    a, c = load(tmp_path / "tiles", "uni2h"), load(tmp_path / "ctx", "uni2h")
    assert c.shape == (2, 8, 12) and np.allclose(c[:, :, :4], a)                 # first block = the tile itself
    assert index(tmp_path / "tiles") == index(tmp_path / "ctx")
    with pytest.raises(RuntimeError):
        build_context(wsi[::-1], ["uni2h"], emb, tmp_path / "ctx2", 8, (1,), tmp_path / "tiles", workers=1)


@pytest.mark.parametrize("cand", ["K1:raw", "K2:ple"])
def test_ctx_experts_valid_and_cached_separately(cache, cand, tmp_path):  # noqa: F811
    X, y, _, out = cache
    blocks = Blocks.from_sizes({"uni2h": 3, "virchow2": 2, ID: 1, META: 2})
    cfg = _cfg(out) | {"families": ["mexp", "ctx"], "ctx_cache": str(out)}  # same rows; content irrelevant here
    cfg["expert_cache"] = str(tmp_path / "experts")
    train = np.arange(len(y)) % 3 != 0
    P = build_candidate(cand, "screen", blocks, cfg, seed=0).fit(X[train], y[train]).predict_proba(X[~train])
    assert np.allclose(P.sum(axis=1), 1)
    cfg["ctx_cache"] = str(tmp_path / "elsewhere")                            # a different cache: a different key
    with pytest.raises(FileNotFoundError):
        build_candidate(cand, "screen", blocks, cfg, seed=0).fit(X[train], y[train]).predict_proba(X[~train])


def test_v08_config():
    cfg = load_config("conf/versions/v08.yaml")
    c = all_candidates(cfg)
    assert len(c) == 62 and c[-4:] == ["K1:raw", "K1:ple", "K2:raw", "K2:ple"]
    assert {"K1:raw", "K2:raw"} <= set(cfg["opening"]) and cfg["ctx_cache"] != cfg["tile_cache"]
