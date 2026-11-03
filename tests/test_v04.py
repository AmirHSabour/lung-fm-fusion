"""v04: tile cache, the ID block, and multiple-instance recipes (synthetic cache, CPU)."""
from __future__ import annotations

import copy

import h5py
import numpy as np
import pytest

from src.embeddings.tile_cache import build, index, tile_subset
from src.fusion.blocks import ID, META, Blocks
from src.fusion.candidates import build_candidate
from src.utils.config import load_config

K, PER, T = 7, 6, 16


@pytest.fixture()
def cache(tmp_path):
    """A synthetic tile cache of 42 slides x 16 tiles for two models with class-dependent tiles."""
    rng = np.random.default_rng(0)
    y = np.repeat(np.arange(K), PER)
    wsi = [f"WSI-{i + 1}" for i in range(len(y))]
    emb = tmp_path / "emb"
    for m, D in (("uni2h", 12), ("virchow2", 8)):
        (emb / m).mkdir(parents=True)
    for w, c in zip(wsi, y):
        n_t = 40
        coords = np.arange(2 * n_t).reshape(n_t, 2).astype(np.int32)
        with h5py.File(emb / "uni2h" / f"{w}.h5", "w") as f:
            f["coords"] = coords
            f["cls"] = rng.normal(size=(n_t, 12)).astype(np.float32) + c * 0.3
        with h5py.File(emb / "virchow2" / f"{w}.h5", "w") as f:
            f["coords"] = coords
            f["cls"] = rng.normal(size=(n_t, 4)).astype(np.float32)
            f["patch_mean"] = rng.normal(size=(n_t, 4)).astype(np.float32) + c * 0.2
    out = tmp_path / "tiles"
    build(wsi, ["uni2h", "virchow2"], emb, out, T)
    pos = index(out)
    age = rng.integers(30, 80, len(y)).astype(float)
    sex = rng.integers(0, 2, len(y)).astype(float)
    X = np.hstack([rng.normal(size=(len(y), 5)) + 0.4 * y[:, None], np.array([[pos[w]] for w in wsi], float),
                   age[:, None], sex[:, None]])
    blocks = Blocks.from_sizes({"uni2h": 5, ID: 1, META: 2})
    return X, y, blocks, out


def _cfg(cache_dir) -> dict:
    cfg = copy.deepcopy(load_config("conf/versions/v04.yaml"))
    cfg["tile_cache"] = str(cache_dir)
    cfg["mil"] |= {"epochs": 5, "n_seeds": 2, "tiles_per_step": 8}
    cfg["phase1"]["size"] = 8
    cfg["phase2"]["inner_folds"] = 3                      # 4 training slides per class
    return cfg


def test_tile_subset_is_deterministic_and_seeded_by_slide():
    a, b = tile_subset("WSI-140", 10000, 1024), tile_subset("WSI-140", 10000, 1024)
    assert np.array_equal(a, b) and len(np.unique(a)) == 1024 and np.all(np.diff(a) > 0)
    assert not np.array_equal(a, tile_subset("WSI-141", 10000, 1024))


def test_cache_holds_both_models_for_the_same_tiles(cache):
    _, _, _, out = cache
    u, v = np.load(out / "uni2h.npy"), np.load(out / "virchow2.npy")
    assert u.shape == (K * PER, T, 12) and v.shape == (K * PER, T, 8)


@pytest.mark.parametrize("cand", ["M1:ft", "M2:ple", "M3:ft"])
def test_mil_candidates_valid_and_deterministic(cache, cand):
    X, y, _, out = cache
    # MIL reads the image models from the block names: use the two cached models as image blocks.
    blocks = Blocks.from_sizes({"uni2h": 3, "virchow2": 2, ID: 1, META: 2})
    cfg, train = _cfg(out), np.arange(len(y)) % 3 != 0
    P1 = build_candidate(cand, "screen", blocks, cfg, seed=0).fit(X[train], y[train]).predict_proba(X[~train])
    P2 = build_candidate(cand, "screen", blocks, cfg, seed=0).fit(X[train], y[train]).predict_proba(X[~train])
    assert P1.shape == ((~train).sum(), K) and np.allclose(P1.sum(axis=1), 1) and np.allclose(P1, P2)


@pytest.mark.parametrize("cand", ["E1:raw", "E3:ple", "L2:raw", "L4:ple", "P3:raw", "N1:ft"])
def test_other_recipes_ignore_the_id_block(cache, cand):
    X, y, blocks, out = cache
    cfg, train = _cfg(out), np.arange(len(y)) % 3 != 0
    cfg["phase1"]["size"] = 3
    no_id = Blocks.from_sizes({"uni2h": 5, META: 2})
    X_no = np.delete(X, 5, axis=1)
    a = build_candidate(cand, "screen", blocks, cfg, seed=0).fit(X[train], y[train]).predict_proba(X[~train])
    b = build_candidate(cand, "screen", no_id, cfg, seed=0).fit(X_no[train], y[train]).predict_proba(X_no[~train])
    assert np.allclose(a, b)


# ---- v05: pooled MIL experts ------------------------------------------------------------

@pytest.mark.parametrize("cand", ["X1:raw", "X3:ple", "X4:raw"])
def test_mil_expert_pool_valid_and_cached(cache, cand, tmp_path):
    X, y, _, out = cache
    blocks = Blocks.from_sizes({"uni2h": 3, "virchow2": 2, ID: 1, META: 2})
    cfg = _cfg(out) | {"families": ["early", "late", "pool", "learned", "mil", "mexp"]}
    cfg["expert_cache"] = str(tmp_path / "experts")
    train = np.arange(len(y)) % 3 != 0
    P1 = build_candidate(cand, "screen", blocks, cfg, seed=0).fit(X[train], y[train]).predict_proba(X[~train])
    n_cached = len(list((tmp_path / "experts").glob("*.npy")))
    P2 = build_candidate(cand, "screen", blocks, cfg, seed=0).fit(X[train], y[train]).predict_proba(X[~train])
    assert np.allclose(P1.sum(axis=1), 1) and np.allclose(P1, P2) and n_cached > 0
    assert len(list((tmp_path / "experts").glob("*.npy"))) == n_cached          # second fit read the cache


def test_policy_skips_untunable_families_when_tuning():
    from src.agent.policies import get_policy
    from src.agent.policy import Decide
    cfg = copy.deepcopy(load_config("conf/versions/v05.yaml"))
    cands = ["X1:raw", "M1:ft", "L2:raw"]
    recs = [{"type": "header"}, {"type": "decision", "id": "s0-D001", "rule": "open", "based_on": [], "statistics": {},
                                 "action": {"candidates": cands}}]
    for i, (c, v) in enumerate(zip(cands, [0.9, 0.85, 0.8])):
        recs.append({"type": "experiment", "id": f"s0-T{i + 1:03d}", "candidate": c, "mode": "screen", "status": "done",
                     "scores": {"macro_auroc": {"oof": v}, "balanced_accuracy": {"oof": 0.5}}})
    cfg["search"]["budget"] = 6                      # leaves no exploration budget: next step is tune_start
    nxt = get_policy("v02")(recs, cfg | {"opening": cands}, lambda a, b: {"share_won": 0.5, "gain": 0.0, "ci": [0, 0]})
    while isinstance(nxt, Decide) and nxt.rule in ("leader", "drop"):
        recs.append({"type": "decision", "id": f"s0-D{len(recs):03d}", "rule": nxt.rule, "based_on": nxt.based_on,
                     "statistics": nxt.statistics, "action": nxt.action})
        nxt = get_policy("v02")(recs, cfg | {"opening": cands}, lambda a, b: {"share_won": 0.5, "gain": 0.0, "ci": [0, 0]})
    assert isinstance(nxt, Decide) and nxt.rule == "tune_start" and "X1:raw" not in nxt.action["tune"]
