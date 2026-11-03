"""v09: ensembles of expert types per model (product of experts) (CPU)."""
from __future__ import annotations

import numpy as np

from src.fusion.blocks import ID, META, Blocks
from src.fusion.candidates import all_candidates, build_candidate
from src.fusion.pool import _logp
from src.utils.config import load_config
from tests.test_v04 import _cfg, cache  # noqa: F401  (fixture)

FAMS = ["mexp", "clam", "ctx", "ens"]


def _setup(cache, tmp_path):  # noqa: F811
    X, y, _, out = cache
    blocks = Blocks.from_sizes({"uni2h": 3, "virchow2": 2, ID: 1, META: 2})
    cfg = _cfg(out) | {"families": FAMS, "ctx_cache": str(out)}
    cfg["expert_cache"] = str(tmp_path / "experts")
    return X, y, blocks, cfg, np.arange(len(y)) % 3 != 0


def test_ensemble_is_the_mean_of_member_log_probabilities(cache, tmp_path):  # noqa: F811
    X, y, blocks, cfg, train = _setup(cache, tmp_path)
    fit = lambda c, b=blocks: build_candidate(c, "screen", b, cfg, seed=0).fit(X[train], y[train])
    img = Blocks.from_sizes({"uni2h": 3, "virchow2": 2, ID: 1})
    cols = blocks.columns(["uni2h", "virchow2", ID])
    Xi = X[:, cols]
    fit_i = lambda c: build_candidate(c, "screen", img, cfg, seed=0).fit(Xi[train], y[train]).predict_proba(Xi[~train])
    Pa, Pc, Pz = fit_i("X1:raw"), fit_i("C1:raw"), fit_i("Z1:raw")
    n_cached = len(list((tmp_path / "experts").glob("*.npy")))
    expected = np.exp((_logp(Pa) + _logp(Pc)) / 2)
    assert np.allclose(Pz, expected / expected.sum(axis=1, keepdims=True), atol=1e-6)
    assert n_cached == 4                                     # Z1 reused the X1 and C1 experts (2 models x 2 types)
    P = fit("Z2:raw").predict_proba(X[~train])               # with age and sex, and context members
    assert np.allclose(P.sum(axis=1), 1)


def test_default_expert_pool_unchanged(cache, tmp_path):  # noqa: F811
    X, y, blocks, cfg, train = _setup(cache, tmp_path)
    P1 = build_candidate("X1:raw", "screen", blocks, cfg, seed=0).fit(X[train], y[train]).predict_proba(X[~train])
    cfg["expert_cache"] = None
    P2 = build_candidate("X1:raw", "screen", blocks, cfg, seed=0).fit(X[train], y[train]).predict_proba(X[~train])
    assert np.allclose(P1, P2, atol=1e-6)


def test_v09_config():
    cfg = load_config("conf/versions/v09.yaml")
    c = all_candidates(cfg)
    assert len(c) == 66 and c[-4:] == ["Z1:raw", "Z1:ple", "Z2:raw", "Z2:ple"]
    assert {"Z1:raw", "Z2:raw"} <= set(cfg["opening"]) and "ens" in cfg["untunable_families"]
