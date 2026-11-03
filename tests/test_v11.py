"""v11: auxiliary grade-group head for the tile-level experts (CPU)."""
from __future__ import annotations

import numpy as np
import pytest

from src.data.cohort import CLASSES
from src.fusion.blocks import ID, META, Blocks
from src.fusion.candidates import all_candidates, build_candidate
from src.fusion.mil import GRADE_GROUPS, MILFusion
from src.utils.config import load_config
from tests.test_v04 import _cfg, cache  # noqa: F401  (fixture)


def test_grade_groups_follow_the_iaslc_pattern_groups():
    g = dict(zip(CLASSES, GRADE_GROUPS))
    assert g["In situ"] == 0 and g["Lepidic"] == 1 and g["Acinar"] == g["Papillary"] == 2
    assert g["Solid"] == g["Micropapillary"] == g["Cribriform"] == 3


def test_group_head_changes_training_not_the_interface(cache):  # noqa: F811
    X, y, _, out = cache
    blocks = Blocks.from_sizes({"uni2h": 5, ID: 1})
    Xi = X[:, :6]
    kw = dict(cache_dir=str(out), encoding=None, epochs=5, n_seeds=1, tiles_per_step=8, d=8)
    for agg in ("attention", "clam_sb"):
        a = MILFusion(blocks, aggregator=agg, **kw).fit(Xi, y).predict_proba(Xi)
        b = MILFusion(blocks, aggregator=agg, aux_groups=True, **kw).fit(Xi, y).predict_proba(Xi)
        assert a.shape == b.shape == (len(y), 7) and np.allclose(b.sum(1), 1) and not np.allclose(a, b)
    with pytest.raises(ValueError):
        MILFusion(Blocks.from_sizes({"uni2h": 5, ID: 1, META: 2}), aux_groups=True, **kw).fit(X, y)


@pytest.mark.parametrize("cand", ["H1:raw", "H2:ple"])
def test_hier_experts_valid_deterministic_cached_apart(cache, cand, tmp_path):  # noqa: F811
    X, y, _, out = cache
    blocks = Blocks.from_sizes({"uni2h": 3, "virchow2": 2, ID: 1, META: 2})
    cfg = _cfg(out) | {"families": ["mexp", "clam", "hier"]}
    cfg["expert_cache"] = str(tmp_path / "experts")
    train = np.arange(len(y)) % 3 != 0
    run = lambda c: build_candidate(c, "screen", blocks, cfg, seed=0).fit(X[train], y[train]).predict_proba(X[~train])
    P = run(cand)
    base = run("X1:raw" if cand.startswith("H1") else "C1:raw")
    assert len(list((tmp_path / "experts").glob("*.npy"))) == 4 and not np.allclose(P, base)
    cfg["expert_cache"] = None
    assert np.allclose(run(cand), P, atol=1e-6) and np.allclose(P.sum(1), 1)


def test_v11_config():
    cfg = load_config("conf/versions/v11.yaml")
    c = all_candidates(cfg)
    assert len(c) == 74 and c[-4:] == ["H1:raw", "H1:ple", "H2:raw", "H2:ple"]
    assert {"H1:raw", "H2:raw"} <= set(cfg["opening"])
