"""v06: composition aggregators (mixtures of tile class distributions) and their expert pool (CPU)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from src.fusion.blocks import ID, META, Blocks
from src.fusion.candidates import all_candidates, build_candidate
from src.fusion.mil import CompositionNet, MILFusion
from src.utils.config import load_config
from tests.test_v04 import _cfg, cache  # noqa: F401  (fixture)


def test_comp_is_the_mean_of_tile_distributions():
    torch.manual_seed(0)
    net = CompositionNet(6, 4, 0.0, 7, weighted=False).eval()
    x = torch.randn(3, 10, 6)
    q = torch.softmax(net.head(net.proj(x)), dim=-1)
    assert torch.allclose(net([x]).exp(), q.mean(dim=1), atol=1e-6)
    assert torch.allclose(net([x]).exp().sum(dim=1), torch.ones(3), atol=1e-6)


def test_wcomp_with_constant_attention_equals_comp():
    torch.manual_seed(0)
    a, b = CompositionNet(6, 4, 0.0, 7, weighted=False).eval(), CompositionNet(6, 4, 0.0, 7, weighted=True).eval()
    b.proj.load_state_dict(a.proj.state_dict())
    b.head.load_state_dict(a.head.state_dict())
    torch.nn.init.zeros_(b.w.weight)
    torch.nn.init.zeros_(b.w.bias)
    x = torch.randn(2, 9, 6)
    assert torch.allclose(a([x]), b([x]), atol=1e-6)


def test_composition_refuses_age_and_sex(cache):  # noqa: F811
    X, y, _, out = cache
    blocks = Blocks.from_sizes({"uni2h": 5, ID: 1, META: 2})
    with pytest.raises(ValueError):
        MILFusion(blocks, encoding="ft", cache_dir=str(out), aggregator="comp", epochs=1, n_seeds=1).fit(X, y)


@pytest.mark.parametrize("cand", ["A1:raw", "A2:ple"])
def test_agg_expert_pool_valid_deterministic_and_cached(cache, cand, tmp_path):  # noqa: F811
    X, y, _, out = cache
    blocks = Blocks.from_sizes({"uni2h": 3, "virchow2": 2, ID: 1, META: 2})
    cfg = _cfg(out) | {"families": ["early", "late", "pool", "learned", "mil", "mexp", "agg"]}
    cfg["expert_cache"] = str(tmp_path / "experts")
    train = np.arange(len(y)) % 3 != 0
    P1 = build_candidate(cand, "screen", blocks, cfg, seed=0).fit(X[train], y[train]).predict_proba(X[~train])
    n_cached = len(list((tmp_path / "experts").glob("*.npy")))
    cfg["expert_cache"] = None
    P2 = build_candidate(cand, "screen", blocks, cfg, seed=0).fit(X[train], y[train]).predict_proba(X[~train])
    assert P1.shape == (int((~train).sum()), 7) and np.allclose(P1.sum(axis=1), 1)
    assert np.allclose(P1, P2, atol=1e-6) and n_cached == 2           # one expert per model; recomputed identically


def test_agg_and_attention_experts_have_distinct_cache_entries(cache, tmp_path):  # noqa: F811
    X, y, _, out = cache
    blocks = Blocks.from_sizes({"uni2h": 3, "virchow2": 2, ID: 1, META: 2})
    cfg = _cfg(out) | {"families": ["early", "late", "pool", "learned", "mil", "mexp", "agg"]}
    cfg["expert_cache"] = str(tmp_path / "experts")
    train = np.arange(len(y)) % 3 != 0
    Pa = build_candidate("A1:raw", "screen", blocks, cfg, seed=0).fit(X[train], y[train]).predict_proba(X[~train])
    Px = build_candidate("X1:raw", "screen", blocks, cfg, seed=0).fit(X[train], y[train]).predict_proba(X[~train])
    assert len(list((tmp_path / "experts").glob("*.npy"))) == 4 and not np.allclose(Pa, Px)


def test_v06_config_candidates():
    cfg = load_config("conf/versions/v06.yaml")
    c = all_candidates(cfg)
    assert len(c) == 54 and c[-4:] == ["A1:raw", "A1:ple", "A2:raw", "A2:ple"]
    assert cfg["models"] == ["uni2h", "virchow2", "genbio_pathfm"] and "A1:raw" in cfg["opening"]
