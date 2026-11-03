"""v07: CLAM experts (single- and multi-branch attention, instance clustering loss) (CPU)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from src.fusion.blocks import ID, META, Blocks
from src.fusion.candidates import all_candidates, build_candidate
from src.fusion.mil import ClamNet
from src.utils.config import load_config
from tests.test_v04 import _cfg, cache  # noqa: F401  (fixture)


@pytest.mark.parametrize("mb", [False, True])
def test_clam_shapes_and_attention(mb):
    torch.manual_seed(0)
    net = ClamNet(6, 4, 0.0, 7, multi_branch=mb).eval()
    x = torch.randn(3, 20, 6)
    h, a = net.attend([x])
    assert h.shape == (3, 20, 4) and a.shape == (3, 20, 7 if mb else 1)
    assert torch.allclose(a.sum(dim=1), torch.ones_like(a.sum(dim=1)), atol=1e-6)
    assert net([x]).shape == (3, 7)
    loss = net.instance_loss(h, a, torch.tensor([0, 3, 6]))
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_instance_loss_labels_the_right_tiles():
    """One slide of class 1, SB: class 1's classifier sees the top tiles as positive and the bottom
    tiles as negative; every other classifier sees the top tiles as negative. A classifier that
    outputs exactly those labels has a lower loss than its sign-flipped version."""
    torch.manual_seed(0)
    net = ClamNet(2, 2, 0.0, 3, multi_branch=False, k_sample=2).eval()
    T = 8
    h = torch.zeros(1, T, 2)
    h[0, :, 0] = torch.linspace(-1, 1, T)                      # feature 0 increases with the tile index
    a = torch.softmax(torch.linspace(-3, 3, T), dim=0).view(1, T, 1)   # attention increases with the index
    y = torch.tensor([1])
    with torch.no_grad():
        net.inst.zero_(); net.inst_b.zero_()
        net.inst[1, 0] = torch.tensor([-5.0, 5.0])             # class 1: high feature -> positive
        for k in (0, 2):
            net.inst[k, 0] = torch.tensor([5.0, -5.0])         # others: high feature -> negative
        good = net.instance_loss(h, a, y)
        net.inst.mul_(-1)
        bad = net.instance_loss(h, a, y)
    assert good < 0.05 < bad


@pytest.mark.parametrize("cand", ["C1:raw", "C2:ple"])
def test_clam_expert_pool_valid_deterministic_and_cached(cache, cand, tmp_path):  # noqa: F811
    X, y, _, out = cache
    blocks = Blocks.from_sizes({"uni2h": 3, "virchow2": 2, ID: 1, META: 2})
    cfg = _cfg(out) | {"families": ["early", "late", "pool", "learned", "mil", "mexp", "agg", "clam"]}
    cfg["expert_cache"] = str(tmp_path / "experts")
    train = np.arange(len(y)) % 3 != 0
    P1 = build_candidate(cand, "screen", blocks, cfg, seed=0).fit(X[train], y[train]).predict_proba(X[~train])
    n_cached = len(list((tmp_path / "experts").glob("*.npy")))
    cfg["expert_cache"] = None
    P2 = build_candidate(cand, "screen", blocks, cfg, seed=0).fit(X[train], y[train]).predict_proba(X[~train])
    assert P1.shape == (int((~train).sum()), 7) and np.allclose(P1.sum(axis=1), 1)
    assert np.allclose(P1, P2, atol=1e-6) and n_cached == 2


def test_v07_config():
    cfg = load_config("conf/versions/v07.yaml")
    c = all_candidates(cfg)
    assert len(c) == 58 and c[-4:] == ["C1:raw", "C1:ple", "C2:raw", "C2:ple"]
    assert {"C1:raw", "C2:raw"} <= set(cfg["opening"]) and "mil" in cfg["untunable_families"]
