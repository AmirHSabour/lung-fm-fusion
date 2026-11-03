"""v10: cross-attention pooling experts; age and sex as attention queries (CPU)."""
from __future__ import annotations

import numpy as np
import torch

from src.fusion.blocks import ID, META, Blocks
from src.fusion.candidates import all_candidates, build_candidate
from src.fusion.mil import CrossAttnNet
from src.utils.config import load_config
from tests.test_v04 import _cfg, cache  # noqa: F401  (fixture)

FAMS = ["mexp", "xq", "xqm"]


def _meta(n):
    return {"age_z": torch.randn(n), "sex": torch.randint(0, 2, (n,)), "sex_z": torch.randn(n),
            "age_ple": torch.rand(n, 8)}


def test_queries_and_shapes():
    torch.manual_seed(0)
    plain, tok = CrossAttnNet(6, 8, 0.0, 7).eval(), CrossAttnNet(6, 8, 0.0, 7, encoding="ft").eval()
    x = torch.randn(3, 20, 6)
    assert plain.queries(3, None).shape == (3, 1, 8) and tok.queries(3, _meta(3)).shape == (3, 3, 8)
    assert plain([x]).shape == (3, 7) and tok([x], _meta(3)).shape == (3, 7)


def test_age_and_sex_change_the_attention_and_the_output():
    torch.manual_seed(0)
    net = CrossAttnNet(6, 8, 0.0, 7, encoding="ft").eval()
    x = torch.randn(2, 20, 6)
    m1, m2 = _meta(2), _meta(2)
    assert not torch.allclose(net([x], m1), net([x], m2))


def _setup(cache, tmp_path):  # noqa: F811
    X, y, _, out = cache
    blocks = Blocks.from_sizes({"uni2h": 3, "virchow2": 2, ID: 1, META: 2})
    cfg = _cfg(out) | {"families": FAMS}
    cfg["encodings"] = cfg["encodings"] | {"xq": ["raw", "ple"], "xqm": ["ft", "ple"]}
    cfg["expert_cache"] = str(tmp_path / "experts")
    return X, y, blocks, cfg, np.arange(len(y)) % 3 != 0


def test_q1_and_q2_valid_deterministic_cached(cache, tmp_path):  # noqa: F811
    X, y, blocks, cfg, train = _setup(cache, tmp_path)
    run = lambda c: build_candidate(c, "screen", blocks, cfg, seed=0).fit(X[train], y[train]).predict_proba(X[~train])
    P1, P2 = run("Q1:raw"), run("Q2:ft")
    n = len(list((tmp_path / "experts").glob("*.npy")))
    assert np.allclose(P1.sum(1), 1) and np.allclose(P2.sum(1), 1) and n == 4      # Q2 experts keyed apart from Q1's
    cfg["expert_cache"] = None
    assert np.allclose(run("Q2:ft"), P2, atol=1e-6)


def test_q2_without_age_and_sex_equals_q1_experts_without_prior(cache, tmp_path):  # noqa: F811
    X, y, blocks, cfg, train = _setup(cache, tmp_path)
    img = Blocks.from_sizes({"uni2h": 3, "virchow2": 2, ID: 1})
    Xi = X[:, blocks.columns(["uni2h", "virchow2", ID])]
    run = lambda c: build_candidate(c, "screen", img, cfg, seed=0).fit(Xi[train], y[train]).predict_proba(Xi[~train])
    assert np.allclose(run("Q1:raw"), run("Q2:ft"), atol=1e-6)


def test_v10_config():
    cfg = load_config("conf/versions/v10.yaml")
    c = all_candidates(cfg)
    assert len(c) == 70 and c[-4:] == ["Q1:raw", "Q1:ple", "Q2:ft", "Q2:ple"]
    assert {"Q1:raw", "Q2:ft"} <= set(cfg["opening"])
