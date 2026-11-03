"""v02: age-and-sex encodings, candidates, the token network, and the v02 policy (fake evaluator)."""
from __future__ import annotations

import copy

import numpy as np
import pytest
import torch
from sklearn.model_selection import GridSearchCV

from src.agent.policies import get_policy
from src.agent.policy import Decide, Done
from src.agent.store import TrialStore
from src.fusion.blocks import META, Blocks
from src.fusion.candidates import all_candidates, build_candidate, family
from src.fusion.learned import FusionNetTok, LearnedFusionTok
from src.fusion.metadata import (PLE_BINS, MetaColumns, MetaTokenizer, PiecewiseLinear, encoded_blocks)
from src.fusion.registry import build
from src.utils.config import load_config

CFG = load_config("conf/versions/v02.yaml")
K = 7


def _cfg() -> dict:
    cfg = copy.deepcopy(CFG)
    cfg["network"] |= {"epochs": 5, "n_seeds": 2}
    cfg["phase2"]["inner_folds"] = 3
    cfg["phase1"]["size"] = 8                            # synthetic blocks are narrow
    cfg["phase2"]["size_grid"] = [4, 8]
    return cfg


def _data(n_per_class: int = 12, seed: int = 0):
    rng = np.random.default_rng(seed)
    y = np.repeat(np.arange(K), n_per_class)
    A = rng.normal(size=(len(y), 20)) + 0.5 * y[:, None]
    B = rng.normal(size=(len(y), 30)) + 0.3 * y[:, None]
    age = rng.integers(30, 80, len(y)).astype(float)
    sex = rng.integers(0, 2, len(y)).astype(float)
    X = np.hstack([A, B, age[:, None], sex[:, None]])
    return X, y, Blocks.from_sizes({"A": 20, "B": 30, META: 2})


def test_piecewise_linear_uses_training_quantiles_and_is_monotone():
    a = np.arange(20, 80, dtype=float)
    ple = PiecewiseLinear().fit(a[:40])
    assert np.allclose(ple.edges_, np.quantile(a[:40], np.linspace(0, 1, PLE_BINS + 1)))
    Z = ple.transform(np.array([10.0, 40.0, 100.0]))
    assert Z.shape == (3, PLE_BINS) and Z.min() >= 0 and Z.max() <= 1
    assert np.all(Z[0] == 0) and np.all(Z[2] == 1) and np.all(np.diff(Z.sum(axis=1)) >= 0)


def test_linear_encodings_and_layout():
    X, y, b = _data()
    M = b.take(X, META)
    assert np.array_equal(MetaColumns("raw").fit(M).transform(M), M)
    assert MetaColumns("ple").fit(M).transform(M).shape == (len(X), PLE_BINS + 1)
    assert encoded_blocks(b, "ple").bounds[-1] == (50, 50 + PLE_BINS + 1)


def test_raw_encoding_reproduces_v01_early_fusion():
    X, y, b = _data()
    cfg, train = _cfg(), np.arange(len(y)) % 4 != 0
    v1 = build("E1", b, 1, None, load_config("conf/agent.yaml"), seed=0).fit(X[train], y[train])
    v2 = build_candidate("E1:raw", "screen", b, cfg, seed=0).fit(X[train], y[train])
    assert np.allclose(v1.predict_proba(X[~train]), v2.predict_proba(X[~train]))


@pytest.mark.parametrize("cand", all_candidates(CFG))
def test_every_screen_candidate_gives_valid_deterministic_probabilities(cand):
    X, y, b = _data()
    cfg, train = _cfg(), np.arange(len(y)) % 4 != 0
    P1 = build_candidate(cand, "screen", b, cfg, seed=0).fit(X[train], y[train]).predict_proba(X[~train])
    P2 = build_candidate(cand, "screen", b, cfg, seed=0).fit(X[train], y[train]).predict_proba(X[~train])
    assert P1.shape == ((~train).sum(), K) and np.all(P1 >= 0) and np.allclose(P1.sum(axis=1), 1)
    assert np.allclose(P1, P2)


def test_tuned_candidates_search_regularisation_and_size_by_inner_cv():
    X, y, b = _data()
    cfg = _cfg()
    e3 = build_candidate("E3:ple", "tuned", b, cfg, seed=0)
    assert isinstance(e3.inner, GridSearchCV) and set(e3.inner.param_grid) == {"lr__C", "blocks__k"}
    n2 = build_candidate("N2:plr", "tuned", b, cfg, seed=0)
    assert isinstance(n2, GridSearchCV) and set(n2.param_grid) == {"weight_decay", "d"}
    assert e3.fit(X, y).best_params_["blocks__k"] in cfg["phase2"]["size_grid"]


@pytest.mark.parametrize("enc", ["joint", "ft", "plr", "ple"])
def test_tokenizer_shapes(enc):
    tok = MetaTokenizer(enc, 16, 0.0)
    n = 5
    out = tok(torch.randn(n), torch.tensor([0, 1, 0, 1, 1]), torch.randn(n), torch.rand(n, PLE_BINS))
    assert out.shape == (n, 1 if enc == "joint" else 2, 16)


def test_token_network_handles_missing_sources_for_matched_rows():
    X, y, b = _data()
    meta_only = b.subset([META])
    img_only = b.subset(["A"])
    for blocks, cols in ((meta_only, b.columns([META])), (img_only, b.columns(["A"]))):
        m = LearnedFusionTok(blocks, kind="gated", encoding="plr", epochs=3, n_seeds=1).fit(X[:, cols], y)
        assert np.allclose(m.predict_proba(X[:, cols]).sum(axis=1), 1)
    with pytest.raises(ValueError):
        FusionNetTok([20], None, "bilinear", 16, 0.0, K)


# ---- policy v02 with a fake evaluator ------------------------------------------------------

HEADER = {"seed": 0, "mode": "dev", "models": ["a"], "code": "x", "features": {"a": "1"}, "config": "c"}


def fake_auroc(cand: str, mode: str, table: dict) -> float:
    base = table.get(cand, table.get(family(cand), 0.75))
    return base + (table.get("tune_gain", 0.005) if mode == "tuned" else 0)


def fake_compare(records):
    by_id = {r["id"]: r for r in records if r["type"] == "experiment"}

    def compare(a, b):
        d = by_id[a]["scores"]["macro_auroc"]["oof"] - by_id[b]["scores"]["macro_auroc"]["oof"]
        return {"gain": d, "ci": [d - 0.05, d + 0.05], "share_won": 0.99 if d > 0.03 else 0.6}
    return compare


def run(store, table, cfg=CFG, max_steps=None):
    nxt = get_policy("v02")
    store.open_run(HEADER)
    steps = 0
    while not isinstance(a := nxt(store.records(), cfg, fake_compare(store.records())), Done):
        if max_steps is not None and steps == max_steps:
            return None
        if isinstance(a, Decide):
            store.append({"type": "decision", "id": store.next_id("D"), "rule": a.rule, "based_on": a.based_on,
                          "statistics": a.statistics, "action": a.action, "reason": a.reason})
        else:
            v = fake_auroc(a.candidate, a.mode, table)
            store.append({"type": "experiment", "id": store.next_id("T"), "candidate": a.candidate, "mode": a.mode,
                          "status": "done", "based_on": a.based_on, "reason": a.reason,
                          "scores": {"macro_auroc": {"oof": v}, "balanced_accuracy": {"oof": 0.4}}})
        steps += 1
    return a


def test_v02_search_follows_the_protocol(tmp_path):
    store = TrialStore(tmp_path / "t.jsonl", 0)
    run(store, {"late": 0.80, "L2:raw": 0.81, "L4:ple": 0.83, "early": 0.70, "learned": 0.79})
    recs = store.records()
    exps = [r for r in recs if r["type"] == "experiment"]
    rules = [r["rule"] for r in recs if r["type"] == "decision"]
    assert [e["candidate"] for e in exps[:3]] == CFG["opening"]
    assert len(exps) <= CFG["search"]["budget"] and rules[-1] == "final_selection"
    assert "drop" in rules and "tune_start" in rules and "leader" in rules
    dropped = {r["action"]["family"]: i for i, r in enumerate(recs) if r["type"] == "decision" and r["rule"] == "drop"}
    for f, i in dropped.items():                       # a dropped family is never screened again
        assert not any(r["type"] == "experiment" and family(r["candidate"]) == f for r in recs[i:])
    assert sum(e["mode"] == "tuned" for e in exps) == CFG["search"]["tune_count"]
    seen = set()
    for r in recs[1:]:
        assert set(r["based_on"]) <= seen
        seen.add(r["id"])
    assert any(e["candidate"] == "L4:ple" for e in exps)        # encodings of the leading family are explored


def test_v02_resume_gives_the_same_record(tmp_path):
    table = {"late": 0.80, "L2:raw": 0.81, "early": 0.78, "learned": 0.79}
    a = TrialStore(tmp_path / "a.jsonl", 0)
    run(a, table)
    b = TrialStore(tmp_path / "b.jsonl", 0)
    assert run(b, table, max_steps=9) is None
    run(b, table)
    assert a.records() == b.records()


# ---- v03: log-linear pools --------------------------------------------------------------

def test_fit_log_linear_recovers_a_known_weighting():
    from src.fusion.pool import fit_log_linear
    from scipy.special import log_softmax
    rng = np.random.default_rng(0)
    n, Kc = 2000, 4
    L1, L2 = log_softmax(rng.normal(size=(n, Kc)) * 2, axis=1), log_softmax(rng.normal(size=(n, Kc)) * 2, axis=1)
    z = 1.5 * L1 + 0.0 * L2
    y = np.array([rng.choice(Kc, p=p) for p in np.exp(log_softmax(z, axis=1))])
    w, c = fit_log_linear([L1, L2], y, Kc, bias=False)
    assert abs(w[0] - 1.5) < 0.25 and w[1] < 0.2 and np.all(c == 0)


def test_product_pool_is_the_sum_of_expert_log_probabilities():
    from src.fusion.pool import LogLinearPool, _logp
    from scipy.special import log_softmax
    X, y, b = _data()
    m = LogLinearPool(b, pool="product").fit(X, y)
    lp = {n: _logp(e.predict_proba(b.take(X, n))) for n, e in m.experts_.items()}
    z = (lp["A"] + lp["B"]) / 2 + lp[META]
    assert np.allclose(m.predict_proba(X), np.exp(log_softmax(z, axis=1)))


@pytest.mark.parametrize("cand", ["P1:raw", "P2:ple", "P3:raw", "P4:ple"])
def test_pool_candidates_give_valid_probabilities(cand):
    X, y, b = _data()
    cfg = _cfg() | {"families": ["early", "late", "pool", "learned"]}
    train = np.arange(len(y)) % 4 != 0
    P = build_candidate(cand, "screen", b, cfg, seed=0).fit(X[train], y[train]).predict_proba(X[~train])
    assert np.allclose(P.sum(axis=1), 1) and np.all(P >= 0)
