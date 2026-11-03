"""Fusion recipes on synthetic data: block scaling, valid probabilities, determinism, tuning wrappers."""
from __future__ import annotations

import copy

import numpy as np
import pytest
from sklearn.model_selection import GridSearchCV

from src.fusion.blocks import META, Blocks, BlockTransformer
from src.fusion.late import LateFusion
from src.fusion.registry import RECIPES, build
from src.utils.config import load_config

K = 7


def _cfg() -> dict:
    cfg = copy.deepcopy(load_config("conf/agent.yaml"))
    cfg["network"] |= {"epochs": 5, "n_seeds": 2}       # fast; behaviour, not accuracy, is tested
    cfg["phase2"]["inner_folds"] = 3
    return cfg


def _data(n_per_class: int = 12, seed: int = 0):
    rng = np.random.default_rng(seed)
    y = np.repeat(np.arange(K), n_per_class)
    sizes = {"A": 20, "B": 30, META: 2}
    X = np.hstack([rng.normal(size=(len(y), p)) + 0.5 * y[:, None] * (j + 1) for j, p in enumerate(sizes.values())])
    return X, y, Blocks.from_sizes(sizes)


def test_blocks_layout_and_meta_last():
    _, _, b = _data()
    assert b.names == ("A", "B", META) and b.images == ("A", "B") and b.slice("B") == slice(20, 50)
    with pytest.raises(ValueError):
        Blocks.from_sizes({META: 2, "A": 3})
    sub = b.subset(["B", META])
    assert sub.bounds == ((0, 30), (30, 32)) and list(b.columns(["B", META])) == list(range(20, 52))


@pytest.mark.parametrize("l2,pca", [(False, False), (True, False), (False, True), (True, True)])
def test_block_scaling_equal_variance_per_block(l2, pca):
    X, y, b = _data()
    Z = BlockTransformer(b, l2=l2, pca=pca, k=8).fit_transform(X)
    widths = [8 if pca else 20, 8 if pca else 30, 2]
    edges = np.cumsum([0] + widths)
    per_block = [Z[:, edges[i]:edges[i + 1]].var(axis=0).sum() for i in range(3)]
    assert np.allclose(per_block, Z.shape[1] / 3) and np.isclose(sum(per_block), Z.shape[1])


def test_pca_needs_more_patients_than_components():
    X, y, b = _data()
    with pytest.raises(ValueError):
        BlockTransformer(b, pca=True, k=len(X)).fit(X)


@pytest.mark.parametrize("name", load_config("conf/agent.yaml")["order"])     # the v01 recipes
def test_phase1_recipe_gives_valid_deterministic_probabilities(name):
    X, y, b = _data()
    cfg, train = _cfg(), np.arange(len(y)) % 4 != 0
    size = 8 if RECIPES[name].has_size else None
    P1 = build(name, b, 1, size, cfg, seed=0).fit(X[train], y[train]).predict_proba(X[~train])
    P2 = build(name, b, 1, size, cfg, seed=0).fit(X[train], y[train]).predict_proba(X[~train])
    assert P1.shape == ((~train).sum(), K) and np.all(P1 >= 0) and np.allclose(P1.sum(axis=1), 1)
    assert np.allclose(P1, P2)


def test_phase2_wrappers_tune_the_regularisation():
    X, y, b = _data()
    cfg = _cfg()
    early = build("E1", b, 2, None, cfg, seed=0)
    assert isinstance(early, GridSearchCV) and early.param_grid == {"lr__C": cfg["phase2"]["grid"]}
    net = build("N1", b, 2, 16, cfg, seed=0)
    assert isinstance(net, GridSearchCV) and net.param_grid == {"weight_decay": cfg["phase2"]["grid"]}
    late = build("L4", b, 2, None, cfg, seed=0).fit(X, y)
    assert isinstance(late, LateFusion) and set(late.best_params_) == {"A", "B", META, "stack"}
    assert early.fit(X, y).best_params_["lr__C"] in cfg["phase2"]["grid"]


def test_late_fusion_base_models_see_only_their_block_and_vote_weights_sum_to_one():
    X, y, b = _data()
    m = LateFusion(b, combine="weighted_vote").fit(X, y)
    assert {n: m.bases_[n].named_steps["lr"].coef_.shape[1] for n in b.names} == {"A": 20, "B": 30, META: 2}
    assert np.isclose(m.weights_.sum(), 1) and np.all(m.weights_ >= 0)
