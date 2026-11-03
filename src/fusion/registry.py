"""The 11 fusion recipes, registered by name (docs/agent.md).

`build` returns a scikit-learn estimator for a recipe in a given phase. Phase 1 uses the
fixed values of `conf/agent.yaml`; Phase 2 tunes C (or weight decay) by inner
cross-validation over the 5-value grid, exactly as the baselines tune C.
"""
from __future__ import annotations

from dataclasses import dataclass

from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from src.fusion.blocks import Blocks
from src.fusion.early import early_fusion
from src.fusion.late import LateFusion
from src.fusion.learned import LearnedFusion


@dataclass(frozen=True)
class Recipe:
    family: str            # early | late | learned
    options: dict
    has_size: bool         # PCA k (E3, E4) or projection d (N1–N3)


RECIPES: dict[str, Recipe] = {
    "E1": Recipe("early", dict(l2=False, pca=False), False),
    "E2": Recipe("early", dict(l2=True, pca=False), False),
    "E3": Recipe("early", dict(l2=False, pca=True), True),
    "E4": Recipe("early", dict(l2=True, pca=True), True),
    "L1": Recipe("late", dict(combine="prob_mean"), False),
    "L2": Recipe("late", dict(combine="logit_mean"), False),
    "L3": Recipe("late", dict(combine="weighted_vote"), False),
    "L4": Recipe("late", dict(combine="stack"), False),
    "N1": Recipe("learned", dict(kind="gated"), True),
    "N2": Recipe("learned", dict(kind="attention"), True),
    "N3": Recipe("learned", dict(kind="bilinear"), True),
    # v03: log-linear pools (age and sex as a Bayesian prior); not used by v01/v02 configs.
    "P1": Recipe("pool", dict(pool="product"), False),
    "P2": Recipe("pool", dict(pool="two"), False),
    "P3": Recipe("pool", dict(pool="per_source"), False),
    "P4": Recipe("pool", dict(pool="per_source_bias"), False),
    # v04: tile-level multiple-instance fusion (attention pooling over cached tiles).
    "M1": Recipe("mil", dict(kind="concat"), False),
    "M2": Recipe("mil", dict(kind="branches"), False),
    "M3": Recipe("mil", dict(kind="film"), False),
    # v05: separately trained MIL experts combined with the age-and-sex expert by log-linear pooling.
    "X1": Recipe("mexp", dict(pool="product"), False),
    "X2": Recipe("mexp", dict(pool="two"), False),
    "X3": Recipe("mexp", dict(pool="per_source"), False),
    "X4": Recipe("mexp", dict(pool="per_source_bias"), False),
    # v06: experts with composition aggregators (mixtures of tile class distributions), X1 product.
    "A1": Recipe("agg", dict(pool="product", aggregator="comp"), False),
    "A2": Recipe("agg", dict(pool="product", aggregator="wcomp"), False),
    # v07: CLAM experts (single- and multi-branch, instance clustering loss), X1 product.
    "C1": Recipe("clam", dict(pool="product", aggregator="clam_sb"), False),
    "C2": Recipe("clam", dict(pool="product", aggregator="clam_mb"), False),
    # v08: experts on spatial-context tokens (tile + 3x3 and 7x7 neighbourhood means), X1 product.
    "K1": Recipe("ctx", dict(pool="product", aggregator="attention", context=True), False),
    "K2": Recipe("ctx", dict(pool="product", aggregator="clam_sb", context=True), False),
    # v09: ensembles of expert types per model (mean log-probability), X1 product with age and sex.
    "Z1": Recipe("ens", dict(pool="product", experts=("attention", "clam_sb")), False),
    "Z2": Recipe("ens", dict(pool="product", experts=("attention", "clam_sb", "attention@ctx", "clam_sb@ctx")), False),
    # v10: cross-attention pooling experts; Q2 with age and sex tokens as queries inside each expert.
    "Q1": Recipe("xq", dict(pool="product", aggregator="xattn"), False),
    "Q2": Recipe("xqm", dict(pool="product", aggregator="xattn", meta_queries=True), False),
    # v11: experts trained with an auxiliary grade-group head, X1 product.
    "H1": Recipe("hier", dict(pool="product", aux_groups=True), False),
    "H2": Recipe("hier", dict(pool="product", aggregator="clam_sb", aux_groups=True), False),
}


def settings(name: str, phase: int, size: int | None, cfg: dict) -> dict:
    """The settings that define an experiment (recorded and fingerprinted)."""
    r, p1 = RECIPES[name], cfg["phase1"]
    s: dict = {"recipe": name, "family": r.family, **r.options, "phase": phase,
               "size": size if r.has_size else None}
    if phase == 1:
        s |= {"early": {"C": p1["C_img"]},
              "late": {"C_img": p1["C_img"], "C_meta": p1["C_meta"], "C_stack": p1["C_stack"]},
              "learned": {"weight_decay": p1["weight_decay"]}}[r.family]
    else:
        s |= {"tuned_by_inner_cv": cfg["phase2"]["grid"], "inner_folds": cfg["phase2"]["inner_folds"]}
    if r.family == "learned":
        s |= {"network": cfg["network"]}
    return s


def _tuned(est: BaseEstimator, param: str, cfg: dict, seed: int, n_jobs: int) -> GridSearchCV:
    cv = StratifiedKFold(cfg["phase2"]["inner_folds"], shuffle=True, random_state=seed)
    return GridSearchCV(est, {param: list(cfg["phase2"]["grid"])}, cv=cv, scoring="roc_auc_ovr",
                        n_jobs=n_jobs, refit=True)


def build(name: str, blocks: Blocks, phase: int, size: int | None, cfg: dict, seed: int,
          device: str = "cpu") -> BaseEstimator:
    r, p1 = RECIPES[name], cfg["phase1"]
    if r.family == "early":
        est = early_fusion(blocks, k=size or p1["size"], C=p1["C_img"], max_iter=cfg["max_iter"], **r.options)
        return est if phase == 1 else _tuned(est, "lr__C", cfg, seed, n_jobs=8)
    if r.family == "late":
        return LateFusion(blocks, C_img=p1["C_img"], C_meta=p1["C_meta"], C_stack=p1["C_stack"],
                          tune=phase == 2, C_grid=tuple(cfg["phase2"]["grid"]),
                          inner_folds=cfg["phase2"]["inner_folds"], seed=seed, max_iter=cfg["max_iter"],
                          **r.options)
    est = LearnedFusion(blocks, d=size or p1["size"], weight_decay=p1["weight_decay"], seed=seed,
                        device=device, **cfg["network"], **r.options)
    return est if phase == 1 else _tuned(est, "weight_decay", cfg, seed, n_jobs=1)
