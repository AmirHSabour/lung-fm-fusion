"""v02 candidates: a recipe combined with an age-and-sex encoding, e.g. `N2:plr` (decisions/v02.md §2).

`build_candidate` returns a scikit-learn estimator in one of two modes:
- `screen`: the fixed values of the config (v01 Phase 1 values);
- `tuned`: every hyperparameter chosen by inner cross-validation on the training patients
  (C or weight decay over the 5-value grid; the size jointly, for recipes that have one).
"""
from __future__ import annotations

from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from src.fusion.blocks import Blocks
from src.fusion.early import early_fusion
from src.fusion.late import LateFusion
from src.fusion.learned import LearnedFusionTok
from src.fusion.metadata import MetaEncoded, encoded_blocks
from src.fusion.pool import LogLinearPool
from src.fusion.registry import RECIPES

FAMILY_ORDER = ("early", "late", "learned")      # v02; later versions list theirs in the config (`families`)


def families(cfg: dict) -> tuple[str, ...]:
    return tuple(cfg.get("families", FAMILY_ORDER))


def split(candidate: str) -> tuple[str, str]:
    recipe, encoding = candidate.split(":")
    return recipe, encoding


def family(candidate: str) -> str:
    return RECIPES[split(candidate)[0]].family


def all_candidates(cfg: dict) -> list[str]:
    """Every candidate, in simplicity order (family, recipe, encoding)."""
    enc = cfg["encodings"]
    out = []
    for fam in families(cfg):
        for recipe in (r for r in cfg["order"] if RECIPES[r].family == fam):
            out += [f"{recipe}:{e}" for e in enc[fam]]
    return out


def _grid_search(est: BaseEstimator, grid: dict, cfg: dict, seed: int, n_jobs: int) -> GridSearchCV:
    cv = StratifiedKFold(cfg["phase2"]["inner_folds"], shuffle=True, random_state=seed)
    return GridSearchCV(est, grid, cv=cv, scoring="roc_auc_ovr", n_jobs=n_jobs, refit=True)


def build_candidate(candidate: str, mode: str, blocks: Blocks, cfg: dict, seed: int,
                    device: str = "cpu") -> BaseEstimator:
    recipe, encoding = split(candidate)
    r, p1, grid = RECIPES[recipe], cfg["phase1"], list(cfg["phase2"]["grid"])
    sizes = list(cfg["phase2"]["size_grid"])
    tuned = mode == "tuned"
    if r.family == "mil":
        from src.fusion.mil import MILFusion
        est = MILFusion(blocks, encoding=encoding, cache_dir=cfg["tile_cache"], d=p1["size"],
                        weight_decay=p1["weight_decay"], seed=seed, device=device, **cfg["mil"], **r.options)
        return _grid_search(est, {"weight_decay": grid}, cfg, seed, n_jobs=1) if tuned else est
    if r.family == "learned":
        est = LearnedFusionTok(blocks, encoding=encoding, d=p1["size"], weight_decay=p1["weight_decay"],
                               seed=seed, device=device, **cfg["network"], **r.options)
        if not tuned:
            return est
        return _grid_search(est, {"weight_decay": grid, "d": sizes}, cfg, seed, n_jobs=1)
    if r.family == "xqm":                         # v10: age and sex enter each expert as token queries
        from src.fusion.expert_pool import MILExpertPool
        opts = {k: v for k, v in r.options.items() if k != "meta_queries"}
        return MILExpertPool(blocks, C_meta=p1["C_meta"], cache_dir=cfg["tile_cache"], mil=dict(cfg["mil"]),
                             d=p1["size"], weight_decay=p1["weight_decay"], inner_folds=cfg["phase2"]["inner_folds"],
                             seed=seed, device=device, expert_cache=cfg.get("expert_cache"), max_iter=cfg["max_iter"],
                             ctx_cache=cfg.get("ctx_cache"), meta_in_experts=encoding, **opts)
    enc_blocks = encoded_blocks(blocks, encoding)
    if r.family == "early":
        inner = early_fusion(enc_blocks, k=p1["size"], C=p1["C_img"], max_iter=cfg["max_iter"], **r.options)
        if tuned:
            g = {"lr__C": grid} | ({"blocks__k": sizes} if r.has_size else {})
            inner = _grid_search(inner, g, cfg, seed, n_jobs=8)
    elif r.family in ("mexp", "agg", "clam", "ctx", "ens", "xq", "hier"):
        from src.fusion.expert_pool import MILExpertPool
        mil = {k: v for k, v in cfg["mil"].items()}
        opts = {k: v for k, v in r.options.items() if k != "context"}
        cache = cfg["ctx_cache"] if r.options.get("context") else cfg["tile_cache"]      # v08: context tokens
        inner = MILExpertPool(enc_blocks, C_meta=p1["C_meta"], cache_dir=cache, mil=mil, d=p1["size"],
                              weight_decay=p1["weight_decay"], inner_folds=cfg["phase2"]["inner_folds"], seed=seed,
                              device=device, expert_cache=cfg.get("expert_cache"), max_iter=cfg["max_iter"],
                              ctx_cache=cfg.get("ctx_cache"), **opts)
    elif r.family == "pool":
        inner = LogLinearPool(enc_blocks, C_img=p1["C_img"], C_meta=p1["C_meta"], tune=tuned, C_grid=tuple(grid),
                              inner_folds=cfg["phase2"]["inner_folds"], seed=seed, max_iter=cfg["max_iter"],
                              **r.options)
    else:
        inner = LateFusion(enc_blocks, C_img=p1["C_img"], C_meta=p1["C_meta"], C_stack=p1["C_stack"], tune=tuned,
                           C_grid=tuple(grid), inner_folds=cfg["phase2"]["inner_folds"], seed=seed,
                           max_iter=cfg["max_iter"], **r.options)
    return MetaEncoded(inner, blocks, encoding)
