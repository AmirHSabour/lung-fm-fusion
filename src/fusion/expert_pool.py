"""Pooled tile-level experts (v05, decisions/v05.md).

One gated-attention MIL expert per foundation model (src/fusion/mil.py, without age and sex), each
trained on its own tiles, plus a logistic age-and-sex expert. Their log-probabilities are combined
by a log-linear pool (src/fusion/pool.py): weights, and for X4 class offsets, are fitted by
class-weighted NLL on the experts' inner out-of-fold predictions.

The MIL experts are deterministic and identical across the pooled candidates, so their predictions
are cached on disk (`expert_cache`), keyed by model, training rows, predicted rows and settings.

v06: `aggregator` selects the experts' tile→slide aggregator (src/fusion/mil.py); recipes A1, A2.
v09: `experts` lists several expert types per model (`attention`, `clam_sb`, `…@ctx` on the context
cache); a model's term is the mean of its experts' log-probabilities (ensemble); recipes Z1, Z2.
v10: `meta_in_experts` (a token encoding) gives every expert age and sex (as attention queries, Q2);
the pool then has no separate age-and-sex term.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import log_softmax
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src.fusion.blocks import ID, META, Blocks
from src.fusion.classifier import logistic
from src.fusion.mil import MILFusion
from src.fusion.pool import POOLS, _logp, fit_log_linear


class MILExpertPool(BaseEstimator, ClassifierMixin):
    def __init__(self, blocks: Blocks, pool: str = "product", C_meta: float = 1.0,
                 cache_dir: str = "artifacts/features/tiles", mil: dict | None = None, d: int = 32,
                 weight_decay: float = 0.01, inner_folds: int = 5, seed: int = 0, device: str = "cpu",
                 expert_cache: str | None = None, max_iter: int = 5000, aggregator: str = "attention",
                 experts: tuple[str, ...] | None = None, ctx_cache: str | None = None,
                 meta_in_experts: str | None = None, aux_groups: bool = False):
        self.blocks = blocks
        self.pool = pool
        self.C_meta = C_meta
        self.cache_dir = cache_dir
        self.mil = mil
        self.d = d
        self.weight_decay = weight_decay
        self.inner_folds = inner_folds
        self.seed = seed
        self.device = device
        self.expert_cache = expert_cache
        self.max_iter = max_iter
        self.aggregator = aggregator
        self.experts = experts
        self.ctx_cache = ctx_cache
        self.meta_in_experts = meta_in_experts
        self.aux_groups = aux_groups

    # ---- one MIL expert -------------------------------------------------------------------
    def _members(self) -> list[tuple[str, str]]:
        """(aggregator, tile cache) of each expert type; one per model unless `experts` is set."""
        if not self.experts:
            return [(self.aggregator, self.cache_dir)]
        out = []
        for e in self.experts:
            agg, _, where = e.partition("@")
            if where not in ("", "ctx"):
                raise ValueError(f"unknown expert type {e}")
            out.append((agg, self.ctx_cache if where == "ctx" else self.cache_dir))
        return out

    def _model_logp(self, m: str, X: np.ndarray, y: np.ndarray, train: np.ndarray, pred: np.ndarray) -> np.ndarray:
        """Mean log-probability over model `m`'s expert types (one type: that expert's log-probability)."""
        return np.mean([_logp(self._expert_predict(m, X, y, train, pred, a, c)) for a, c in self._members()], axis=0)

    def _expert_predict(self, m: str, X: np.ndarray, y: np.ndarray, train: np.ndarray, pred: np.ndarray,
                        aggregator: str | None = None, cache_dir: str | None = None) -> np.ndarray:
        """Probabilities of the MIL expert of model `m`, trained on rows `train`, for rows `pred`."""
        aggregator = aggregator or self.aggregator
        cache_dir = cache_dir or self.cache_dir
        ids = self.blocks.take(X, ID)[:, 0].astype(int)
        settings = {"model": m, "train": ids[train].tolist(), "pred": ids[pred].tolist(), "mil": self.mil,
                    "d": self.d, "wd": self.weight_decay, "seed": self.seed, "cache": str(cache_dir)}
        if aggregator != "attention":                               # v05 keys unchanged
            settings["aggregator"] = aggregator
        with_meta = bool(self.meta_in_experts) and META in self.blocks.names
        if with_meta:
            settings["meta_in_experts"] = self.meta_in_experts
        if self.aux_groups:
            settings["aux_groups"] = True
        key = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:24]
        path = Path(self.expert_cache) / f"{key}.npy" if self.expert_cache else None
        if path is not None and path.exists():
            return np.load(path)
        names = [m, ID] + ([META] if with_meta else [])
        sub = self.blocks.subset(names)
        cols = self.blocks.columns(names)
        est = MILFusion(sub, kind="concat", encoding=self.meta_in_experts if with_meta else None,
                        cache_dir=cache_dir, d=self.d,
                        weight_decay=self.weight_decay, seed=self.seed, device=self.device, aggregator=aggregator,
                        aux_groups=self.aux_groups,
                        **(self.mil or {}))
        P = est.fit(X[train][:, cols], y[train]).predict_proba(X[pred][:, cols])
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, P)
        return P

    def _terms(self, logp: dict[str, np.ndarray]) -> list[np.ndarray]:
        imgs = [logp[n] for n in self.blocks.images]
        meta = [logp[META]] if META in logp else []
        if self.pool in ("product", "two"):
            return ([np.mean(imgs, axis=0)] if imgs else []) + meta
        return imgs + meta

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MILExpertPool":
        if self.pool not in POOLS:
            raise ValueError(f"pool must be one of {POOLS}")
        self.classes_ = np.unique(y)
        yi = np.searchsorted(self.classes_, y)
        K = len(self.classes_)
        self.X_, self.y_ = X, y                       # the final experts are trained lazily in predict_proba
        self.prior_ = META in self.blocks.names and not self.meta_in_experts    # separate age-and-sex expert
        if self.prior_:
            self.meta_ = logistic(self.C_meta, self.max_iter).fit(self.blocks.take(X, META), y)
        if self.pool == "product":
            names = [n for n in self.blocks.features if n != ID and (n != META or self.prior_)]
            self.weights_ = np.ones(len(self._terms({n: np.zeros((1, K)) for n in names})))
            self.offsets_ = np.zeros(K)
        else:
            cv = StratifiedKFold(self.inner_folds, shuffle=True, random_state=self.seed)
            logp = {m: np.zeros((len(y), K)) for m in self.blocks.images}
            for tr, va in cv.split(X, y):
                for m in self.blocks.images:
                    logp[m][va] = self._model_logp(m, X, y, tr, va)
            if self.prior_:
                logp[META] = _logp(cross_val_predict(clone(logistic(self.C_meta, self.max_iter)),
                                                     self.blocks.take(X, META), y, cv=cv, method="predict_proba"))
            self.weights_, self.offsets_ = fit_log_linear(self._terms(logp), yi, K, self.pool == "per_source_bias")
        self.best_params_ = {"weights": [round(float(w), 4) for w in self.weights_]} if self.pool != "product" else None
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xall = np.vstack([self.X_, X])
        n = len(self.X_)
        yall = np.concatenate([self.y_, np.full(len(X), self.y_[0])])      # labels of predicted rows are never used
        logp = {m: self._model_logp(m, Xall, yall, np.arange(n), np.arange(n, len(Xall))) for m in self.blocks.images}
        if self.prior_:
            logp[META] = _logp(self.meta_.predict_proba(self.blocks.take(X, META)))
        z = np.tensordot(self.weights_, np.stack(self._terms(logp)), axes=1) + self.offsets_
        return np.exp(log_softmax(z, axis=1))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[self.predict_proba(X).argmax(axis=1)]
