"""Late fusion (L1–L4): one base model per block, then combine their predictions.

Each base model is the baselines' classifier on one block (each image model, and age and
sex). Combinations (docs/agent.md): probability average, log-probability average,
weighted vote, stacking. Vote weights and stacking features come from inner out-of-fold
predictions on the training patients, so the combiner never learns from predictions a base
model made on patients it was trained on. With `tune=True` (Phase 2) every base model and
the stacking model choose C by their own inner cross-validation (fully nested).
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src.eval.metrics import macro_auroc
from src.fusion.blocks import META, Blocks
from src.fusion.classifier import logistic, tuned_logistic

COMBINE = ("prob_mean", "logit_mean", "weighted_vote", "stack")
EPS = 1e-12


def _log(P: np.ndarray) -> np.ndarray:
    return np.log(np.clip(P, EPS, 1.0))


def _softmax(Z: np.ndarray) -> np.ndarray:
    Z = Z - Z.max(axis=1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)


class LateFusion(BaseEstimator, ClassifierMixin):
    def __init__(self, blocks: Blocks, combine: str = "prob_mean", C_img: float = 0.01,
                 C_meta: float = 1.0, C_stack: float = 1.0, tune: bool = False,
                 C_grid: tuple[float, ...] = (0.0001, 0.001, 0.01, 0.1, 1.0), inner_folds: int = 5,
                 seed: int = 0, max_iter: int = 5000, n_jobs: int = 8):
        self.blocks = blocks
        self.combine = combine
        self.C_img = C_img
        self.C_meta = C_meta
        self.C_stack = C_stack
        self.tune = tune
        self.C_grid = C_grid
        self.inner_folds = inner_folds
        self.seed = seed
        self.max_iter = max_iter
        self.n_jobs = n_jobs

    def _model(self, C: float):
        if self.tune:
            return tuned_logistic(list(self.C_grid), self.inner_folds, self.seed, self.max_iter, self.n_jobs)
        return logistic(C, self.max_iter)

    def _cv(self) -> StratifiedKFold:
        return StratifiedKFold(self.inner_folds, shuffle=True, random_state=self.seed)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LateFusion":
        if self.combine not in COMBINE:
            raise ValueError(f"combine must be one of {COMBINE}")
        self.classes_ = np.unique(y)
        self.bases_, oof = {}, {}
        for name in self.blocks.features:
            Xb = self.blocks.take(X, name)
            base = self._model(self.C_meta if name == META else self.C_img)
            if self.combine in ("weighted_vote", "stack"):
                oof[name] = cross_val_predict(clone(base), Xb, y, cv=self._cv(), method="predict_proba")
            self.bases_[name] = base.fit(Xb, y)
        if self.combine == "weighted_vote":
            w = np.array([max(macro_auroc(y, oof[n]) - 0.5, 0.0) for n in self.blocks.features])
            self.weights_ = w / w.sum() if w.sum() > 0 else np.full(len(w), 1 / len(w))
        if self.combine == "stack":
            self.stacker_ = self._model(self.C_stack).fit(self._stack_features(oof), y)
        self.best_params_ = self._chosen()
        return self

    def _stack_features(self, P: dict[str, np.ndarray]) -> np.ndarray:
        return np.hstack([_log(P[n]) for n in self.blocks.features])

    def _chosen(self) -> dict | None:
        """C picked by each inner CV (Phase 2), for the experiment record."""
        if not self.tune:
            return None
        chosen = {n: m.best_params_["lr__C"] for n, m in self.bases_.items()}
        if self.combine == "stack":
            chosen["stack"] = self.stacker_.best_params_["lr__C"]
        return chosen

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        P = {n: m.predict_proba(self.blocks.take(X, n)) for n, m in self.bases_.items()}
        stacked = np.stack([P[n] for n in self.blocks.features])           # (B, n, K)
        if self.combine == "prob_mean":
            return stacked.mean(axis=0)
        if self.combine == "logit_mean":
            return _softmax(_log(stacked).mean(axis=0))
        if self.combine == "weighted_vote":
            votes = np.eye(stacked.shape[2])[stacked.argmax(axis=2)]    # (B, n, K) one-hot
            return np.tensordot(self.weights_, votes, axes=1)
        return self.stacker_.predict_proba(self._stack_features(P))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[self.predict_proba(X).argmax(axis=1)]
