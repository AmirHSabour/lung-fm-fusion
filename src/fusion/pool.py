"""Log-linear pools (v03, decisions/v03.md): age and sex as a Bayesian prior on a pooled image expert.

Experts are the baselines' class-weighted logistic regressions, one per block. Under
conditional independence of morphology and age/sex given the subtype, Bayes' rule gives
log p(y | x, m) = log p(y | x) + log p(y | m) + const (class-balanced experts imply a uniform
prior). The redundant image models are pooled into one image expert (mean log-probability).
Pools P2–P4 learn a few non-negative weights (and, for P4, class offsets) by minimising the
class-weighted negative log-likelihood on inner out-of-fold expert log-probabilities.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import log_softmax
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src.fusion.blocks import META, Blocks
from src.fusion.classifier import logistic, tuned_logistic

POOLS = ("product", "two", "per_source", "per_source_bias")
EPS = 1e-12


def _logp(P: np.ndarray) -> np.ndarray:
    return np.log(np.clip(P, EPS, 1.0))


def fit_log_linear(L: list[np.ndarray], y: np.ndarray, K: int, bias: bool) -> tuple[np.ndarray, np.ndarray]:
    """Weights w >= 0 (one per term) and offsets c minimising class-weighted NLL of softmax(Σ w_j L_j + c)."""
    n_terms = len(L)
    counts = np.bincount(y, minlength=K)
    sw = (len(y) / (K * counts))[y]
    stack = np.stack(L)                                   # (terms, n, K)

    def nll(theta: np.ndarray) -> float:
        w, c = theta[:n_terms], (theta[n_terms:] if bias else np.zeros(K))
        z = np.tensordot(w, stack, axes=1) + c
        return float(-(sw * log_softmax(z, axis=1)[np.arange(len(y)), y]).sum() / sw.sum())

    x0 = np.r_[np.full(n_terms, 1.0), np.zeros(K if bias else 0)]
    bounds = [(0.0, 10.0)] * n_terms + [(-5.0, 5.0)] * (K if bias else 0)
    res = minimize(nll, x0, method="L-BFGS-B", bounds=bounds)
    return res.x[:n_terms], (res.x[n_terms:] if bias else np.zeros(K))


class LogLinearPool(BaseEstimator, ClassifierMixin):
    def __init__(self, blocks: Blocks, pool: str = "product", C_img: float = 0.01, C_meta: float = 1.0,
                 tune: bool = False, C_grid: tuple[float, ...] = (0.0001, 0.001, 0.01, 0.1, 1.0),
                 inner_folds: int = 5, seed: int = 0, max_iter: int = 5000, n_jobs: int = 8):
        self.blocks = blocks
        self.pool = pool
        self.C_img = C_img
        self.C_meta = C_meta
        self.tune = tune
        self.C_grid = C_grid
        self.inner_folds = inner_folds
        self.seed = seed
        self.max_iter = max_iter
        self.n_jobs = n_jobs

    def _expert(self, C: float):
        if self.tune:
            return tuned_logistic(list(self.C_grid), self.inner_folds, self.seed, self.max_iter, self.n_jobs)
        return logistic(C, self.max_iter)

    def _terms(self, logp: dict[str, np.ndarray]) -> list[np.ndarray]:
        """The pooled terms: image expert (mean) and age and sex (P1, P2), or one term per source (P3, P4)."""
        imgs = [logp[n] for n in self.blocks.images]
        meta = [logp[META]] if META in logp else []
        if self.pool in ("product", "two"):
            return ([np.mean(imgs, axis=0)] if imgs else []) + meta
        return imgs + meta

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogLinearPool":
        if self.pool not in POOLS:
            raise ValueError(f"pool must be one of {POOLS}")
        self.classes_ = np.unique(y)
        yi = np.searchsorted(self.classes_, y)
        K = len(self.classes_)
        cv = StratifiedKFold(self.inner_folds, shuffle=True, random_state=self.seed)
        self.experts_, oof = {}, {}
        for name in self.blocks.features:
            Xb = self.blocks.take(X, name)
            est = self._expert(self.C_meta if name == META else self.C_img)
            if self.pool != "product":
                oof[name] = _logp(cross_val_predict(clone(est), Xb, y, cv=cv, method="predict_proba"))
            self.experts_[name] = est.fit(Xb, y)
        n_terms = len(self._terms({n: np.zeros((1, K)) for n in self.blocks.features}))
        if self.pool == "product":
            self.weights_, self.offsets_ = np.ones(n_terms), np.zeros(K)
        else:
            self.weights_, self.offsets_ = fit_log_linear(self._terms(oof), yi, K, self.pool == "per_source_bias")
        chosen = {"weights": [round(float(w), 4) for w in self.weights_]}
        if self.tune:
            chosen |= {n: m.best_params_["lr__C"] for n, m in self.experts_.items()}
        self.best_params_ = chosen if (self.tune or self.pool != "product") else None
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        logp = {n: _logp(m.predict_proba(self.blocks.take(X, n))) for n, m in self.experts_.items()}
        z = np.tensordot(self.weights_, np.stack(self._terms(logp)), axes=1) + self.offsets_
        return np.exp(log_softmax(z, axis=1))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[self.predict_proba(X).argmax(axis=1)]
