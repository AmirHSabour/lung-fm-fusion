"""Metrics and bootstrap uncertainty (README: macro AUROC, per-class AUROC, balanced accuracy).

All functions take integer labels y (0..K-1) and a probability matrix P (n, K).
Bootstrap resampling is stratified by class, so every resample contains every class and
AUROC is always defined.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score


def macro_auroc(y: np.ndarray, P: np.ndarray) -> float:
    return float(roc_auc_score(y, P, multi_class="ovr", average="macro", labels=np.arange(P.shape[1])))


def per_class_auroc(y: np.ndarray, P: np.ndarray) -> np.ndarray:
    return np.array([roc_auc_score(y == k, P[:, k]) for k in range(P.shape[1])])


def balanced_accuracy(y: np.ndarray, P: np.ndarray) -> float:
    return float(balanced_accuracy_score(y, P.argmax(1)))


def confusion(y: np.ndarray, P: np.ndarray) -> np.ndarray:
    return confusion_matrix(y, P.argmax(1), labels=np.arange(P.shape[1]))


METRICS = {"macro_auroc": macro_auroc, "balanced_accuracy": balanced_accuracy}


def _stratified_resamples(y: np.ndarray, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(y == k) for k in np.unique(y)]
    return np.stack([np.concatenate([rng.choice(g, len(g)) for g in groups]) for _ in range(n)])


def stratified_resamples(y: np.ndarray, n: int, seed: int) -> np.ndarray:
    """(n, len(y)) patient indices, resampled with replacement within each class."""
    return _stratified_resamples(y, n, seed)


def bootstrap_ci(y: np.ndarray, P: np.ndarray, metric: str, n: int = 2000, seed: int = 0,
                 alpha: float = 0.05, idx: np.ndarray | None = None) -> tuple[float, float, float]:
    """(point estimate, lower, upper) percentile CI over patients.

    `idx` (n_resamples, n_patients) reuses stored resamples; otherwise they are drawn from `seed`.
    """
    f = METRICS[metric]
    idx = _stratified_resamples(y, n, seed) if idx is None else idx
    vals = np.array([f(y[i], P[i]) for i in idx])
    return f(y, P), float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


def paired_bootstrap(y: np.ndarray, P_a: np.ndarray, P_b: np.ndarray, metric: str, n: int = 2000,
                     seed: int = 0, idx: np.ndarray | None = None) -> dict:
    """How reliably A beats B on the same patients: share of resamples with A > B, mean gain, 95% CI."""
    f = METRICS[metric]
    idx = _stratified_resamples(y, n, seed) if idx is None else idx
    d = np.array([f(y[i], P_a[i]) - f(y[i], P_b[i]) for i in idx])
    return dict(p_better=float((d > 0).mean()), gain=float(f(y, P_a) - f(y, P_b)),
                ci_low=float(np.quantile(d, 0.025)), ci_high=float(np.quantile(d, 0.975)))
