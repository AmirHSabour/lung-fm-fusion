"""Out-of-fold evaluation on the committed development folds.

In each round the model is fitted on the other folds and predicts the held-out fold, so
every development patient receives exactly one prediction from a model that never saw it.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.base import BaseEstimator

from src.eval.metrics import balanced_accuracy, macro_auroc


def run_cv(X: np.ndarray, y: np.ndarray, folds: np.ndarray,
           make_model: Callable[[], BaseEstimator], n_classes: int) -> dict:
    """Out-of-fold probabilities (n, n_classes), per-fold scores and fitted settings."""
    P = np.full((len(y), n_classes), np.nan)
    per_fold = []
    for k in np.unique(folds):
        train, val = folds != k, folds == k
        model = make_model().fit(X[train], y[train])
        P[val] = model.predict_proba(X[val])
        per_fold.append(dict(fold=int(k), n=int(val.sum()),
                             macro_auroc=macro_auroc(y[val], P[val]),
                             balanced_accuracy=balanced_accuracy(y[val], P[val]),
                             best_params=getattr(model, "best_params_", None)))
    if np.isnan(P).any():
        raise RuntimeError("some development patients received no out-of-fold prediction")
    return dict(P=P, per_fold=per_fold)
