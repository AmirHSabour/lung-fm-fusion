"""The single downstream classifier family used by every row (README line 162).

Class-weighted multinomial logistic regression on standardised inputs. The scaler sits
inside the pipeline, so when the regularisation strength is chosen by an inner CV the
scaler is refitted on each inner training fold and never sees the inner validation fold.
"""
from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def logistic(C: float = 1.0, max_iter: int = 5000) -> Pipeline:
    """Standardise, then class-weighted multinomial logistic regression at a fixed C."""
    return Pipeline([("scale", StandardScaler()),
                     ("lr", LogisticRegression(C=C, class_weight="balanced", max_iter=max_iter))])


def tuned_logistic(C_grid: list[float], inner_folds: int, seed: int, max_iter: int = 5000,
                   n_jobs: int = 8) -> GridSearchCV:
    """Logistic regression whose C is chosen by inner stratified CV on macro one-vs-rest AUROC."""
    return GridSearchCV(logistic(C_grid[0], max_iter), {"lr__C": list(C_grid)},
                        cv=StratifiedKFold(inner_folds, shuffle=True, random_state=seed),
                        scoring="roc_auc_ovr", n_jobs=n_jobs, refit=True)
