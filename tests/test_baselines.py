"""Metrics, development-set loading and out-of-fold evaluation used by the baselines."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.data.splits import load_split
from src.eval.cv import run_cv
from src.eval.dataset import load_dev
from src.eval.metrics import (balanced_accuracy, bootstrap_ci, macro_auroc, paired_bootstrap,
                              per_class_auroc)
from src.fusion.classifier import logistic
from src.utils.config import load_config

DATA = load_config("conf/data.yaml")
TRAIN = load_config("conf/train.yaml")


def test_metrics_on_known_inputs():
    y = np.array([0, 0, 1, 1, 2, 2])
    perfect = np.eye(3)[y]
    assert macro_auroc(y, perfect) == 1.0 and balanced_accuracy(y, perfect) == 1.0
    assert (per_class_auroc(y, perfect) == 1.0).all()
    flat = np.full((6, 3), 1 / 3)
    assert macro_auroc(y, flat) == 0.5
    est, lo, hi = bootstrap_ci(y, perfect, "macro_auroc", n=50)
    assert est == lo == hi == 1.0
    cmp = paired_bootstrap(y, perfect, flat, "macro_auroc", n=50)
    assert cmp["p_better"] == 1.0 and cmp["gain"] == 0.5


def _available_models() -> list[str]:
    return [m for m in TRAIN["models"] if (Path(TRAIN["features_dir"]) / f"{m}.npz").is_file()]


@pytest.mark.parametrize("seed", DATA["split"]["seeds"])
def test_load_dev_has_only_development_patients(seed):
    models = _available_models()
    if not models:
        pytest.skip("no slide features yet (run stage 08)")
    dev = load_dev(seed, models, splits_dir=Path(DATA["splits_dir"]), cohort_csv=Path(DATA["cohort_csv"]),
                   features_dir=Path(TRAIN["features_dir"]))
    split = load_split(DATA["splits_dir"], seed)
    test_ids = set(split.loc[split.split == "test", "WSI_ID"])
    assert not test_ids & set(dev.wsi_ids) and len(dev.wsi_ids) == (split.split == "dev").sum()
    assert all(len(X) == len(dev.y) for X in dev.blocks.values()) and dev.meta.shape == (len(dev.y), 2)
    assert set(dev.folds) == set(range(DATA["split"]["n_folds"]))


def test_run_cv_predicts_every_patient_once_and_is_deterministic():
    rng = np.random.default_rng(0)
    y = np.repeat(np.arange(3), 16)
    X = rng.normal(size=(len(y), 5)) + y[:, None]
    folds = np.tile(np.arange(4), 12)
    a = run_cv(X, y, folds, lambda: logistic(1.0), n_classes=3)
    b = run_cv(X, y, folds, lambda: logistic(1.0), n_classes=3)
    assert a["P"].shape == (48, 3) and np.allclose(a["P"].sum(1), 1)
    assert np.array_equal(a["P"], b["P"]) and len(a["per_fold"]) == 4
