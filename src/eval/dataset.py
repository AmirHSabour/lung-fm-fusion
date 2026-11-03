"""Data for one seed: slide vectors, age/sex, labels and CV folds.

`load_dev` loads development patients only; every stage before the final evaluation uses it.
`load_test` loads the seed's test patients and is called only by the final evaluation
(scripts/12_final_eval.py), after the designs are frozen (decisions/SUMMARY.md).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.cohort import CLASSES
from src.data.splits import load_split


@dataclass(frozen=True)
class DevSet:
    wsi_ids: np.ndarray            # (n,) development slides, in split-file order
    blocks: dict[str, np.ndarray]  # model -> (n, d) slide vectors
    meta: np.ndarray               # (n, 2): age in years, sex (1 = male, 0 = female)
    y: np.ndarray                  # (n,) integer labels, index into CLASSES
    folds: np.ndarray              # (n,) CV fold 0..K-1
    classes: tuple[str, ...] = CLASSES


def load_dev(seed: int, models: list[str], *, splits_dir: Path, cohort_csv: Path,
             features_dir: Path) -> DevSet:
    """Aligned development data for `models` (each must have a features file)."""
    return _load(seed, models, "dev", splits_dir=splits_dir, cohort_csv=cohort_csv, features_dir=features_dir)


def load_test(seed: int, models: list[str], *, splits_dir: Path, cohort_csv: Path,
              features_dir: Path) -> DevSet:
    """The seed's test patients (folds = -1). Final evaluation only."""
    return _load(seed, models, "test", splits_dir=splits_dir, cohort_csv=cohort_csv, features_dir=features_dir)


def _load(seed: int, models: list[str], part: str, *, splits_dir: Path, cohort_csv: Path,
          features_dir: Path) -> DevSet:
    split = load_split(splits_dir, seed)
    dev = split[split["split"] == part].reset_index(drop=True)
    cohort = pd.read_csv(cohort_csv).set_index("WSI_ID").loc[dev["WSI_ID"]]
    blocks = {}
    for m in models:
        f = np.load(Path(features_dir) / f"{m}.npz")
        index = {w: i for i, w in enumerate(f["WSI_ID"])}
        blocks[m] = f["X"][[index[w] for w in dev["WSI_ID"]]].astype(np.float64)
    meta = np.column_stack([cohort["age"].to_numpy(float), (cohort["sex"] == "Male").to_numpy(float)])
    y = np.array([CLASSES.index(lbl) for lbl in dev["label"]])
    folds = dev["fold"].to_numpy(int) if part == "dev" else np.full(len(dev), -1)
    return DevSet(wsi_ids=dev["WSI_ID"].to_numpy(), blocks=blocks, meta=meta, y=y, folds=folds)
