"""Patient-level splits: a stratified test set plus stratified CV folds on the rest.

See docs/methods.md D12. For a seed s:
  test  = train_test_split(stratify=label, test_size=test_fraction, random_state=s)
  folds = StratifiedKFold(n_folds, shuffle=True, random_state=s) on the remaining patients.
Each split file has one row per patient; round k uses fold == k as validation and the
other development patients as training.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

SPLIT_COLUMNS = ["SampleNumber", "WSI_ID", "label", "split", "fold"]


def make_split(cohort: pd.DataFrame, seed: int, test_fraction: float, n_folds: int) -> pd.DataFrame:
    """Assign every patient to 'test' or to a development fold 0..n_folds-1."""
    df = cohort.sort_values("SampleNumber", ignore_index=True)  # output depends on patients, not row order
    dev_idx, test_idx = train_test_split(np.arange(len(df)), test_size=test_fraction,
                                         stratify=df["label"], random_state=seed)
    fold = pd.Series(pd.NA, index=df.index, dtype="Int64")
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for k, (_, val) in enumerate(skf.split(dev_idx, df["label"].to_numpy()[dev_idx])):
        fold.iloc[dev_idx[val]] = k
    out = df.assign(split="dev", fold=fold)
    out.loc[test_idx, "split"] = "test"
    return out[SPLIT_COLUMNS]


def check_split(df: pd.DataFrame, n_folds: int, test_fraction: float) -> None:
    """Raise if the split breaks any rule of the partitioning protocol."""
    def fail(msg: str) -> None:
        raise AssertionError(f"split check failed: {msg}")

    if df["SampleNumber"].duplicated().any():
        fail("a patient appears more than once")
    test, dev = df[df.split == "test"], df[df.split == "dev"]
    if len(test) + len(dev) != len(df):
        fail("split must be 'test' or 'dev'")
    if test["fold"].notna().any() or dev["fold"].isna().any():
        fail("test rows must have no fold; dev rows must have one")
    if set(dev["fold"]) != set(range(n_folds)):
        fail(f"dev folds are {sorted(set(dev['fold']))}, expected 0..{n_folds - 1}")
    classes = set(df["label"])
    if set(test["label"]) != classes:
        fail(f"classes missing from test: {sorted(classes - set(test['label']))}")
    for k, val in dev.groupby("fold"):
        if set(val["label"]) != classes:
            fail(f"classes missing from validation fold {k}: {sorted(classes - set(val['label']))}")
    if abs(len(test) - test_fraction * len(df)) > 1:
        fail(f"test has {len(test)} patients, expected about {test_fraction * len(df):.1f}")
    sizes = dev["fold"].value_counts()
    if sizes.max() - sizes.min() > 1:
        fail(f"fold sizes are unbalanced: {sizes.sort_index().tolist()}")


def balance_table(df: pd.DataFrame, cohort: pd.DataFrame) -> pd.DataFrame:
    """Class counts, mean age and % female for test, dev and each validation fold."""
    d = df.merge(cohort[["SampleNumber", "age", "sex"]], on="SampleNumber", validate="one_to_one")
    groups = {"test": d[d.split == "test"], "dev": d[d.split == "dev"]}
    groups |= {f"fold_{k}": g for k, g in d[d.split == "dev"].groupby("fold")}
    rows = []
    for name, g in groups.items():
        row = {"set": name, "n": len(g), "mean_age": round(g["age"].mean(), 1),
               "pct_female": round(100 * (g["sex"] == "Female").mean(), 1)}
        row |= g["label"].value_counts().to_dict()
        rows.append(row)
    labels = sorted(d["label"].unique())
    return pd.DataFrame(rows)[["set", "n", *labels, "mean_age", "pct_female"]].fillna(0)


def split_path(splits_dir: Path, seed: int) -> Path:
    return Path(splits_dir) / f"seed_{seed}.csv"


def load_split(splits_dir: Path, seed: int) -> pd.DataFrame:
    """Read a committed split file; the only way later stages should obtain a split."""
    return pd.read_csv(split_path(splits_dir, seed), dtype={"fold": "Int64"})


def fold_ids(df: pd.DataFrame, k: int) -> tuple[list[int], list[int]]:
    """(train, validation) patient IDs for CV round k."""
    dev = df[df.split == "dev"]
    return dev.loc[dev.fold != k, "SampleNumber"].tolist(), dev.loc[dev.fold == k, "SampleNumber"].tolist()


def dev_ids(df: pd.DataFrame) -> list[int]:
    """All development patient IDs (used to refit the selected model)."""
    return df.loc[df.split == "dev", "SampleNumber"].tolist()


def test_ids(df: pd.DataFrame) -> list[int]:
    """Held-out test patient IDs; to be used once, after selection is frozen."""
    return df.loc[df.split == "test", "SampleNumber"].tolist()
