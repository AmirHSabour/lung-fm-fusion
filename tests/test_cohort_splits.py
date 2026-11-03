"""The committed cohort and split files are correct and exactly reproducible.

These tests use only committed files (no images, no network), so anyone can run them.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from src.data.cohort import build_cohort
from src.data.splits import check_split, dev_ids, fold_ids, load_split, make_split
from src.data.splits import test_ids as get_test_ids  # aliased so pytest does not collect it
from src.utils.config import load_config

CFG = load_config("conf/data.yaml")
SPLIT = CFG["split"]
COHORT = pd.read_csv(CFG["cohort_csv"])


def test_cohort_rebuilds_byte_for_byte():
    cohort, excluded = build_cohort(pd.read_csv(CFG["metadata_csv"]), pd.read_csv(CFG["slide_scale_csv"]))
    assert cohort.to_csv(index=False) == Path(CFG["cohort_csv"]).read_text()
    assert excluded.to_csv(index=False) == Path(CFG["cohort_excluded_csv"]).read_text()


def test_cohort_decisions():
    meta = pd.read_csv(CFG["metadata_csv"])
    assert COHORT["SampleNumber"].is_unique and len(COHORT) == 203                 # D4: 6 of 209 dropped
    assert meta["SampleNumber"].nunique() == 209
    row = COHORT.set_index("SampleNumber").loc[8377886]
    assert (row.WSI_ID, row.age) == ("WSI-35", 68)                                  # D10: one patient
    assert COHORT.set_index("SampleNumber").loc[8365118, "WSI_ID"] == "WSI-9"        # D5: numeric, not text
    chosen = COHORT.merge(meta, on="WSI_ID")
    assert (chosen["label"] == chosen["Tumor Subtype"]).all()                       # D11: chosen slide's label
    assert COHORT["label_conflict"].sum() == 5


@pytest.mark.parametrize("seed", SPLIT["seeds"])
def test_split_file_is_valid_and_reproducible(seed):
    path = Path(CFG["splits_dir"]) / f"seed_{seed}.csv"
    committed = load_split(CFG["splits_dir"], seed)
    check_split(committed, SPLIT["n_folds"], SPLIT["test_fraction"])
    regenerated = make_split(COHORT, seed, SPLIT["test_fraction"], SPLIT["n_folds"])
    assert regenerated.to_csv(index=False) == path.read_text()
    meta = json.loads(path.with_suffix(".json").read_text())
    assert meta["cohort_sha256"] == hashlib.sha256(Path(CFG["cohort_csv"]).read_bytes()).hexdigest()


@pytest.mark.parametrize("seed", SPLIT["seeds"])
def test_split_helpers_partition_patients(seed):
    df = load_split(CFG["splits_dir"], seed)
    test, dev = set(get_test_ids(df)), set(dev_ids(df))
    assert not test & dev and test | dev == set(COHORT["SampleNumber"])
    seen_in_val = set()
    for k in range(SPLIT["n_folds"]):
        train, val = map(set, fold_ids(df, k))
        assert not train & val and train | val == dev and not (train | val) & test
        assert not val & seen_in_val
        seen_in_val |= val
    assert seen_in_val == dev
