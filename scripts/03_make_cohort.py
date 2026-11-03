"""Stage 03: build the patient cohort (artifacts/cohort.csv) from the slide spreadsheet.

Usage (from the repo root):  uv run python -m scripts.03_make_cohort [--config conf/data.yaml]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data.cohort import build_cohort
from src.utils.config import load_config


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, default=Path("conf/data.yaml"))
    cfg = load_config(ap.parse_args().config)

    cohort, excluded = build_cohort(pd.read_csv(cfg["metadata_csv"]),
                                    pd.read_csv(cfg["slide_scale_csv"]),
                                    Path(cfg["images_dir"]))
    for key, df in (("cohort_csv", cohort), ("cohort_excluded_csv", excluded)):
        Path(cfg[key]).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cfg[key], index=False)

    print(f"cohort: {len(cohort)} patients -> {cfg['cohort_csv']}")
    print(f"excluded: {len(excluded)} patients -> {cfg['cohort_excluded_csv']}")
    print("classes:", cohort["label"].value_counts().to_dict())
    print("patients with conflicting slide labels:", int(cohort["label_conflict"].sum()))


if __name__ == "__main__":
    main()
