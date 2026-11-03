"""Stage 04: write one frozen split file per seed (artifacts/splits/seed_<s>.csv).

Write-once: an existing split file is never overwritten. On a rerun the split is
recomputed and must match the file byte for byte, and the cohort checksum must match
the one recorded when the file was created; otherwise the script stops with an error.

Usage (from the repo root):  uv run python -m scripts.04_make_splits [--config conf/data.yaml]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import sklearn

from src.data.splits import balance_table, check_split, make_split, split_path
from src.utils.config import load_config


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, default=Path("conf/data.yaml"))
    cfg = load_config(ap.parse_args().config)
    sc, out_dir = cfg["split"], Path(cfg["splits_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    cohort_bytes = Path(cfg["cohort_csv"]).read_bytes()
    cohort_sha = hashlib.sha256(cohort_bytes).hexdigest()
    cohort = pd.read_csv(cfg["cohort_csv"])

    for seed in sc["seeds"]:
        df = make_split(cohort, seed, sc["test_fraction"], sc["n_folds"])
        check_split(df, sc["n_folds"], sc["test_fraction"])
        csv_text = df.to_csv(index=False)
        path, meta_path = split_path(out_dir, seed), split_path(out_dir, seed).with_suffix(".json")

        if path.exists():
            recorded = json.loads(meta_path.read_text())["cohort_sha256"]
            if recorded != cohort_sha:
                raise SystemExit(f"seed {seed}: cohort.csv changed since {path} was written; "
                                 "splits are frozen, refusing to continue")
            if path.read_text() != csv_text:
                raise SystemExit(f"seed {seed}: recomputed split differs from {path}; refusing to overwrite")
            print(f"seed {seed}: verified, unchanged ({path})")
            continue

        path.write_text(csv_text)
        sizes = df["split"].value_counts().to_dict() | {
            "folds": df["fold"].value_counts().sort_index().tolist()}
        meta_path.write_text(json.dumps({
            "seed": seed, "test_fraction": sc["test_fraction"], "n_folds": sc["n_folds"],
            "n_patients": len(df), "sizes": sizes, "cohort_csv": cfg["cohort_csv"],
            "cohort_sha256": cohort_sha, "scikit_learn": sklearn.__version__,
        }, indent=2) + "\n")
        balance_table(df, cohort).to_csv(out_dir / f"seed_{seed}_balance.csv", index=False)
        print(f"seed {seed}: wrote {path} | test {sizes['test']}, dev {sizes['dev']}, "
              f"fold sizes {sizes['folds']}")


if __name__ == "__main__":
    main()
