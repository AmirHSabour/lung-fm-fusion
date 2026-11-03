"""Stage 08: one mean-pooled vector per slide per model (artifacts/features/<model>.npz).

Uses no labels. A model is skipped, with a message, until its embeddings exist for every
cohort slide.

Usage (from the repo root):
    uv run python -m scripts.08_slide_features [--models uni2h virchow2]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.embeddings.aggregate import mean_pool
from src.utils.config import load_config


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, default=Path("conf/train.yaml"))
    ap.add_argument("--models", nargs="+")
    args = ap.parse_args()
    cfg, emb = load_config(args.config), load_config("conf/embeddings.yaml")
    wsis = pd.read_csv(emb["cohort_csv"])["WSI_ID"].tolist()
    out_dir = Path(cfg["features_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    for key in args.models or cfg["models"]:
        paths = [Path(emb["out_dir"]) / key / f"{w}.h5" for w in wsis]
        missing = [p.stem for p in paths if not p.is_file()]
        if missing:
            print(f"{key}: skipped, {len(missing)} of {len(wsis)} slides not embedded yet")
            continue
        pooled = [mean_pool(p, cfg["embedding"][key]) for p in paths]
        X = np.stack([v for v, _ in pooled])
        n_tiles = np.array([n for _, n in pooled])
        if not np.isfinite(X).all():
            raise SystemExit(f"{key}: non-finite slide vectors")
        np.savez(out_dir / f"{key}.npz", WSI_ID=np.array(wsis), X=X, n_tiles=n_tiles,
                 fields=np.array(cfg["embedding"][key]))
        print(f"{key}: {X.shape[0]} slides x {X.shape[1]} dims -> {out_dir / f'{key}.npz'}")


if __name__ == "__main__":
    main()
