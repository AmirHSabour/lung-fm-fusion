"""Stage 08b: cache a fixed random subset of tiles per slide for multiple-instance recipes (v04).

For every cohort slide, 1,024 tiles chosen with a generator seeded by the slide number, with each
model's embedding of exactly those tiles. Label-free; reads the stage-07 embeddings only.

Usage: uv run --no-sync python -m scripts.08b_tile_cache [--models uni2h virchow2] [--n 1024]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.embeddings.tile_cache import build


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models", nargs="+", default=["uni2h", "virchow2"])
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--out", type=Path, default=Path("artifacts/features/tiles"))
    args = ap.parse_args()
    wsi = pd.read_csv("artifacts/cohort.csv")["WSI_ID"].tolist()
    build(wsi, args.models, Path("artifacts/embeddings"), args.out, args.n)
    print(f"-> {args.out}: {len(wsi)} slides x {args.n} tiles, models {args.models}")


if __name__ == "__main__":
    main()
