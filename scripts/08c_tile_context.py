"""Stage 08c: spatial-context tokens for the cached tiles (v08, decisions/v08.md).

For the same tiles as stage 08b (same row order), each model's embedding concatenated with the means
of that model's embeddings over the 3×3 and 7×7 grid neighbourhoods (tissue tiles only, from all of
the slide's tiles). Parameter-free; reads the stage-07 embeddings only; CPU.

Usage: uv run --no-sync python -m scripts.08c_tile_context [--models uni2h virchow2 genbio_pathfm]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.embeddings.tile_cache import build_context


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models", nargs="+", default=["uni2h", "virchow2", "genbio_pathfm"])
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--radii", type=int, nargs="+", default=[1, 3])
    ap.add_argument("--tile-cache", type=Path, default=Path("artifacts/features/tiles"))
    ap.add_argument("--out", type=Path, default=Path("artifacts/features/tiles_ctx"))
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    wsi = pd.read_csv("artifacts/cohort.csv")["WSI_ID"].tolist()
    build_context(wsi, args.models, Path("artifacts/embeddings"), args.out, args.n, tuple(args.radii),
                  args.tile_cache, args.workers)
    print(f"-> {args.out}: {len(wsi)} slides x {args.n} tiles, radii {args.radii}, models {args.models}")


if __name__ == "__main__":
    main()
