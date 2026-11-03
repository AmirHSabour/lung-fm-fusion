"""Stage 07: embed every kept tile of every cohort slide with one frozen encoder.

Writes artifacts/embeddings/<model>/<WSI_ID>.h5 (tile coordinates + every tile embedding;
see src/embeddings/store.py). Slides already embedded for the same tile list are skipped,
so the stage can be resumed and is never repeated. Slides can be split across GPUs with
--shard/--num-shards; --summarize writes artifacts/embeddings_manifest.csv (tiles, dims,
seconds and GPU per slide and model).

Usage (from the repo root):
    uv run python -m scripts.07_extract_embeddings --model uni2h --device cuda:0 --shard 0 --num-shards 2
    uv run python -m scripts.07_extract_embeddings --summarize
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import h5py
import pandas as pd
import torch

from src.embeddings.extract import embed_slide
from src.embeddings.models import load_genbio_pathfm, load_uni2h, load_virchow2, model_transform
from src.embeddings.store import code_version, is_complete, sha256_file, write_embeddings
from src.utils.config import load_config
from src.utils.gpu import gpu_name

LOADERS = {"uni2h": load_uni2h, "virchow2": load_virchow2, "genbio_pathfm": load_genbio_pathfm}


def shard_slides(cohort_csv: Path, tiles_dir: Path, shard: int, num_shards: int) -> list[str]:
    """Deterministic, load-balanced shard: slides sorted by tile count, dealt round-robin."""
    sizes = {w: sum(1 for _ in open(tiles_dir / f"{w}.csv")) - 1 for w in pd.read_csv(cohort_csv)["WSI_ID"]}
    ordered = sorted(sizes, key=lambda w: (-sizes[w], w))
    return ordered[shard::num_shards]


def extract(cfg: dict, key: str, device: str, shard: int, num_shards: int, slides: list[str] | None) -> None:
    models = load_config(cfg["models_config"])
    spec = models["models"][key]
    stride = load_config("conf/preprocess.yaml")["tiling"]["stride_px"]
    tiles_dir, slides_dir = Path(cfg["tiles_dir"]), Path(cfg["slides_dir"])
    out_dir = Path(cfg["out_dir"]) / key
    slides = slides or shard_slides(Path(cfg["cohort_csv"]), tiles_dir, shard, num_shards)

    todo = []
    for w in slides:
        tiles_csv = tiles_dir / f"{w}.csv"
        n = sum(1 for _ in open(tiles_csv)) - 1
        if not is_complete(out_dir / f"{w}.h5", n, sha256_file(tiles_csv)):
            todo.append(w)
    print(f"{key} on {device}: shard {shard}/{num_shards}, {len(slides)} slides, "
          f"{len(slides) - len(todo)} already done, {len(todo)} to embed", flush=True)
    if not todo:
        return

    gpu_name(device, cfg["gpu_type"])
    torch.cuda.set_device(device)
    model = LOADERS[key](Path(models["root"]) / spec["local_dir"]).to(device)
    transform = model_transform(model)
    base_attrs = dict(model=key, repo_id=spec["repo_id"], revision=spec["revision"] or "unpinned",
                      precision=cfg["precision"], tile_px=cfg["tile_px"], stride_px=stride, mpp=0.5,
                      transform=str(transform), code_version=code_version(),
                      device=torch.cuda.get_device_name(device))
    t_all, n_all = time.time(), 0
    for i, w in enumerate(todo, 1):
        tiles_csv = tiles_dir / f"{w}.csv"
        xy = pd.read_csv(tiles_csv)[["x", "y"]].to_numpy()
        t0 = time.time()
        feats = embed_slide(model, key, transform, slides_dir / f"{w}.tif", xy, tile_px=cfg["tile_px"],
                            batch_size=cfg["batch_size"], workers=cfg["workers"], device=device,
                            precision=cfg["precision"])
        secs = time.time() - t0
        write_embeddings(out_dir / f"{w}.h5", xy, feats,
                         base_attrs | dict(WSI_ID=w, n_tiles=len(xy), seconds=round(secs, 1),
                                           tiles_sha256=sha256_file(tiles_csv)))
        n_all += len(xy)
        el = time.time() - t_all
        print(f"[{i}/{len(todo)}] {key} {w}: {len(xy)} tiles, {secs:.0f}s ({len(xy) / secs:.0f} tiles/s) "
              f"| elapsed {el / 60:.1f} min, mean {n_all / el:.0f} tiles/s", flush=True)
    print(f"{key} shard {shard}: done, {n_all} tiles in {(time.time() - t_all) / 60:.1f} min", flush=True)


def summarize(cfg: dict) -> None:
    rows = []
    for path in sorted(Path(cfg["out_dir"]).glob("*/*.h5")):
        with h5py.File(path, "r") as f:
            a = dict(f.attrs)
            rows.append(dict(model=a["model"], WSI_ID=a["WSI_ID"], n_tiles=a["n_tiles"],
                             dims=" ".join(f"{k}:{f[k].shape[1]}" for k in f if k != "coords"),
                             seconds=a["seconds"], device=a["device"], precision=a["precision"],
                             revision=a["revision"], code_version=a["code_version"]))
    df = pd.DataFrame(rows)
    if df.device.nunique() != 1 or set(df.precision) != {cfg["precision"]}:      # one GPU type, one precision
        raise SystemExit(f"inconsistent embeddings: devices {set(df.device)}, precisions {set(df.precision)}")
    out = Path(cfg["out_dir"]).with_name("embeddings_manifest.csv")
    df.to_csv(out, index=False)
    print(df.groupby("model").agg(slides=("WSI_ID", "size"), tiles=("n_tiles", "sum"),
                                  gpu_hours=("seconds", lambda s: round(s.sum() / 3600, 2))).to_string())
    print(f"-> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, default=Path("conf/embeddings.yaml"))
    ap.add_argument("--model", choices=sorted(LOADERS))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--slides", nargs="+", help="specific WSI_IDs (default: this shard of the cohort)")
    ap.add_argument("--summarize", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.summarize:
        summarize(cfg)
    else:
        extract(cfg, args.model, args.device, args.shard, args.num_shards, args.slides)


if __name__ == "__main__":
    main()
