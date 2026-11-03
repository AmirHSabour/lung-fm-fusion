"""Stage 01: download the pinned CLWD JPG release into ``data/raw/CLWD``.

Usage (from the repo root):
    uv run python -m scripts.01_download_dataset [--workers 16] [--retries 3]

Idempotent and resumable: files already on disk with the size the Hub reports are
skipped. Exits 1 if any file is missing or has the wrong size at the end.
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from huggingface_hub import hf_hub_download

from src.utils.config import load_config
from src.utils.hub import hub_file_sizes, incomplete_files


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, default=Path("conf/data.yaml"))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--retries", type=int, default=3)
    args = ap.parse_args()
    cfg = load_config(args.config)
    repo, rev, root = cfg["dataset"]["repo_id"], cfg["dataset"]["revision"], Path(cfg["images_dir"])
    root.mkdir(parents=True, exist_ok=True)

    sizes = hub_file_sizes(repo, rev, repo_type="dataset")
    todo = incomplete_files(root, sizes)
    total = sum(sizes[f] for f in todo)
    print(f"{repo} @ {rev}: {len(sizes)} files ({sum(sizes.values()) / 1e9:.1f} GB) | "
          f"{len(sizes) - len(todo)} complete | {len(todo)} to download ({total / 1e9:.1f} GB)", flush=True)

    def fetch(name: str) -> None:
        for attempt in range(1, args.retries + 1):
            try:
                hf_hub_download(repo, name, repo_type="dataset", revision=rev, local_dir=str(root))
                if not incomplete_files(root, {name: sizes[name]}):
                    return
                raise OSError("size mismatch after download")
            except Exception as exc:  # noqa: BLE001 - retry, then report
                if attempt == args.retries:
                    raise
                print(f"  retry {attempt}/{args.retries} {name}: {type(exc).__name__}: {exc}", flush=True)
                time.sleep(5 * attempt)

    t0, done = time.time(), 0
    with ThreadPoolExecutor(args.workers) as pool:
        futures = {pool.submit(fetch, f): f for f in todo}
        for fut in as_completed(futures):
            name = futures[fut]
            try:
                fut.result()
                done += sizes[name]
                rate = done / (time.time() - t0) / 1e6
                print(f"{name:14s} {sizes[name] / 1e9:5.2f} GB | {done / 1e9:6.1f}/{total / 1e9:.1f} GB "
                      f"| {rate:5.1f} MB/s", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"FAILED {name}: {type(exc).__name__}: {exc}", flush=True)

    bad = incomplete_files(root, sizes)
    print(f"done in {(time.time() - t0) / 60:.1f} min | complete {len(sizes) - len(bad)}/{len(sizes)}", flush=True)
    for f in bad:
        print("  MISSING/BAD:", f)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
