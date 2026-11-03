"""Stage 00: download the pinned foundation-model weights into ``models/``.

Usage (from the repo root):
    uv run python -m scripts.00_download_models [--models uni2h virchow2] [--strict]

Idempotent: complete files are skipped and interrupted downloads resume. Gated
repositories still awaiting approval are reported and skipped (an error only with
``--strict``). Every file is checked against the size the Hub reports.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

from src.utils.config import load_config
from src.utils.hub import check_access, hub_file_sizes, incomplete_files


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, default=Path("conf/models.yaml"))
    ap.add_argument("--models", nargs="+", help="keys from the config (default: all)")
    ap.add_argument("--strict", action="store_true", help="exit 1 if any model is skipped or incomplete")
    args = ap.parse_args()
    cfg = load_config(args.config)
    keys = args.models or list(cfg["models"])

    failures = []
    for key in keys:
        spec = cfg["models"][key]
        local_dir = Path(cfg["root"]) / spec["local_dir"]
        ignore = tuple(spec["ignore_patterns"])
        print(f"=== {spec['repo_id']} @ {spec['revision'] or 'latest'} -> {local_dir}")
        ok, msg = check_access(spec["repo_id"])
        if not ok:
            print(f"  SKIPPED: {msg}")
            failures.append(key)
            continue
        if spec["revision"] is None:
            print("  WARNING: no pinned revision; record the downloaded commit in conf/models.yaml")
        snapshot_download(spec["repo_id"], revision=spec["revision"], local_dir=str(local_dir),
                          ignore_patterns=list(ignore) or None)
        sizes = hub_file_sizes(spec["repo_id"], spec["revision"], ignore_patterns=ignore)
        bad = incomplete_files(local_dir, sizes)
        print(f"  {len(sizes) - len(bad)}/{len(sizes)} files match the Hub sizes"
              + (f"; bad: {bad}" if bad else ""))
        if bad:
            failures.append(key)

    print("incomplete: " + ", ".join(failures) if failures else "all requested models verified")
    return 1 if failures and args.strict else 0


if __name__ == "__main__":
    sys.exit(main())
