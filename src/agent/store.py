"""The agent's record: an append-only JSON-lines file (docs/agent.md).

The first line is a header (seed, models, code version, feature and config checksums).
Every later line is an experiment, a decision or the post-search ablation. Lines are never
edited. The file is the index: IDs, resuming and caching are all derived from it.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

HEADER_KEYS = ("seed", "mode", "models", "code", "features", "config")
CODE_PATHS = ("src", "scripts", "conf", "pyproject.toml", "uv.lock")


def code_version(paths: tuple[str, ...] = CODE_PATHS) -> str:
    """The last commit that changed code or configuration (not documentation), plus '-dirty' if
    those paths have uncommitted changes. Documentation commits therefore never block a resume."""
    import subprocess
    try:
        rev = subprocess.run(["git", "log", "-1", "--format=%H", "--", *paths], capture_output=True, text=True,
                             check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no", "--", *paths],
                               capture_output=True, text=True, check=True).stdout.strip()
        return rev + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def fingerprint(obj: dict) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


class TrialStore:
    def __init__(self, path: Path, seed: int):
        self.path = Path(path)
        self.seed = seed
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def records(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]

    def of_type(self, kind: str) -> list[dict]:
        return [r for r in self.records() if r["type"] == kind]

    def next_id(self, letter: str) -> str:
        n = sum(1 for r in self.records() if r.get("id", "").split("-")[-1].startswith(letter))
        return f"s{self.seed}-{letter}{n + 1:03d}"

    def append(self, record: dict) -> dict:
        with self.path.open("a") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")
            f.flush()
            os.fsync(f.fileno())
        return record

    def open_run(self, header: dict) -> None:
        """Start a new record, or resume one made with the same code, data and config."""
        header = {"type": "header", **{k: header[k] for k in HEADER_KEYS}}
        existing = self.records()
        if not existing:
            self.append(header)
            return
        old = existing[0]
        diff = [k for k in HEADER_KEYS if old.get(k) != header[k]]
        if diff:
            raise SystemExit(f"{self.path} was made with different {diff}; refusing to mix versions. "
                             "Start a new run directory instead.")
