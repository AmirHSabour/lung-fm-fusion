"""Configuration loading."""
from __future__ import annotations

from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    """Read a YAML config file into a dict."""
    return yaml.safe_load(Path(path).read_text())
