"""Hugging Face Hub helpers shared by the model and dataset downloaders.

Authentication uses only the standard locations (``HF_TOKEN`` or the Hugging Face
token cache); nothing is written to the repository.
"""
from __future__ import annotations

from fnmatch import fnmatch
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.utils import GatedRepoError, HfHubHTTPError


def check_access(repo_id: str, repo_type: str = "model") -> tuple[bool, str]:
    """Return (ok, message) without downloading anything."""
    try:
        HfApi().auth_check(repo_id, repo_type=repo_type)
        return True, "access OK"
    except GatedRepoError:
        return False, "gated: approval pending. Request access on the Hub page and re-run."
    except HfHubHTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "?"
        if status == 401:
            return False, "not authenticated: run `uv run hf auth login` or set HF_TOKEN."
        return False, f"HTTP {status}: {exc}"


def hub_file_sizes(repo_id: str, revision: str | None, repo_type: str = "model",
                   ignore_patterns: tuple[str, ...] = ()) -> dict[str, int]:
    """File name -> size in bytes, as reported by the Hub for this revision."""
    api = HfApi()
    info = (api.dataset_info if repo_type == "dataset" else api.model_info)(
        repo_id, revision=revision, files_metadata=True)
    return {s.rfilename: (s.size or 0) for s in info.siblings
            if s.rfilename != ".gitattributes" and not any(fnmatch(s.rfilename, p) for p in ignore_patterns)}


def incomplete_files(local_dir: Path, sizes: dict[str, int]) -> list[str]:
    """Files that are missing locally or whose size differs from the Hub's."""
    return [f for f, n in sizes.items()
            if not (local_dir / f).is_file() or (local_dir / f).stat().st_size != n]
