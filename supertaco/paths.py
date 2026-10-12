"""Repo-root anchoring for run artifacts (cwd-independent)."""

from pathlib import Path


def repo_root() -> Path:
    """Absolute path of the repository root (parent of the supertaco package)."""
    return Path(__file__).resolve().parents[1]
