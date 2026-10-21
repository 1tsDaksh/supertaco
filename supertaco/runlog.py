"""Run identity and per-run summary artifacts (WS4 traceability primitive)."""

import contextvars
import json
import os
import uuid
import warnings
from pathlib import Path
from typing import Any

from supertaco.paths import repo_root

_RUN_ID: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "supertaco_run_id", default=None
)


def _logs_dir() -> Path:
    """Audit-log directory, anchored to the repo root (cwd-independent). Test seam."""
    return repo_root() / "logs"


def current_run_id() -> str | None:
    """Resolved run id for audit stamping: contextvar first, then env var, else None."""
    value = _RUN_ID.get()
    if value:
        return value
    env = os.environ.get("SUPERTACO_RUN_ID")
    return env or None


def resolve_run_id(explicit: str | None = None) -> str:
    """Priority: explicit argument, contextvar, env var, else a fresh 8-hex id."""
    for candidate in (explicit, _RUN_ID.get(), os.environ.get("SUPERTACO_RUN_ID")):
        if candidate:
            return candidate
    return uuid.uuid4().hex[:8]


def set_run_id(run_id: str) -> contextvars.Token:
    """Bind run_id to the current context; pass the token to reset_run_id()."""
    return _RUN_ID.set(run_id)


def reset_run_id(token: contextvars.Token) -> None:
    """Restore the contextvar to its previous value."""
    _RUN_ID.reset(token)


def write_run_artifact(run_id: str, data: dict[str, Any]) -> Path:
    """Write logs/runs/<run_id>.json (UTF-8, ensure_ascii=False, indent 2).

    Caller must pass a resolved id (see resolve_run_id); raw ids must not contain
    path separators or "..".

    Returns the intended path. An OSError is warned, never raised (spec 6), matching
    the audit writer's behavior.
    """
    path = _logs_dir() / "runs" / f"{run_id}.json"
    tmp = path.with_suffix(".json.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2, default=str)
            fh.write("\n")
        os.replace(tmp, path)
    except OSError as exc:
        warnings.warn(f"run artifact write failed: {exc}", stacklevel=2)
    return path
