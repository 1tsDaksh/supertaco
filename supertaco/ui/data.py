"""Data access for the redesigned dashboard pages.

Every number on the Landing and Results pages traces to a schema-v1 run artifact
(the same honesty contract as RESULTS.md): nothing is fabricated here. Showcase
mode reads the committed ``examples/runs`` recordings; normal mode reads
``logs/runs`` with the same fallback.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]


def live_runs_dir() -> Path:
    """Artifact directory for this checkout (cwd-relative so live runs are found)."""
    return Path.cwd() / "logs" / "runs"


def examples_runs_dir() -> Path:
    """Committed recordings shipped with the repo (SUPERTACO_EXAMPLES_RUNS overrides)."""
    override = os.getenv("SUPERTACO_EXAMPLES_RUNS")
    if override:
        return Path(override)
    return REPO_ROOT / "examples" / "runs"


def load_artifacts(runs_dir: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Load schema-v1 artifacts; returns (artifacts, warnings-for-skipped-files).

    Corrupt files and non-run files (e.g. *.timing.json) never become artifacts;
    the caller renders warnings so a broken recording is visible, not silent.
    """
    artifacts: list[dict[str, Any]] = []
    warnings: list[str] = []
    if not runs_dir.is_dir():
        return artifacts, warnings
    for path in sorted(runs_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            warnings.append(f"Skipped unreadable recorded run `{path.name}`: {exc}")
            continue
        if not isinstance(data, dict) or data.get("schema") != 1:
            continue  # non-run files (e.g. *.timing.json) are not recordings
        artifacts.append(data)
    return artifacts, warnings


def collection_dirs(showcase: bool) -> list[Path]:
    """Directories to read for the evidence/results views.

    Normal mode reads only the live ``logs/runs`` artifacts; showcase mode reads
    only the committed recordings. No cross-fallback: bundled examples must not
    masquerade as fresh results (and empty states must stay reachable).
    """
    if showcase:
        return [examples_runs_dir()]
    return [live_runs_dir()]


def load_collection(showcase: bool) -> tuple[list[dict[str, Any]], list[str]]:
    """Artifacts across all collection dirs (live first, then bundled fallback)."""
    artifacts: list[dict[str, Any]] = []
    warnings: list[str] = []
    for directory in collection_dirs(showcase):
        found, skipped = load_artifacts(directory)
        artifacts.extend(found)
        warnings.extend(skipped)
    return artifacts, warnings


def token_totals(artifacts: list[dict[str, Any]]) -> dict[str, int]:
    """Sum llm_calls tokens per model_key (classify/judge/patch), honest zeros."""
    totals: dict[str, int] = {}
    for art in artifacts:
        for call in art.get("llm_calls") or []:
            if not isinstance(call, dict):
                continue
            key = str(call.get("model_key") or "?")
            try:
                tokens = int(call.get("tokens") or 0)
            except (TypeError, ValueError):
                tokens = 0
            totals[key] = totals.get(key, 0) + tokens
    return totals


def sweep_summary(artifacts: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate strip for Landing/Results: counts, avg delta, tokens, wall time."""
    evals = [a.get("eval") or {} for a in artifacts]
    deltas = [e["delta"] for e in evals if isinstance(e.get("delta"), (int, float))]
    improved = sum(1 for d in deltas if d > 0)
    regressed = sum(1 for d in deltas if d < 0)
    flat = sum(1 for d in deltas if d == 0)
    walls = [
        a["wall_clock_s"]
        for a in artifacts
        if isinstance(a.get("wall_clock_s"), (int, float))
    ]
    return {
        "runs": len(artifacts),
        "successful": sum(1 for a in artifacts if a.get("success")),
        "improved": improved,
        "flat": flat,
        "regressed": regressed,
        "avg_delta": (sum(deltas) / len(deltas)) if deltas else None,
        "tokens": token_totals(artifacts),
        "wall_s": sum(walls) if walls else None,
        "judge_backends": sorted({str(a.get("judge_backend") or "direct") for a in artifacts}),
    }


def verify_commands(run_id: str) -> str:
    """Copy-paste receipts tying a run row to the audit log and its artifact."""
    return (
        f'grep -c \'"run_id": "{run_id}"\' logs/llm_calls.jsonl\n'
        f"cat logs/runs/{run_id}.json"
    )


def eval_row(artifact: dict[str, Any]) -> dict[str, Any] | None:
    """Normalized eval block for table rows (None when the run never got judged)."""
    ev = artifact.get("eval") or {}
    if not isinstance(ev.get("delta"), (int, float)):
        return None
    return {
        "run_id": artifact.get("run_id", "?"),
        "goal": (artifact.get("goal") or "").strip(),
        "attempts": artifact.get("attempts", 0),
        "healed": bool(artifact.get("healed")),
        "base": ev.get("base_avg"),
        "ft": ev.get("ft_avg"),
        "delta": ev["delta"],
        "wall_s": artifact.get("wall_clock_s"),
        "judge_backend": artifact.get("judge_backend") or "direct",
    }
