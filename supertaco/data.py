"""Synthetic training-row generation for goal-driven runs (spec 2026-10-09)."""

from __future__ import annotations

import json
from typing import Any

MAX_ROW_CHARS = 1500


def _extract_array(text: str) -> list:
    """Pull the first JSON array out of a model reply."""
    start = text.find("[")
    if start == -1:
        raise ValueError("generation reply contains no JSON array")
    arr, _ = json.JSONDecoder().raw_decode(text[start:])
    if not isinstance(arr, list):
        raise ValueError("generation reply is not a JSON array")
    return arr


def _clean_row(rec: object) -> dict | None:
    """Validate one generated record into the alpaca shape, or reject it."""
    if not isinstance(rec, dict):
        return None
    instruction = str(rec.get("instruction") or "").strip()
    output = str(rec.get("output") or "").strip()
    if len(instruction) < 3 or len(output) < 3:
        return None
    if len(instruction) + len(output) > MAX_ROW_CHARS:
        return None
    return {"instruction": instruction, "input": str(rec.get("input") or ""), "output": output}


def generate_dataset(
    goal: str,
    llm: Any,
    *,
    target: int = 150,
    chunk: int = 50,
    floor: int = 100,
    max_calls: int = 6,
) -> list[dict]:
    """Generate alpaca-shaped rows for a goal.

    Returns [] when the quality floor is not met after max_calls chunks (G8:
    the caller falls back to the alpaca slice with a flag).
    """
    rows: list[dict] = []
    banned: list[str] = []
    for _ in range(max_calls):
        if len(rows) >= target:
            break
        count = min(chunk, target - len(rows))
        try:
            batch = _extract_array(llm.generate_rows(goal, count, banned))
        except Exception:
            # G8: any generation/parse failure ends chunking; the floor check
            # below decides between returning partial rows or [] so the caller
            # can fall back to the alpaca slice (never raises).
            break
        for rec in batch:
            row = _clean_row(rec)
            if row is None or row["instruction"].lower() in banned:
                continue
            banned.append(row["instruction"].lower())
            rows.append(row)
            if len(rows) >= target:
                break
    return rows if len(rows) >= floor else []
