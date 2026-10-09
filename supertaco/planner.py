"""Turn a free-text goal into a validated run plan (spec 2026-10-09-goal-driven-runs)."""

from __future__ import annotations

import json
import math
from typing import Any, Optional

from pydantic import BaseModel, Field

from supertaco.data import generate_dataset
from supertaco.errors import ConfigurationError
from supertaco.eval.harness import DEFAULT_PROMPTS, PROMPT_SUITE_SIZE
from supertaco.gpu.transport import resolve_model


class DatasetSpec(BaseModel):
    """Where training data comes from (G4: rows use the alpaca shape)."""

    kind: str = "hf"  # "hf" | "synthetic"
    hf_id: Optional[str] = None
    max_rows: int = 256
    rows: Optional[list[dict]] = None
    note: Optional[str] = None


class RunPlan(BaseModel):
    """One planned run: knobs + data source + the 5 judge prompts."""

    model: str = "qwen2.5-0.5b"
    learning_rate: float = 2e-4
    batch_size: int = 4
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    num_epochs: int = 1
    chat_template: str = "llama-3"
    dataset: DatasetSpec = Field(
        default_factory=lambda: DatasetSpec(kind="hf", hf_id="yahma/alpaca-cleaned", max_rows=256)
    )
    eval_prompts: list[str] = Field(default_factory=lambda: list(DEFAULT_PROMPTS))
    plan_mode: str = "real"  # "real" | "fallback"


def default_plan() -> RunPlan:
    """Shipped defaults flagged as a planner fallback (G8)."""
    return RunPlan(plan_mode="fallback")


def _extract_json(text: str) -> dict:
    start = text.find("{")
    if start == -1:
        raise ValueError("planner reply contains no JSON object")
    obj, _ = json.JSONDecoder().raw_decode(text[start:])
    if not isinstance(obj, dict):
        raise ValueError("planner reply is not a JSON object")
    return obj


def _dataset_spec(raw: object) -> DatasetSpec:
    """Map the planner's dataset blob (key `id`, int `rows`) to DatasetSpec
    (`hf_id`, `max_rows`).
    """
    if not isinstance(raw, dict):
        raise ValueError("dataset must be an object")
    if raw.get("kind") == "hf":
        hf_id = raw.get("id")
        if not isinstance(hf_id, str) or not hf_id.strip():
            raise ValueError("hf dataset requires an id")
        rows = raw.get("rows")
        if not isinstance(rows, int):
            rows = 256
        return DatasetSpec(kind="hf", hf_id=hf_id.strip(), max_rows=min(max(rows, 50), 512))
    return DatasetSpec(kind="synthetic")


def _finite(value: float, fallback: float) -> float:
    """Replace NaN/inf with a safe default before clamping (G10)."""
    return value if math.isfinite(value) else fallback


def _clamp(plan: RunPlan) -> None:
    """G10: keep planner values inside T4-safe ranges."""
    plan.learning_rate = min(max(_finite(plan.learning_rate, 2e-4), 1e-5), 1e-3)
    plan.batch_size = min(max(plan.batch_size, 1), 16)
    plan.lora_r = min(max(plan.lora_r, 4), 32)
    plan.lora_alpha = min(max(plan.lora_alpha, 8), 64)
    plan.lora_dropout = min(max(_finite(plan.lora_dropout, 0.05), 0.0), 0.3)
    plan.num_epochs = min(max(plan.num_epochs, 1), 3)


def _validate_model(plan: RunPlan) -> None:
    try:
        plan.model = resolve_model(plan.model)
    except ConfigurationError:
        plan.model = "qwen2.5-0.5b"


def _validate_prompts(plan: RunPlan) -> None:
    prompts = plan.eval_prompts
    if len(prompts) != PROMPT_SUITE_SIZE or not all(
        isinstance(p, str) and p.strip() for p in prompts
    ):
        plan.eval_prompts = list(DEFAULT_PROMPTS)


def plan_run(goal: str, llm: Any) -> RunPlan:
    """Ask Nemotron for a run plan; validate/clamp it, falling back to defaults.

    Never raises (G8): any failure yields default_plan() with plan_mode="fallback".
    """
    try:
        blob = _extract_json(str(llm.plan_goal(goal)))
        fields = {
            k: blob[k]
            for k in blob
            if k in RunPlan.model_fields and k not in ("dataset", "plan_mode")
        }
        plan = RunPlan(**fields, plan_mode="real")
        if "dataset" in blob:
            plan.dataset = _dataset_spec(blob["dataset"])
        _validate_model(plan)
        _clamp(plan)
        _validate_prompts(plan)
        return plan
    except Exception:
        return default_plan()


def plan_to_config(plan: RunPlan) -> dict:
    """Loop-compatible config dict: the 8 knobs plus the dataset spec (G3)."""
    cfg = {
        name: getattr(plan, name)
        for name in (
            "model",
            "learning_rate",
            "batch_size",
            "lora_r",
            "lora_alpha",
            "lora_dropout",
            "num_epochs",
            "chat_template",
        )
    }
    cfg["dataset"] = plan.dataset.model_dump(exclude_none=True)
    return cfg


def materialize_dataset(plan: RunPlan, goal: str, llm: Any) -> RunPlan:
    """Fill synthetic rows for a plan, or fall back to the alpaca slice (G8).

    Shared by the CLI `plan` command and the dashboard plan button (DRY).
    """
    if plan.dataset.kind != "synthetic":
        return plan
    rows = generate_dataset(goal, llm)
    if rows:
        plan.dataset.rows = rows
    else:
        plan.dataset = DatasetSpec(
            kind="hf",
            hf_id="yahma/alpaca-cleaned",
            max_rows=256,
            note=(
                "synthetic generation returned fewer than 100 usable rows; "
                "fell back to the alpaca slice"
            ),
        )
    return plan
