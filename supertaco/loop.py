"""Real training loop: transport attempts supervised by the Token Factory.

Replaces runner.py + simlogs: every attempt runs on a real Colab T4 via the
injected transport; failures are detected from real logs, classified by
Nemotron, patched through the playbook, and retrained within the shared
initial + <= max_retries budget. After healthy logs the judge scores real
base vs fine-tuned answers emitted by the VM; regression feeds back as
EVAL_REGRESSION (spec 2026-09-29-colab-real-loop-design.md).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Optional, Protocol

import yaml

from supertaco.agent.playbook import (  # noqa: F401 # wired in Task 4
    apply_default_fix,
    detect_failure,
)
from supertaco.errors import ConfigurationError

LOSS_RE = re.compile(r"\bloss\s+([0-9]+\.[0-9]+|NaN)\b")


@dataclass
class Event:
    """One thing that happened during a run, in order."""

    type: str
    data: dict[str, Any] = field(default_factory=dict)


OnEvent = Callable[[Event], None]
OnLine = Callable[[str], None]


@dataclass
class AttemptResult:
    logs: str
    artifact: Optional[Path] = None
    error: Optional[str] = None


@dataclass
class AttemptRecord:
    attempt: int
    failure_key: Optional[str]
    verdict: Optional[str]
    loss_points: list[float]
    config_before: dict
    config_after: Optional[dict] = None


@dataclass
class LoopResult:
    success: bool
    attempts: int
    final_config: dict
    configs_written: list[str]
    failure_key: Optional[str]
    error: Optional[str]
    llm_calls: list[dict]
    eval_results: Optional[dict] = None
    attempt_ledger: list[AttemptRecord] = field(default_factory=list)
    artifact: Optional[Path] = None


class AttemptTransport(Protocol):
    def run_attempt(
        self, config: dict, prompts: list[str], on_line: Optional[OnLine] = None
    ) -> AttemptResult: ...


def _emit(on_event: Optional[OnEvent], event_type: str, **data: Any) -> None:
    if on_event is not None:
        on_event(Event(type=event_type, data=data))


def validate_config(config: dict) -> None:
    """Raise ConfigurationError before any run starts (spec 6)."""
    if not isinstance(config, dict) or not config:
        raise ConfigurationError("Config must be a non-empty mapping")
    if config.get("learning_rate") is None:
        raise ConfigurationError("Config is missing 'learning_rate'")


def _write_config(config: dict, failure_key: str, runs_dir: Path) -> str:
    """Write a NEW patched config file (invariant 3: never overwrite)."""
    runs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
    path = runs_dir / f"{timestamp}_{failure_key}_patched.yaml"
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(config, fh, sort_keys=True, allow_unicode=True)
    return str(path)


def extract_loss_points(log_text: str) -> list[float]:
    """Extract loss values in emission order; NaN is preserved as float('nan')."""
    points: list[float] = []
    for match in LOSS_RE.finditer(log_text):
        raw = match.group(1)
        points.append(float("nan") if raw == "NaN" else float(raw))
    return points


def make_llm():
    from supertaco.agent.llm import NemotronClient
    from supertaco.settings import settings

    return NemotronClient(
        base_url=settings.token_factory_base_url,
        api_key=settings.nebius_api_key.get_secret_value(),
    )
