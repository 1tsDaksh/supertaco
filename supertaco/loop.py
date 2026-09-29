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

from supertaco.agent.playbook import (
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
    """Outcome of one transport attempt: logs plus optional artifact or error."""

    logs: str
    artifact: Optional[Path] = None
    error: Optional[str] = None


@dataclass
class AttemptRecord:
    """Ledger entry for one attempt: failure key, verdict, losses, config."""

    attempt: int
    failure_key: Optional[str]
    verdict: Optional[str]
    loss_points: list[float]
    config_before: dict
    config_after: Optional[dict] = None


@dataclass
class LoopResult:
    """Final outcome of a loop run: success, attempts, evidence, and artifacts."""

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
    """Build the Nemotron client used for classification and judging."""
    from supertaco.agent.llm import NemotronClient
    from supertaco.settings import settings

    return NemotronClient(
        base_url=settings.token_factory_base_url,
        api_key=settings.nebius_api_key.get_secret_value(),
    )


def run_training_loop(
    config: dict,
    *,
    transport: AttemptTransport,
    llm: Any = None,
    on_event: Optional[OnEvent] = None,
    on_line: Optional[OnLine] = None,
    max_retries: int = 3,
    runs_dir: str = "configs/runs",
) -> LoopResult:
    """Run attempts until the judge passes or the relaunch budget is spent.

    max_retries counts RELAUNCHES: initial attempt + up to max_retries
    relaunches. Validation failures and transport errors return a failed
    LoopResult (never raise) with an actionable message (spec D9).
    """
    from supertaco.eval.harness import DEFAULT_PROMPTS, run_eval_suite, split_responses
    from supertaco.gpu.transport import resolve_model

    llm = llm or make_llm()
    _emit(on_event, "run_started", max_retries=max_retries, mode="colab_t4")

    def _failed(attempts, error, failure_key=None, configs=None, ledger=None):
        _emit(
            on_event,
            "run_failed",
            error=error,
            attempts=attempts,
            failure_key=failure_key,
            path=(configs or [])[-1] if configs else None,
        )
        return LoopResult(
            False,
            attempts,
            dict(config),
            configs or [],
            failure_key,
            error,
            list(getattr(llm, "call_log", [])),
            attempt_ledger=ledger or [],
        )

    try:
        validate_config(config)
        model_hf = resolve_model(config.get("model"))
    except ConfigurationError as exc:
        return _failed(0, str(exc))

    prompts = list(DEFAULT_PROMPTS)
    current = dict(config)
    configs_written: list[str] = []
    ledger: list[AttemptRecord] = []
    attempt = 0
    attempt_limit = max_retries + 1

    while attempt < attempt_limit:
        attempt += 1
        _emit(
            on_event,
            "job_launched",
            attempt=attempt,
            payload={"config": dict(current), "model": model_hf},
        )
        try:
            result = transport.run_attempt(current, prompts, on_line=on_line)
        except Exception as exc:  # infra problem: honest failure, no fallback (D9)
            return _failed(attempt, str(exc), configs=configs_written, ledger=ledger)

        if result.error:
            return _failed(attempt, result.error, configs=configs_written, ledger=ledger)

        train_logs, responses = split_responses(result.logs)
        points = extract_loss_points(train_logs)
        _emit(
            on_event,
            "logs_produced",
            attempt=attempt,
            text=train_logs,
            loss_points=points,
        )

        failure_key = detect_failure(train_logs)
        if failure_key is None:
            if responses is None:
                return _failed(
                    attempt,
                    "training produced no ###RESPONSES_JSON### block; cannot evaluate",
                    configs=configs_written,
                    ledger=ledger,
                )
            eval_results = run_eval_suite(
                prompts, responses["base"], responses["fine_tuned"], llm=llm
            )
            if not eval_results["regression_flagged"]:
                ledger.append(AttemptRecord(attempt, None, None, points, dict(current)))
                _emit(
                    on_event,
                    "run_succeeded",
                    attempts=attempt,
                    configs_written=list(configs_written),
                )
                return LoopResult(
                    True,
                    attempt,
                    dict(current),
                    configs_written,
                    None,
                    None,
                    list(getattr(llm, "call_log", [])),
                    eval_results=eval_results,
                    attempt_ledger=ledger,
                    artifact=result.artifact,
                )
            failure_key = "EVAL_REGRESSION"
            _emit(
                on_event,
                "failure_detected",
                attempt=attempt,
                failure_key=failure_key,
            )
        else:
            _emit(
                on_event,
                "failure_detected",
                attempt=attempt,
                failure_key=failure_key,
            )

        ledger.append(AttemptRecord(attempt, failure_key, None, points, dict(current)))
        verdict = llm.classify_failure(train_logs)
        last = llm.call_log[-1] if getattr(llm, "call_log", None) else {}
        _emit(
            on_event,
            "classified",
            attempt=attempt,
            failure_key=failure_key,
            nemotron_verdict=verdict,
            mode=last.get("mode", "unknown"),
            diverged=verdict != failure_key,
        )
        ledger[-1].verdict = verdict

        if attempt >= attempt_limit:
            break

        proposal = llm.propose_patch(failure_key, current, train_logs)
        _emit(
            on_event,
            "patch_proposed",
            attempt=attempt,
            failure_key=failure_key,
            proposal=proposal,
        )
        new_config = apply_default_fix(failure_key, current)
        path = _write_config(new_config, failure_key, Path(runs_dir))
        configs_written.append(path)
        _emit(
            on_event,
            "patch_written",
            attempt=attempt,
            path=path,
            failure_key=failure_key,
            before=current,
            after=new_config,
        )
        ledger[-1].config_after = dict(new_config)
        current = new_config
        _emit(on_event, "retry_scheduled", next_attempt=attempt + 1)

    error = f"MaxRetriesExceeded: {max_retries} relaunches exhausted"
    return _failed(attempt, error, failure_key=failure_key, configs=configs_written, ledger=ledger)
