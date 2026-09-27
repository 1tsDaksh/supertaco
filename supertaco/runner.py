"""Single run path shared by dashboard and CLI. Emits typed events.

Dry-run only: Nebius launch builds a payload with zero network I/O.
Real GPU jobs are handoff Gate 1 and intentionally unimplemented.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Optional

import yaml

from supertaco.agent.playbook import apply_default_fix, detect_failure
from supertaco.agent.simlogs import extract_loss_points, generate_logs
from supertaco.errors import ConfigurationError
from supertaco.nebius.jobs import NebiusJobClient


@dataclass
class Event:
    """One thing that happened during a run, in order."""

    type: str
    data: dict[str, Any] = field(default_factory=dict)


OnEvent = Callable[[Event], None]


@dataclass
class RunResult:
    success: bool
    attempts: int
    final_config: dict
    configs_written: list[str]
    failure_key: Optional[str]
    error: Optional[str]
    llm_calls: list[dict]


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


def _default_llm():
    from supertaco.agent.llm import NemotronClient
    from supertaco.settings import settings

    return NemotronClient(
        base_url=settings.token_factory_base_url,
        api_key=settings.nebius_api_key.get_secret_value(),
    )


def run(
    config: dict,
    *,
    max_retries: int = 3,
    dry_run: bool = True,
    on_event: Optional[OnEvent] = None,
    log_fn: Optional[Callable[[dict], str]] = None,
    llm: Any = None,
    runs_dir: str = "configs/runs",
) -> RunResult:
    """Run monitor -> classify -> patch -> relaunch until healthy or capped.

    max_retries counts RELAUNCHES: initial attempt + up to max_retries
    relaunches; if the final attempt's logs still fail -> run_failed
    (handoff invariant 1). Validation raises ConfigurationError before any
    event; every other failure returns success=False instead of raising.
    """
    validate_config(config)
    log_fn = log_fn or generate_logs
    llm = llm or _default_llm()
    _emit(on_event, "run_started", max_retries=max_retries, dry_run=dry_run)

    if not dry_run:
        error = "Real Nebius jobs are not implemented yet (Gate 1)"
        _emit(on_event, "run_failed", error=error, attempts=0, failure_key=None, path=None)
        return RunResult(False, 0, dict(config), [], None, error, [])

    from supertaco.settings import settings

    job_client = NebiusJobClient(
        base_url=settings.token_factory_base_url,
        api_key=settings.nebius_api_key.get_secret_value(),
        project_id=settings.nebius_project_id,
    )

    current = dict(config)
    configs_written: list[str] = []
    attempt = 0
    attempt_limit = max_retries + 1
    failure_key: Optional[str] = None

    while attempt < attempt_limit:
        attempt += 1
        launch = job_client.launch_job(current, dry_run=True)  # zero network
        _emit(on_event, "job_launched", attempt=attempt, payload=launch["payload"])

        logs = log_fn(current)
        _emit(
            on_event,
            "logs_produced",
            attempt=attempt,
            text=logs,
            loss_points=extract_loss_points(logs),
        )

        failure_key = detect_failure(logs)
        if failure_key is None:
            _emit(
                on_event, "run_succeeded", attempts=attempt, configs_written=list(configs_written)
            )
            return RunResult(
                True, attempt, current, configs_written, None, None, list(llm.call_log)
            )

        _emit(on_event, "failure_detected", attempt=attempt, failure_key=failure_key)

        verdict = llm.classify_failure(logs)
        last = llm.call_log[-1] if llm.call_log else {}
        _emit(
            on_event,
            "classified",
            attempt=attempt,
            failure_key=failure_key,
            nemotron_verdict=verdict,
            mode=last.get("mode", "unknown"),
            diverged=verdict != failure_key,
        )

        if attempt < attempt_limit:
            proposal = llm.propose_patch(failure_key, current, logs)
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
            current = new_config
            _emit(on_event, "retry_scheduled", next_attempt=attempt + 1)

    error = f"MaxRetriesExceeded: {max_retries} relaunches exhausted"
    _emit(
        on_event,
        "run_failed",
        error=error,
        attempts=attempt,
        failure_key=failure_key,
        path=configs_written[-1] if configs_written else None,
    )
    return RunResult(
        False, attempt, current, configs_written, failure_key, error, list(llm.call_log)
    )
