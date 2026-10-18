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
from supertaco.errors import ConfigurationError, JudgeBackendError
from supertaco.paths import repo_root
from supertaco.text import ascii_fold

LOSS_RE = re.compile(r"\bloss\s+([0-9]+\.[0-9]+|NaN)\b", re.I)


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
    """Ledger entry for one attempt: failure key, verdict, losses, config, sources."""

    attempt: int
    failure_key: Optional[str]
    verdict: Optional[str]
    loss_points: list[float]
    config_before: dict
    config_after: Optional[dict] = None
    citations: list[dict] = field(default_factory=list)


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


def _citation_query(failure_key: str | None, text: str) -> str:
    """Build the Tavily query for a failure (WIN_PLAN §5 query shape)."""
    if failure_key is not None:
        from supertaco.agent.playbook import PLAYBOOK

        spec = PLAYBOOK.get(failure_key)
        desc = spec["description"] if spec else ""
        return f"{failure_key}: {desc} pytorch LoRA fine-tuning".strip()
    return f"pytorch fine-tuning crash: {text[:300]}"


def validate_config(config: dict) -> None:
    """Raise ConfigurationError before any run starts (spec 6)."""
    if not isinstance(config, dict) or not config:
        raise ConfigurationError("Config must be a non-empty mapping")
    if config.get("learning_rate") is None:
        raise ConfigurationError("Config is missing 'learning_rate'")


def _resolve_runs_dir(runs_dir: str | Path) -> Path:
    """Anchor a relative runs_dir to the repo root; absolute paths pass through."""
    path = Path(runs_dir)
    return path if path.is_absolute() else repo_root() / path


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
    prompts: Optional[list[str]] = None,
    judge_backend: str = "direct",
    citations: Any = None,
) -> LoopResult:
    """Run attempts until the judge passes or the relaunch budget is spent.

    max_retries counts RELAUNCHES: initial attempt + up to max_retries
    relaunches. Validation failures and transport errors return a failed
    LoopResult (never raise) with an actionable message (spec D9).
    citations: optional CitationClient; None builds one from settings (or skips when no key).
    """
    from supertaco.eval.harness import DEFAULT_PROMPTS, run_eval_suite, split_responses
    from supertaco.gpu.transport import resolve_model

    _emit(on_event, "run_started", max_retries=max_retries, mode="colab_t4")
    llm = llm or make_llm()
    if citations is None:
        from supertaco.agent.citations import client_from_settings

        citations = client_from_settings(getattr(llm, "call_log", None))
    runs_path = _resolve_runs_dir(runs_dir)

    def _failed(
        attempts: int,
        error: str,
        failure_key: Optional[str] = None,
        configs: Optional[list[str]] = None,
        ledger: Optional[list[AttemptRecord]] = None,
        eval_results: Optional[dict] = None,
    ) -> LoopResult:
        safe_config = dict(config) if isinstance(config, dict) else {}
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
            safe_config,
            configs or [],
            failure_key,
            error,
            list(getattr(llm, "call_log", [])),
            eval_results=eval_results,
            attempt_ledger=ledger or [],
        )

    try:
        validate_config(config)
        model_hf = resolve_model(config.get("model"))
    except ConfigurationError as exc:
        return _failed(0, str(exc))

    # fold to ASCII once here: render and judge must share the exact keys the
    # VM will echo (the colab CLI reads scripts with the cp1252 locale codec)
    prompts = [ascii_fold(p) for p in (list(prompts) if prompts else list(DEFAULT_PROMPTS))]
    current = dict(config)
    configs_written: list[str] = []
    ledger: list[AttemptRecord] = []
    citation_cache: dict[str, list[dict]] = {}
    attempt = 0
    attempt_limit = max_retries + 1
    failure_key: Optional[str] = None
    last_eval: Optional[dict] = None

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
            ledger.append(AttemptRecord(attempt, None, None, [], dict(current)))
            return _failed(
                attempt,
                str(exc),
                configs=configs_written,
                ledger=ledger,
                eval_results=last_eval,
            )

        responses = None
        if result.error:
            # `colab exec` exits 0 even when a cell raises (kernel survives), so
            # a hard VM crash (e.g. CUDA OOM) never arrives as a nonzero exit —
            # it surfaces as a download-miss error with the training tail glued
            # on. Route playbook-detectable crashes into the heal path below;
            # unrecognized errors keep the D9 honest fail-fast.
            tail = result.logs or ""
            if not isinstance(tail, str):
                tail = "\n".join(str(line) for line in tail)
            crash_text = f"{result.error}\n{tail}"
            crash_key = detect_failure(crash_text)
            if crash_key is None:
                # D9: unrecognized attempt failure — honest fail, no fallback.
                ledger.append(
                    AttemptRecord(attempt, None, None, extract_loss_points(tail), dict(current))
                )
                return _failed(
                    attempt,
                    result.error,
                    configs=configs_written,
                    ledger=ledger,
                    eval_results=last_eval,
                )
            # crash produced no responses block; the eval branch below is
            # unreachable (failure_key is set), so carry the crash text as
            # train_logs for classify/patch evidence.
            train_logs = crash_text
            failure_key = crash_key
            points = extract_loss_points(crash_text)
        else:
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
                ledger.append(AttemptRecord(attempt, None, None, points, dict(current)))
                return _failed(
                    attempt,
                    "training produced no ###RESPONSES_JSON### block; cannot evaluate",
                    configs=configs_written,
                    ledger=ledger,
                    eval_results=last_eval,
                )
            judge = None
            if judge_backend == "nebius_job":
                from supertaco.eval.nebius_backend import NebiusJudgeBackend

                try:
                    judge = NebiusJudgeBackend(on_event=None).judge
                except JudgeBackendError as exc:
                    _emit(on_event, "judge_fallback", reason=str(exc), stage="setup")
            try:
                try:
                    eval_results = run_eval_suite(
                        prompts, responses["base"], responses["fine_tuned"], llm=llm, judge=judge
                    )
                except JudgeBackendError as exc:
                    _emit(on_event, "judge_fallback", reason=str(exc), stage="job")
                    eval_results = run_eval_suite(
                        prompts, responses["base"], responses["fine_tuned"], llm=llm
                    )
                    eval_results["judge_backend"] = "direct_fallback"
            except KeyError as exc:
                ledger.append(AttemptRecord(attempt, None, None, points, dict(current)))
                return _failed(
                    attempt,
                    f"responses payload missing prompt key: {exc}",
                    configs=configs_written,
                    ledger=ledger,
                    eval_results=last_eval,
                )
            if judge is None and judge_backend == "nebius_job":
                eval_results["judge_backend"] = "direct_fallback"
            for entry in eval_results.pop("call_log", []) or []:
                llm.call_log.append(entry)
            last_eval = eval_results
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
        found: list[dict] = []
        if citations is not None and failure_key is not None:
            if failure_key in citation_cache:
                found = citation_cache[failure_key]
            else:
                query = _citation_query(failure_key, train_logs)
                found = [c.model_dump() for c in citations.search_citations(query)]
                citation_cache[failure_key] = found
                _emit(
                    on_event,
                    "citations_found",
                    failure_key=failure_key,
                    query=query,
                    citations=list(found),
                )
            ledger[-1].citations = list(found)
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
            citations=list(found),
        )
        new_config = apply_default_fix(failure_key, current)
        path = _write_config(new_config, failure_key, runs_path)
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
    return _failed(
        attempt,
        error,
        failure_key=failure_key,
        configs=configs_written,
        ledger=ledger,
        eval_results=last_eval,
    )
