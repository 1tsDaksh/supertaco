# Colab Real Training Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the simulated loop + single-shot Colab button with one real pipeline: dashboard config → Colab T4 attempts supervised by Token Factory (classify → playbook patch → relaunch ≤3) → judge eval → detailed attempt-ledger report.

**Architecture:** New `supertaco/loop.py` (Approach B) composes *reused* pieces — `playbook.py`, `judges.py`/`harness.py`, timeline rendering — around a transport seam. The real transport (`supertaco/gpu/transport.py`) renders the user's config into `colab/train_lora.py`, runs it on a fresh T4 VM per attempt via the proven adapter, and returns streamed logs + an on-VM-generated `###RESPONSES_JSON###` block that the loop feeds to the Token Factory judge. Simulation (`runner.py`, `simlogs.py`), Nebius stubs, fixtures, and the old panels are deleted.

**Tech Stack:** Python 3.11+ (runs on 3.14), Streamlit, plotly, pytest, ruff, google-colab-cli (Windows venv `~/.venv-colab`), Token Factory Nemotron via `.env`.

**Spec:** `docs/superpowers/specs/2026-09-29-colab-real-loop-design.md` (commit `806218a`). Milestones 1–6 map to Tasks 1–14.

---

## Context the executor must know (zero-context briefing)

- **Shell:** Windows PowerShell 5.1. Use `;` not `&&`. `python -c "..."` is broken — run scripts as files.
- **Gates (run every task):** `python -m pytest tests/ -q`, `python -m ruff check <touched files>`, `python -m ruff format --check <touched files>`. Repo-wide bare ruff has pre-existing debt in files this plan deletes (graph/tools/nebius) — scoped ruff on touched files is the bar; after Task 13 the repo debt is gone.
- **Never commit:** `.env`, `logs/`, `supertaco/logs/`, `supertaco/configs/runs/*_patched.yaml`, `configs/runs/*_patched.yaml`. **Never modify:** `configs/base/`. Rendered attempt scripts land in `configs/runs/*.py` (already gitignored by `configs/runs/*`).
- **PROJECT.md:** every task's commit step appends one line under `## Recent changes (newest first)` (heading at `docs/PROJECT.md:26`), format: `- 2026-09-29: <area>: <change> (plan Task N).` Use ASCII-only anchor lines when inserting (the file has mojibake from a concurrent session).
- **Concurrent session junk:** untracked `NUL` and `typescript` files are NOT ours — never stage them. The deletion `docs/superpowers/specs/2026-09-28-real-gpu-autoheal-design.md` is restored + bannered in Task 14 only.
- **Event catalog frozen at 10 types** (`run_started, job_launched, logs_produced, failure_detected, classified, patch_proposed, patch_written, retry_scheduled, run_succeeded, run_failed`); payload keys may change.
- **Token Factory:** `make_llm()` builds `NemotronClient` from `.env` (`TOKEN_FACTORY_BASE_URL`, `NEBIUS_API_KEY`). Offline → fallback mode recorded in `call_log`; judge marks hash modes. Tests use fakes; only the final live smoke (Task 14) spends tokens.
- **Colab prerequisites (already true on this machine):** authed `~/.venv-colab/Scripts/colab.exe` (probe: `'' | & "$env:USERPROFILE\.venv-colab\Scripts\colab.exe" usage` prints `Current balance:` — no OAuth URL), termios stub written by the adapter.
- **Live evidence:** one real T4 run already succeeded (PROJECT.md 2026-09-29 FIRST FULL LIVE COLAB RUN) — training prints `step N loss X`, ~4–6 min/attempt.
- **Existing test seams:** `AppTest.from_file` + `offline` fixture in `tests/test_dashboard.py`; adapter fake-CLI fixture in `tests/test_gpu_colab.py`.

### File structure (locked)

| File | Responsibility |
|---|---|
| Create `supertaco/loop.py` | `Event`, `LoopResult`, `AttemptRecord`, `AttemptResult`, `AttemptTransport` protocol, `validate_config`, `_write_config`, `extract_loss_points`, `make_llm`, `run_training_loop` |
| Create `supertaco/gpu/transport.py` | `MODEL_WHITELIST`, `resolve_model`, `render_script`, `ColabTransport` |
| Create `configs/defaults/colab_t4.yaml` | tracked sane default config |
| Modify `colab/train_lora.py` | marker regions (`CFG`, `EVAL_PROMPTS`) + phase 2 generation + responses block |
| Modify `supertaco/eval/harness.py` | add `RESPONSES_START/END`, `split_responses`; later delete `build_responses` tiers |
| Modify `supertaco/agent/playbook.py` | real-log phrasings (lowercase `nan`) |
| Modify `supertaco/ui/dashboard.py` | sidebar config/model/knobs, single launch button, `execute_launch` → loop, report section; delete eval/GPU panels |
| Modify `supertaco/cli.py` | `run` → real loop, no `--dry-run` |
| Delete | `supertaco/runner.py`, `supertaco/agent/simlogs.py`, `supertaco/agent/graph.py`, `supertaco/agent/tools.py`, `supertaco/nebius/*`, `scripts/make_broken_configs.py`, `configs/runs/*_broken_model.yaml`, `tests/test_runner.py`, `tests/test_simlogs.py` |
| Rewrite `tests/test_cli.py`, `tests/test_dashboard.py`; edit `tests/test_judge.py`, `tests/test_playbook.py`; Create `tests/test_loop.py`, `tests/test_transport.py` | |

---

### Task 1: `split_responses` marker parser (Milestone 3, isolated first)

**Files:**
- Modify: `supertaco/eval/harness.py` (append)
- Test: `tests/test_harness_responses.py` (new)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_harness_responses.py`:

```python
"""###RESPONSES_JSON### marker extraction (spec 5.4)."""

import json

from supertaco.eval.harness import RESPONSES_END, RESPONSES_START, split_responses


def _block(payload: dict) -> str:
    return f"{RESPONSES_START}\n{json.dumps(payload)}\n{RESPONSES_END}"


PAYLOAD = {"base": {"p": "a"}, "fine_tuned": {"p": "b"}}


def test_split_returns_train_logs_and_payload():
    logs = "step 4 loss 1.92\n" + _block(PAYLOAD) + "\ntraining finished"
    train_logs, parsed = split_responses(logs)
    assert parsed == PAYLOAD
    assert RESPONSES_START not in train_logs
    assert "step 4 loss 1.92" in train_logs
    assert "training finished" in train_logs


def test_no_block_returns_logs_unchanged():
    logs = "step 4 loss 1.92"
    assert split_responses(logs) == (logs, None)


def test_unterminated_block_returns_none_and_keeps_logs():
    logs = f"loss\n{RESPONSES_START}\n{{\"base\": {{}}"
    train_logs, parsed = split_responses(logs)
    assert parsed is None
    assert train_logs == logs


def test_bad_json_returns_none():
    logs = f"loss\n{RESPONSES_START}\nnot-json\n{RESPONSES_END}"
    train_logs, parsed = split_responses(logs)
    assert parsed is None
    assert RESPONSES_START not in train_logs  # block still stripped


def test_non_object_payload_returns_none():
    assert split_responses(f"{RESPONSES_START}\n[1, 2]\n{RESPONSES_END}")[1] is None


def test_payload_keys_must_be_strings():
    assert split_responses(f"{RESPONSES_START}\n{{}}\n{RESPONSES_END}")[1] is None
```

Note: the last test expects `{}` (empty dict) to be **rejected** because it lacks the required `base`/`fine_tuned` string-valued mappings — the contract below enforces that.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_harness_responses.py -q`
Expected: FAIL — `ImportError: cannot import name 'RESPONSES_START'`

- [ ] **Step 3: Implement the parser**

Append to `supertaco/eval/harness.py`:

```python
RESPONSES_START = "###RESPONSES_JSON###"
RESPONSES_END = "###END_RESPONSES_JSON###"


def split_responses(log_text: str) -> tuple[str, dict | None]:
    """Strip the responses marker block from train logs and parse it.

    Tolerant by design: unterminated block, bad JSON, or a payload without
    string-valued `base`/`fine_tuned` dicts returns (logs, None); the block is
    still stripped when the markers are present and terminated.
    """
    start = log_text.find(RESPONSES_START)
    if start < 0:
        return log_text, None
    end = log_text.find(RESPONSES_END, start)
    train_logs = log_text
    if end < 0:
        return train_logs, None
    inner = log_text[start + len(RESPONSES_START) : end].strip()
    train_logs = (log_text[:start] + log_text[end + len(RESPONSES_END) :]).strip()
    try:
        payload = json.loads(inner)
    except (json.JSONDecodeError, ValueError):
        return train_logs, None
    if not isinstance(payload, dict):
        return train_logs, None
    base, ft = payload.get("base"), payload.get("fine_tuned")
    if not isinstance(base, dict) or not isinstance(ft, dict):
        return train_logs, None
    if not base or not ft or not all(isinstance(v, str) for v in base.values()):
        return train_logs, None
    if not all(isinstance(v, str) for v in ft.values()):
        return train_logs, None
    return train_logs, payload
```

Add `import json` to the harness imports (module currently imports only `typing`). Note: empty `base`/`ft` dicts are rejected (`not base`) — matches test_payload_keys_must_be_strings (`{}` fails on `base` missing/empty).

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_harness_responses.py -q`
Expected: 6 passed. Then `python -m pytest tests/ -q` — all pre-existing tests still pass (parser is additive).

- [ ] **Step 5: Commit**

```powershell
git add supertaco/eval/harness.py tests/test_harness_responses.py
git commit -m "feat: split_responses marker parser (real-loop spec 5.4)"
```

---

### Task 2: `loop.py` — Event, results, moved helpers (Milestone 1)

**Files:**
- Create: `supertaco/loop.py`
- Test: `tests/test_loop.py` (new)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_loop.py`:

```python
"""Real training loop: transport seam, events, budget, eval gate."""

import json

import pytest

from supertaco.errors import ConfigurationError
from supertaco.eval.harness import RESPONSES_END, RESPONSES_START
from supertaco.loop import (
    AttemptResult,
    Event,
    LoopResult,
    extract_loss_points,
    run_training_loop,
    validate_config,
)

HEALTHY_LOG = (
    "config: {'model': 'Qwen/Qwen2.5-0.5B-Instruct'}\n"
    "gpu: Tesla T4\n"
    "step 4 loss 1.9218\n"
    "step 8 loss 1.6969\n"
    "step 12 loss 1.3245\n"
    "training finished\n"
)

NAN_LOG = "step 4 loss NaN\nstep 8 loss NaN\ntraining finished\n"


def _block(base: dict, ft: dict) -> str:
    return f"{RESPONSES_START}\n{json.dumps({'base': base, 'fine_tuned': ft})}\n{RESPONSES_END}"


def _answers(score_word: str) -> dict:
    return {p: f"{score_word} answer" for p in ["a", "b", "c", "d", "e"]}


def healthy_responses() -> str:
    # prompts are substituted by the loop; build keys dynamically in transport fakes
    return ""


class FakeTransport:
    """Returns scripted AttemptResults in order; last entry repeats."""

    def __init__(self, results, fail_on=None):
        self.results = list(results)
        self.fail_on = fail_on or []
        self.calls = []

    def run_attempt(self, config, prompts, on_line=None):
        self.calls.append(dict(config))
        if len(self.calls) in self.fail_on:
            raise RuntimeError("transport exploded")
        idx = min(len(self.calls) - 1, len(self.results) - 1)
        res = self.results[idx]
        if callable(res):
            return res(prompts)
        return res


def make_result(logs, responses_payload=None, artifact=None, error=None):
    text = logs
    if responses_payload is not None:
        text += "\n" + _block(responses_payload["base"], responses_payload["fine_tuned"])
    return AttemptResult(logs=text, artifact=artifact, error=error)


def responses_for(prompts, base="base", ft="fine"):
    return {
        "base": {p: f"{base} answer to {p}" for p in prompts},
        "fine_tuned": {p: f"{ft} answer to {p}" for p in prompts},
    }


class FakeLLM:
    """Judge scores alternate base/ft calls; classify/patch are deterministic."""

    def __init__(self, healthy=True):
        self.call_log = []
        self.healthy = healthy
        self._n = 0

    def classify_failure(self, logs):
        return "NAN_LOSS"

    def propose_patch(self, failure_key, config, logs):
        return {"fix": f"fix-{failure_key}", "reason": "because"}

    def _call(self, model_key, prompt, **kw):
        self.call_log.append(
            {"model_key": model_key, "mode": "real", "tokens": 1, "latency": 0.01, "error": None}
        )
        is_base = self._n % 2 == 0
        self._n += 1
        if self.healthy:
            return "9/10" if not is_base else "5/10"
        return "3/10" if not is_base else "9/10"  # ft worse -> regression


def collect_events():
    events = []
    return events, lambda ev: events.append(ev)


# ---------- Task 2: helpers ----------


def test_validate_config_rejects_missing_learning_rate():
    with pytest.raises(ConfigurationError, match="learning_rate"):
        validate_config({"batch_size": 8})


def test_validate_config_rejects_empty():
    with pytest.raises(ConfigurationError):
        validate_config({})


def test_extract_loss_points_from_real_lines():
    pts = extract_loss_points("step 4 loss 1.9218\nstep 8 loss 0.5")
    assert pts == [1.9218, 0.5]


def test_extract_loss_points_preserves_nan():
    pts = extract_loss_points("step 4 loss NaN")
    assert len(pts) == 1 and pts[0] != pts[0]  # NaN
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_loop.py -q`
Expected: FAIL — `ImportError: cannot import name 'AttemptResult' ... from 'supertaco.loop'`

- [ ] **Step 3: Implement `loop.py` skeleton (helpers only)**

Create `supertaco/loop.py`:

```python
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

from supertaco.agent.playbook import apply_default_fix, detect_failure
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
```

(`run_training_loop` lands in Task 3; protocol/`AttemptResult` defined now so Task 1-2 tests compile.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_loop.py -q`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```powershell
git add supertaco/loop.py tests/test_loop.py
git commit -m "feat: loop module skeleton — events, LoopResult, moved validate/_write/extract helpers"
```

---

### Task 3: `run_training_loop` happy path (Milestone 1)

**Files:**
- Modify: `supertaco/loop.py`
- Test: `tests/test_loop.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_loop.py`:

```python
def test_happy_path_first_attempt_success():
    transport = FakeTransport([lambda prompts: make_result(HEALTHY_LOG, responses_for(prompts))])
    llm = FakeLLM()
    events, on_event = collect_events()
    result = run_training_loop(
        {"learning_rate": 2e-4, "model": "qwen2.5-0.5b"},
        transport=transport,
        llm=llm,
        on_event=on_event,
        runs_dir=str(_tmp_runs()),
    )
    assert result.success is True
    assert result.attempts == 1
    assert result.failure_key is None
    assert result.eval_results is not None
    assert result.eval_results["regression_flagged"] is False
    assert len(result.attempt_ledger) == 1
    assert result.attempt_ledger[0].failure_key is None
    types = [e.type for e in events]
    assert types[0] == "run_started"
    assert "job_launched" in types and "logs_produced" in types
    assert types[-1] == "run_succeeded"
    launch = next(e for e in events if e.type == "job_launched")
    assert launch.data["payload"]["model"] == "Qwen/Qwen2.5-0.5B-Instruct"
    assert launch.data["payload"]["config"]["learning_rate"] == 2e-4


def test_transport_error_becomes_run_failed_never_raises():
    transport = FakeTransport([AttemptResult(logs="", error="Colab CLI is not authenticated")])
    events, on_event = collect_events()
    result = run_training_loop(
        {"learning_rate": 2e-4, "model": "qwen2.5-0.5b"},
        transport=transport,
        llm=FakeLLM(),
        on_event=on_event,
    )
    assert result.success is False
    assert "not authenticated" in (result.error or "")
    assert [e.type for e in events][-1] == "run_failed"


def test_invalid_model_fails_before_any_attempt():
    transport = FakeTransport([make_result(HEALTHY_LOG)])
    events, on_event = collect_events()
    result = run_training_loop(
        {"learning_rate": 2e-4, "model": "llama-3-8b"},
        transport=transport,
        llm=FakeLLM(),
        on_event=on_event,
    )
    assert result.success is False
    assert "T4-viable" in (result.error or "")
    assert transport.calls == []
    assert [e.type for e in events][-1] == "run_failed"
```

Add at module top of `tests/test_loop.py` (helper for runs_dir isolation):

```python
from pathlib import Path
import tempfile


def _tmp_runs():
    d = tempfile.mkdtemp(prefix="st_runs_")
    return d
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_loop.py -q`
Expected: 3 FAIL with `TypeError: run_training_loop() missing ...` / ImportError for `run_training_loop`.

- [ ] **Step 3: Implement `run_training_loop` (happy path + guards)**

Append to `supertaco/loop.py`:

```python
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
```

Note `resolve_model` is imported here but implemented in Task 6 — **do Task 6 first if imports fail**; to keep TDD order green, temporarily the import is inside the function so tests only fail when the path executes. For now add Task 6's `resolve_model` stub ONLY if Step 2 shows ImportError on module load (it won't — function-level import). For test_invalid_model_fails_before_any_attempt to pass, implement the real `resolve_model` now (Task 6 shrinks to whitelist tests only):

Create `supertaco/gpu/transport.py` with just (full file grows in Task 8):

```python
"""Colab transport: render config -> script -> run on a fresh T4 VM."""

from __future__ import annotations

from supertaco.errors import ConfigurationError

MODEL_WHITELIST: dict[str, str] = {
    "qwen2.5-0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen2.5-3b": "Qwen/Qwen2.5-3B-Instruct",
    "tinyllama-1.1b": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
}


def resolve_model(name: object) -> str:
    """Map a config model name to a T4-viable HF id or raise before provisioning."""
    if not name or not isinstance(name, str):
        raise ConfigurationError(
            f"model {name!r} is not T4-viable; pick from: "
            f"{', '.join(sorted(MODEL_WHITELIST))}"
        )
    key = name.strip().lower()
    if key in MODEL_WHITELIST:
        return MODEL_WHITELIST[key]
    for hf_id in MODEL_WHITELIST.values():
        if key == hf_id.lower():
            return hf_id
    raise ConfigurationError(
        f"model {name!r} is not T4-viable; pick from: "
        f"{', '.join(sorted(MODEL_WHITELIST))}"
    )
```

Also create `supertaco/gpu/__init__.py` if missing (it exists from Task be35c47 — do not recreate).

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_loop.py -q`
Expected: 7 passed (4 from Task 2 + 3 new).

- [ ] **Step 5: Commit**

```powershell
git add supertaco/loop.py supertaco/gpu/transport.py tests/test_loop.py
git commit -m "feat: run_training_loop happy path + pre-launch model validation"
```

---

### Task 4: failure → classify → patch → retry + budget (Milestone 1)

**Files:**
- Modify: `supertaco/loop.py` (no change expected — behavior already coded; tests pin it)
- Test: `tests/test_loop.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_loop.py`:

```python
def test_nan_failure_patches_and_heals():
    def attempt_fn(prompts):
        if transport.calls and ...  # placeholder replaced below
    transport = FakeTransport(
        [
            make_result(NAN_LOG),
            lambda prompts: make_result(HEALTHY_LOG, responses_for(prompts)),
        ]
    )
    llm = FakeLLM()
    events, on_event = collect_events()
    result = run_training_loop(
        {"learning_rate": 0.1, "batch_size": 8, "model": "qwen2.5-0.5b"},
        transport=transport,
        llm=llm,
        on_event=on_event,
        runs_dir=str(_tmp_runs()),
    )
    assert result.success is True
    assert result.attempts == 2
    types = [e.type for e in events]
    assert types.index("failure_detected") < types.index("patch_written") < types.index(
        "retry_scheduled"
    )
    assert types.count("job_launched") == 2
    assert result.final_config["learning_rate"] == 0.01  # playbook NAN_LOSS: lr / 10
    assert len(result.configs_written) == 1
    rec = result.attempt_ledger[0]
    assert rec.failure_key == "NAN_LOSS"
    assert rec.verdict == "NAN_LOSS"
    assert rec.config_after["learning_rate"] == 0.01


def test_budget_exhausted_returns_run_failed_with_ledger():
    transport = FakeTransport([make_result(NAN_LOG)])  # always NaN
    events, on_event = collect_events()
    result = run_training_loop(
        {"learning_rate": 0.1, "batch_size": 8, "model": "qwen2.5-0.5b"},
        transport=transport,
        llm=FakeLLM(),
        on_event=on_event,
        max_retries=2,
        runs_dir=str(_tmp_runs()),
    )
    assert result.success is False
    assert "MaxRetriesExceeded" in (result.error or "")
    assert result.attempts == 3  # initial + 2 relaunches
    assert len(result.attempt_ledger) == 3
    assert all(r.failure_key == "NAN_LOSS" for r in result.attempt_ledger)
    assert [e.type for e in events][-1] == "run_failed"
    assert len(result.configs_written) == 2  # patched configs feed RELAUNCHES 2 and 3; attempt 3 breaks before patching (review fix: was == 3)
```

- [ ] **Step 2: Run tests to verify they fail/pass**

Run: `python -m pytest tests/test_loop.py -q`
Expected: these may PASS immediately (Task 3 implemented the loop fully). If they pass, that is the pin — mark Step 3 as "verify only" and still record expectations. If `test_nan_failure...` fails on the leftover placeholder line, remove the dead `def attempt_fn` block above (it is not used).

- [ ] **Step 3: Remove the placeholder block if present**

If you included the `def attempt_fn` / `...` lines, delete them so the file contains only the two real tests.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_loop.py -q`
Expected: 9 passed.

- [ ] **Step 5: Commit**

```powershell
git add tests/test_loop.py
git commit -m "test: pin failure->classify->patch->retry loop and shared relaunch budget"
```

---

### Task 5: eval gate — EVAL_REGRESSION heals within budget (Milestone 3)

**Files:**
- Test: `tests/test_loop.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_loop.py`:

```python
def test_eval_regression_triggers_patch_then_heals():
    transport = FakeTransport(
        [
            lambda prompts: make_result(HEALTHY_LOG, responses_for(prompts)),
            lambda prompts: make_result(HEALTHY_LOG, responses_for(prompts)),
        ]
    )
    llm = FakeLLM(healthy=False)  # first eval: ft < base -> regression; second call reuses n
    events, on_event = collect_events()
    result = run_training_loop(
        {"learning_rate": 2e-4, "lora_r": 8, "lora_alpha": 16, "num_epochs": 3,
         "model": "qwen2.5-0.5b"},
        transport=transport,
        llm=llm,
        on_event=on_event,
        runs_dir=str(_tmp_runs()),
    )
    types = [e.type for e in events]
    failure_events = [e for e in events if e.type == "failure_detected"]
    assert failure_events[0].data["failure_key"] == "EVAL_REGRESSION"
    patch = next(e for e in events if e.type == "patch_written")
    assert patch.data["failure_key"] == "EVAL_REGRESSION"
    assert patch.data["after"]["lora_r"] == 8          # max(8, 8//2) floors at 8
    assert patch.data["after"]["lora_alpha"] == 16     # floor at 16
    assert patch.data["after"]["num_epochs"] == 2      # -1 epoch
    assert result.success is True
    assert result.attempts == 2


def test_healthy_but_no_responses_block_is_honest_failure():
    transport = FakeTransport([make_result(HEALTHY_LOG)])  # no block
    events, on_event = collect_events()
    result = run_training_loop(
        {"learning_rate": 2e-4, "model": "qwen2.5-0.5b"},
        transport=transport,
        llm=FakeLLM(),
        on_event=on_event,
    )
    assert result.success is False
    assert "RESPONSES_JSON" in (result.error or "")
    assert [e.type for e in events][-1] == "run_failed"
```

- [ ] **Step 2: Run tests to verify they fail/pass**

Run: `python -m pytest tests/test_loop.py -q`
Expected: both PASS (loop already implements the eval gate) — they are pins. If `test_eval_regression...` fails because `FakeLLM(healthy=False)` alternation gives regression only on even call counts, note: eval call 1 = base(9),ft(3)... with `healthy=False`, `_n=0` base→"9/10", ft→"3/10" (is_base False → returns "3/10"? check FakeLLM: `return "3/10" if not is_base else "9/10"` → base gets 9, ft gets 3 → ft<base → regression ✓). Second eval (attempt 2): `_n` continues 2,3 → base 9, ft 3 → regression AGAIN → attempt 3... but transport has only 2 scripted results (2nd repeats) and budget default 3 allows a 3rd. **The test as written would exhaust differently than asserted.** Fix: make FakeLLM heal after the first eval — add to FakeLLM:

```python
class FakeLLM:
    def __init__(self, healthy=True, heal_after_first_eval=False):
        ...
        self.heal_after_first_eval = heal_after_first_eval
        self._evals = 0

    def _call(self, model_key, prompt, **kw):
        self.call_log.append(...)
        is_base = self._n % 2 == 0
        self._n += 1
        # count full evals: every 10 calls = 5 prompts x 2 -> too clever; instead:
        regressed = not self.healthy and self.heal_after_first_eval and self._n <= 10
        if self.heal_after_first_eval:
            healthy_now = self._n > 10
        else:
            healthy_now = self.healthy
        if healthy_now:
            return "9/10" if not is_base else "5/10"
        return "3/10" if not is_base else "9/10"
```

Use `FakeLLM(heal_after_first_eval=True)` in `test_eval_regression_triggers_patch_then_heals` (first 10 calls = eval 1 regressed; calls 11+ = eval 2 healthy).

- [ ] **Step 3: Update FakeLLM as above and re-run**

Run: `python -m pytest tests/test_loop.py -q`
Expected: 11 passed.

- [ ] **Step 4: Commit**

```powershell
git add tests/test_loop.py
git commit -m "test: pin eval-regression feedback loop (EVAL_REGRESSION shares relaunch budget)"
```

---

### Task 6: whitelist tests for `resolve_model` (Milestone 2)

**Files:**
- Create: `tests/test_transport.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_transport.py`:

```python
"""Config rendering + Colab transport (spec 5.2/5.3)."""

import pytest

from supertaco.errors import ConfigurationError
from supertaco.gpu.transport import MODEL_WHITELIST, resolve_model


def test_whitelist_keys_resolve_to_hf_ids():
    for key, hf in MODEL_WHITELIST.items():
        assert resolve_model(key) == hf
        assert resolve_model(key.upper()) == hf


def test_hf_id_passthrough_is_case_insensitive():
    assert resolve_model("qwen/qwen2.5-1.5b-instruct") == "Qwen/Qwen2.5-1.5B-Instruct"


def test_llama_3_8b_rejected_with_options():
    with pytest.raises(ConfigurationError, match="T4-viable"):
        resolve_model("llama-3-8b")


def test_missing_model_rejected():
    with pytest.raises(ConfigurationError, match="T4-viable"):
        resolve_model(None)
    with pytest.raises(ConfigurationError, match="T4-viable"):
        resolve_model("")
```

- [ ] **Step 2: Run tests to verify they fail/pass**

Run: `python -m pytest tests/test_transport.py -q`
Expected: PASS (resolve_model landed in Task 3) — pins.

- [ ] **Step 3: Commit**

```powershell
git add tests/test_transport.py
git commit -m "test: pin T4 model whitelist + pre-launch rejection"
```

---

### Task 7: template markers + `render_script` (Milestone 2)

**Files:**
- Modify: `colab/train_lora.py`
- Modify: `supertaco/gpu/transport.py`
- Test: `tests/test_transport.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_transport.py`:

```python
import json
from pathlib import Path

from supertaco.gpu.transport import render_script

REPO = Path(__file__).resolve().parents[1]
TEMPLATE = REPO / "colab" / "train_lora.py"


def test_render_script_embeds_config_and_prompts(tmp_path):
    cfg = {
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "learning_rate": 0.0005,
        "batch_size": 2,
        "lora_r": 4,
        "lora_alpha": 8,
        "num_epochs": 2,
    }
    prompts = ["p one", "p two"]
    out = render_script(cfg, prompts, TEMPLATE, tmp_path)
    text = out.read_text(encoding="utf-8")
    assert out.parent == tmp_path
    assert "'model': 'Qwen/Qwen2.5-1.5B-Instruct'" in text
    assert "'learning_rate': 0.0005" in text
    assert "'batch_size': 2" in text
    assert "EVAL_PROMPTS = ['p one', 'p two']" in text
    # template file itself untouched
    assert "SUPER_TACO_CFG" in TEMPLATE.read_text(encoding="utf-8")


def test_render_script_is_deterministic_per_config(tmp_path):
    cfg = {"model": "Qwen/Qwen2.5-0.5B-Instruct", "learning_rate": 2e-4}
    a = render_script(cfg, ["x"], TEMPLATE, tmp_path / "a").read_text(encoding="utf-8")
    b = render_script(cfg, ["x"], TEMPLATE, tmp_path / "b").read_text(encoding="utf-8")
    assert a == b
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_transport.py -q`
Expected: FAIL — `ImportError: cannot import name 'render_script'`

- [ ] **Step 3: Add marker regions to `colab/train_lora.py`**

In `colab/train_lora.py`, inside `main()` right after the heavy imports (`from transformers import (...)` block), wrap the existing CFG dict:

```python
    # >>> SUPER_TACO_CFG >>>
    CFG = {
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "learning_rate": 2e-4,
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "num_epochs": 1,
        "batch_size": 4,
        "grad_accum": 2,
        "max_rows": 256,
    }
    # <<< SUPER_TACO_CFG <<<
    print("config:", CFG, flush=True)
```

Directly after the CFG end marker add:

```python
    # >>> SUPER_TACO_PROMPTS >>>
    EVAL_PROMPTS = [
        "Explain what LoRA fine-tuning is in two sentences.",
        "Write a Python function that loads a Hugging Face dataset.",
        "Summarize why gradient clipping prevents divergence.",
        "How does a chat template differ from a tokenizer?",
        "Give one reason eval scores can regress after training.",
    ]
    # <<< SUPER_TACO_PROMPTS <<<
```

Add `import json` to the module-level imports (top of file, next to `import os`).

- [ ] **Step 4: Implement `render_script`**

Append to `supertaco/gpu/transport.py`:

```python
import pprint
import re
from datetime import UTC, datetime
from pathlib import Path

CFG_START = "# >>> SUPER_TACO_CFG >>>"
CFG_END = "# <<< SUPER_TACO_CFG <<<"
PROMPTS_START = "# >>> SUPER_TACO_PROMPTS >>>"
PROMPTS_END = "# <<< SUPER_TACO_PROMPTS <<<"


def _swap(text: str, start: str, end: str, replacement: str) -> str:
    pattern = re.compile(re.escape(start) + r".*?" + re.escape(end), re.S)
    if not pattern.search(text):
        raise ValueError(f"template markers missing: {start} .. {end}")
    return pattern.sub(lambda _m: replacement, text, count=1)


def render_script(config: dict, prompts: list[str], template: Path, out_dir: Path) -> Path:
    """Render a per-attempt training script with the config and prompts embedded."""
    text = template.read_text(encoding="utf-8")
    cfg_block = CFG_START + "\n    CFG = " + pprint.pformat(dict(config), sort_dicts=True) + "\n" + CFG_END
    text = _swap(text, CFG_START, CFG_END, cfg_block)
    prompts_block = (
        PROMPTS_START + "\n    EVAL_PROMPTS = " + repr(list(prompts)) + "\n" + PROMPTS_END
    )
    text = _swap(text, PROMPTS_START, PROMPTS_END, prompts_block)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
    path = out_dir / f"attempt_{stamp}.py"
    path.write_text(text, encoding="utf-8")
    return path
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_transport.py -q`
Expected: 6 passed. Sanity: `python -c` is broken — instead run `python -m ruff check colab/train_lora.py supertaco/gpu/transport.py` (clean) and confirm the template still compiles: `python -m py_compile colab/train_lora.py`.

- [ ] **Step 6: Commit**

```powershell
git add colab/train_lora.py supertaco/gpu/transport.py tests/test_transport.py
git commit -m "feat: template marker regions + deterministic render_script"
```

---

### Task 8: `ColabTransport` + adapter logs passthrough (Milestone 2/6)

**Files:**
- Modify: `supertaco/gpu/colab.py` (TrainingOutcome gains `logs`)
- Modify: `supertaco/gpu/transport.py` (`ColabTransport`)
- Test: `tests/test_transport.py`, `tests/test_gpu_colab.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_transport.py`:

```python
from supertaco.gpu.transport import ColabTransport


class _Outcome:
    def __init__(self, success=True, artifact=None, error=None, logs=None):
        self.success = success
        self.artifact = artifact
        self.error = error
        self.logs = logs or []


def test_transport_renders_and_calls_adapter(tmp_path, monkeypatch):
    captured = {}

    def fake_run(script, *, output_dir, on_line=None):
        captured["script"] = script
        captured["output_dir"] = output_dir
        if on_line:
            on_line("step 4 loss 1.9218")
        return _Outcome(success=True, artifact=tmp_path / "a.zip", logs=["step 4 loss 1.9218"])

    monkeypatch.setattr("supertaco.gpu.colab.run_training", fake_run)
    t = ColabTransport(script_template=TEMPLATE, output_dir=tmp_path / "out")
    lines = []
    res = t.run_attempt(
        {"model": "qwen2.5-0.5b", "learning_rate": 2e-4}, ["p1", "p2"], on_line=lines.append
    )
    assert lines == ["step 4 loss 1.9218"]
    assert res.error is None
    assert res.artifact == tmp_path / "a.zip"
    assert "step 4 loss 1.9218" in res.logs
    rendered = captured["script"].read_text(encoding="utf-8")
    assert "'model': 'Qwen/Qwen2.5-0.5B-Instruct'" in rendered
    assert "EVAL_PROMPTS = ['p1', 'p2']" in rendered


def test_transport_maps_failed_outcome_to_error(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "supertaco.gpu.colab.run_training",
        lambda script, *, output_dir, on_line=None: _Outcome(
            success=False, error="download failed (training tail: boom)", logs=["boom"]
        ),
    )
    t = ColabTransport(script_template=TEMPLATE, output_dir=tmp_path / "out")
    res = t.run_attempt({"model": "qwen2.5-0.5b", "learning_rate": 2e-4}, ["p"])
    assert "download failed" in res.error
    assert res.artifact is None
```

In `tests/test_gpu_colab.py` add:

```python
def test_outcome_carries_collected_logs(fake_cli, tmp_path):
    script = tmp_path / "t.py"
    script.write_text("print('x')", encoding="utf-8")
    outcome = run_training(
        script, cli=fake_cli, output_dir=tmp_path / "out",
        provision_deadline_s=30, exec_deadline_s=30, download_deadline_s=30,
    )
    assert outcome.success is True
    assert any("step 4 loss" in ln for ln in outcome.logs)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_transport.py tests/test_gpu_colab.py -q`
Expected: FAIL — `ImportError: cannot import name 'ColabTransport'`; `AttributeError: 'TrainingOutcome' object has no attribute 'logs'`.

- [ ] **Step 3: Adapter collects logs**

In `supertaco/gpu/colab.py`:
1. `@dataclass class TrainingOutcome:` — add field `logs: list = field(default_factory=list)` (import `field` from dataclasses).
2. In `run_training`, create `all_lines: list[str] = []` next to `artifact: Optional[Path] = None`.
3. After every `_stream(...)` call (`new`, `exec`, `download`, both `stop`s), add `all_lines.extend(lines)`.
4. Every `return TrainingOutcome(...)` gains `logs=list(all_lines)`:
   - provisioning timeout → `TrainingOutcome(False, error=..., logs=list(all_lines))`
   - new failed → same
   - exec timeout / rc!=0 → same
   - download fail → same
   - success → `TrainingOutcome(True, artifact=artifact, logs=list(all_lines))`

- [ ] **Step 4: Implement `ColabTransport`**

Append to `supertaco/gpu/transport.py`:

```python
from typing import Callable, Optional

from supertaco.loop import AttemptResult


class ColabTransport:
    """One attempt = fresh T4 VM: render -> train+generate -> download -> stop."""

    def __init__(self, script_template: Optional[Path] = None, output_dir: Optional[Path] = None):
        repo = Path(__file__).resolve().parents[2]
        self.script_template = script_template or repo / "colab" / "train_lora.py"
        self.output_dir = output_dir or repo / "colab" / "output"
        self.render_dir = (repo / "configs" / "runs")

    def run_attempt(
        self, config: dict, prompts: list[str], on_line: Optional[Callable[[str], None]] = None
    ) -> AttemptResult:
        from supertaco.gpu.colab import run_training

        resolved = {**config, "model": resolve_model(config.get("model"))}
        script = render_script(resolved, prompts, self.script_template, self.render_dir)
        outcome = run_training(script, output_dir=self.output_dir, on_line=on_line)
        return AttemptResult(
            logs="\n".join(outcome.logs),
            artifact=outcome.artifact,
            error=outcome.error if not outcome.success else None,
        )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_transport.py tests/test_gpu_colab.py tests/test_loop.py -q`
Expected: all passed. Then `python -m ruff check supertaco/gpu tests/test_transport.py` + `python -m ruff format supertaco/gpu tests/test_transport.py tests/test_gpu_colab.py`.

- [ ] **Step 6: Commit**

```powershell
git add supertaco/gpu/colab.py supertaco/gpu/transport.py tests/test_transport.py tests/test_gpu_colab.py
git commit -m "feat: ColabTransport + adapter log passthrough (AttemptResult.logs)"
```

---

### Task 9: VM script phase 2 — on-VM base/ft generation (Milestone 3)

**Files:**
- Modify: `colab/train_lora.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_transport.py`:

```python
def test_template_contains_phase2_generation_and_block(tmp_path):
    text = TEMPLATE.read_text(encoding="utf-8")
    assert "disable_adapter()" in text
    assert "###RESPONSES_JSON###" in text
    assert "###END_RESPONSES_JSON###" in text
    assert 'json.dumps({"base"' in text
    assert "max_new_tokens=150" in text
    assert "do_sample=False" in text
    # responses block prints BEFORE the zip so a later download miss still evals
    assert text.index("###RESPONSES_JSON###") < text.index('shutil.make_archive')
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_transport.py -q`
Expected: FAIL (markers absent).

- [ ] **Step 3: Implement phase 2 in `colab/train_lora.py`**

Inside `main()`, after `trainer.train()` + `print("training finished", ...)` and BEFORE the `out = "/content/lora_adapter"` save block, insert:

```python
    # ---- phase 2: base vs fine-tuned answers for the judge ----
    print("phase 2: generating eval answers", flush=True)
    chat_inputs = [
        tokenizer.apply_chat_template(
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": p},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        for p in EVAL_PROMPTS
    ]

    def _gen(prompt_text: str) -> str:
        enc = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        out_ids = model.generate(
            **enc, max_new_tokens=150, do_sample=False, pad_token_id=tokenizer.pad_token_id
        )
        return tokenizer.decode(out_ids[0][enc["input_ids"].shape[1] :], skip_special_tokens=True).strip()

    base_answers: dict = {}
    with model.disable_adapter():
        for p, t in zip(EVAL_PROMPTS, chat_inputs):
            base_answers[p] = _gen(t)
    ft_answers: dict = {}
    for p, t in zip(EVAL_PROMPTS, chat_inputs):
        ft_answers[p] = _gen(t)
    print("###RESPONSES_JSON###", flush=True)
    print(json.dumps({"base": base_answers, "fine_tuned": ft_answers}), flush=True)
    print("###END_RESPONSES_JSON###", flush=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_transport.py -q` and `python -m py_compile colab/train_lora.py`
Expected: all passed, no syntax errors.

- [ ] **Step 5: Commit**

```powershell
git add colab/train_lora.py tests/test_transport.py
git commit -m "feat: on-VM base/ft answer generation + responses block (spec 5.5)"
```

---

### Task 10: playbook real-log phrasings (Milestone 1)

**Files:**
- Modify: `supertaco/agent/playbook.py`
- Test: `tests/test_playbook.py` (append)
- Create: `tests/fixtures/real_colab_healthy.log`

- [ ] **Step 1: Create the real-log fixture**

Create `tests/fixtures/real_colab_healthy.log` (trimmed from the successful live run, loss 1.92→1.32):

```
config: {'model': 'Qwen/Qwen2.5-0.5B-Instruct', 'learning_rate': 0.0002, 'lora_r': 8}
gpu: Tesla T4
dataset rows: 256
trainable params: 1,081,344 || all params: 495,114,112 || trainable%: 0.2184
[ 4/32 00:01 < 00:16, 1.78 it/s, Epoch 0.12/1]
step 4 loss 1.9218
[ 8/32 00:03 < 00:13, 1.97 it/s, Epoch 0.25/1]
step 8 loss 1.6969
[ 12/32 00:05 < 00:11, 1.78 it/s, Epoch 0.34/1]
step 12 loss 1.7044
[ 16/32 00:07 < 00:08, 1.82 it/s, Epoch 0.47/1]
step 16 loss 1.4154
[ 24/32 00:11 < 00:04, 1.90 it/s, Epoch 0.72/1]
step 24 loss 1.2329
[ 32/32 00:16, Epoch 1/1]
step 32 loss 1.3245
training finished
artifact: /content/lora_adapter.zip
```

- [ ] **Step 2: Write the failing tests**

Append to `tests/test_playbook.py`:

```python
from pathlib import Path

from supertaco.agent.playbook import detect_failure


def test_detects_lowercase_nan_from_torch_trainer():
    log = "step 4 loss nan\nstep 8 loss nan\nTraceback: loss became nan"
    assert detect_failure(log) == "NAN_LOSS"


def test_real_healthy_colab_log_detects_no_failure():
    fixture = Path(__file__).parent / "fixtures" / "real_colab_healthy.log"
    assert detect_failure(fixture.read_text(encoding="utf-8")) is None


def test_real_oom_traceback_detected():
    log = "step 4 loss 1.92\nCUDA out of memory. Tried to allocate 2.00 GiB"
    assert detect_failure(log) == "OOM"
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m pytest tests/test_playbook.py -q`
Expected: `test_detects_lowercase_nan_from_torch_trainer` FAILS (`NaN` capital check); the other two likely already pass — they are pins.

- [ ] **Step 4: Extend NAN_LOSS detection**

In `supertaco/agent/playbook.py`, add `import re` after the ConfigurationError import, and replace the NAN_LOSS detection lambda:

```python
    "NAN_LOSS": {
        "detection": lambda log: any(
            "loss" in line.lower() and re.search(r"\bnan\b", line, re.I)
            for line in log.splitlines()[-20:]
        ),
        "default_fix": "lower_lr_10x",
        "description": "Loss is NaN in first N steps",
    },
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_playbook.py tests/test_simlogs.py -q`
Expected: ALL pass (existing sim-phrased tests still match the broadened pattern; `test_simlogs` still exists until Task 13).

- [ ] **Step 6: Commit**

```powershell
git add supertaco/agent/playbook.py tests/test_playbook.py tests/fixtures/real_colab_healthy.log
git commit -m "feat: playbook detects torch-style lowercase nan; pin real healthy Colab log"
```

---

### Task 11: Dashboard rework (Milestone 4)

**Files:**
- Modify: `supertaco/ui/dashboard.py`
- Rewrite: `tests/test_dashboard.py`

- [ ] **Step 1: Rewrite `tests/test_dashboard.py`**

Full replacement:

```python
"""AppTest: config -> real loop (faked transport + LLM) -> report."""

import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from supertaco.eval.harness import RESPONSES_END, RESPONSES_START

REPO = Path(__file__).resolve().parents[1]
DASHBOARD = REPO / "supertaco" / "ui" / "dashboard.py"

HEALTHY_LOG = (
    "config: {'model': 'Qwen/Qwen2.5-0.5B-Instruct'}\ngpu: Tesla T4\n"
    "step 4 loss 1.9218\nstep 8 loss 1.6969\nstep 12 loss 1.3245\n"
    "step 32 loss 1.3245\ntraining finished\n"
)
NAN_LOG = "step 4 loss NaN\nstep 8 loss NaN\ntraining finished\n"


@pytest.fixture()
def offline(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    import supertaco.settings as settings_mod

    monkeypatch.setattr(
        settings_mod.settings, "token_factory_base_url", "http://127.0.0.1:9", raising=True
    )


class FakeLLM:
    def __init__(self, healthy=True, heal_after_first_eval=False):
        self.call_log = []
        self.healthy = healthy
        self.heal = heal_after_first_eval
        self._n = 0

    def classify_failure(self, logs):
        return "NAN_LOSS"

    def propose_patch(self, failure_key, config, logs):
        return {"fix": f"fix-{failure_key}", "reason": "because"}

    def _call(self, model_key, prompt, **kw):
        self.call_log.append(
            {"model_key": model_key, "mode": "real", "tokens": 1, "latency": 0.01, "error": None}
        )
        is_base = self._n % 2 == 0
        self._n += 1
        healthy_now = (self._n > 10) if self.heal else self.healthy
        if healthy_now:
            return "9/10" if not is_base else "5/10"
        return "3/10" if not is_base else "9/10"


class FakeTransport:
    def __init__(self, results):
        self.results = list(results)
        self.calls = 0

    def run_attempt(self, config, prompts, on_line=None):
        from supertaco.loop import AttemptResult

        self.calls += 1
        idx = min(self.calls - 1, len(self.results) - 1)
        item = self.results[idx]
        if isinstance(item, Exception):
            raise item
        logs, payload, error = item
        text = logs
        if payload:
            block = json.dumps(
                {"base": {p: "base a" for p in prompts}, "fine_tuned": {p: "ft a" for p in prompts}}
            )
            text += f"\n{RESPONSES_START}\n{block}\n{RESPONSES_END}"
        return AttemptResult(logs=text, artifact=None, error=error)


def _patch_run(monkeypatch, transport, llm):
    import supertaco.gpu.transport as transport_mod
    import supertaco.loop as loop_mod

    monkeypatch.setattr(transport_mod.ColabTransport, "run_attempt", transport.run_attempt)
    monkeypatch.setattr(loop_mod, "make_llm", lambda: llm)


@pytest.fixture()
def app(offline):
    at = AppTest.from_file(str(DASHBOARD), default_timeout=60)
    at.run()
    assert not at.exception, f"initial render crashed: {[e.value for e in at.exception]}"
    return at


def _launch(app):
    btn = next(b for b in app.button if "Fine-tune on Colab" in b.label)
    btn.click().run()
    return app


def test_initial_render_no_exceptions(app):
    assert any("Fine-tune on Colab" in b.label for b in app.button)
    assert not any("Load Fixture" in b.label for b in app.button)  # fixtures gone


def test_success_flow_renders_report(app, monkeypatch, tmp_path):
    _patch_run(
        monkeypatch,
        FakeTransport([(HEALTHY_LOG, "ok", None)]),
        FakeLLM(),
    )
    _launch(app)
    assert not app.exception, [e.value for e in app.exception]
    result = app.session_state["run_result"]
    assert result.success is True
    assert result.eval_results is not None
    assert result.attempt_ledger[0].failure_key is None
    assert any("Training Report" in m.value for m in app.markdown)


def test_nan_failure_heals_with_patch(app, monkeypatch):
    _patch_run(
        monkeypatch,
        FakeTransport([(NAN_LOG, None, None), (HEALTHY_LOG, "ok", None)]),
        FakeLLM(),
    )
    _launch(app)
    assert not app.exception, [e.value for e in app.exception]
    result = app.session_state["run_result"]
    assert result.success is True
    assert result.attempts == 2
    types = [e.type for e in app.session_state["events"]]
    assert "patch_written" in types
    assert result.attempt_ledger[0].failure_key == "NAN_LOSS"


def test_budget_exhaustion_shows_error_without_traceback(app, monkeypatch):
    _patch_run(monkeypatch, FakeTransport([(NAN_LOG, None, None)]), FakeLLM())
    _launch(app)
    assert not app.exception, [e.value for e in app.exception]
    assert any("MaxRetriesExceeded" in e.value for e in app.error)
    assert app.session_state["run_result"].success is False


def test_transport_error_shows_actionable_message(app, monkeypatch):
    _patch_run(
        monkeypatch,
        FakeTransport([RuntimeError("Colab CLI is not authenticated. Run: colab usage")]),
        FakeLLM(),
    )
    _launch(app)
    assert not app.exception, [e.value for e in app.exception]
    assert any("not authenticated" in e.value for e in app.error)


def test_invalid_model_blocked_before_launch(app, monkeypatch):
    transport = FakeTransport([(HEALTHY_LOG, "ok", None)])
    _patch_run(monkeypatch, transport, FakeLLM())
    app.session_state["config"] = {"learning_rate": 2e-4, "model": "llama-3-8b"}
    app.run()
    _launch(app)
    assert not app.exception, [e.value for e in app.exception]
    assert any("T4-viable" in e.value for e in app.error)
    assert transport.calls == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_dashboard.py -q`
Expected: FAIL — button label `Fine-tune on Colab` not found (old buttons still there).

- [ ] **Step 3: Rework `supertaco/ui/dashboard.py`**

3a. **Imports (lines ~20-21):** replace

```python
from supertaco.runner import Event, make_llm
from supertaco.runner import run as runner_run
```

with

```python
from supertaco.loop import Event, extract_loss_points, make_llm, run_training_loop
```

3b. **`init_session2` defaults:** add `"cfg_path": "configs/defaults/colab_t4.yaml"` and `"knobs": {}` to the defaults dict.

3c. **`render_sidebar`:** delete the `Environment` selectbox block (lines 108-115), the `Demo` block (lines 135-178), and the `Broken Fixtures` block (lines 180-203). After the `Max Retries` block insert:

```python
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Training Config")
    repo = Path(__file__).resolve().parents[2]
    cfg_path = st.sidebar.text_input(
        "Config YAML path", value=st.session_state.get("cfg_path", "configs/defaults/colab_t4.yaml")
    )
    st.session_state["cfg_path"] = cfg_path
    if st.sidebar.button("Load config"):
        target = Path(cfg_path) if Path(cfg_path).is_absolute() else repo / cfg_path
        try:
            loaded = yaml.safe_load(target.read_text(encoding="utf-8-sig"))
        except Exception as exc:
            st.sidebar.error(f"Could not load {cfg_path}: {exc}")
        else:
            if not isinstance(loaded, dict):
                st.sidebar.error(f"{cfg_path} is not a config mapping")
            else:
                st.session_state["config"] = loaded
                st.sidebar.success(f"Loaded {cfg_path}")

    from supertaco.gpu.transport import MODEL_WHITELIST

    st.sidebar.markdown("### Model (T4-viable)")
    st.session_state["model_sel"] = st.sidebar.selectbox(
        "model", list(MODEL_WHITELIST), index=0, help="Whitelisted HF models that fit a free T4"
    )
    st.sidebar.markdown("### Hyperparameters")
    knobs = st.session_state.setdefault("knobs", {})
    knobs["learning_rate"] = st.sidebar.number_input(
        "learning_rate", value=float(knobs.get("learning_rate", 2e-4)), format="%.6f"
    )
    knobs["batch_size"] = st.sidebar.number_input(
        "batch_size", min_value=1, max_value=64, value=int(knobs.get("batch_size", 4))
    )
    knobs["lora_r"] = st.sidebar.number_input(
        "lora_r", min_value=4, max_value=64, value=int(knobs.get("lora_r", 8))
    )
    knobs["lora_alpha"] = st.sidebar.number_input(
        "lora_alpha", min_value=4, max_value=128, value=int(knobs.get("lora_alpha", 16))
    )
    knobs["num_epochs"] = st.sidebar.number_input(
        "num_epochs", min_value=1, max_value=10, value=int(knobs.get("num_epochs", 1))
    )
```

Return value: keep `{"settings": _settings, "env": env}` — but `env` no longer exists; change return to `{"settings": _settings}` (check `main()` does not use the return — it currently ignores it).

3d. **Delete `render_eval_panel` and `render_gpu_panel` + `_gpu_figure` usage decision:** delete both functions entirely; KEEP `_gpu_figure` (used for the live stream chart).

3e. **`execute_launch`:** replace signature + body:

```python
def execute_launch(config, max_retries, timeline_slot, logs_slot, chart_slot):
    """Run the real loop, streaming events + live loss into the placeholders."""
    from supertaco.gpu.transport import ColabTransport

    events: list[Event] = []
    live_points: list[float] = []

    def on_line(ln: str) -> None:
        logs_slot.code(ln[-500:], language="log")
        pts = extract_loss_points(ln)
        if pts:
            live_points.extend(p for p in pts if p == p)
            if live_points:
                chart_slot.plotly_chart(
                    _gpu_figure(live_points), width="stretch", key="live_loss_chart"
                )

    def on_event(ev: Event) -> None:
        events.append(ev)
        st.session_state["events"] = events  # survive a mid-run crash (spec 6)
        timeline_slot.markdown(_render_events(events))
        if ev.type == "logs_produced":
            logs_slot.code(ev.data["text"], language="log")
            chart_slot.plotly_chart(_loss_figure(events), use_container_width=True)

    result = run_training_loop(
        config,
        transport=ColabTransport(),
        on_event=on_event,
        on_line=on_line,
        llm=make_llm(),
        max_retries=max_retries,
        runs_dir="configs/runs",
    )
    return events, result
```

3f. **`_render_events`:** fix the `job_launched` line (line ~291-295) — replace the dry-run wording and mangled arrows for ALL lines by rewriting the function body:

```python
def _render_events(events: list) -> str:
    """Build the full timeline markdown from collected events."""
    if not events:
        return "No decisions yet - launch a job to see the loop."
    lines = []
    for ev in events:
        d = ev.data
        if ev.type == "job_launched":
            payload = d.get("payload", {})
            lines.append(
                f"**Attempt {d['attempt']}** · 🚀 launch `{payload.get('model', '?')}` on T4 "
                f"({len(payload.get('config', {}))} config keys)"
            )
        elif ev.type == "failure_detected":
            lines.append(f"**Attempt {d['attempt']}** · ❌ detected `{d['failure_key']}`")
        elif ev.type == "classified":
            mark = "⚠️ diverges from playbook" if d["diverged"] else "agrees with playbook"
            lines.append(
                f"**Attempt {d['attempt']}** · 🤖 Nemotron → "
                f"`{d['nemotron_verdict']}` ({d['mode']}, {mark})"
            )
        elif ev.type == "patch_proposed":
            reason = (
                d["proposal"].get("reason", d["proposal"])
                if isinstance(d["proposal"], dict)
                else d["proposal"]
            )
            lines.append(f"**Attempt {d['attempt']}** · 🧩 model reasoning: {reason}")
        elif ev.type == "patch_written":
            before, after = d.get("before", {}), d.get("after", {})
            changed = ", ".join(
                f"{k}: {before.get(k)!r}->{after[k]!r}"
                for k in after
                if before.get(k) != after[k]
            )
            lines.append(
                f"**Attempt {d['attempt']}** · 💾 patched `{d['failure_key']}` → "
                f"`{d['path']}`\n   `{changed}`"
            )
        elif ev.type == "retry_scheduled":
            lines.append(f"🔁 relaunching (attempt {d['next_attempt']})")
        elif ev.type == "run_succeeded":
            lines.append(f"✅ **Healed** after {d['attempts']} attempt(s)")
        elif ev.type == "run_failed":
            detail = f" · attempts {d['attempts']}" if d.get("attempts") is not None else ""
            last = f" · last `{d['path']}`" if d.get("path") else ""
            lines.append(f"🛑 **Failed**: {d['error']}{detail}{last}")
    return "\n\n".join(lines)
```

3g. **Add `render_report`:** new function (placed where `render_eval_panel` was):

```python
def render_report() -> None:
    """Full attempt ledger + judge table + final config + artifact (spec D5)."""
    st.markdown("### 📋 Training Report")
    result = st.session_state.get("run_result")
    if result is None:
        st.info("Run the supervisor loop to see attempts, judge scores and config changes.")
        return

    if result.attempt_ledger:
        st.markdown(
            f"**Attempts used:** {result.attempts} — "
            + ("✅ passed the judge" if result.success else f"❌ {result.error}")
        )
        rows: dict[str, list] = {
            "attempt": [],
            "failure": [],
            "verdict": [],
            "config changes": [],
            "loss points": [],
        }
        for rec in result.attempt_ledger:
            rows["attempt"].append(rec.attempt)
            rows["failure"].append(rec.failure_key or "—")
            rows["verdict"].append(rec.verdict or "—")
            if rec.config_after:
                changed = ", ".join(
                    f"{k}: {rec.config_before.get(k)!r}->{rec.config_after[k]!r}"
                    for k in rec.config_after
                    if rec.config_before.get(k) != rec.config_after[k]
                )
            else:
                changed = "—"
            rows["config changes"].append(changed)
            rows["loss points"].append(len(rec.loss_points))
        st.dataframe(rows, use_container_width=True, hide_index=True)

    results = result.eval_results
    if not results:
        st.caption("No judge evaluation — the run never reached a healthy attempt.")
    else:
        base_avg = sum(results["base_scores"]) / len(results["base_scores"])
        ft_avg = sum(results["fine_tuned_scores"]) / len(results["fine_tuned_scores"])
        col1, col2, col3 = st.columns(3)
        col1.metric("Judge: base", f"{base_avg:.2f}")
        col2.metric("Judge: fine-tuned", f"{ft_avg:.2f}")
        col3.metric("Improvement", f"{ft_avg - base_avg:+.2f}")
        if results["regression_flagged"]:
            st.error("🚨 EVAL REGRESSION: fine-tuned is >10% below baseline.")
        else:
            st.caption("✅ Regression check passed (fine-tuned ≥ 90% of base).")
        if any("hash" in m for m in results["modes"]):
            st.warning(
                "⚠ Judge fell back to hash mode (LLM unavailable) — "
                "scores are deterministic placeholders."
            )
        st.dataframe(
            {
                "prompt": results["prompts"],
                "base": results["base_scores"],
                "fine_tuned": results["fine_tuned_scores"],
                "judge": results["modes"],
            },
            use_container_width=True,
            hide_index=True,
        )

    if result.success:
        st.markdown("**Final config**")
        st.code(yaml.dump(result.final_config, sort_keys=True), language="yaml")
        if result.artifact and Path(result.artifact).exists():
            artifact = Path(result.artifact)
            st.download_button(
                f"⬇️ Download {artifact.name}",
                data=artifact.read_bytes(),
                file_name=artifact.name,
                mime="application/zip",
                key="report_download_btn",
            )
```

3h. **`main()`:** replace the Launch Job container's checkbox with model/knob merge, drop `dry_run=dry_run_on`, drop eval/gpu panel calls:

```python
    with st.container(border=True):
        st.markdown("### 🚦 Fine-tune on Colab")
        cfg = render_config_editor(st.session_state["config"])
        knobs = st.session_state.get("knobs") or {}
        model_sel = st.session_state.get("model_sel")
        cfg_for_run = {**cfg, **knobs}
        if model_sel:
            cfg_for_run["model"] = model_sel
        run_clicked = st.button(
            "🚀 Fine-tune on Colab (supervisor loop)", type="primary", use_container_width=True
        )
```

In the launch handler: `execute_launch(cfg_for_run, st.session_state.get("max_retries", 3), timeline_slot, logs_slot, chart_slot)` — drop the `dry_run=` kwarg. Toast text: `st.toast("✅ Run complete", icon="✅")`.

Bottom of `main()`:

```python
    st.markdown("---")
    render_report()
```

(delete the `render_eval_panel()` and `render_gpu_panel()` calls).

3i. **Create `configs/defaults/colab_t4.yaml`:**

```yaml
# Sane default for a free Colab T4 (supervisor loop spec 2026-09-29)
model: qwen2.5-0.5b
learning_rate: 0.0002
batch_size: 4
lora_r: 8
lora_alpha: 16
lora_dropout: 0.05
num_epochs: 1
chat_template: llama-3
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_dashboard.py -q`
Expected: 6 passed. Then `python -m ruff check supertaco/ui/dashboard.py tests/test_dashboard.py` + `python -m ruff format supertaco/ui/dashboard.py tests/test_dashboard.py` — fix any line-length fallout.

- [ ] **Step 5: Full suite + commit**

Run: `python -m pytest tests/ -q` — old `test_runner`/`test_simlogs`/`test_cli` still pass (runner untouched until Task 13); `test_judge` still passes (build_responses still present).

```powershell
git add supertaco/ui/dashboard.py tests/test_dashboard.py configs/defaults/colab_t4.yaml
git commit -m "feat: dashboard single real flow — config+knoobs+model, loop button, attempt-ledger report"
```

---

### Task 12: CLI runs the real loop (Milestone 4)

**Files:**
- Modify: `supertaco/cli.py`
- Rewrite: `tests/test_cli.py`
- Modify: `docs/PROJECT.md` (one-line entry)

- [ ] **Step 1: Rewrite `tests/test_cli.py`**

```python
"""CLI: run executes the real loop with an injected transport (tests fake it)."""

import json

import pytest
import yaml

from supertaco import cli
from supertaco.eval.harness import RESPONSES_END, RESPONSES_START
from supertaco.loop import AttemptResult

HEALTHY_LOG = (
    "config: {'model': 'Qwen/Qwen2.5-0.5B-Instruct'}\n"
    "step 4 loss 1.9218\n"
    "step 8 loss 1.6969\n"
    "training finished\n"
)


def _responses_block(prompts):
    payload = {
        "base": {p: f"base answer to {p}" for p in prompts},
        "fine_tuned": {p: f"ft answer to {p}" for p in prompts},
    }
    return f"\n{RESPONSES_START}\n{json.dumps(payload)}\n{RESPONSES_END}"


class FakeTransport:
    def __init__(self, logs=HEALTHY_LOG, error=None):
        self.logs = logs
        self.error = error
        self.calls = 0

    def run_attempt(self, config, prompts, on_line=None):
        self.calls += 1
        text = self.logs
        if self.error is None:
            text += _responses_block(prompts)
        return AttemptResult(logs=text, artifact="adapter.zip", error=self.error)


class FakeLLM:
    """Healthy judge: base 5/10, ft 9/10 on alternating _call slots."""

    def __init__(self):
        self.call_log = []
        self._n = 0

    def classify_failure(self, logs):
        return "NAN_LOSS"

    def propose_patch(self, failure_key, config, logs):
        return {"fix": f"fix-{failure_key}", "reason": "because"}

    def _call(self, model_key, prompt, **kw):
        self.call_log.append(
            {"model_key": model_key, "mode": "real", "tokens": 1, "latency": 0.01, "error": None}
        )
        is_base = self._n % 2 == 0
        self._n += 1
        return "5/10" if is_base else "9/10"


def _write_cfg(tmp_path, **overrides):
    cfg = {"model": "qwen2.5-0.5b", "learning_rate": 2e-4, "batch_size": 8}
    cfg.update(overrides)
    path = tmp_path / "ok.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


def test_run_config_success_prints_events(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)  # patched configs land in tmp cwd
    transport = FakeTransport()
    monkeypatch.setattr(cli, "_make_transport", lambda: transport)

    cli.run_config(str(_write_cfg(tmp_path)), llm=FakeLLM())
    out = capsys.readouterr().out
    assert "[run_started]" in out
    assert "[run_succeeded]" in out
    assert "Success after 1 attempt(s)" in out
    assert transport.calls == 1


def test_missing_config_exits_cleanly(tmp_path):
    with pytest.raises(SystemExit) as exc:
        cli.run_config(str(tmp_path / "nope.yaml"))
    assert "Error loading" in str(exc.value)


def test_config_without_learning_rate_exits_cleanly(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "_make_transport", lambda: FakeTransport())
    bad = tmp_path / "bad.yaml"
    bad.write_text("batch_size: 8\n", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        cli.run_config(str(bad), llm=FakeLLM())
    assert exc.value.code == 1
    assert "learning_rate" in capsys.readouterr().out


def test_transport_failure_exits_one(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    transport = FakeTransport(logs="", error="provisioning timeout")
    monkeypatch.setattr(cli, "_make_transport", lambda: transport)
    with pytest.raises(SystemExit) as exc:
        cli.run_config(str(_write_cfg(tmp_path)), llm=FakeLLM())
    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert "[run_failed]" in out
    assert "Run failed:" in out
    assert transport.calls >= 1


def test_dry_run_flag_is_gone(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["run", "x.yaml", "--dry-run"])
    assert exc.value.code == 2
    assert "unrecognized arguments" in capsys.readouterr().err
```

Notes for the executor:
- The `_call` alternation mirrors `run_eval_suite`'s strict base→ft ordering per prompt (`supertaco/eval/harness.py:151-157`), so healthy judge = base 5 / ft 9 → no regression flag.
- `run_config` loads the file before touching the transport, so `test_missing_config` never constructs one.
- `test_transport_failure_exits_one` pins only the exit contract, not whether the loop fails fast or spends the relaunch budget on infra errors — tighten to the actual `LoopResult.error` wording if it proves stable.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_cli.py -q`
Expected: FAIL — old `cli.py` has no `_make_transport` (AttributeError through monkeypatch) and argparse still accepts `--dry-run`.

- [ ] **Step 3: Rewrite `supertaco/cli.py`**

Replace the whole file:

```python
"""SuperTaco CLI entry point.

Usage:
    supertaco run <config>
    supertaco --help
"""

from __future__ import annotations

import argparse
import json
import sys

import yaml

from supertaco.errors import ConfigurationError
from supertaco.loop import make_llm, run_training_loop


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="SuperTaco: AI-supervised fine-tuning sandbox")
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser(
        "run", help="Fine-tune on a Colab T4 with the supervisor loop"
    )
    run_parser.add_argument("config", help="Path to YAML config file")

    args = parser.parse_args(argv)
    if args.command == "run":
        run_config(args.config)
    else:
        parser.print_help()
        sys.exit(0)


def _make_transport():
    from supertaco.gpu.transport import ColabTransport

    return ColabTransport()


def run_config(config_path: str, llm=None, transport=None) -> None:
    """Run the real supervisor loop for a config file.

    llm/transport default to the Token Factory client and the Colab
    transport; tests inject fakes through these parameters or by
    monkeypatching _make_transport.
    """
    try:
        config = _load_config(config_path)
    except (OSError, yaml.YAMLError) as exc:
        raise SystemExit(f"Error loading {config_path}: {exc}") from exc
    print(f"SuperTaco: running {config_path}")
    llm = llm if llm is not None else make_llm()
    transport = transport if transport is not None else _make_transport()

    def printer(event) -> None:
        print(f"[{event.type}] {json.dumps(event.data, default=str)}")

    try:
        result = run_training_loop(config, transport=transport, llm=llm, on_event=printer)
    except ConfigurationError as exc:
        raise SystemExit(str(exc)) from exc
    if result.success:
        print(f"Success after {result.attempts} attempt(s). Configs: {result.configs_written}")
    else:
        print(f"Run failed: {result.error}")
        sys.exit(1)


def _load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    return config or {}


if __name__ == "__main__":
    main()
```

Decisions baked in: no `--dry-run` anywhere (spec D9 — nothing fake remains); `max_retries` uses the loop default (3 relaunches); `make_llm` comes from `supertaco.loop` (Task 2), not the doomed `runner.py`; `_make_transport` is the seam tests monkeypatch.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_cli.py -q`
Expected: 5 passed.

If `test_run_config_success_prints_events` fails on eval/ledger assertions, the loop's `run_eval_suite` wiring is wrong — fix `loop.py`, not the test.

- [ ] **Step 5: Gates + commit**

```powershell
python -m ruff check supertaco/cli.py tests/test_cli.py
python -m ruff format --check supertaco/cli.py tests/test_cli.py
python -m pytest tests/ -q
```
Expected: ruff clean on both files; full suite still green (`test_runner.py` survives until Task 13).

Append to `docs/PROJECT.md` under `## Recent changes (newest first)`:
`- 2026-09-29: cli: run executes the real supervisor loop; --dry-run removed (plan Task 12).`

```powershell
git add supertaco/cli.py tests/test_cli.py docs/PROJECT.md
git commit -m "cli: run executes the real loop (no --dry-run); transport/llm seams for tests"
```

---

### Task 13: Delete simulation + Nebius machinery (Milestone 5)

**Files:**
- Delete: `supertaco/runner.py`, `supertaco/agent/simlogs.py`, `supertaco/agent/graph.py`, `supertaco/agent/tools.py`, `supertaco/nebius/`, `scripts/make_broken_configs.py`, `configs/runs/*_broken_model.yaml`, `tests/test_runner.py`, `tests/test_simlogs.py`
- Modify: `supertaco/eval/harness.py`, `tests/test_judge.py`, `docs/PROJECT.md`

- [ ] **Step 1: Delete the dead modules and fixtures**

```powershell
git rm supertaco/runner.py supertaco/agent/simlogs.py supertaco/agent/graph.py supertaco/agent/tools.py
git rm -r supertaco/nebius
git rm scripts/make_broken_configs.py
git rm tests/test_runner.py tests/test_simlogs.py
git rm "configs/runs/*_broken_model.yaml"
```
(8 broken fixtures under `configs/runs/` are the only tracked files there; the wildcard is a git pathspec.)

Prove no importer survived:
```powershell
python -m ruff check .
```
Expected: no F401/F821 against deleted modules. Known importers already fixed by earlier tasks: `cli.py` (Task 12), `dashboard.py` + `test_dashboard.py` (Task 11), `harness.py` loses its `simlogs` import in Step 2 below (run ruff after Step 2 for a clean result). `supertaco/agent/__init__.py` and `supertaco/__init__.py` import nothing — safe.

- [ ] **Step 2: Strip `build_responses` from `supertaco/eval/harness.py`**

Delete: `BASE_ANSWERS`, `HEALTHY_ANSWERS`, `UNHEALTHY_ANSWERS`, and `build_responses` including its `from supertaco.agent.simlogs import classify_config_failure` import. Keep `DEFAULT_PROMPTS`, `PROMPT_SUITE_SIZE`, `run_eval_suite`, `check_regression`, plus Task 1's `RESPONSES_START/END`/`split_responses`.

- [ ] **Step 3: Fix `tests/test_judge.py`**

1. Import line: drop `build_responses` → `from supertaco.eval.harness import DEFAULT_PROMPTS, run_eval_suite`.
2. Delete `test_build_responses_shapes_and_health_dependence` and `test_build_responses_answers_are_distinct_and_nonempty` entirely.
3. In `test_run_eval_suite_regression_flag` and `test_suite_without_llm_reports_hash_modes`, replace the `base, ft = build_responses(...)` line with:

```python
    base = {p: f"base answer to {p}" for p in DEFAULT_PROMPTS}
    ft = {p: f"ft answer to {p}" for p in DEFAULT_PROMPTS}
```

(`test_suite_caps_at_five_prompts` already builds its dicts inline — untouched.)

- [ ] **Step 4: Run gates**

```powershell
python -m pytest tests/ -q
python -m ruff check .
python -m ruff format --check supertaco tests scripts
```
Expected: full suite green without `test_runner.py`/`test_simlogs.py`; repo-wide ruff check clean (the debt lived only in files this step deletes). If format findings remain in files this plan never touched, record them in `docs/PROJECT.md` — do not drive-by fix (rule 5).

- [ ] **Step 5: Commit**

Append to `docs/PROJECT.md`:
`- 2026-09-29: cleanup: delete simulated runner, Nebius stubs, broken fixtures; harness loses build_responses (plan Task 13).`

```powershell
git add supertaco/eval/harness.py tests/test_judge.py docs/PROJECT.md
git commit -m "chore: remove simulation loop, Nebius stubs, broken fixtures; harness loses build_responses"
```
Deletions are already staged by `git rm`. **Never** use `git add -A` / `git commit -a` here: the concurrent session's unstaged spec deletion and the junk `NUL`/`typescript` files must stay out.

---

### Task 14: Docs, banners, final gates + live smoke (Milestone 6)

**Files:**
- Modify: `docs/AGENTS.md`, `docs/PROJECT.md`
- Restore + banner: `docs/superpowers/specs/2026-09-28-real-gpu-autoheal-design.md`
- Modify: `docs/superpowers/plans/2026-09-28-real-gpu-autoheal.md`

- [ ] **Step 1: Restore + banner the old spec**

The concurrent session deleted it from the worktree (unstaged). Restore, then prepend this banner:

```markdown
> **SUPERSEDED (2026-09-29):** This Nebius autoheal design was never executed; the approved
> replacement is `docs/superpowers/specs/2026-09-29-colab-real-loop-design.md`.
```

```powershell
git restore docs/superpowers/specs/2026-09-28-real-gpu-autoheal-design.md
```

- [ ] **Step 2: Banner the old plan**

Prepend to `docs/superpowers/plans/2026-09-28-real-gpu-autoheal.md`:

```markdown
> **ABANDONED (2026-09-29):** Superseded by `docs/superpowers/plans/2026-09-29-colab-real-loop.md`.
```

- [ ] **Step 3: Rewrite AGENTS.md rule 2 + CLI examples**

Rule 2 promises a Nebius `--dry-run` that no longer exists. Replace rule 2 with:

```markdown
2. **Never spend real resources without asking.** Colab T4 runs are free but take ~5-15 min each;
   Token Factory judge calls are the only metered resource. Tests fake the transport and the LLM;
   only a documented, user-approved live smoke touches either.
```

In "Testing & validation commands", replace the two `supertaco.cli run ... --dry-run` example lines with one:

```bash
# Real supervisor loop on the default T4 config (free GPU, ~15 min, spends judge tokens - ask first)
python -m supertaco.cli run configs/defaults/colab_t4.yaml
```

Leave the rest of AGENTS.md alone (the mission text still frames Nebius jobs — stale but out of scope; note that in PROJECT.md).

- [ ] **Step 4: Final gates**

```powershell
python -m pytest tests/ -q
python -m ruff check .
python -m ruff format --check supertaco tests scripts
```
Expected: all green, zero scoped exclusions. Record any residual format findings in untouched files in `docs/PROJECT.md` rather than fixing them.

- [ ] **Step 5: ⏸ APPROVAL GATE — live end-to-end smoke**

Costs judge tokens and ~10-20 min of a free T4. Ask the user first: *"Run `python -m supertaco.cli run configs/defaults/colab_t4.yaml` live? (~15 min, judge tokens only)."* Record yes + timestamp in `docs/PROJECT.md`.

Success criteria:
- exit code 0; stdout shows `run_started → job_launched → logs_produced → ... → run_succeeded` with streamed `step N loss X` lines;
- `Success after N attempt(s)` printed; patched configs under `configs/runs/` only if a failure actually occurred;
- eval results show judge modes `base:real / ft:real` (or a documented hash/fallback mode if `.env` is unavailable — never silently);
- adapter stopped the VM afterwards (no session left running).

If it fails: debug through the loop's own error path first (that is the product working); patch code only for genuine code bugs, then rerun the same smoke.

- [ ] **Step 6: PROJECT.md + commit**

Append (adjust wording to what actually happened):
`- 2026-09-29: docs: superseded banners, AGENTS rule 2 rewritten for Colab; live loop smoke passed (plan Task 14).`

```powershell
git add docs/AGENTS.md docs/PROJECT.md docs/superpowers/specs/2026-09-28-real-gpu-autoheal-design.md docs/superpowers/plans/2026-09-28-real-gpu-autoheal.md
git commit -m "docs: supersede nebius autoheal spec/plan; AGENTS rule 2 matches colab reality; live smoke record"
```

---

## Self-review (writing-plans checklist)

- **Spec coverage:** D1/D2 → Tasks 3+6; D3 cleanup → Task 13; D4 judge real-only → Tasks 3/11 (`make_llm`), no fallbacks added anywhere; D5 attempt-ledger report → Task 11 `render_report`; D6 (Approach B) → whole structure; D7 on-VM responses → Tasks 1+9; D8 shared budget → Tasks 4+5; D9 no-fake fallbacks → Tasks 12 (no `--dry-run`) + 14 (hash modes only surfaced, never silent); D10 fixed dataset → template already pinned in Task 7's `render_script`.
- **Type consistency:** `AttemptResult(logs, artifact, error)` (Task 2) used by every fake in Tasks 8/9/11/12; `Event`/`LoopResult` come only from `supertaco.loop` after Task 2; `run_eval_suite(prompts, base, ft, llm=)` matches `harness.py:138`; `LoopResult.artifact` consumed by Task 11's report.
- **Test seams:** transport faked via constructor injection or `cli._make_transport` monkeypatch; LLM faked via `loop.make_llm` monkeypatch or constructor args; no test contacts Colab or Token Factory before the Task 14 approval gate.
- **Assumptions:** removing `--dry-run` entirely is approved (D3/D9); `configs/defaults/colab_t4.yaml` is tracked (`configs/runs/*` gitignore does not cover `configs/defaults/`); AGENTS mission text left stale deliberately.
- **Scope boundaries:** no edits to `configs/base/`; no formatting of untouched debt; concurrent-session files touched only by the documented Task 14 restore; junk `NUL`/`typescript` never staged.
- **Ordering:** deletions (13) come after every consumer is rewired (cli 12, dashboard 11); live smoke is last and gated on explicit user approval.