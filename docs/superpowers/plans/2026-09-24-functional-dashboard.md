# Functional Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the dashboard actually run the agent pipeline — config-derived dry-run logs, playbook detection, real Nemotron calls with fallback, config patching to `configs/runs/`, live event-driven panels, real judge eval — plus a working CLI.

**Architecture:** One `supertaco.runner.run()` owns launch→log→detect→classify→patch→relaunch and emits typed `Event`s via callback; dashboard and CLI both consume it. `simlogs.py` deterministically maps config→logs (rule table in spec §5.1). `NemotronClient` makes real Token Factory HTTP calls with per-run circuit breaker + heuristic fallback.

**Tech Stack:** Python 3.14 (env), streamlit, plotly, openai SDK, pyyaml, pytest, ruff

**Spec:** `docs/superpowers/specs/2026-09-24-functional-dashboard-design.md` (§ references below)

**Event types:** `run_started, job_launched, logs_produced, failure_detected, classified, patch_proposed, patch_written, retry_scheduled, run_succeeded, run_failed` (`classified` carries playbook key + Nemotron verdict + mode + diverged flag; `logs_produced` carries `loss_points` for the chart). Deliberate deviation from spec §4: the `llm_call` and `eval_scored` events are dropped — the same data reaches the UI via `RunResult.llm_calls` (LLM table + banner) and `st.session_state["eval_results"]` (eval panel).

**Per docs/AGENTS.md:** after each task, append a one-line entry to `docs/PROJECT.md` "Recent changes" and include that file in the task's commit.

**RunResult contract (used by tasks 5, 7, 8):**
```python
@dataclass
class RunResult:
    success: bool
    attempts: int                 # attempts actually made
    final_config: dict
    configs_written: list[str]    # paths in runs_dir
    failure_key: str | None       # last detected failure
    error: str | None             # None on success
    llm_calls: list[dict]         # copy of NemotronClient.call_log
```

---

## File structure

| File | Responsibility | Action |
|---|---|---|
| `supertaco/agent/simlogs.py` | config→logs generator, `classify_config_failure`, `extract_loss_points` | Create |
| `supertaco/runner.py` | single run path, `Event`, `RunResult`, config writing | Create |
| `supertaco/agent/llm.py` | real HTTP `_call`, circuit breaker, JSONL logging | Modify |
| `supertaco/agent/playbook.py` | EVAL_REGRESSION str detection, `ConfigurationError` import | Modify |
| `supertaco/eval/judges.py` | real Nemotron judge + hash fallback, `List` fix | Modify |
| `supertaco/eval/harness.py` | 5 fixed prompts, template responses, real-scored suite | Modify |
| `supertaco/cli.py` | rewire to runner (fixes 2 F821 crashes) | Modify |
| `supertaco/ui/dashboard.py` | event-driven live panels, fixture loader, eval suite | Modify |
| `tests/test_*.py` | unit + AppTest suites | Create |
| `docs/HANDOFF.md`, `docs/PROJECT.md` | invariant/status updates | Modify |

---

### Task 1: Environment setup + commit pending bug-hunt fixes

**Files:**
- Modify: `pyproject.toml` (pytest config only)
- Commit: existing working-tree fixes (dashboard crash fixes, install-chain fixes, HANDOFF paths)

- [ ] **Step 1: Install the package and test runner**

```powershell
pip install -e .
pip install pytest
```

Expected: `Successfully installed supertaco-0.1.0 ...` (pyproject was fixed in the bug hunt; this validates it).

If any dependency fails to build on Python 3.14, fall back to:

```powershell
pip install -e . --no-deps
pip install openai httpx pydantic pydantic-settings pyyaml structlog
pip install pytest
```

- [ ] **Step 2: Verify imports the run path needs**

```powershell
python -c "import openai, httpx, yaml, pytest; import supertaco.settings; print('setup ok')"
```

Expected: `setup ok`

- [ ] **Step 3: Add pytest marker config to `pyproject.toml`**

Append at end of `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
    "integration: hits real APIs and costs money (deselect with -m 'not integration')",
]
```

- [ ] **Step 4: Track the broken-config fixtures (they are gitignored today)**

`.gitignore` line `configs/runs/*` ignores everything in `configs/runs/`, so the 8
`*_broken_model.yaml` fixtures the tests and fixture loader depend on are untracked.
Append this exception to `.gitignore` right after the `!configs/runs/*_example.yaml` line:

```gitignore
!configs/runs/*_broken_model.yaml
```

Then:

```powershell
git add .gitignore configs/runs/
git commit -m "chore: track broken-config fixtures used by tests and the demo fixture loader"
```

Expected: 8 files staged (`git show --stat --oneline HEAD` shows 8 fixtures + `.gitignore`).

- [ ] **Step 5: Commit the pending bug-hunt fixes (3 commits, staged separately)**

```powershell
git add supertaco/ui/dashboard.py
git commit -m "fix(dashboard): syntax error, missing yaml import, hex-score crash, detect_failure, deprecations, lint"
git add pyproject.toml
git commit -m "fix(build): valid pyproject TOML, setuptools backend, package discovery, runtime deps, pytest config"
git add docs/HANDOFF.md docs/AGENTS.md docs/CONVENTIONS.md docs/PROJECT.md docs/SPEC.md docs/STACK.md docs/superpowers/plans/
git commit -m "docs: add project handoff docs, correct dashboard run path, add implementation plan"
```

Expected: `git status --short` shows only untracked `__pycache__`/generated files — tree clean except future work.

- [ ] **Step 6: Run the existing dashboard AppTest suite as baseline**

```powershell
python $env:TEMP\opencode\test_dashboard.py 2>&1 | Select-String -Pattern "^\[|DONE"
```

Expected: 13× `[ok]` + `DONE`. (This harness is superseded by `tests/test_dashboard.py` in Tasks 8–9.)

---

### Task 2: Playbook — string EVAL_REGRESSION detection + ConfigurationError import

**Files:**
- Modify: `supertaco/agent/playbook.py:1,39-43`
- Test: `tests/test_playbook.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_playbook.py`:

```python
"""Playbook detection fixes required by the runner path."""
from supertaco.agent.playbook import PLAYBOOK, apply_default_fix, detect_failure


def test_detects_eval_regression_from_log_text():
    log = "eval prompt_suite score 3.40\npost-train eval regression flagged vs baseline"
    assert detect_failure(log) == "EVAL_REGRESSION"


def test_eval_regression_does_not_match_unrelated_text():
    # Steep decay: a shallow series here would trip LOSS_PLATEAU, so the text
    # must avoid the plateau detector (last-10 window must differ by >= 0.5).
    assert detect_failure("step 1 loss 3.000\nstep 2 loss 2.000\nstep 3 loss 1.000") is None


def test_apply_default_fix_unknown_key_raises_configuration_error():
    from supertaco.errors import ConfigurationError
    try:
        apply_default_fix("NOT_A_MODE", {"learning_rate": 1e-4})
    except ConfigurationError:
        pass
    else:
        raise AssertionError("ConfigurationError not raised")


def test_all_eight_modes_have_detection_and_fix():
    assert len(PLAYBOOK) == 8
    for key in PLAYBOOK:
        apply_default_fix(key, {"learning_rate": 1e-4, "batch_size": 8,
                                "lora_r": 8, "lora_alpha": 16, "num_epochs": 3})
```

- [ ] **Step 2: Run test to verify it fails**

```powershell
python -m pytest tests/test_playbook.py -v
```

Expected: FAIL — `test_detects_eval_regression_from_log_text` (detect returns `None`) and `test_apply_default_fix_unknown_key_raises_configuration_error` (`NameError: ConfigurationError`).

- [ ] **Step 3: Fix `supertaco/agent/playbook.py`**

Replace line 1:

```python
from supertaco.errors import ConfigurationError
```

Replace the `EVAL_REGRESSION` entry (lines 39-43) with:

```python
    "EVAL_REGRESSION": {
        "detection": lambda log: (
            (isinstance(log, dict) and log.get("regression", False))
            or (isinstance(log, str) and "eval" in log.lower() and "regression" in log.lower())
        ),
        "default_fix": "lower_lora_rank_alpha_fewer_epochs_stronger_regularization",
        "description": "Post-train eval < baseline",
    },
```

- [ ] **Step 4: Run test to verify it passes**

```powershell
python -m pytest tests/test_playbook.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Lint + commit**

```powershell
python -m ruff check supertaco/agent/playbook.py tests/test_playbook.py
python -m ruff format supertaco/agent/playbook.py tests/test_playbook.py
git add supertaco/agent/playbook.py tests/test_playbook.py
git commit -m "feat(playbook): detect EVAL_REGRESSION from log text, fix ConfigurationError import"
```

Expected: All checks passed.

---

### Task 3: `simlogs.py` — deterministic config→log generator

**Files:**
- Create: `supertaco/agent/simlogs.py`
- Test: `tests/test_simlogs.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_simlogs.py`:

```python
"""Config -> log generation: classification, healing, determinism, shapes."""
import math
from pathlib import Path

import pytest

from supertaco.agent.playbook import apply_default_fix, detect_failure
from supertaco.agent.simlogs import (
    classify_config_failure,
    extract_loss_points,
    generate_logs,
)

RUNS = Path(__file__).resolve().parents[1] / "configs" / "runs"

FIXTURE_CASES = [
    "NAN_LOSS", "OOM", "LOSS_DIVERGENCE", "LOSS_PLATEAU",
    "EVAL_REGRESSION", "TOKENIZER_MISMATCH", "DATALOADER_STALL", "GRADIENT_EXPLOSION",
]

# Sidebar demo configs, copied verbatim from ui/dashboard.py demo_options
DEMO_CASES = {
    "NAN_LOSS": {"learning_rate": 0.1, "batch_size": 8, "lora_r": 8, "lora_alpha": 16, "num_epochs": 3},
    "OOM": {"learning_rate": 5e-5, "batch_size": 64, "lora_r": 8, "lora_alpha": 16, "num_epochs": 3},
    "LOSS_DIVERGENCE": {"learning_rate": 0.1, "batch_size": 8, "lora_r": 8, "lora_alpha": 16, "num_epochs": 10},
    "LOSS_PLATEAU": {"learning_rate": 1e-6, "batch_size": 8, "lora_r": 8, "lora_alpha": 16, "num_epochs": 1},
    "EVAL_REGRESSION": {"learning_rate": 1e-4, "batch_size": 8, "lora_r": 64, "lora_alpha": 32, "num_epochs": 10},
}

HEALTHY = {"learning_rate": 2e-5, "batch_size": 8, "lora_r": 8,
           "lora_alpha": 16, "num_epochs": 3, "gradient_clip_norm": 1.0,
           "chat_template": "llama-3"}


def _fixture(mode: str) -> dict:
    import yaml
    matches = [p for p in RUNS.glob("*.yaml") if mode in p.name]
    assert matches, f"no fixture for {mode}"
    return yaml.safe_load(matches[0].read_text(encoding="utf-8"))


@pytest.mark.parametrize("mode", FIXTURE_CASES)
def test_fixture_classifies_as_its_mode(mode):
    config = _fixture(mode)
    assert classify_config_failure(config) == mode
    assert detect_failure(generate_logs(config)) == mode


@pytest.mark.parametrize("mode", list(DEMO_CASES))
def test_sidebar_demo_classifies_as_its_mode(mode):
    config = DEMO_CASES[mode]
    assert classify_config_failure(config) == mode
    assert detect_failure(generate_logs(config)) == mode


@pytest.mark.parametrize("mode", FIXTURE_CASES)
def test_healing_property_fixture_then_fix_is_healthy(mode):
    config = _fixture(mode)
    healed = apply_default_fix(mode, config)
    assert classify_config_failure(healed) is None
    assert detect_failure(generate_logs(healed)) is None


def test_healthy_config_generates_no_failure():
    assert classify_config_failure(HEALTHY) is None
    assert detect_failure(generate_logs(HEALTHY)) is None


def test_generation_is_deterministic():
    config = _fixture("NAN_LOSS")
    assert generate_logs(config) == generate_logs(config)


def test_healthy_loss_series_decays_and_is_not_plateau():
    pts = extract_loss_points(generate_logs(HEALTHY))
    assert len(pts) == 30
    assert pts[0] > pts[-1]
    assert pts[-10] - pts[-1] > 0.5  # playbook plateau guard


def test_divergence_series_increases():
    pts = extract_loss_points(generate_logs(_fixture("LOSS_DIVERGENCE")))
    mid = len(pts) // 2
    assert sum(pts[mid:]) / len(pts[mid:]) > sum(pts[:mid]) / mid


def test_nan_series_contains_nan():
    pts = extract_loss_points(generate_logs(_fixture("NAN_LOSS")))
    assert any(math.isnan(v) for v in pts)
```

- [ ] **Step 2: Run test to verify it fails**

```powershell
python -m pytest tests/test_simlogs.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'supertaco.agent.simlogs'`.

- [ ] **Step 3: Write `supertaco/agent/simlogs.py`**

```python
"""Deterministic dry-run training-log generator.

Maps a training config to the log text a run would produce so the playbook
can classify dry-run output without a GPU. Rule precedence and healing
thresholds: docs/superpowers/specs/2026-09-24-functional-dashboard-design.md
section 5.1.
"""

from __future__ import annotations

import re
from typing import Optional

LOSS_RE = re.compile(r"\bloss\s+([0-9]+\.[0-9]+|NaN)\b")


def classify_config_failure(config: dict) -> Optional[str]:
    """Return the failure mode a config would trigger, or None if healthy.

    Missing keys are read as None/0. Precedence order is significant: it is
    the order modes surface across relaunches for multi-trigger configs.
    """
    lr = float(config.get("learning_rate") or 0)
    batch = int(config.get("batch_size") or 0)
    workers = int(config.get("num_workers") or 0)
    epochs = int(config.get("num_epochs") or 3)
    lora_r = int(config.get("lora_r") or 0)
    clip = config.get("gradient_clip_norm")
    template = config.get("chat_template")

    if batch > 32:
        return "OOM"
    if workers > 16:
        return "DATALOADER_STALL"
    if template not in (None, "llama-3"):
        return "TOKENIZER_MISMATCH"
    if lr >= 0.05 and clip is None and epochs >= 8:
        return "LOSS_DIVERGENCE"
    if lora_r >= 32 and epochs >= 10:
        return "EVAL_REGRESSION"
    if lr >= 0.05 and clip is None and template == "llama-3":
        return "GRADIENT_EXPLOSION"
    if lr >= 0.05 and template in (None, ""):
        return "NAN_LOSS"
    if lr < 1.5e-6:
        return "LOSS_PLATEAU"
    return None


def _header(config: dict) -> str:
    return (
        "=== SuperTaco dry-run training (simulated Nebius job) ===\n"
        f"model={config.get('model', 'llama-3-8b')} "
        f"lr={config.get('learning_rate')} batch_size={config.get('batch_size')} "
        f"epochs={config.get('num_epochs')}\n"
    )


def _healthy_logs() -> str:
    # Decay chosen so playbook's plateau window (last 10, must differ by >= 0.5)
    # and divergence check (second half not > 1.1x first) both stay negative.
    lines = [f"step {i} loss {2.4 - i * 0.06:.3f}" for i in range(30)]
    return _header({}) + "\n".join(lines) + "\ntraining finished (dry-run)\n"


def _nan_logs(config: dict) -> str:
    lines = [f"step {i} loss {2.4 - i * 0.05:.3f}" for i in range(12)]
    lines += ["step 13 loss NaN", "step 14 loss NaN", "FATAL: loss is NaN at step 14"]
    return _header(config) + "\n".join(lines) + "\n"


def _oom_logs(config: dict) -> str:
    lines = [f"step {i} loss {2.4 - i * 0.05:.3f}" for i in range(8)]
    lines += [
        "Traceback (most recent call last):",
        '  File "llamafactory/train/tuner.py", line 88, in train',
        "torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB",
    ]
    return _header(config) + "\n".join(lines) + "\n"


def _divergence_logs(config: dict) -> str:
    lines = [f"step {i} loss {1.0 + i * 0.18:.3f}" for i in range(30)]
    lines.append("WARN: loss trend rising over window")
    return _header(config) + "\n".join(lines) + "\n"


def _plateau_logs(config: dict) -> str:
    lines = [f"step {i} loss {2.0 + (0.001 if i % 2 else 0.0):.3f}" for i in range(25)]
    lines.append("WARN: no improvement in eval metric for 5 consecutive evals")
    return _header(config) + "\n".join(lines) + "\n"


def _grad_logs(config: dict) -> str:
    lines = [f"step {i} loss {1.8 - i * 0.01:.3f}" for i in range(10)]
    spikes = [8.42, 15.7, 42.11, 310.55]
    lines += [f"step {10 + i} lr 0.100 grad_norm {spikes[i % len(spikes)]}" for i in range(12)]
    lines.append("WARN: grad_norm spike detected, gradient explosion likely")
    return _header(config) + "\n".join(lines) + "\n"


def _eval_logs(config: dict) -> str:
    return _header(config) + "\n".join([
        "eval prompt_suite score 3.40 (baseline 6.10)",
        "post-train eval regression flagged vs baseline",
        "aborting: eval regression guard triggered",
    ]) + "\n"


def _tokenizer_logs(config: dict) -> str:
    template = config.get("chat_template")
    return _header(config) + "\n".join([
        "starting dataset formatting pass",
        f"ValueError: chat template '{template}' not found in template registry",
        "tokenizer mismatch: dataset uses chat format but template is unknown",
        "aborting dataloader setup",
    ]) + "\n"


def _stall_logs(config: dict) -> str:
    return _header(config) + "\n".join([
        "epoch 1/3 step 400/5000 metric unavailable (worker queue empty)",
        "no step progress in 15 minutes",
        "WARNING: dataloader workers idle; consider reducing num_workers",
        "no step progress in 16 minutes - stalling",
    ]) + "\n"


_EMITTERS = {
    "NAN_LOSS": _nan_logs,
    "OOM": _oom_logs,
    "LOSS_DIVERGENCE": _divergence_logs,
    "LOSS_PLATEAU": _plateau_logs,
    "EVAL_REGRESSION": _eval_logs,
    "TOKENIZER_MISMATCH": _tokenizer_logs,
    "DATALOADER_STALL": _stall_logs,
    "GRADIENT_EXPLOSION": _grad_logs,
}


def generate_logs(config: dict) -> str:
    """Return the training log a run with this config would produce."""
    mode = classify_config_failure(config)
    if mode is None:
        return _healthy_logs()
    return _EMITTERS[mode](config)


def extract_loss_points(log_text: str) -> list[float]:
    """Extract loss values in emission order; NaN is preserved as float('nan')."""
    points: list[float] = []
    for match in LOSS_RE.finditer(log_text):
        raw = match.group(1)
        points.append(float("nan") if raw == "NaN" else float(raw))
    return points
```

- [ ] **Step 4: Run test to verify it passes**

```powershell
python -m pytest tests/test_simlogs.py -v
```

Expected: 26 passed (8 fixtures + 5 demos + 8 healing + 5 singles). If `test_healthy... plateau` fails, adjust healthy decay constant (`0.06` → steeper) — never the rule table.

- [ ] **Step 5: Lint + commit**

```powershell
python -m ruff check supertaco/agent/simlogs.py tests/test_simlogs.py
python -m ruff format supertaco/agent/simlogs.py tests/test_simlogs.py
git add supertaco/agent/simlogs.py tests/test_simlogs.py
git commit -m "feat(agent): deterministic config-derived dry-run log generator"
```

---

### Task 4: `llm.py` — real Token Factory calls, circuit breaker, JSONL audit log

**Files:**
- Modify: `supertaco/agent/llm.py` (`__init__`, `_call`, new `_http_call`/`_record`)
- Test: `tests/test_llm.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_llm.py`:

```python
"""Real-call path with fallback: modes, circuit breaker, JSONL audit."""
import json
from pathlib import Path

from supertaco.agent.llm import NemotronClient


def _client() -> NemotronClient:
    return NemotronClient(base_url="http://token-factory.invalid/v1", api_key="k", timeout=1)


def test_real_call_records_mode_and_tokens(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    client = _client()
    monkeypatch.setattr(
        NemotronClient, "_http_call",
        lambda self, model_id, prompt, max_tokens, temperature: ("7", 42),
    )
    text = client.classify_failure("step 1 loss NaN")
    assert text == "7"
    entry = client.call_log[-1]
    assert entry["mode"] == "real"
    assert entry["tokens"] == 42
    assert entry["error"] is None
    lines = (tmp_path / "logs" / "llm_calls.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[-1])["mode"] == "real"


def test_failed_call_falls_back_and_opens_circuit(monkeypatch):
    client = _client()
    calls = []

    def boom(self, model_id, prompt, max_tokens, temperature):
        calls.append(model_id)
        raise RuntimeError("connection refused")

    monkeypatch.setattr(NemotronClient, "_http_call", boom)
    first = client.classify_failure("loss is NaN")
    assert first.strip() == "NAN_LOSS"          # heuristic simulation
    assert client.call_log[-1]["mode"] == "fallback"
    assert "connection refused" in client.call_log[-1]["error"]
    assert client._circuit_open is True

    second = client.classify_failure("CUDA out of memory")
    assert second.strip() == "OOM"
    assert client.call_log[-1]["mode"] == "fallback"
    assert "circuit_open" in client.call_log[-1]["error"]
    assert len(calls) == 1                       # second call skipped HTTP


def test_jsonl_write_failure_never_kills_run(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    client = _client()
    monkeypatch.setattr(
        NemotronClient, "_http_call",
        lambda self, model_id, prompt, max_tokens, temperature: ("1", 1),
    )
    (tmp_path / "logs").write_text("", encoding="utf-8")  # logs is a FILE -> mkdir fails
    text = client.classify_failure("x")
    assert text == "1"
    assert client.call_log[-1].get("jsonl_written") is False
```

- [ ] **Step 2: Run test to verify it fails**

```powershell
python -m pytest tests/test_llm.py -v
```

Expected: FAIL — `_http_call` missing (AttributeError) / no `mode` key.

- [ ] **Step 3: Rewrite the call core of `supertaco/agent/llm.py`**

Replace `__init__` (lines 18-21) with:

```python
    def __init__(self, base_url: str, api_key: str, timeout: float = 30.0):
        self.base_url = base_url
        self.api_key = api_key
        self.timeout = timeout
        self.call_log: list[dict] = []
        self._circuit_open = False
```

Replace `_call` (lines 23-29) with:

```python
    def _http_call(self, model_id: str, prompt: str, max_tokens: int, temperature: float) -> tuple[str, int]:
        """Real Token Factory HTTP call. Returns (text, total_tokens). Test seam."""
        from openai import OpenAI

        client = OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=self.timeout)
        response = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        text = response.choices[0].message.content or ""
        tokens = response.usage.total_tokens if response.usage else 0
        return text, tokens

    def _call(self, model_key: str, prompt: str, max_tokens: int = 1024, temperature: float = 0.0) -> str:
        """Call Nemotron via Token Factory; on any failure fall back to heuristics.

        First failure opens the circuit for the rest of this client's life so a
        dead API costs one timeout instead of one per call (spec D3).
        """
        model_id = self.MODELS[model_key]
        start = time.monotonic()
        if self._circuit_open:
            text = self._simulate_response(model_id, prompt)
            self._record(model_key, model_id, text, mode="fallback",
                         latency=0.0, tokens=0, error="circuit_open: earlier call failed")
            return text
        try:
            text, tokens = self._http_call(model_id, prompt, max_tokens, temperature)
        except Exception as exc:
            self._circuit_open = True
            text = self._simulate_response(model_id, prompt)
            self._record(model_key, model_id, text, mode="fallback",
                         latency=time.monotonic() - start, tokens=0,
                         error=f"{type(exc).__name__}: {exc}")
            return text
        self._record(model_key, model_id, text, mode="real",
                     latency=time.monotonic() - start, tokens=tokens, error=None)
        return text

    def _record(self, model_key: str, model_id: str, text: str, *, mode: str,
                latency: float, tokens: int, error: str | None) -> None:
        """Append a call entry to the in-memory log and logs/llm_calls.jsonl.

        JSONL write failure is flagged, never raised (spec 6: warning only).
        """
        entry = {
            "model_key": model_key,
            "model_id": model_id,
            "mode": mode,
            "tokens": tokens,
            "latency": round(latency, 3),
            "error": error,
            "timestamp": time.time(),
            "response_preview": text[:200],
        }
        self.call_log.append(entry)
        try:
            from pathlib import Path

            import json

            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            with open(log_dir / "llm_calls.jsonl", "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry) + "\n")
            entry["jsonl_written"] = True
        except OSError:
            entry["jsonl_written"] = False
```

Keep `_simulate_response`, `classify_failure`, `propose_patch`, `deep_reason`, `log_call` unchanged.

- [ ] **Step 4: Run test to verify it passes**

```powershell
python -m pytest tests/test_llm.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Lint + commit**

```powershell
python -m ruff check supertaco/agent/llm.py tests/test_llm.py
python -m ruff format supertaco/agent/llm.py tests/test_llm.py
git add supertaco/agent/llm.py tests/test_llm.py
git commit -m "feat(agent): real Token Factory calls with circuit-breaker fallback and JSONL audit"
```

---

### Task 5: `runner.py` — the single event-emitting run path

**Files:**
- Create: `supertaco/runner.py`
- Test: `tests/test_runner.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_runner.py`:

```python
"""Runner: event sequences, healing loop, retry cap, validation."""
from pathlib import Path

import pytest
import yaml

from supertaco.errors import ConfigurationError
from supertaco.runner import Event, RunResult, run

HEALTHY = {"learning_rate": 2e-5, "batch_size": 8, "lora_r": 8, "lora_alpha": 16,
           "num_epochs": 3, "gradient_clip_norm": 1.0, "chat_template": "llama-3"}
BROKEN_NAN = {"learning_rate": 0.1, "batch_size": 8, "lora_r": 8, "lora_alpha": 16,
              "num_epochs": 3, "chat_template": None}


class FakeLLM:
    """Minimal NemotronClient stand-in used by runner tests."""

    def __init__(self, verdict: str | None = None):
        self.verdict = verdict
        self.call_log: list[dict] = []

    def classify_failure(self, logs: str) -> str:
        self.call_log.append({"model_key": "classify", "mode": "real", "tokens": 1,
                              "latency": 0.1, "error": None})
        return self.verdict or "NAN_LOSS"

    def propose_patch(self, failure_key: str, config: dict, logs: str) -> dict:
        self.call_log.append({"model_key": "patch", "mode": "real", "tokens": 1,
                              "latency": 0.2, "error": None})
        return {"fix": "lower_lr", "reason": f"model proposes fix for {failure_key}"}


def _collect():
    events: list[Event] = []
    return events, lambda ev: events.append(ev)


def test_validation_rejects_empty_config(tmp_path):
    with pytest.raises(ConfigurationError):
        run({}, runs_dir=tmp_path)
    with pytest.raises(ConfigurationError):
        run({"batch_size": 8}, runs_dir=tmp_path)  # no learning_rate


def test_success_path_healthy_config(tmp_path):
    events, on_event = _collect()
    result = run(HEALTHY, on_event=on_event, llm=FakeLLM(), runs_dir=tmp_path)
    assert result.success is True
    assert result.attempts == 1
    assert result.error is None
    assert [e.type for e in events] == ["run_started", "job_launched", "logs_produced", "run_succeeded"]
    logs_event = events[2]
    assert len(logs_event.data["loss_points"]) == 30


def test_heal_path_writes_patch_file(tmp_path):
    events, on_event = _collect()
    result = run(BROKEN_NAN, on_event=on_event, llm=FakeLLM(verdict="NAN_LOSS"),
                 runs_dir=tmp_path)
    assert result.success is True
    assert result.attempts == 2
    assert len(result.configs_written) == 1
    path = Path(result.configs_written[0])
    assert path.exists()
    patched = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert patched["learning_rate"] == pytest.approx(0.01)
    types = [e.type for e in events]
    assert types == [
        "run_started", "job_launched", "logs_produced",
        "failure_detected", "classified", "patch_proposed", "patch_written",
        "retry_scheduled", "job_launched", "logs_produced", "run_succeeded",
    ]
    assert events[3].data["failure_key"] == "NAN_LOSS"
    assert events[4].data["nemotron_verdict"] == "NAN_LOSS"
    assert events[4].data["diverged"] is False
    assert result.llm_calls[-1]["model_key"] == "patch"


def test_retry_cap_emits_run_failed(tmp_path):
    events, on_event = _collect()

    def always_nan(config: dict) -> str:
        return "step 0 loss NaN"

    result = run(BROKEN_NAN, on_event=on_event, llm=FakeLLM(), log_fn=always_nan,
                 max_retries=3, runs_dir=tmp_path)
    assert result.success is False
    assert result.attempts == 4            # initial + 3 relaunches (spec 4.5)
    assert "MaxRetriesExceeded" in result.error
    assert events[-1].type == "run_failed"
    assert len(result.configs_written) == 3  # one patch per relaunch decision


def test_divergence_flag_when_nemotron_disagrees(tmp_path):
    events, on_event = _collect()
    run(BROKEN_NAN, on_event=on_event, llm=FakeLLM(verdict="OOM"), runs_dir=tmp_path)
    classified = [e for e in events if e.type == "classified"][0]
    assert classified.data["diverged"] is True
    assert classified.data["nemotron_verdict"] == "OOM"


def test_real_jobs_not_implemented_returns_failed_result(tmp_path):
    events, on_event = _collect()
    result = run(HEALTHY, dry_run=False, on_event=on_event, llm=FakeLLM(),
                 runs_dir=tmp_path)
    assert result.success is False
    assert "Gate 1" in result.error
    assert events[-1].type == "run_failed"
```

- [ ] **Step 2: Run test to verify it fails**

```powershell
python -m pytest tests/test_runner.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'supertaco.runner'`.

- [ ] **Step 3: Write `supertaco/runner.py`**

```python
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
    (handoff invariant 1). Never raises for expected failures.
    """
    validate_config(config)
    log_fn = log_fn or generate_logs
    llm = llm or _default_llm()
    _emit(on_event, "run_started", max_retries=max_retries, dry_run=dry_run)

    if not dry_run:
        error = "Real Nebius jobs are not implemented yet (Gate 1)"
        _emit(on_event, "run_failed", error=error, attempts=0, failure_key=None)
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
        _emit(on_event, "logs_produced", attempt=attempt, text=logs,
              loss_points=extract_loss_points(logs))

        failure_key = detect_failure(logs)
        if failure_key is None:
            _emit(on_event, "run_succeeded", attempts=attempt,
                  configs_written=list(configs_written))
            return RunResult(True, attempt, current, configs_written, None, None,
                             list(llm.call_log))

        _emit(on_event, "failure_detected", attempt=attempt, failure_key=failure_key)

        verdict = llm.classify_failure(logs)
        last = llm.call_log[-1] if llm.call_log else {}
        _emit(on_event, "classified", attempt=attempt, failure_key=failure_key,
              nemotron_verdict=verdict, mode=last.get("mode", "unknown"),
              diverged=verdict != failure_key)

        proposal = llm.propose_patch(failure_key, current, logs)
        _emit(on_event, "patch_proposed", attempt=attempt, failure_key=failure_key,
              proposal=proposal)

        new_config = apply_default_fix(failure_key, current)
        path = _write_config(new_config, failure_key, Path(runs_dir))
        configs_written.append(path)
        _emit(on_event, "patch_written", attempt=attempt, path=path,
              failure_key=failure_key, before=current, after=new_config)
        current = new_config

        if attempt < attempt_limit:
            _emit(on_event, "retry_scheduled", next_attempt=attempt + 1)

    error = f"MaxRetriesExceeded: {max_retries} relaunches exhausted"
    _emit(on_event, "run_failed", error=error, attempts=attempt,
          failure_key=failure_key,
          path=configs_written[-1] if configs_written else None)
    return RunResult(False, attempt, current, configs_written, failure_key, error,
                     list(llm.call_log))
```

- [ ] **Step 4: Run test to verify it passes**

```powershell
python -m pytest tests/test_runner.py -v
```

Expected: 6 passed.

- [ ] **Step 5: Lint + commit**

```powershell
python -m ruff check supertaco/runner.py tests/test_runner.py
python -m ruff format supertaco/runner.py tests/test_runner.py
git add supertaco/runner.py tests/test_runner.py
git commit -m "feat(runner): event-emitting agent run path with retry cap and config patching"
```

---

### Task 6: Eval — real Nemotron judge + template responses

**Files:**
- Modify: `supertaco/eval/judges.py` (full rewrite)
- Modify: `supertaco/eval/harness.py` (full rewrite)
- Test: `tests/test_judge.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_judge.py`:

```python
"""Real judge scoring with hash fallback, clamping, and suite regression flag."""
from supertaco.eval.harness import DEFAULT_PROMPTS, build_responses, run_eval_suite
from supertaco.eval.judges import JudgeResult, score_response


class FakeLLM:
    def __init__(self, replies):
        self.replies = list(replies)
        self.call_log = []

    def _call(self, model_key, prompt, max_tokens=1024, temperature=0.0):
        text = self.replies.pop(0)
        self.call_log.append({"model_key": model_key, "mode": "real", "tokens": 1,
                              "latency": 0.1, "error": None})
        return text


def test_hash_fallback_without_client():
    result = score_response("p", "r")
    assert isinstance(result, JudgeResult)
    assert result.mode == "hash"
    assert 2.0 <= result.score <= 10.0


def test_real_judge_score_parsed_and_clamped():
    llm = FakeLLM(["8"])  # first prompt, base
    result = score_response("p", "r", llm=llm)
    assert result.score == 8.0
    assert result.mode == "real"


def test_clamp_out_of_range():
    llm = FakeLLM(["15", "Score: -2"])
    assert score_response("p", "r", llm=llm).score == 10.0
    assert score_response("p", "r", llm=llm).score == 0.0


def test_unparseable_judge_output_falls_back_to_hash():
    llm = FakeLLM(["no numbers here"])
    result = score_response("p", "r", llm=llm)
    assert result.mode == "hash"
    assert 2.0 <= result.score <= 10.0


def test_build_responses_shapes_and_health_dependence():
    healthy = {"learning_rate": 2e-5, "batch_size": 8, "lora_r": 8,
               "lora_alpha": 16, "num_epochs": 3}
    broken = {"learning_rate": 0.1, "batch_size": 8, "lora_r": 8,
              "lora_alpha": 16, "num_epochs": 3}
    base_h, ft_h = build_responses(healthy)
    base_b, ft_b = build_responses(broken)
    assert len(base_h) == len(DEFAULT_PROMPTS) == 5
    assert ft_h[DEFAULT_PROMPTS[0]] != ft_b[DEFAULT_PROMPTS[0]]
    assert "structured" in ft_h[DEFAULT_PROMPTS[0]].lower() or "1)" in ft_h[DEFAULT_PROMPTS[0]]
    assert "uncertain" in ft_b[DEFAULT_PROMPTS[0]]


def test_run_eval_suite_regression_flag():
    # base always scores 9, fine-tuned always scores 3 -> regression True
    llm = FakeLLM(["9"] * 5 + ["3"] * 5)
    base, ft = build_responses({"learning_rate": 2e-5, "batch_size": 8})
    results = run_eval_suite(DEFAULT_PROMPTS, base, ft, llm=llm)
    assert len(results["base_scores"]) == 5
    assert results["regression_flagged"] is True
    assert all(0.0 <= s <= 10.0 for s in results["base_scores"] + results["fine_tuned_scores"])
    assert all(m == "real" for m in results["modes"])
```

- [ ] **Step 2: Run test to verify it fails**

```powershell
python -m pytest tests/test_judge.py -v
```

Expected: FAIL — `ImportError: cannot import name 'JudgeResult'` (and `DEFAULT_PROMPTS`).

- [ ] **Step 3: Rewrite `supertaco/eval/judges.py`**

```python
"""Nemotron judge scoring with deterministic hash fallback (spec 5.4)."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Optional

JUDGE_PROMPT = """You are a strict evaluation judge. Score the response from 0 to 10
(10 = excellent, 0 = unusable). Reply with a single number only.

Prompt: {prompt}
Response: {response}"""


@dataclass
class JudgeResult:
    score: float
    mode: str  # "real" | "fallback" (HTTP level) | "hash"
    error: Optional[str] = None


def _hash_score(prompt: str, response: str) -> float:
    digest = int(hashlib.sha256(f"{prompt}{response}".encode()).hexdigest()[:8], 16)
    return round(2.0 + (digest % 81) / 10, 2)  # 2.0 - 10.0


def score_response(prompt: str, response: str, llm=None,
                   judge_model_key: str = "patch") -> JudgeResult:
    """Score one response. With an llm client: real Nemotron judge call.

    Parse failure or missing client falls back to the deterministic hash;
    the HTTP-level mode (real/fallback) is inherited from the client.
    """
    if llm is None:
        return JudgeResult(_hash_score(prompt, response), "hash", "no llm client")

    text = llm._call(judge_model_key, JUDGE_PROMPT.format(prompt=prompt, response=response),
                     max_tokens=16, temperature=0.0)
    http_mode = "real"
    http_error = None
    if llm.call_log:
        http_mode = llm.call_log[-1].get("mode", "real")
        http_error = llm.call_log[-1].get("error")

    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)  # signed so "Score: -2" clamps to 0.0
    if not numbers:
        return JudgeResult(_hash_score(prompt, response), "hash",
                           f"unparseable judge output: {text[:80]!r}")
    raw = float(numbers[-1])  # last number: avoids "0 to 10" echoes
    score = max(0.0, min(10.0, raw))
    return JudgeResult(round(score, 2), http_mode, http_error)


def compare_models(base_responses: dict, fine_tuned_responses: dict,
                   prompts: list, llm=None) -> dict:
    """Compare base vs fine-tuned responses across prompts."""
    results = {"prompt_results": [], "base_total": 0.0, "ft_total": 0.0,
               "regression": False}
    for prompt in prompts:
        base = score_response(prompt, base_responses.get(prompt, ""), llm=llm)
        ft = score_response(prompt, fine_tuned_responses.get(prompt, ""), llm=llm)
        results["prompt_results"].append({
            "prompt": prompt, "base_score": base.score, "fine_tuned_score": ft.score,
            "improvement": ft.score > base.score,
        })
        results["base_total"] += base.score
        results["ft_total"] += ft.score
    results["regression"] = results["ft_total"] < results["base_total"] * 0.9
    return results
```

- [ ] **Step 4: Rewrite `supertaco/eval/harness.py`**

```python
"""Fixed 5-prompt eval suite: template responses scored by real judge."""

from __future__ import annotations

from typing import Any

PROMPT_SUITE_SIZE = 5

DEFAULT_PROMPTS = [
    "Explain what LoRA fine-tuning is in two sentences.",
    "Write a Python function that loads a Hugging Face dataset.",
    "Summarize why gradient clipping prevents divergence.",
    "How does a chat template differ from a tokenizer?",
    "Give one reason eval scores can regress after training.",
]


def build_responses(final_config: dict) -> tuple[dict, dict]:
    """Deterministic base vs fine-tuned template responses.

    Fine-tuned quality depends on whether the final config is healthy, so a
    still-broken config produces visibly worse answers (spec 5.4).
    """
    from supertaco.agent.simlogs import classify_config_failure

    healthy = classify_config_failure(final_config) is None
    base = {
        p: f"Base model answer to '{p[:48]}': a generic, unstructured reply "
           f"without concrete details."
        for p in DEFAULT_PROMPTS
    }
    if healthy:
        fine_tuned = {
            p: f"Fine-tuned answer to '{p[:48]}': 1) precise structure "
               f"2) concrete details 3) correct terminology."
            for p in DEFAULT_PROMPTS
        }
    else:
        fine_tuned = {
            p: f"I'm not really sure about '{p[:48]}'... an uncertain, "
               f"rambling reply that hedges and repeats itself."
            for p in DEFAULT_PROMPTS
        }
    return base, fine_tuned


def run_eval_suite(prompts, base_responses: dict, fine_tuned_responses: dict,
                   llm: Any = None) -> dict:
    """Run the fixed suite; every score comes from judges.score_response."""
    from supertaco.eval.judges import score_response

    results: dict[str, Any] = {
        "prompts": [], "base_scores": [], "fine_tuned_scores": [],
        "modes": [], "regression_flagged": False,
    }
    for prompt in prompts[:PROMPT_SUITE_SIZE]:
        base = score_response(prompt, base_responses[prompt], llm=llm)
        ft = score_response(prompt, fine_tuned_responses[prompt], llm=llm)
        results["prompts"].append(prompt)
        results["base_scores"].append(base.score)
        results["fine_tuned_scores"].append(ft.score)
        results["modes"].append(f"base:{base.mode} / ft:{ft.mode}")

    if results["fine_tuned_scores"] and results["base_scores"]:
        avg_ft = sum(results["fine_tuned_scores"]) / len(results["fine_tuned_scores"])
        avg_base = sum(results["base_scores"]) / len(results["base_scores"])
        results["regression_flagged"] = avg_ft < avg_base * 0.9
    return results


def check_regression(eval_results: dict) -> bool:
    """True when fine-tuned average is >10%% below baseline."""
    return eval_results.get("regression_flagged", False)
```

- [ ] **Step 5: Run test to verify it passes**

```powershell
python -m pytest tests/test_judge.py -v
```

Expected: 6 passed.

- [ ] **Step 6: Lint + commit**

```powershell
python -m ruff check supertaco/eval/judges.py supertaco/eval/harness.py tests/test_judge.py
python -m ruff format supertaco/eval/judges.py supertaco/eval/harness.py tests/test_judge.py
git add supertaco/eval/judges.py supertaco/eval/harness.py tests/test_judge.py
git commit -m "feat(eval): real Nemotron judge with fallback, template responses, fixed prompt suite"
```

---

### Task 7: CLI — rewire to runner (fixes both F821 crashes)

**Files:**
- Modify: `supertaco/cli.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_cli.py`:

```python
"""CLI smoke: runs the shared runner and prints events."""
from pathlib import Path

import yaml

from supertaco import cli


def test_run_config_success_prints_events(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)  # configs/runs + logs land in tmp
    fixture = next(p for p in (Path(__file__).resolve().parents[1] / "configs" / "runs").glob("*NAN_LOSS*"))

    class FakeLLM:
        call_log = []
        def classify_failure(self, logs): return "NAN_LOSS"
        def propose_patch(self, key, config, logs): return {"fix": "x", "reason": "y"}

    monkeypatch.setattr(cli, "_make_llm", lambda: FakeLLM())

    cli.run_config(str(fixture), dry_run=True)
    out = capsys.readouterr().out
    assert "[run_succeeded]" in out
    assert "[patch_written]" in out
    assert "Success" in out
    written = list((tmp_path / "configs" / "runs").glob("*_patched.yaml"))
    assert len(written) == 1
    patched = yaml.safe_load(written[0].read_text(encoding="utf-8"))
    assert patched["learning_rate"] == 0.01
```

- [ ] **Step 2: Run test to verify it fails**

```powershell
python -m pytest tests/test_cli.py -v
```

Expected: FAIL — `AttributeError: module 'supertaco.cli' has no attribute '_make_llm'`.

- [ ] **Step 3: Rewrite `supertaco/cli.py`**

```python
"""SuperTaco CLI entry point.

Usage:
    supertaco run <config> [--dry-run]
    supertaco --help
"""

from __future__ import annotations

import argparse
import json
import sys

from supertaco.agent.llm import NemotronClient
from supertaco.runner import run as runner_run
from supertaco.settings import settings


def main() -> None:
    parser = argparse.ArgumentParser(description="SuperTaco: AI-supervised fine-tuning sandbox")
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run a fine-tuning job")
    run_parser.add_argument("config", help="Path to YAML config file")
    run_parser.add_argument("--dry-run", action="store_true",
                            help="Simulate without launching real jobs")

    args = parser.parse_args()
    if args.command == "run":
        run_config(args.config, args.dry_run)
    else:
        parser.print_help()
        sys.exit(0)


def _make_llm() -> NemotronClient:
    return NemotronClient(
        base_url=settings.token_factory_base_url,
        api_key=settings.nebius_api_key.get_secret_value(),
    )


def run_config(config_path: str, dry_run: bool = True) -> None:
    """Run a fine-tuning job from a config file via the shared runner."""
    config = _load_config(config_path)
    print(f"SuperTaco: running {config_path}")
    llm = _make_llm()

    def printer(event) -> None:
        print(f"[{event.type}] {json.dumps(event.data, default=str)}")

    result = runner_run(config, max_retries=3, dry_run=dry_run, on_event=printer, llm=llm)
    if result.success:
        print(f"Success after {result.attempts} attempt(s). "
              f"Configs: {result.configs_written}")
    else:
        print(f"Run failed: {result.error}")
        sys.exit(1)


def _load_config(config_path: str) -> dict:
    import yaml

    with open(config_path, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    return config or {}


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

```powershell
python -m pytest tests/test_cli.py -v
```

Expected: 1 passed.

- [ ] **Step 5: Lint + commit**

```powershell
python -m ruff check supertaco/cli.py tests/test_cli.py
python -m ruff format supertaco/cli.py tests/test_cli.py
git add supertaco/cli.py tests/test_cli.py
git commit -m "fix(cli): rewire to shared runner, resolve undefined NemotronClient/run_agent_loop"
```

---

### Task 8: Dashboard — live event-driven launch, timeline, logs, loss chart

**Files:**
- Modify: `supertaco/ui/dashboard.py`
- Test: `tests/test_dashboard.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_dashboard.py`:

```python
"""AppTest: launch a broken fixture and watch the real pipeline heal it."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

REPO = Path(__file__).resolve().parents[1]
DASHBOARD = REPO / "supertaco" / "ui" / "dashboard.py"
NAN_FIXTURE = next((REPO / "configs" / "runs").glob("*NAN_LOSS*"))


@pytest.fixture()
def offline(monkeypatch, tmp_path):
    """Unreachable Token Factory + tmp working dir: no credits, no repo litter."""
    monkeypatch.chdir(tmp_path)
    import supertaco.settings as settings_mod
    monkeypatch.setattr(settings_mod.settings, "token_factory_base_url",
                        "http://127.0.0.1:9", raising=True)


@pytest.fixture()
def app(offline):
    at = AppTest.from_file(str(DASHBOARD), default_timeout=60)
    at.run()
    assert not at.exception, f"initial render crashed: {[e.value for e in at.exception]}"
    return at


def _launch_nan(app):
    import yaml
    config = yaml.safe_load(NAN_FIXTURE.read_text(encoding="utf-8"))
    app.session_state["config"] = config
    app.run()
    launch = next(b for b in app.button if "Launch Job" in b.label)
    launch.click().run()
    app.run()  # sidebar rendered before the handler; rerun so it sees post-run state
    return app


def test_initial_render_no_exceptions(app):
    assert not app.exception


def test_launch_heals_nan_fixture(app):
    app = _launch_nan(app)
    assert not app.exception, [e.value for e in app.exception]
    result = app.session_state["run_result"]
    assert result.success is True
    assert result.attempts == 2
    assert len(result.configs_written) == 1

    types = [e.type for e in app.session_state["events"]]
    assert "failure_detected" in types
    assert "patch_written" in types
    assert types[-1] == "run_succeeded"

    failure = next(e for e in app.session_state["events"] if e.type == "failure_detected")
    assert failure.data["failure_key"] == "NAN_LOSS"

    # patched config actually landed in tmp cwd's configs/runs
    patched = list((Path.cwd() / "configs" / "runs").glob("*_patched.yaml"))
    assert len(patched) == 1


def test_llm_calls_recorded_with_fallback_mode(app):
    app = _launch_nan(app)
    calls = app.session_state["run_result"].llm_calls
    assert len(calls) >= 2  # classify + patch
    assert all(c["mode"] == "fallback" for c in calls)  # offline URL
    assert any(c["error"] for c in calls)


def test_loss_chart_receives_real_points(app):
    app = _launch_nan(app)
    loss_events = [e for e in app.session_state["events"] if e.type == "logs_produced"]
    assert len(loss_events) == 2
    assert len(loss_events[0].data["loss_points"]) >= 10
```

- [ ] **Step 2: Run test to verify it fails**

```powershell
python -m pytest tests/test_dashboard.py -v
```

Expected: FAIL — `KeyError: 'run_result'` (current launch stores only fake `sim_` data).

- [ ] **Step 3: Rework `supertaco/ui/dashboard.py`**

Changes (top to bottom):

1. Add to the normal import block (after the plotly import — these imports have no
   settings side effect; `supertaco.settings` is deliberately NOT imported here, see
   helper below, so it initializes after `_load_env()`):

```python
from supertaco.agent.llm import NemotronClient
from supertaco.errors import ConfigurationError
from supertaco.runner import Event, run as runner_run
```

2. Delete the whole local failure-detection block (`_extract_losses` + `detect_failure` under the "kept in sync manually" banner) — the runner/playbook own this now. Leave `_load_env`, `get_settings`, session helpers intact.

3. Add `"events": []` and `"run_result": None` to `init_session2` defaults.

4. Add these helpers above `def main():`:

```python
def _make_llm() -> NemotronClient:
    # Imported here so package settings initialize AFTER _load_env() above.
    from supertaco.settings import settings

    return NemotronClient(
        base_url=settings.token_factory_base_url,
        api_key=settings.nebius_api_key.get_secret_value(),
    )


def _render_events(events: list) -> str:
    """Build the full timeline markdown from collected events."""
    if not events:
        return "No decisions yet — launch a job to see the loop."
    lines = []
    for ev in events:
        d = ev.data
        if ev.type == "job_launched":
            lines.append(f"**Attempt {d['attempt']}** · 🚀 dry-run launch "
                         f"({len(d['payload'].get('config', {}))} config keys)")
        elif ev.type == "failure_detected":
            lines.append(f"**Attempt {d['attempt']}** · ❌ detected `{d['failure_key']}`")
        elif ev.type == "classified":
            mark = "⚠️ diverges from playbook" if d["diverged"] else "agrees with playbook"
            lines.append(f"**Attempt {d['attempt']}** · 🤖 Nemotron nano → "
                         f"`{d['nemotron_verdict']}` ({d['mode']}, {mark})")
        elif ev.type == "patch_proposed":
            reason = d["proposal"].get("reason", d["proposal"]) if isinstance(d["proposal"], dict) else d["proposal"]
            lines.append(f"**Attempt {d['attempt']}** · 🧩 model reasoning: {reason}")
        elif ev.type == "patch_written":
            before, after = d.get("before", {}), d.get("after", {})
            changed = ", ".join(
                f"{k}: {before.get(k)!r}→{after[k]!r}"
                for k in after if before.get(k) != after.get(k)
            )
            lines.append(f"**Attempt {d['attempt']}** · 💾 patched `{d['failure_key']}` → "
                         f"`{d['path']}`\n   `{changed}`")
        elif ev.type == "retry_scheduled":
            lines.append(f"🔁 relaunching (attempt {d['next_attempt']})")
        elif ev.type == "run_succeeded":
            lines.append(f"✅ **Healed** after {d['attempts']} attempt(s)")
        elif ev.type == "run_failed":
            lines.append(f"🛑 **Failed**: {d['error']}")
    return "\n\n".join(lines)


def _loss_figure(events: list):
    """Per-attempt loss series parsed from generated logs (spec 5.5)."""
    import math
    fig = go.Figure()
    for ev in events:
        if ev.type != "logs_produced":
            continue
        points = [None if math.isnan(v) else v for v in ev.data["loss_points"]]
        fig.add_trace(go.Scatter(y=points, mode="lines",
                                 name=f"attempt {ev.data['attempt']}"))
    fig.update_layout(title="Training loss (dry-run, per attempt)",
                      xaxis_title="Step", yaxis_title="Loss",
                      hovermode="x unified", height=300)
    return fig


def _llm_rows(calls: list) -> list:
    return [
        {"model": c.get("model_key"), "id": c.get("model_id"),
         "mode": c.get("mode"), "tokens": c.get("tokens"),
         "latency_s": c.get("latency"),
         "error": (c.get("error") or "")[:60]}
        for c in calls
    ]


def execute_launch(config: dict, max_retries: int, timeline_slot, logs_slot, chart_slot):
    """Run the pipeline, streaming events into the status placeholders."""
    events: list[Event] = []

    def on_event(ev: Event) -> None:
        events.append(ev)
        st.session_state["events"] = events  # survive a mid-run crash (spec 6)
        timeline_slot.markdown(_render_events(events))
        if ev.type == "logs_produced":
            logs_slot.code(ev.data["text"], language="log")
            chart_slot.plotly_chart(_loss_figure(events), use_container_width=True)

    result = runner_run(config, max_retries=max_retries, on_event=on_event,
                        llm=_make_llm())
    return events, result
```

5. Replace the dry-run checkbox label (spec 5.5) inside the launch container:

```python
            st.checkbox("Dry-run (Nebius GPU jobs stay dry-run until Gate 1)", value=True)
```

6. Replace the launch handler block in `main()` (from `if st.button("▶ Launch Job"...` through `st.rerun()`) with capture-only:

```python
        run_clicked = st.button("▶ Launch Job", type="primary", use_container_width=True)
        cfg_for_run = cfg
```

7. Replace the "Job Status & Logs" container content (`render_log_stream(...)` / `render_loss_curve([])` / `render_timeline(...)`) with:

```python
    with st.container(border=True):
        st.markdown("### 📡 Job Status & Logs")
        timeline_slot = st.empty()
        logs_slot = st.empty()
        chart_slot = st.empty()

        if run_clicked:
            try:
                events, result = execute_launch(
                    cfg_for_run, st.session_state.get("max_retries", 3),
                    timeline_slot, logs_slot, chart_slot,
                )
                st.session_state["events"] = events
                st.session_state["run_result"] = result
                if not result.success:
                    st.error(f"🛑 {result.error}")
                else:
                    st.toast("✅ Run healed", icon="✅")
            except ConfigurationError as exc:
                st.error(f"❌ {exc}")
            except Exception as exc:  # spec 6: never a Streamlit traceback
                st.error(f"❌ Run crashed: {exc}")
                timeline_slot.markdown(_render_events(st.session_state.get("events", [])))
        elif st.session_state.get("events"):
            timeline_slot.markdown(_render_events(st.session_state["events"]))
            latest_logs = next(
                (e.data["text"] for e in reversed(st.session_state["events"])
                 if e.type == "logs_produced"), "")
            logs_slot.code(latest_logs or "—", language="log")
            chart_slot.plotly_chart(_loss_figure(st.session_state["events"]),
                                    use_container_width=True)
        else:
            timeline_slot.info("No run yet — load a config and press Launch.")
```

8. Remove now-unused `render_log_stream`, `render_loss_curve`, `render_timeline`, and `init_session`/`reset_session` (the duplicate pair; keep `init_session2`/`reset_session2`).

- [ ] **Step 4: Run test to verify it passes**

```powershell
python -m pytest tests/test_dashboard.py -v
```

Expected: 4 passed. (Judge/eval panel still the old hash one — replaced in Task 9.)

- [ ] **Step 5: Lint + baseline + commit**

```powershell
python -m ruff check supertaco/ui/dashboard.py tests/test_dashboard.py
python -m ruff format supertaco/ui/dashboard.py tests/test_dashboard.py
python -m pytest tests/ -m "not integration" -q
git add supertaco/ui/dashboard.py tests/test_dashboard.py
git commit -m "feat(dashboard): live event-driven launch, timeline, logs, and loss chart"
```

Expected: all tests green.

---

### Task 9: Dashboard — LLM table, fallback banner, fixture loader, real eval suite

**Files:**
- Modify: `supertaco/ui/dashboard.py`
- Test: `tests/test_dashboard.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_dashboard.py`:

```python
def test_fixture_loader_and_eval_suite(app):
    # Fixture loader present: selectbox labeled "Fixture" offers the yaml fixtures.
    # (s.label is the widget label; the filenames live in s.options.)
    fixture_sb = next(s for s in app.selectbox if s.label == "Fixture")
    assert any("NAN_LOSS" in opt for opt in fixture_sb.options)

    # complete a run first
    app = _launch_nan(app)
    assert app.session_state["run_result"].success is True

    # real eval suite (offline -> hash/fallback modes, but structured scores)
    eval_button = next(b for b in app.button if "Eval" in b.label)
    eval_button.click().run()
    assert not app.exception, [e.value for e in app.exception]
    results = app.session_state["eval_results"]
    assert len(results["base_scores"]) == 5
    assert len(results["fine_tuned_scores"]) == 5
    assert all(0.0 <= s <= 10.0 for s in results["base_scores"] + results["fine_tuned_scores"])
    assert isinstance(results["regression_flagged"], bool)


def test_fallback_banner_shows_after_offline_run(app):
    app = _launch_nan(app)
    # sidebar should surface the fallback state (error element)
    banner_texts = [e.value for e in app.sidebar.error]
    assert any("fallback" in t.lower() for t in banner_texts)
```

- [ ] **Step 2: Run test to verify it fails**

```powershell
python -m pytest tests/test_dashboard.py -v -k "fixture_loader or fallback_banner"
```

Expected: FAIL — no `eval_results` key / no fixture selectbox / no banner.

- [ ] **Step 3: Implement in `supertaco/ui/dashboard.py`**

1. **Fixture loader** — inside `render_sidebar()`, after the demo-config block, add:

```python
    st.sidebar.markdown("### Broken Fixtures")
    from pathlib import Path as _Path
    fixtures = sorted((_Path(__file__).resolve().parents[2] / "configs" / "runs").glob("*.yaml"))
    if fixtures:
        chosen = st.sidebar.selectbox("Fixture", [f.name for f in fixtures],
                                      format_func=lambda n: n.split("_", 2)[-1].rsplit("_", 1)[0])
        if st.sidebar.button("Load Fixture"):
            import yaml as _yaml
            st.session_state["config"] = _yaml.safe_load(
                (_Path(__file__).resolve().parents[2] / "configs" / "runs" / chosen).read_text(encoding="utf-8"))
            st.sidebar.success(f"Loaded {chosen}")
```

2. **Fallback banner** — in `render_sidebar()` at the top after the title:

```python
    calls = st.session_state.get("llm_calls") or []
    fallbacks = [c for c in calls if c.get("mode") != "real"]
    if fallbacks:
        st.sidebar.error(
            f"⚠️ Nemotron fallback: {len(fallbacks)}/{len(calls)} call(s) used heuristics. "
            f"First error: {fallbacks[0].get('error')}"
        )
    elif calls:
        st.sidebar.success(f"✅ Nemotron: {len(calls)} real call(s)")
```

3. Store calls after launch — in the `run_clicked` handler after computing `result`:

```python
                st.session_state["llm_calls"] = result.llm_calls
```

(Also add `"llm_calls"` to `init_session2` defaults if not present — it already exists.)

4. **Replace `render_eval_panel`** entirely with a suite-based panel:

```python
def render_eval_panel() -> None:
    """Before/after eval over the fixed 5-prompt suite (real judge)."""
    st.markdown("### 📊 Before / After Evaluation")
    result = st.session_state.get("run_result")
    if result is None or not result.success:
        st.info("Run a job to completion first — eval needs a final config.")
        return

    if st.button("🧪 Run Eval Suite (5 prompts, real Nemotron judge)",
                 use_container_width=True, key="eval_suite_btn"):
        from supertaco.eval.harness import DEFAULT_PROMPTS, build_responses, run_eval_suite

        with st.spinner("Scoring with Nemotron judge..."):
            base_responses, ft_responses = build_responses(result.final_config)
            eval_llm = _make_llm()  # fresh client -> fresh circuit breaker
            results = run_eval_suite(DEFAULT_PROMPTS, base_responses, ft_responses,
                                     llm=eval_llm)
            st.session_state["eval_results"] = results
            st.session_state["eval_llm_calls"] = eval_llm.call_log

    results = st.session_state.get("eval_results")
    if not results:
        return

    base_avg = sum(results["base_scores"]) / len(results["base_scores"])
    ft_avg = sum(results["fine_tuned_scores"]) / len(results["fine_tuned_scores"])
    col1, col2, col3 = st.columns(3)
    col1.metric("Before (base)", f"{base_avg:.2f}")
    col2.metric("After (fine-tuned)", f"{ft_avg:.2f}")
    col3.metric("Improvement", f"{ft_avg - base_avg:+.2f}")
    if results["regression_flagged"]:
        st.error("🚨 EVAL REGRESSION: fine-tuned is >10% below baseline.")
    st.dataframe(
        {"prompt": results["prompts"],
         "base": results["base_scores"],
         "fine_tuned": results["fine_tuned_scores"],
         "judge": results["modes"]},
        use_container_width=True, hide_index=True,
    )
```

5. Replace the call site in `main()`:

```python
    st.markdown("---")
    render_eval_panel(
        st.session_state.get("before_scores", []),
        st.session_state.get("after_scores", []),
    )
```

becomes:

```python
    st.markdown("---")
    render_eval_panel()
```

6. **LLM-calls table** — in the status container after `chart_slot`, add:

```python
        calls = st.session_state.get("llm_calls") or []
        if calls:
            with st.expander(f"🧠 LLM Calls ({len(calls)})"):
                st.dataframe(_llm_rows(calls), use_container_width=True, hide_index=True)
```

7. Remove the old single-prompt hash scoring (`import hashlib` / `_score` inside the old panel) — gone with `render_eval_panel` rewrite. Then remove `List` and `Optional` from the `typing` import at the top of the file if `ruff check` flags them as unused (`Dict` stays — `render_sidebar` returns it).

- [ ] **Step 4: Run full test suite**

```powershell
python -m pytest tests/ -m "not integration" -v
```

Expected: all passed (playbook 4, simlogs 26, llm 3, runner 6, judge 6, cli 1, dashboard 6) = 52 tests.

- [ ] **Step 5: Lint + commit**

```powershell
python -m ruff check supertaco/ui/dashboard.py tests/test_dashboard.py
python -m ruff format supertaco/ui/dashboard.py tests/test_dashboard.py
git add supertaco/ui/dashboard.py tests/test_dashboard.py
git commit -m "feat(dashboard): fixture loader, fallback banner, LLM call table, real judge eval suite"
```

---

### Task 10: Docs, quality gates, live smoke

**Files:**
- Modify: `docs/HANDOFF.md`, `docs/PROJECT.md`
- Verify: full gates

- [ ] **Step 1: Update HANDOFF invariants (§5)**

Replace invariant 4 line:

```markdown
4. Dry-run mode never makes a real Nebius network call (client wrapper may be
   constructed; `launch_job(dry_run=True)` is payload-only).
```

Replace the "Known gaps" bullet "Dashboard is self-contained and does not import
the package" with:

```markdown
- Dashboard consumes `supertaco.runner` (typed-event seam) plus `supertaco.eval`
  for the judge panel; it no longer duplicates pipeline logic.
```

Update Quick Start dashboard command if still stale, and the §2 status table row
`Dashboard` to `✅ Live event-driven panels (dry-run pipeline + real Nemotron)`.

- [ ] **Step 2: Update PROJECT.md recent changes**

Append under `## Recent changes (newest first)`:

```markdown
- 2026-09-24: Functional dashboard: shared event-emitting runner, config-derived
  dry-run logs, real Nemotron calls with fallback, real judge eval, CLI rewired.
```

- [ ] **Step 3: Run every gate**

```powershell
python -m pytest tests/ -m "not integration" -q
python -m ruff check supertaco tests docs/superpowers 2>$null; python -m ruff check supertaco/runner.py supertaco/agent/simlogs.py supertaco/agent/llm.py supertaco/agent/playbook.py supertaco/cli.py supertaco/ui/dashboard.py supertaco/eval/judges.py supertaco/eval/harness.py tests/
python -m ruff format --check supertaco/runner.py supertaco/agent/simlogs.py supertaco/agent/llm.py supertaco/agent/playbook.py supertaco/cli.py supertaco/ui/dashboard.py supertaco/eval/judges.py supertaco/eval/harness.py tests/
python -m mypy supertaco/runner.py supertaco/agent/simlogs.py 2>$null
```

Expected: pytest all green; ruff `All checks passed!` on touched files; mypy
clean on the two new modules if installed (skip with a note if not — repo-wide
mypy debt is out of scope).

- [ ] **Step 4: Server smoke test**

```powershell
$p = Start-Process -FilePath python -ArgumentList "-m","streamlit","run","supertaco/ui/dashboard.py","--server.headless","true","--server.port","8512" -PassThru -WindowStyle Hidden
Start-Sleep -Seconds 8
try { $r = Invoke-WebRequest -Uri "http://localhost:8512" -UseBasicParsing -TimeoutSec 10; Write-Output "HTTP $($r.StatusCode)" } finally { Stop-Process -Id $p.Id -Force }
```

Expected: `HTTP 200`.

- [ ] **Step 5: Optional live Token Factory check (ASK USER FIRST — spends credits)**

Only after explicit user approval: run one nano classification against the real
API from a Python one-liner and confirm `call_log[-1]["mode"] == "real"`.

- [ ] **Step 6: Commit docs + final status**

```powershell
git add docs/HANDOFF.md docs/PROJECT.md
git commit -m "docs: update handoff invariants and status for functional dashboard"
git log --oneline -10
```

---

## Self-review notes (plan author)

- **Spec coverage:** §2 decisions → Tasks 1 (A+B install), 3 (D2), 4+9 (D3), 6+9 (D4), 7 (D5), 5/8 (D6). §5.1 rule table → Task 3 verbatim (fixture-by-fixture classification + healing re-verified against all 8 YAMLs this review). §5.2–5.7 → Tasks 4,5,6,7. §6 error table → validation (T5), circuit breaker (T4), run_failed-not-crash (T5), broad `except Exception` + event preservation (T8), eval gating (T9). §7 test pyramid → Tasks 2–9 (52 tests). §8 docs → Task 10. §9 success criteria → Task 8 asserts + Task 10 smoke. §5.5 checkbox relabel → T8 step 5; before/after diff in timeline → T8 `_render_events`.
- **Type consistency:** `Event(type, data)`, `RunResult(success, attempts, final_config, configs_written, failure_key, error, llm_calls)`, `score_response(...) -> JudgeResult(score, mode, error)`, `generate_logs(config) -> str`, `classify_config_failure(config) -> str|None`, `extract_loss_points(str) -> list[float]` — names identical across tasks.
- **Review-round fixes applied:** plateau trap in Task 2's negative test; simlogs expected count 21→26; judge regex now signed so `-2` clamps to 0.0; CLI test dead imports removed; retry_scheduled timeline rendering used a key the runner never emits; dashboard package imports sorted with `supertaco.settings` deferred into `_make_llm()` (E402 + env-precedence after `_load_env()`); `except Exception` fallback + per-event session stash; `_launch_nan` extra rerun so the sidebar banner test sees post-run state; fixture-loader assertion reads `selectbox.options` not `.label`; fixtures themselves were gitignored — Task 1 now un-ignores and commits them.
- **Known plan-level risks:** Python 3.14 wheel availability for `pip install -e .` (fallback documented Task 1 Step 1); streamlit `st.empty()` streaming inside the post-button block relies on delta-per-call rendering (AppTest Task 8 asserts final state; visual streaming verified in Task 10 smoke by watching the browser); a handful of plan code lines may exceed the 100-char ruff limit — wrap them when `ruff check` flags E501.
