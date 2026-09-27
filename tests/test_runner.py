"""Runner: event sequences, healing loop, retry cap, validation."""

from pathlib import Path

import pytest
import yaml

from supertaco.errors import ConfigurationError
from supertaco.runner import Event, RunResult, run  # noqa: F401

HEALTHY = {
    "learning_rate": 2e-5,
    "batch_size": 8,
    "lora_r": 8,
    "lora_alpha": 16,
    "num_epochs": 3,
    "gradient_clip_norm": 1.0,
    "chat_template": "llama-3",
}
BROKEN_NAN = {
    "learning_rate": 0.1,
    "batch_size": 8,
    "lora_r": 8,
    "lora_alpha": 16,
    "num_epochs": 3,
    "chat_template": None,
}


class FakeLLM:
    """Minimal NemotronClient stand-in used by runner tests."""

    def __init__(self, verdict: str | None = None):
        self.verdict = verdict
        self.call_log: list[dict] = []

    def classify_failure(self, logs: str) -> str:
        self.call_log.append(
            {"model_key": "classify", "mode": "real", "tokens": 1, "latency": 0.1, "error": None}
        )
        return self.verdict or "NAN_LOSS"

    def propose_patch(self, failure_key: str, config: dict, logs: str) -> dict:
        self.call_log.append(
            {"model_key": "patch", "mode": "real", "tokens": 1, "latency": 0.2, "error": None}
        )
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
    assert [e.type for e in events] == [
        "run_started",
        "job_launched",
        "logs_produced",
        "run_succeeded",
    ]
    logs_event = events[2]
    assert len(logs_event.data["loss_points"]) == 30


def test_heal_path_writes_patch_file(tmp_path):
    events, on_event = _collect()
    result = run(BROKEN_NAN, on_event=on_event, llm=FakeLLM(verdict="NAN_LOSS"), runs_dir=tmp_path)
    assert result.success is True
    assert result.attempts == 2
    assert len(result.configs_written) == 1
    path = Path(result.configs_written[0])
    assert path.exists()
    patched = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert patched["learning_rate"] == pytest.approx(0.01)
    types = [e.type for e in events]
    assert types == [
        "run_started",
        "job_launched",
        "logs_produced",
        "failure_detected",
        "classified",
        "patch_proposed",
        "patch_written",
        "retry_scheduled",
        "job_launched",
        "logs_produced",
        "run_succeeded",
    ]
    assert events[3].data["failure_key"] == "NAN_LOSS"
    assert events[4].data["nemotron_verdict"] == "NAN_LOSS"
    assert events[4].data["diverged"] is False
    assert result.llm_calls[-1]["model_key"] == "patch"


def test_retry_cap_emits_run_failed(tmp_path):
    events, on_event = _collect()

    def always_nan(config: dict) -> str:
        return "step 0 loss NaN"

    result = run(
        BROKEN_NAN,
        on_event=on_event,
        llm=FakeLLM(),
        log_fn=always_nan,
        max_retries=3,
        runs_dir=tmp_path,
    )
    assert result.success is False
    assert result.attempts == 4  # initial + 3 relaunches (spec 4.5)
    assert "MaxRetriesExceeded" in result.error
    assert events[-1].type == "run_failed"
    assert len(result.configs_written) == 3  # one patch per relaunch decision
    assert sum(1 for e in events if e.type == "patch_written") == 3


def test_divergence_flag_when_nemotron_disagrees(tmp_path):
    events, on_event = _collect()
    run(BROKEN_NAN, on_event=on_event, llm=FakeLLM(verdict="OOM"), runs_dir=tmp_path)
    classified = [e for e in events if e.type == "classified"][0]
    assert classified.data["diverged"] is True
    assert classified.data["nemotron_verdict"] == "OOM"


def test_real_jobs_not_implemented_returns_failed_result(tmp_path):
    events, on_event = _collect()
    result = run(HEALTHY, dry_run=False, on_event=on_event, llm=FakeLLM(), runs_dir=tmp_path)
    assert result.success is False
    assert "Gate 1" in result.error
    assert events[-1].type == "run_failed"
