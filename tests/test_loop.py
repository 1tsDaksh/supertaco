"""Real training loop: transport seam, events, budget, eval gate."""

import json
import tempfile

import pytest

from supertaco.errors import ConfigurationError
from supertaco.eval.harness import DEFAULT_PROMPTS, RESPONSES_END, RESPONSES_START
from supertaco.loop import (
    AttemptResult,
    extract_loss_points,
    run_training_loop,
    validate_config,
)


def _tmp_runs():
    d = tempfile.mkdtemp(prefix="st_runs_")
    return d


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
        self.call_log.append(
            {"model_key": "classify", "mode": "real", "tokens": 1, "latency": 0.01, "error": None}
        )
        return "NAN_LOSS"

    def propose_patch(self, failure_key, config, logs):
        self.call_log.append(
            {"model_key": "patch", "mode": "real", "tokens": 1, "latency": 0.01, "error": None}
        )
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


# ---------- Task 3: run_training_loop ----------


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


# ---------- Task 3 review fixes ----------


def test_transport_exception_fails_fast_with_ledger():
    transport = FakeTransport([make_result(HEALTHY_LOG)], fail_on=[1])
    events, on_event = collect_events()
    result = run_training_loop(
        {"learning_rate": 2e-4, "model": "qwen2.5-0.5b"},
        transport=transport,
        llm=FakeLLM(),
        on_event=on_event,
    )
    assert result.success is False
    assert "transport exploded" in (result.error or "")
    assert result.attempts == 1
    assert [e.type for e in events][-1] == "run_failed"
    assert len(result.attempt_ledger) == 1  # evidence survives fail-fast (fix 3)


def test_non_dict_config_never_raises():
    transport = FakeTransport([make_result(HEALTHY_LOG)])
    result = run_training_loop(
        "not-a-dict",
        transport=transport,
        llm=FakeLLM(),
        on_event=lambda ev: None,
    )
    assert result.success is False
    assert "non-empty mapping" in (result.error or "")
    assert transport.calls == []


def test_negative_max_retries_returns_failed_result():
    transport = FakeTransport([make_result(HEALTHY_LOG)])
    result = run_training_loop(
        {"learning_rate": 2e-4, "model": "qwen2.5-0.5b"},
        transport=transport,
        llm=FakeLLM(),
        max_retries=-1,
    )
    assert result.success is False
    assert "MaxRetriesExceeded" in (result.error or "")
    assert result.attempts == 0


# ---------- Task 4: plan pins ----------


def test_nan_failure_patches_and_heals():
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
    assert (
        types.index("failure_detected")
        < types.index("patch_written")
        < types.index("retry_scheduled")
    )
    assert types.count("job_launched") == 2
    assert result.final_config["learning_rate"] == 0.01  # playbook NAN_LOSS: lr / 10
    assert len(result.configs_written) == 1
    rec = result.attempt_ledger[0]
    assert rec.failure_key == "NAN_LOSS"
    assert rec.verdict == "NAN_LOSS"
    assert rec.config_after["learning_rate"] == 0.01
    # classify-mode pin (review addendum 7 — needs the FakeLLM change):
    classified = next(e for e in events if e.type == "classified")
    assert classified.data["mode"] == "real"
    assert classified.data["diverged"] is False
    assert transport.calls[0]["learning_rate"] == 0.1  # first attempt ran the broken config
    assert transport.calls[1]["learning_rate"] == 0.01  # relaunch received the PATCHED config
    assert "patch_proposed" in types  # frozen 10-event catalog includes this
    assert len(result.attempt_ledger[0].loss_points) == 2  # NAN_LOG has 2 loss lines; D5 evidence


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
    # patched configs feed RELAUNCHES 2 and 3; attempt 3 breaks before patching
    assert len(result.configs_written) == 2


# ---------- Task 4: review-required tests ----------


def test_responses_missing_prompt_key_honest_failure():
    prompts_only_first = {
        "base": {DEFAULT_PROMPTS[0]: "base answer"},
        "fine_tuned": {DEFAULT_PROMPTS[0]: "ft answer"},
    }
    transport = FakeTransport([make_result(HEALTHY_LOG, prompts_only_first)])
    events, on_event = collect_events()
    result = run_training_loop(
        {"learning_rate": 2e-4, "model": "qwen2.5-0.5b"},
        transport=transport,
        llm=FakeLLM(),
        on_event=on_event,
    )
    assert result.success is False
    assert "missing prompt key" in (result.error or "")
    assert len(result.attempt_ledger) == 1
    assert [e.type for e in events][-1] == "run_failed"


def test_failed_result_carries_prior_eval_results():
    transport = FakeTransport(
        [
            lambda prompts: make_result(HEALTHY_LOG, responses_for(prompts)),
            make_result(HEALTHY_LOG),
        ],
        fail_on=[2],
    )
    result = run_training_loop(
        {
            "learning_rate": 2e-4,
            "lora_r": 8,
            "lora_alpha": 16,
            "num_epochs": 3,
            "model": "qwen2.5-0.5b",
        },
        transport=transport,
        llm=FakeLLM(healthy=False),
        on_event=lambda ev: None,
        runs_dir=str(_tmp_runs()),
    )
    assert result.success is False
    assert "transport exploded" in (result.error or "")
    assert len(result.attempt_ledger) == 2
    assert result.attempt_ledger[0].failure_key == "EVAL_REGRESSION"
    assert result.attempt_ledger[1].failure_key is None
    assert result.eval_results is not None
    assert result.eval_results["regression_flagged"] is True


def test_error_result_captures_partial_log_points():
    transport = FakeTransport(
        [AttemptResult(logs="step 4 loss 1.9\ndownload failed", error="download failed")]
    )
    result = run_training_loop(
        {"learning_rate": 2e-4, "model": "qwen2.5-0.5b"},
        transport=transport,
        llm=FakeLLM(),
        on_event=lambda ev: None,
    )
    assert result.success is False
    assert len(result.attempt_ledger) == 1
    assert len(result.attempt_ledger[0].loss_points) == 1
    assert result.eval_results is None


def test_error_result_with_none_logs_never_raises():
    transport = FakeTransport([AttemptResult(logs=None, error="download failed")])
    result = run_training_loop(
        {"learning_rate": 2e-4, "model": "qwen2.5-0.5b"},
        transport=transport,
        llm=FakeLLM(),
        on_event=lambda ev: None,
    )
    assert result.success is False
    assert "download failed" in (result.error or "")
    assert len(result.attempt_ledger) == 1


def test_run_started_emitted_before_llm_construction(monkeypatch):
    events, on_event = collect_events()
    seen_at_construction: list[str] = []

    class RecordingLLM(FakeLLM):
        def __init__(self):
            super().__init__()
            seen_at_construction.extend(e.type for e in events)

    monkeypatch.setattr("supertaco.loop.make_llm", RecordingLLM)
    result = run_training_loop(
        {"learning_rate": 2e-4, "model": "qwen2.5-0.5b"},
        transport=FakeTransport([make_result(HEALTHY_LOG)]),  # no responses block -> honest failure
        on_event=on_event,
    )
    assert result.success is False
    assert seen_at_construction, "make_llm must be constructed (proves llm=None path)"
    assert seen_at_construction[0] == "run_started"
