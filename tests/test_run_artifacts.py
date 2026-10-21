"""Schema-v1 per-run artifacts written by run_training_loop (WS4 receipts)."""

import json
import re

import pytest

from supertaco.eval.harness import RESPONSES_END, RESPONSES_START
from supertaco.loop import AttemptResult, run_training_loop

CFG = {"model": "qwen2.5-0.5b", "learning_rate": 2e-4, "batch_size": 8}
HEALTHY_LOG = (
    "config: {'model': 'Qwen/Qwen2.5-0.5B-Instruct'}\n"
    "step 4 loss 1.9218\nstep 8 loss 1.6969\ntraining finished\n"
)


def _responses(prompts):
    payload = {
        "base": {p: "base a" for p in prompts},
        "fine_tuned": {p: "ft a" for p in prompts},
    }
    return f"\n{RESPONSES_START}\n{json.dumps(payload)}\n{RESPONSES_END}"


class FakeTransport:
    def __init__(self, logs=HEALTHY_LOG, error=None):
        self.logs = logs
        self.error = error

    def run_attempt(self, config, prompts, on_line=None):
        text = self.logs
        if self.error is None:
            text += _responses(prompts)
        return AttemptResult(logs=text, error=self.error)


class FakeLLM:
    def __init__(self):
        self.call_log = []

    def classify_failure(self, logs):
        return "NAN_LOSS"

    def propose_patch(self, failure_key, config, logs):
        return {"fix": f"fix-{failure_key}", "reason": "because"}

    def _call(self, model_key, prompt, **kw):
        self.call_log.append(
            {"model_key": model_key, "mode": "real", "tokens": 1, "latency": 0.01, "error": None}
        )
        return "5/10"


def _read(tmp_path, run_id):
    path = tmp_path / "logs" / "runs" / f"{run_id}.json"
    assert path.exists(), f"missing artifact {path}"
    return json.loads(path.read_text(encoding="utf-8"))


def test_success_run_writes_schema_v1_artifact(tmp_path):
    result = run_training_loop(
        dict(CFG),
        transport=FakeTransport(),
        llm=FakeLLM(),
        goal="answer like a pirate chef",
        run_id="artok001",
        config_path="configs/runs/plan_demo.yaml",
    )
    assert result.success is True
    art = _read(tmp_path, "artok001")
    assert art["schema"] == 1
    assert art["run_id"] == "artok001"
    assert art["goal"] == "answer like a pirate chef"
    assert art["config_path"] == "configs/runs/plan_demo.yaml"
    assert art["success"] is True
    assert art["attempts"] == 1
    assert art["failure_keys"] == []
    assert art["healed"] is False
    assert art["error"] is None
    assert art["judge_backend"] == "direct"
    assert art["eval"] is not None
    assert art["eval"]["base_avg"] == art["eval"]["ft_avg"]
    assert art["eval"]["delta"] == 0.0
    assert art["events"][0]["type"] == "run_started"
    assert art["events"][-1]["type"] == "run_succeeded"
    assert art["attempt_ledger"] and art["attempt_ledger"][0]["attempt"] == 1
    assert art["llm_calls"]  # FakeLLM scored the eval through _call
    assert art["wall_clock_s"] >= 0
    assert art["started"] <= art["finished"]


def test_failure_run_writes_artifact_with_error(tmp_path):
    result = run_training_loop(
        dict(CFG),
        transport=FakeTransport(logs="", error="provisioning timeout"),
        llm=FakeLLM(),
        run_id="artfail01",
    )
    assert result.success is False
    art = _read(tmp_path, "artfail01")
    assert art["success"] is False
    assert art["error"] == "provisioning timeout"
    assert art["goal"] is None
    assert art["config_path"] is None
    assert art["eval"] is None
    assert art["judge_backend"] is None
    assert art["events"][0]["type"] == "run_started"
    assert art["events"][-1]["type"] == "run_failed"


def test_artifact_written_on_unexpected_exception(tmp_path):
    class BoomLLM(FakeLLM):
        def classify_failure(self, logs):
            raise RuntimeError("boom")

    transport = FakeTransport(logs="step 4 loss NaN\ntraining finished\n", error="CUDA exploded")
    with pytest.raises(RuntimeError, match="boom"):
        run_training_loop(dict(CFG), transport=transport, llm=BoomLLM(), run_id="artcrash1")
    art = _read(tmp_path, "artcrash1")
    assert art["success"] is False
    assert "RuntimeError: boom" in art["error"]
    assert art["attempts"] == 1
    assert art["failure_keys"] == ["NAN_LOSS"]
    assert art["events"]  # partial timeline kept


def test_default_run_generates_run_id_and_null_goal(tmp_path, monkeypatch):
    monkeypatch.delenv("SUPERTACO_RUN_ID", raising=False)
    result = run_training_loop(dict(CFG), transport=FakeTransport(), llm=FakeLLM())
    assert result.success is True
    files = list((tmp_path / "logs" / "runs").glob("*.json"))
    assert len(files) == 1
    art = json.loads(files[0].read_text(encoding="utf-8"))
    assert re.fullmatch(r"[0-9a-f]{8}", art["run_id"])
    assert art["goal"] is None
    assert art["config_path"] is None


def test_run_id_contextvar_not_leaked_after_success_and_crash(tmp_path, monkeypatch):
    from supertaco import runlog

    monkeypatch.delenv("SUPERTACO_RUN_ID", raising=False)
    run_training_loop(dict(CFG), transport=FakeTransport(), llm=FakeLLM(), run_id="leaksucc1")
    assert runlog.current_run_id() is None

    class BoomLLM(FakeLLM):
        def classify_failure(self, logs):
            raise RuntimeError("boom")

    transport = FakeTransport(logs="step 4 loss NaN\ntraining finished\n", error="CUDA exploded")
    with pytest.raises(RuntimeError, match="boom"):
        run_training_loop(dict(CFG), transport=transport, llm=BoomLLM(), run_id="leakcrash")
    assert runlog.current_run_id() is None


def test_wrapper_resolves_run_id_from_env(tmp_path, monkeypatch):
    monkeypatch.setenv("SUPERTACO_RUN_ID", "envart99")
    result = run_training_loop(dict(CFG), transport=FakeTransport(), llm=FakeLLM())
    assert result.success is True
    path = tmp_path / "logs" / "runs" / "envart99.json"
    assert path.exists(), f"missing artifact {path}"
    art = json.loads(path.read_text(encoding="utf-8"))
    assert art["run_id"] == "envart99"


def test_success_path_receipt_failure_never_fails_run(tmp_path, monkeypatch):
    import supertaco.runlog as runlog

    def _boom(run_id, data):
        raise RuntimeError("receipt exploded")

    monkeypatch.setattr(runlog, "write_run_artifact", _boom)
    with pytest.warns(UserWarning, match="run artifact write failed"):
        result = run_training_loop(
            dict(CFG), transport=FakeTransport(), llm=FakeLLM(), run_id="artreceipt"
        )
    assert result.success is True  # receipt failure must never fail a successful run


def test_crash_path_receipt_failure_still_raises_primary(tmp_path, monkeypatch):
    import supertaco.runlog as runlog

    def _boom(run_id, data):
        raise RuntimeError("receipt exploded")

    monkeypatch.setattr(runlog, "write_run_artifact", _boom)

    class BoomLLM(FakeLLM):
        def classify_failure(self, logs):
            raise RuntimeError("boom")

    transport = FakeTransport(logs="step 4 loss NaN\ntraining finished\n", error="CUDA exploded")
    with pytest.raises(RuntimeError, match="boom"):
        run_training_loop(dict(CFG), transport=transport, llm=BoomLLM(), run_id="artmask01")
