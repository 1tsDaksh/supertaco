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
        self.last_config = None

    def run_attempt(self, config, prompts, on_line=None):
        from supertaco.loop import AttemptResult

        self.calls += 1
        self.last_config = dict(config)
        idx = min(self.calls - 1, len(self.results) - 1)
        item = self.results[idx]
        if isinstance(item, Exception):
            raise item
        if len(item) == 4:
            logs, payload, error, artifact = item
        else:
            logs, payload, error = item
            artifact = None
        text = logs
        if payload:
            block = json.dumps(
                {"base": {p: "base a" for p in prompts}, "fine_tuned": {p: "ft a" for p in prompts}}
            )
            text += f"\n{RESPONSES_START}\n{block}\n{RESPONSES_END}"
        return AttemptResult(logs=text, artifact=artifact, error=error)


class StreamingFakeTransport(FakeTransport):
    """FakeTransport that also drives the live on_line streaming path."""

    def run_attempt(self, config, prompts, on_line=None):
        res = super().run_attempt(config, prompts, on_line=None)
        if on_line:
            on_line("step 4 loss 1.9218")
            on_line("step 8 loss 1.6969")
            on_line("training finished")
        return res


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
    artifact_path = tmp_path / "eval.zip"
    artifact_path.write_bytes(b"PK\x03\x04fake")
    _patch_run(
        monkeypatch,
        FakeTransport([(HEALTHY_LOG, "ok", None, artifact_path)]),
        FakeLLM(),
    )
    _launch(app)
    assert not app.exception, [e.value for e in app.exception]
    result = app.session_state["run_result"]
    assert result.success is True
    assert result.eval_results is not None
    assert result.attempt_ledger[0].failure_key is None
    assert any("Training Report" in m.value for m in app.markdown)
    assert len(app.dataframe) >= 2  # attempt ledger + judge table
    assert len(app.metric) >= 3  # judge base / fine-tuned / improvement
    assert len(app.get("download_button")) == 1  # adapter artifact download


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


def test_live_streaming_survives_multiple_loss_lines(app, monkeypatch):
    _patch_run(monkeypatch, StreamingFakeTransport([(HEALTHY_LOG, "ok", None)]), FakeLLM())
    _launch(app)
    assert not app.exception, [e.value for e in app.exception]
    result = app.session_state["run_result"]
    assert result.success is True, result.error  # would be False with the duplicate-key bug
    assert any("training finished" in c.value for c in app.code)  # streamed logs landed


def test_knobs_overlay_config_yaml(app, monkeypatch):
    transport = FakeTransport([(HEALTHY_LOG, "ok", None)])
    _patch_run(monkeypatch, transport, FakeLLM())
    app.session_state["config"] = {
        "learning_rate": 0.0007,
        "lora_r": 32,
        "model": "qwen2.5-0.5b",
        "batch_size": 16,
    }
    app.run()
    _launch(app)
    assert not app.exception, [e.value for e in app.exception]
    seen = transport.last_config
    # spec D2: knobs overlay the YAML — even at their defaults
    assert seen["learning_rate"] == pytest.approx(2e-4)  # knob default beats YAML 0.0007
    assert seen["batch_size"] == 4  # knob default beats YAML 16
    assert seen["lora_r"] == 8  # knob default beats YAML 32
    assert seen["model"] == "qwen2.5-0.5b"
    # an explicitly-set knob wins over both YAML and the default
    lr_knob = next(w for w in app.number_input if w.label == "learning_rate")
    lr_knob.set_value(0.0009)
    _launch(app)
    assert not app.exception, [e.value for e in app.exception]
    assert transport.last_config["learning_rate"] == pytest.approx(0.0009)


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
