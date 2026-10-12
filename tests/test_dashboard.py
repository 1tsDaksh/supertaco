"""AppTest: config -> real loop (faked transport + LLM) -> report."""

import json
import re
from pathlib import Path

import pytest
import yaml
from streamlit.testing.v1 import AppTest

from supertaco.eval.harness import DEFAULT_PROMPTS, RESPONSES_END, RESPONSES_START

REPO = Path(__file__).resolve().parents[1]
DASHBOARD = REPO / "supertaco" / "ui" / "dashboard.py"

HEALTHY_LOG = (
    "config: {'model': 'Qwen/Qwen2.5-0.5B-Instruct'}\ngpu: Tesla T4\n"
    "step 4 loss 1.9218\nstep 8 loss 1.6969\nstep 12 loss 1.3245\n"
    "step 32 loss 1.3245\ntraining finished\n"
)
NAN_LOG = "step 4 loss NaN\nstep 8 loss NaN\ntraining finished\n"
OOM_CRASH = (
    "Traceback (most recent call last):\n"
    "RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB\n"
)


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


class UnparseableJudgeLLM(FakeLLM):
    """Real replies (call_log mode real) that the judge cannot parse."""

    def _call(self, model_key, prompt, **kw):
        self.call_log.append(
            {"model_key": model_key, "mode": "real", "tokens": 1, "latency": 0.01, "error": None}
        )
        return "no numbers in this reply"


class TransportFailLLM(FakeLLM):
    """HTTP-level failure: call_log mode fallback, simulated unparseable replies."""

    def _call(self, model_key, prompt, **kw):
        self.call_log.append(
            {
                "model_key": model_key,
                "mode": "fallback",
                "tokens": 0,
                "latency": 0.0,
                "error": "ConnectError: connection refused",
            }
        )
        return '{"fix": "lower_learning_rate", "reason": "model suggests LR reduction"}'


def test_page_config_declared_before_any_other_st_call():
    src = DASHBOARD.read_text(encoding="utf-8")
    idx = src.index("st.set_page_config(")
    assert idx > 0
    assert not re.search(r"\bst\.\w+\s*\(", src[:idx])


def test_default_config_preloaded_into_session(app):
    expected = yaml.safe_load(
        (REPO / "configs" / "defaults" / "colab_t4.yaml").read_text(encoding="utf-8-sig")
    )
    assert app.session_state["config"] == expected
    assert app.session_state["config"]["model"] == "qwen2.5-0.5b"


def test_default_config_helper_returns_empty_on_read_error(monkeypatch):
    import supertaco.ui.dashboard as dash

    def _boom(*args, **kwargs):
        raise OSError("disk on fire")

    monkeypatch.setattr(dash.yaml, "safe_load", _boom)
    assert dash._default_config() == {}


def test_sidebar_omits_vestigial_tavily_caption(app):
    assert not any("TAVILY" in c.value for c in app.sidebar.caption)


def test_hash_judge_alert_says_unparseable_not_llm_unavailable(app, monkeypatch):
    _patch_run(monkeypatch, FakeTransport([(HEALTHY_LOG, "ok", None)]), UnparseableJudgeLLM())
    _launch(app)
    assert not app.exception, [e.value for e in app.exception]
    warnings_text = "\n".join(w.value for w in app.warning)
    assert "unparseable" in warnings_text
    assert "LLM unavailable" not in warnings_text
    # the call itself was real — sidebar and report must not contradict each other
    app.run()  # sidebar renders before the run in a pass; re-render to see its calls
    assert any("real call" in s.value for s in app.sidebar.success)
    judge_df = next(df for df in app.dataframe if "judge" in df.value.columns)
    assert all("†" in str(m) for m in judge_df.value["judge"])
    assert any("†" in c.value for c in app.caption)


def test_http_failure_alert_distinguishes_transport_from_hash(app, monkeypatch):
    _patch_run(monkeypatch, FakeTransport([(HEALTHY_LOG, "ok", None)]), TransportFailLLM())
    _launch(app)
    assert not app.exception, [e.value for e in app.exception]
    warnings_text = "\n".join(w.value for w in app.warning)
    assert "HTTP failure" in warnings_text
    assert "unparseable" not in warnings_text


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


def test_clean_first_attempt_says_no_heal_needed(app, monkeypatch):
    _patch_run(monkeypatch, FakeTransport([(HEALTHY_LOG, "ok", None)]), FakeLLM())
    _launch(app)
    assert not app.exception, [e.value for e in app.exception]
    result = app.session_state["run_result"]
    assert result.success is True
    assert result.attempts == 1
    md = "\n".join(m.value for m in app.markdown)
    assert "first attempt" in md
    assert "Healed" not in md
    assert app.session_state["run_active"] is False  # flag round-trips after the run


def test_oom_crash_heal_says_healed_after_2_and_advisory_caption(app, monkeypatch):
    _patch_run(
        monkeypatch,
        FakeTransport([(OOM_CRASH, None, "artifact download missing"), (HEALTHY_LOG, "ok", None)]),
        FakeLLM(),
    )
    _launch(app)
    assert not app.exception, [e.value for e in app.exception]
    result = app.session_state["run_result"]
    assert result.success is True
    assert result.attempts == 2
    md = "\n".join(m.value for m in app.markdown)
    assert "Healed" in md
    assert "after 2 attempt(s)" in md
    assert result.attempt_ledger[0].failure_key == "OOM"
    assert any("advisory" in c.value for c in app.caption)


def test_run_in_progress_status_while_active(app, monkeypatch):
    transport = FakeTransport([(HEALTHY_LOG, "ok", None)])
    _patch_run(monkeypatch, transport, FakeLLM())

    import streamlit as st

    import supertaco.loop as loop_mod

    seen = {}
    real_loop = loop_mod.run_training_loop

    def spy(*args, **kwargs):
        seen["active"] = st.session_state.get("run_active")
        return real_loop(*args, **kwargs)

    monkeypatch.setattr(loop_mod, "run_training_loop", spy)
    _launch(app)
    assert not app.exception, [e.value for e in app.exception]
    assert seen["active"] is True  # set before the blocking loop
    assert app.session_state["run_active"] is False  # cleared in finally


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


class PlanFakeLLM(FakeLLM):
    """Judge behavior from FakeLLM plus planner/generation fakes."""

    def __init__(self):
        super().__init__()
        self._row_n = 0

    def plan_goal(self, goal):
        return json.dumps(
            {
                "model": "qwen2.5-1.5b",
                "learning_rate": 5e-4,
                "batch_size": 8,
                "lora_r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.1,
                "num_epochs": 2,
                "dataset": {"kind": "synthetic"},
                "eval_prompts": ["gp1", "gp2", "gp3", "gp4", "gp5"],
            }
        )

    def generate_rows(self, goal, count, banned):
        start = self._row_n
        rows = [
            {"instruction": f"pirate request {n}", "input": "", "output": f"arr {n}"}
            for n in range(start, start + count)
        ]
        self._row_n = start + count
        return json.dumps(rows)


def test_goal_plan_then_run_threads_dataset_and_prompts(app, monkeypatch):
    class Recorder(FakeTransport):
        def run_attempt(self, config, prompts, on_line=None):
            self.prompts = list(prompts)
            return super().run_attempt(config, prompts, on_line)

    transport = Recorder([(HEALTHY_LOG, "ok", None)])
    _patch_run(monkeypatch, transport, PlanFakeLLM())

    goal_box = next(ta for ta in app.text_area if "model learn" in ta.label)
    goal_box.set_value("answer like a pirate chef").run()
    next(b for b in app.button if "Plan with Nemotron" in b.label).click().run()
    assert not app.exception, [e.value for e in app.exception]
    assert any("Plan ready" in s.value for s in app.success)
    assert any('Planned for: "answer like a pirate chef"' in s.value for s in app.success)
    assert app.session_state["plan_goal"] == "answer like a pirate chef"
    assert app.session_state["plan_dataset"]["kind"] == "synthetic"
    assert len(app.session_state["plan_dataset"]["rows"]) == 150
    assert app.session_state["plan_prompts"] == ["gp1", "gp2", "gp3", "gp4", "gp5"]
    assert app.session_state["config"]["model"] == "qwen2.5-1.5b"
    assert app.session_state["knobs"]["learning_rate"] == pytest.approx(5e-4)

    _launch(app)
    assert not app.exception, [e.value for e in app.exception]
    seen = transport.last_config
    assert seen["dataset"]["kind"] == "synthetic"
    assert len(seen["dataset"]["rows"]) == 150
    assert seen["learning_rate"] == pytest.approx(5e-4)  # plan knobs survive the overlay
    assert seen["model"] == "qwen2.5-1.5b"
    assert transport.prompts == ["gp1", "gp2", "gp3", "gp4", "gp5"]
    assert app.session_state["run_result"].success is True


def test_prompts_editor_labels_wrong_counts(app, monkeypatch):
    _patch_run(monkeypatch, FakeTransport([(HEALTHY_LOG, "ok", None)]), PlanFakeLLM())
    goal_box = next(ta for ta in app.text_area if "model learn" in ta.label)
    goal_box.set_value("g").run()
    next(b for b in app.button if "Plan with Nemotron" in b.label).click().run()
    assert not app.exception, [e.value for e in app.exception]

    editor = next(ta for ta in app.text_area if "Eval prompts" in ta.label)
    editor.set_value("p1\np2\np3\np4\np5\np6").run()
    assert not app.exception, [e.value for e in app.exception]
    assert any("Expected exactly 5" in w.value for w in app.warning)
    assert app.session_state["plan_prompts"] == ["p1", "p2", "p3", "p4", "p5", "p6"]

    # The write-back above changed session content, so the editor's value-derived
    # element id shifts on the next run - re-render once so set_value("") below
    # targets the id the following run will actually compute.
    app.run()
    editor = next(ta for ta in app.text_area if "Eval prompts" in ta.label)
    editor.set_value("").run()
    assert not app.exception, [e.value for e in app.exception]
    assert app.session_state["plan_prompts"] == []
    assert any("Prompts cleared" in c.value for c in app.caption)
    # editor stays recoverable after clearing
    assert any("Eval prompts" in ta.label for ta in app.text_area)


def test_plan_fallback_shows_warning(app, monkeypatch):
    class FallbackLLM(PlanFakeLLM):
        def plan_goal(self, goal):
            return "I cannot help with that."

    _patch_run(monkeypatch, FakeTransport([(HEALTHY_LOG, "ok", None)]), FallbackLLM())
    goal_box = next(ta for ta in app.text_area if "model learn" in ta.label)
    goal_box.set_value("g").run()
    next(b for b in app.button if "Plan with Nemotron" in b.label).click().run()
    assert not app.exception, [e.value for e in app.exception]
    assert any("fell back" in w.value for w in app.warning)
    assert any('Planned for: "g"' in w.value for w in app.warning)
    assert app.session_state["plan_goal"] == "g"
    assert app.session_state["plan_prompts"] == DEFAULT_PROMPTS
