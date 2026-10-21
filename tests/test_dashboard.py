"""AppTest: config -> real loop (faked transport + LLM) -> report."""

import json
import re
from pathlib import Path

import pytest
import yaml
from streamlit.testing.v1 import AppTest

from supertaco.eval.harness import DEFAULT_PROMPTS, RESPONSES_END, RESPONSES_START
from supertaco.loop import LoopResult

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
    at.session_state["judge_on_nebius"] = False
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


def test_sidebar_shows_tavily_status_caption(app):
    import supertaco.ui.dashboard as dash
    from supertaco.ui.dashboard import _mask_key

    caps = [c.value for c in app.sidebar.caption]
    status = next(c for c in caps if c.startswith("🔎 Tavily citations:"))
    key = dash._TAVILY_API_KEY
    if key:
        assert f"on ({_mask_key(key)})" in status
        assert "****" in status
        assert key not in status  # full key never rendered
    else:
        assert status == "🔎 Tavily citations: off (no TAVILY_API_KEY)"
    # mask edges (folded in to keep the suite count at +2 total for this plan)
    assert _mask_key("tvly-short") == "****"  # len < 12
    assert _mask_key("tvly-abcdefghijkl") == "tvly-****ijkl"


def test_sidebar_omits_api_key_block(app):
    texts = [m.value for m in app.sidebar.markdown] + [c.value for c in app.sidebar.caption]
    assert not any("API Keys" in t for t in texts)
    assert not any("NEBIUS_PROJECT_ID" in t for t in texts)


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


class MixedJudgeLLM(FakeLLM):
    """Prompt 2: base transport-fails, ft real-unparseable. Others: real 10/10."""

    def _call(self, model_key, prompt, **kw):
        mode, error, reply = "real", None, "10/10"
        if "Write a Python function" in prompt:
            if "ft answer" in prompt:
                reply = "no numbers in this reply"
            else:
                mode, error, reply = "fallback", "ConnectError: connection refused", "no numbers"
        self.call_log.append(
            {"model_key": model_key, "mode": mode, "tokens": 1, "latency": 0.01, "error": error}
        )
        return reply


def test_mixed_hash_and_fallback_cells_render_both_warnings_with_selective_dagger(app, monkeypatch):
    class AnswerTransport(FakeTransport):
        def run_attempt(self, config, prompts, on_line=None):
            from supertaco.loop import AttemptResult

            self.calls += 1
            block = json.dumps(
                {
                    "base": {p: f"base answer to {p}" for p in prompts},
                    "fine_tuned": {p: f"ft answer to {p}" for p in prompts},
                }
            )
            text = f"{HEALTHY_LOG}\n{RESPONSES_START}\n{block}\n{RESPONSES_END}"
            return AttemptResult(logs=text, artifact=None, error=None)

    _patch_run(monkeypatch, AnswerTransport([(HEALTHY_LOG, "ok", None)]), MixedJudgeLLM())
    _launch(app)
    assert not app.exception, [e.value for e in app.exception]
    result = app.session_state["run_result"]
    assert result.success is True  # hash of prompt-2 responses must not trip regression
    warnings_text = "\n".join(w.value for w in app.warning)
    assert "HTTP failure" in warnings_text
    assert "unparseable" in warnings_text
    judge_df = next(df for df in app.dataframe if "judge" in df.value.columns)
    cells = [str(m) for m in judge_df.value["judge"]]
    assert cells.count("base:fallback / ft:hash †") == 1
    assert cells.count("base:real / ft:real") == 4
    for cell in cells:
        assert cell.count("†") == (1 if "hash" in cell else 0)
        assert "fallback †" not in cell
        assert "real †" not in cell


def test_config_yaml_read_only_when_key_missing(monkeypatch, offline):
    reads = []
    real_read_text = Path.read_text

    def spy(path_self, *args, **kwargs):
        if "colab_t4" in str(path_self):
            reads.append(str(path_self))
        return real_read_text(path_self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", spy)
    at = AppTest.from_file(str(DASHBOARD), default_timeout=60)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(reads) == 1  # first load preloads the shipped default
    at.run()
    assert len(reads) == 1  # config already in session state: no re-read


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


def test_feedback_ui_hidden_without_run_result(app):
    assert not any("Re-run with feedback" in b.label for b in app.button)
    assert not any(ta.label == "Feedback on the result" for ta in app.text_area)


def _seed_plan_keys(app):
    app.session_state["plan_goal"] = "pirate chef"
    app.session_state["plan_dataset"] = {
        "kind": "synthetic",
        "rows": [{"instruction": "old row", "input": "", "output": "arr"}],
    }
    app.session_state["plan_prompts"] = ["a", "b", "c", "d", "e"]


def _prev_result(ft_scores, modes):
    """A LoopResult standing in for a stashed pre-feedback run."""
    return LoopResult(
        success=True,
        attempts=1,
        final_config={},
        configs_written=[],
        failure_key=None,
        error=None,
        llm_calls=[],
        eval_results={
            "prompts": [f"p{i}" for i in range(len(modes))],
            "base_scores": [1.0] * len(modes),
            "fine_tuned_scores": ft_scores,
            "modes": modes,
            "regression_flagged": False,
        },
    )


def _patch_feedback(monkeypatch, *, plan_mode="real", note="shorter, tighter"):
    """Fake revise_run_config + capture the config the re-run receives."""
    import supertaco.loop as loop_mod
    import supertaco.planner as planner_mod
    from supertaco.planner import DatasetSpec, RunPlan, plan_to_config

    seen = {}

    def fake_revise(feedback, config, goal, llm):
        seen["feedback"] = feedback
        seen["prior_cfg"] = dict(config)
        seen["goal"] = goal
        revised = RunPlan(
            model="Qwen/Qwen2.5-1.5B-Instruct",
            learning_rate=3e-4,
            batch_size=2,
            lora_r=16,
            lora_alpha=32,
            num_epochs=2,
            dataset=DatasetSpec(
                kind="synthetic", rows=[{"instruction": "new row", "input": "", "output": "arr"}]
            ),
            eval_prompts=["a", "b", "c", "d", "e"],
            plan_mode=plan_mode,
            revision_note=note,
        )
        new_cfg = plan_to_config(revised)
        new_cfg["goal"] = goal
        new_cfg["eval_prompts"] = list(revised.eval_prompts)
        if note:
            new_cfg["revision_note"] = note
        return new_cfg, revised

    def fake_loop(config, **kwargs):
        seen["run_config"] = dict(config)
        seen["run_prompts"] = kwargs.get("prompts")
        return LoopResult(
            success=True,
            attempts=1,
            final_config=dict(config),
            configs_written=[],
            failure_key=None,
            error=None,
            llm_calls=[],
            eval_results={
                "prompts": ["a"],
                "base_scores": [5.0],
                "fine_tuned_scores": [8.0],
                "modes": ["base:real / ft:real"],
                "regression_flagged": False,
            },
        )

    monkeypatch.setattr(planner_mod, "revise_run_config", fake_revise)
    monkeypatch.setattr(loop_mod, "run_training_loop", fake_loop)
    return seen


def test_feedback_button_revises_and_reruns(app, monkeypatch):
    _patch_run(monkeypatch, FakeTransport([(HEALTHY_LOG, "ok", None)]), FakeLLM())
    _launch(app)
    assert not app.exception, [e.value for e in app.exception]
    old_result = app.session_state["run_result"]
    assert old_result.success is True

    _seed_plan_keys(app)
    seen = _patch_feedback(monkeypatch)
    app.run()  # feedback box appears now that a run finished

    next(sb for sb in app.sidebar.selectbox if sb.label == "model").set_value("tinyllama-1.1b")
    fb = next(ta for ta in app.text_area if ta.label == "Feedback on the result")
    fb.set_value("be more concise")
    next(b for b in app.button if "Re-run with feedback" in b.label).click().run()
    assert not app.exception, [e.value for e in app.exception]

    # revision input: feedback + goal + prior cfg carrying the sidebar model + old dataset
    assert seen["feedback"] == "be more concise"
    assert seen["goal"] == "pirate chef"
    assert seen["prior_cfg"]["model"] == "tinyllama-1.1b"
    assert seen["prior_cfg"]["dataset"]["rows"][0]["instruction"] == "old row"

    # ordering: the re-run received the REVISED config, not the pre-revision snapshot
    run_cfg = seen["run_config"]
    assert run_cfg["model"] == "qwen2.5-1.5b"  # reverse-mapped; beats the stale model_sel
    assert run_cfg["num_epochs"] == 2
    assert run_cfg["batch_size"] == 2
    assert run_cfg["learning_rate"] == pytest.approx(3e-4)
    assert run_cfg["dataset"]["rows"][0]["instruction"] == "new row"
    assert seen["run_prompts"] == ["a", "b", "c", "d", "e"]

    assert app.session_state["plan_goal"] == "pirate chef"
    assert app.session_state["config"]["model"] == "qwen2.5-1.5b"
    assert app.session_state["knobs"]["num_epochs"] == 2
    assert app.session_state["plan_dataset"]["rows"][0]["instruction"] == "new row"
    assert any('Revised with feedback: "be more concise"' in s.value for s in app.success)
    assert any("shorter, tighter" in s.value for s in app.success)
    assert app.session_state["prev_run_result"] is old_result
    assert app.session_state["run_result"] is not old_result


def test_feedback_fallback_keeps_previous_plan_warning(app, monkeypatch):
    _patch_run(monkeypatch, FakeTransport([(HEALTHY_LOG, "ok", None)]), FakeLLM())
    _launch(app)
    assert not app.exception, [e.value for e in app.exception]
    old_result = app.session_state["run_result"]

    _seed_plan_keys(app)
    _patch_feedback(
        monkeypatch,
        plan_mode="fallback",
        note="feedback revision failed - keeping the previous plan",
    )
    app.run()

    fb = next(ta for ta in app.text_area if ta.label == "Feedback on the result")
    fb.set_value("be more concise")
    next(b for b in app.button if "Re-run with feedback" in b.label).click().run()
    assert not app.exception, [e.value for e in app.exception]
    warn = "\n".join(w.value for w in app.warning)
    assert 'Revision for: "be more concise" - kept the previous plan' in warn
    assert "(Nemotron reply unparseable)" in warn
    assert not any("Revised with feedback" in s.value for s in app.success)
    assert app.session_state["prev_run_result"] is old_result


def test_feedback_validation_warnings(app, monkeypatch):
    _patch_run(monkeypatch, FakeTransport([(HEALTHY_LOG, "ok", None)]), FakeLLM())
    _launch(app)
    assert not app.exception, [e.value for e in app.exception]

    import supertaco.loop as loop_mod

    calls = []

    def boom(*args, **kwargs):
        calls.append(args)
        raise AssertionError("loop must not run for invalid feedback")

    monkeypatch.setattr(loop_mod, "run_training_loop", boom)
    app.run()  # feedback box appears now that a run finished

    next(b for b in app.button if "Re-run with feedback" in b.label).click().run()
    assert not app.exception, [e.value for e in app.exception]
    assert any("Describe what to change first." in w.value for w in app.warning)
    assert calls == []

    fb = next(ta for ta in app.text_area if ta.label == "Feedback on the result")
    fb.set_value("make it better")
    next(b for b in app.button if "Re-run with feedback" in b.label).click().run()
    assert not app.exception, [e.value for e in app.exception]
    assert any(
        "Feedback needs a goal-planned run - plan a goal first." in w.value for w in app.warning
    )
    assert calls == []


def test_report_shows_before_after_delta(app, monkeypatch):
    _patch_run(monkeypatch, FakeTransport([(HEALTHY_LOG, "ok", None)]), FakeLLM())
    _launch(app)
    assert not app.exception, [e.value for e in app.exception]
    # this run's ft avg is 9.00 (FakeLLM); pretend the pre-feedback run scored 3.00
    app.session_state["prev_run_result"] = _prev_result(
        [3.0, 3.0], ["base:real / ft:real", "base:real / ft:real"]
    )
    app.run()
    assert not app.exception, [e.value for e in app.exception]
    assert any(
        "Before feedback: fine-tuned avg 3.00 -> after: 9.00 (+6.00)" in c.value
        for c in app.caption
    )


def test_manual_rerun_after_feedback_clears_delta_caption(app, monkeypatch):
    _patch_run(monkeypatch, FakeTransport([(HEALTHY_LOG, "ok", None)]), FakeLLM())
    _launch(app)
    assert not app.exception, [e.value for e in app.exception]

    _seed_plan_keys(app)
    _patch_feedback(monkeypatch)
    app.run()

    fb = next(ta for ta in app.text_area if ta.label == "Feedback on the result")
    fb.set_value("be more concise")
    next(b for b in app.button if "Re-run with feedback" in b.label).click().run()
    assert not app.exception, [e.value for e in app.exception]
    assert any("Before feedback:" in c.value for c in app.caption)

    # a MANUAL run is not feedback-driven: the baseline must be invalidated
    next(b for b in app.button if "Fine-tune on Colab" in b.label).click().run()
    assert not app.exception, [e.value for e in app.exception]
    assert not any("Before feedback:" in c.value for c in app.caption)
    assert app.session_state["prev_run_result"] is None


def test_delta_hidden_when_prev_judge_not_real_or_scores_empty(app, monkeypatch):
    _patch_run(monkeypatch, FakeTransport([(HEALTHY_LOG, "ok", None)]), FakeLLM())
    _launch(app)
    assert not app.exception, [e.value for e in app.exception]
    real_modes = ["base:real / ft:real", "base:real / ft:real"]

    app.session_state["prev_run_result"] = _prev_result([3.0, 3.0], ["base:real / ft:hash"] * 2)
    app.run()
    assert not app.exception, [e.value for e in app.exception]
    assert not any("Before feedback:" in c.value for c in app.caption)

    app.session_state["prev_run_result"] = _prev_result([], real_modes)
    app.run()
    assert not app.exception, [e.value for e in app.exception]
    assert not any("Before feedback:" in c.value for c in app.caption)

    # control: all-real prev with scores renders the caption again
    app.session_state["prev_run_result"] = _prev_result([3.0, 3.0], real_modes)
    app.run()
    assert not app.exception, [e.value for e in app.exception]
    assert any("Before feedback:" in c.value for c in app.caption)


def test_delta_hidden_when_current_judge_not_real(app, monkeypatch):
    _patch_run(monkeypatch, FakeTransport([(HEALTHY_LOG, "ok", None)]), UnparseableJudgeLLM())
    _launch(app)
    assert not app.exception, [e.value for e in app.exception]
    app.session_state["prev_run_result"] = _prev_result(
        [3.0, 3.0], ["base:real / ft:real", "base:real / ft:real"]
    )
    app.run()
    assert not app.exception, [e.value for e in app.exception]
    judge_df = next(df for df in app.dataframe if "judge" in df.value.columns)
    assert all("hash" in str(m) for m in judge_df.value["judge"])  # current run is hash-mode
    assert not any("Before feedback:" in c.value for c in app.caption)


def test_nebius_judge_badge_renders_for_job_backend(app, monkeypatch):
    _patch_run(monkeypatch, FakeTransport([(HEALTHY_LOG, "ok", None)]), FakeLLM())
    import supertaco.eval.nebius_backend as nb

    class FakeJobBackend:
        def judge(self, prompts, base, ft):
            return {
                "prompts": list(prompts),
                "base_scores": [5.0] * len(prompts),
                "fine_tuned_scores": [9.0] * len(prompts),
                "modes": ["base:real / ft:real"] * len(prompts),
                "call_log": [],
                "judge_backend": "nebius_job",
            }

    monkeypatch.setattr(nb, "NebiusJudgeBackend", lambda **kw: FakeJobBackend())
    app.checkbox(key="judge_on_nebius").set_value(True).run()
    assert not app.exception, [e.value for e in app.exception]
    _launch(app)
    assert not app.exception, [e.value for e in app.exception]
    assert any("Judged on Nebius AI Cloud" in str(b.value) for b in app.success)
    assert not any("Nebius judge unavailable" in str(w.value) for w in app.warning)


def test_direct_fallback_warns_about_nebius(app, monkeypatch):
    _patch_run(monkeypatch, FakeTransport([(HEALTHY_LOG, "ok", None)]), FakeLLM())
    import supertaco.eval.nebius_backend as nb
    from supertaco.errors import JudgeBackendError

    def boom(**kw):
        raise JudgeBackendError("job aijob-1 ended FAILED: disk full")

    monkeypatch.setattr(nb, "NebiusJudgeBackend", boom)
    app.checkbox(key="judge_on_nebius").set_value(True).run()
    assert not app.exception, [e.value for e in app.exception]
    _launch(app)
    assert not app.exception, [e.value for e in app.exception]
    assert any("Nebius judge unavailable" in str(w.value) for w in app.warning)
    assert not any("Judged on Nebius AI Cloud" in str(b.value) for b in app.success)


def test_direct_backend_shows_no_nebius_badge(app, monkeypatch):
    _patch_run(monkeypatch, FakeTransport([(HEALTHY_LOG, "ok", None)]), FakeLLM())
    import supertaco.eval.nebius_backend as nb

    calls = []

    def _forbidden(**kw):
        calls.append(kw)
        raise AssertionError("direct-path test must never construct a Nebius backend")

    monkeypatch.setattr(nb, "NebiusJudgeBackend", _forbidden)
    assert app.checkbox(key="judge_on_nebius").value is False
    _launch(app)
    assert not app.exception, [e.value for e in app.exception]
    assert calls == []
    assert not any("Run crashed" in str(e.value) for e in app.error)
    assert not any("Nebius judge unavailable" in str(w.value) for w in app.warning)
    assert not any("Judged on Nebius AI Cloud" in str(b.value) for b in app.success)


def test_citations_render_under_patch_line_and_in_ledger(app, monkeypatch):
    import supertaco.agent.citations as cit

    class _Cites:
        def search_citations(self, query, *, max_results=2):
            from supertaco.agent.citations import Citation

            return [
                Citation(title="HF Trainer docs", url="https://hf.example/trainer"),
                Citation(title="PyTorch OOM", url="https://pytorch.example/oom"),
            ]

    monkeypatch.setattr(cit, "client_from_settings", lambda call_log=None: _Cites())
    _patch_run(
        monkeypatch,
        FakeTransport([(NAN_LOG, None, None), (HEALTHY_LOG, "ok", None)]),
        FakeLLM(),
    )
    _launch(app)
    assert not app.exception, [e.value for e in app.exception]
    result = app.session_state["run_result"]
    assert result.success is True
    md = "\n".join(m.value for m in app.markdown)
    assert "↳ sources:" in md
    assert "[HF Trainer docs](https://hf.example/trainer)" in md
    ledger_df = next(d for d in app.dataframe if "sources" in d.value.columns)
    assert ledger_df.value["sources"][0] == "HF Trainer docs, PyTorch OOM"
    assert ledger_df.value["sources"][1] == "—"


def test_run_failed_shows_sources_for_playbook_miss(app, monkeypatch):
    import supertaco.agent.citations as cit

    class _Cites:
        def search_citations(self, query, *, max_results=2):
            from supertaco.agent.citations import Citation

            return [Citation(title="Crash docs", url="https://crash.example")]

    monkeypatch.setattr(cit, "client_from_settings", lambda call_log=None: _Cites())
    _patch_run(
        monkeypatch,
        FakeTransport([("zorp quuxle", None, "zorp quuxle: totally unrecognized")]),
        FakeLLM(),
    )
    _launch(app)
    assert not app.exception, [e.value for e in app.exception]
    result = app.session_state["run_result"]
    assert result.success is False
    md = "\n".join(m.value for m in app.markdown)
    assert "🛑 **Failed**" in md
    assert "↳ sources:" in md
    assert "[Crash docs](https://crash.example)" in md


def test_run_failed_stays_silent_when_miss_search_is_empty(app, monkeypatch):
    # Stale-source guard: attempt 1 OOM finds citations, attempt 2 honest miss
    # returns [] -> the Failed line must NOT show OOM's sources (wrong failure).
    import supertaco.agent.citations as cit

    class _Cites:
        def search_citations(self, query, *, max_results=2):
            from supertaco.agent.citations import Citation

            if query.startswith("pytorch fine-tuning crash:"):
                return []
            return [Citation(title="OOM docs", url="https://oom.example")]

    monkeypatch.setattr(cit, "client_from_settings", lambda call_log=None: _Cites())
    _patch_run(
        monkeypatch,
        FakeTransport(
            [
                ("RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB", None, None),
                ("zorp quuxle", None, "zorp quuxle: totally unrecognized"),
            ]
        ),
        FakeLLM(),
    )
    _launch(app)
    assert not app.exception, [e.value for e in app.exception]
    result = app.session_state["run_result"]
    assert result.success is False
    md = "\n".join(m.value for m in app.markdown)
    assert "🛑 **Failed**" in md
    assert "OOM docs" not in md.split("🛑 **Failed**")[-1]  # no stale sources


def test_patch_line_silent_when_search_returns_empty(app, monkeypatch):
    import supertaco.agent.citations as cit

    class _Cites:
        def search_citations(self, query, *, max_results=2):
            return []

    monkeypatch.setattr(cit, "client_from_settings", lambda call_log=None: _Cites())
    _patch_run(
        monkeypatch,
        FakeTransport([(NAN_LOG, None, None), (NAN_LOG, None, None)]),
        FakeLLM(),
    )
    _launch(app)
    assert not app.exception, [e.value for e in app.exception]
    result = app.session_state["run_result"]
    assert result.success is False  # 4 NaN attempts (max_retries=3; transport re-serves last)
    md = "\n".join(m.value for m in app.markdown)
    assert "↳ sources:" not in md  # empty search -> silent everywhere
    ledger_df = next(d for d in app.dataframe if "sources" in d.value.columns)
    assert ledger_df.value["sources"][0] == "—"  # empty -> dash


def test_run_failed_shows_keyed_sources_on_budget_exhaustion(app, monkeypatch):
    import supertaco.agent.citations as cit

    class _Cites:
        def search_citations(self, query, *, max_results=2):
            from supertaco.agent.citations import Citation

            return [Citation(title="NaN guide", url="https://nan.example")]

    monkeypatch.setattr(cit, "client_from_settings", lambda call_log=None: _Cites())
    _patch_run(
        monkeypatch,
        FakeTransport([(NAN_LOG, None, None), (NAN_LOG, None, None)]),
        FakeLLM(),
    )
    _launch(app)
    assert not app.exception, [e.value for e in app.exception]
    result = app.session_state["run_result"]
    assert result.success is False
    md = "\n".join(m.value for m in app.markdown)
    # run_failed.failure_key == "NAN_LOSS" matches the attempt-1 citations_found event
    assert "🛑 **Failed**" in md
    assert "[NaN guide](https://nan.example)" in md.split("🛑 **Failed**")[-1]


def test_execute_launch_passes_plan_goal_to_loop(offline, monkeypatch):
    import supertaco.loop as loop_mod
    from supertaco.loop import LoopResult

    seen = {}

    def fake_loop(config, **kwargs):
        seen.update(kwargs)
        return LoopResult(True, 1, dict(config), [], None, None, [])

    monkeypatch.setattr(loop_mod, "run_training_loop", fake_loop)
    at = AppTest.from_file(str(DASHBOARD), default_timeout=60)
    at.session_state["plan_goal"] = "pirate chef"
    at.run()
    assert not at.exception, f"render crashed: {[e.value for e in at.exception]}"
    run_btn = next(b for b in at.button if "Fine-tune on Colab" in b.label)
    run_btn.click()
    at.run()
    assert not at.exception, f"run crashed: {[e.value for e in at.exception]}"
    assert seen.get("goal") == "pirate chef"


def test_showcase_banner_and_disabled_controls(offline, monkeypatch):
    monkeypatch.setenv("SUPERTACO_SHOWCASE", "1")
    at = AppTest.from_file(str(DASHBOARD), default_timeout=60)
    at.run()
    assert not at.exception, f"showcase render crashed: {[e.value for e in at.exception]}"
    assert any("Showcase mode" in i.value for i in at.info)
    run_btn = next(b for b in at.button if "Fine-tune on Colab" in b.label)
    assert run_btn.disabled
    judge_box = next(c for c in at.checkbox if "Judge on Nebius" in c.label)
    assert judge_box.disabled


def test_non_showcase_banner_absent_controls_enabled(app):
    assert not any("Showcase mode" in i.value for i in app.info)
    run_btn = next(b for b in app.button if "Fine-tune on Colab" in b.label)
    assert not run_btn.disabled
    judge_box = next(c for c in app.checkbox if "Judge on Nebius" in c.label)
    assert not judge_box.disabled
