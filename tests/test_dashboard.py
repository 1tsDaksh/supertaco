"""AppTest: launch a broken fixture and watch the real pipeline heal it."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

REPO = Path(__file__).resolve().parents[1]
DASHBOARD = REPO / "supertaco" / "ui" / "dashboard.py"
NAN_FIXTURE = next(
    p for p in (REPO / "configs" / "runs").glob("*NAN_LOSS*") if "_patched" not in p.name
)


@pytest.fixture()
def offline(monkeypatch, tmp_path):
    """Unreachable Token Factory + tmp working dir: no credits, no repo litter."""
    monkeypatch.chdir(tmp_path)
    import supertaco.settings as settings_mod

    monkeypatch.setattr(
        settings_mod.settings, "token_factory_base_url", "http://127.0.0.1:9", raising=True
    )


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
    assert types[0] == "run_started"
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


def test_validation_error_shows_no_traceback(app):
    app.session_state["config"] = {"batch_size": 8}  # missing learning_rate
    app.run()
    launch = next(b for b in app.button if "Launch Job" in b.label)
    launch.click().run()
    assert not app.exception, [e.value for e in app.exception]
    assert any("learning_rate" in e.value for e in app.error)
    assert app.session_state["run_result"] is None


def test_crash_never_shows_traceback(app, monkeypatch):
    import supertaco.runner as runner_mod

    def boom(*args, **kwargs):
        raise RuntimeError("synthetic mid-run boom")

    monkeypatch.setattr(runner_mod, "run", boom)
    launch = next(b for b in app.button if "Launch Job" in b.label)
    launch.click().run()
    assert not app.exception, [e.value for e in app.exception]
    assert any("Run crashed" in e.value for e in app.error)
    assert app.session_state["run_result"] is None


def test_fixture_loader_and_eval_suite(app):
    # Fixture loader present: selectbox labeled "Fixture" offers the yaml fixtures.
    # (s.label is the widget label; s.options holds the format_func LABELS, e.g.
    #  "NAN_LOSS_broken" — the assertion matches "NAN_LOSS" as a substring of it.)
    fixture_sb = next(s for s in app.selectbox if s.label == "Fixture")
    assert any("NAN_LOSS" in opt for opt in fixture_sb.options)

    # eval panel must be gated on a completed run — no Eval button yet
    assert not any("Eval" in b.label for b in app.button)

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
    # banner count must match the state the LLM table renders
    calls = app.session_state["run_result"].llm_calls
    assert any(f"{len(calls)}/{len(calls)}" in t for t in banner_texts)


def test_new_launch_clears_stale_eval(app):
    app.session_state["eval_results"] = {"stale": True}
    app.session_state["config"] = {"batch_size": 8}
    app.run()
    launch = next(b for b in app.button if "Launch Job" in b.label)
    launch.click().run()
    assert app.session_state["eval_results"] is None
