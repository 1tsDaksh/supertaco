"""Redesign module tests: theme, components, data, and the Landing/Results pages."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from streamlit.testing.v1 import AppTest

from supertaco.ui import components, theme
from supertaco.ui import data as data_mod
from supertaco.ui.views import landing

REPO = Path(__file__).resolve().parents[1]
DASHBOARD = REPO / "supertaco" / "ui" / "dashboard.py"


def _ev(*types):
    return [SimpleNamespace(type=t, data={}) for t in types]


# ─── theme ──────────────────────────────────────────────────────────
def test_theme_css_contains_midnight_tokens():
    css = theme.build_theme_css()
    for token in ("#0b1220", "#131f38", "#22c55e", "stx-terminal", "stx-chip", "stx-step"):
        assert token in css


def test_theme_escapes_markup():
    assert theme.esc("<b>&'") == "&lt;b&gt;&amp;&#x27;"


# ─── components: phase timeline ─────────────────────────────────────
def test_phase_states_empty_is_all_pending():
    assert components.phase_states([]) == dict.fromkeys(components.PHASES, "pending")


def test_phase_states_lunched_then_trained_then_done():
    assert components.phase_states(_ev("job_launched"))["train"] == "active"
    mid = components.phase_states(_ev("job_launched", "logs_produced"))
    assert mid["train"] == "done" and mid["eval"] == "active"
    done = components.phase_states(_ev("job_launched", "logs_produced", "run_succeeded"))
    assert set(done.values()) == {"done"}


def test_phase_states_failure_marks_remaining_pipeline():
    states = components.phase_states(_ev("job_launched", "logs_produced", "run_failed"))
    assert states["train"] == "done"
    assert states["eval"] == "failed" and states["report"] == "failed"
    early = components.phase_states(_ev("run_failed"))
    assert early["plan"] == "failed" and early["train"] == "pending"


def test_phase_timeline_html_marks_states():
    html = components.build_phase_timeline_html(
        {
            "plan": "done",
            "data": "active",
            "train": "pending",
            "eval": "failed",
            "report": "pending",
        }
    )
    assert 'stx-step done">plan ✓' in html
    assert 'stx-step active">data ●' in html
    assert 'stx-step pending">train ○' in html
    assert 'stx-step failed">eval ✕' in html


# ─── components: terminal (auto-scroll, full history) ───────────────
def test_terminal_keeps_full_history_and_pins_scroll():
    lines = [f"line {i}" for i in range(50)]
    html = components.build_terminal_html(lines, height=200)
    for line in lines:
        assert line in html  # nothing truncated
    assert "el.scrollTop = el.scrollHeight" in html
    assert "height: 200px" in html


def test_terminal_escapes_html_in_lines():
    html = components.build_terminal_html(["<script>alert(1)</script>"])
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


# ─── components: badges ─────────────────────────────────────────────
def test_delta_badge_classes():
    assert "up" in components.build_delta_badge_html(1.5)
    assert "down" in components.build_delta_badge_html(-0.2)
    assert "flat" in components.build_delta_badge_html(0.0)
    assert "+1.50" in components.build_delta_badge_html(1.5)


# ─── data ───────────────────────────────────────────────────────────
def _artifact(run_id="swzz01", delta=2.0, tokens=100):
    return {
        "schema": 1,
        "run_id": run_id,
        "goal": f"goal {run_id}",
        "success": True,
        "attempts": 1,
        "healed": False,
        "wall_clock_s": 120.0,
        "judge_backend": "nebius_job",
        "eval": {
            "prompts": ["p"],
            "base_scores": [5.0],
            "fine_tuned_scores": [5.0 + delta],
            "base_avg": 5.0,
            "ft_avg": 5.0 + delta,
            "delta": delta,
            "modes": ["base:real / ft:real"],
            "regression_flagged": False,
        },
        "llm_calls": [
            {"model_key": "classify", "tokens": tokens, "mode": "real"},
            {"model_key": "judge", "tokens": tokens // 2, "mode": "real"},
        ],
    }


def test_load_artifacts_skips_corrupt_and_timing(tmp_path):
    (tmp_path / "bad.json").write_text("{nope", encoding="utf-8")
    (tmp_path / "x.timing.json").write_text(json.dumps({"pair_s": 1.0}), encoding="utf-8")
    (tmp_path / "ok.json").write_text(json.dumps(_artifact()), encoding="utf-8")
    artifacts, warnings = data_mod.load_artifacts(tmp_path)
    assert [a["run_id"] for a in artifacts] == ["swzz01"]
    assert len(warnings) == 1 and "bad.json" in warnings[0]


def test_sweep_summary_math():
    arts = [_artifact("a", delta=2.0, tokens=100), _artifact("b", delta=-1.0, tokens=300)]
    s = data_mod.sweep_summary(arts)
    assert s["runs"] == 2 and s["successful"] == 2
    assert s["improved"] == 1 and s["regressed"] == 1 and s["flat"] == 0
    assert s["avg_delta"] == pytest.approx(0.5)
    assert s["tokens"] == {"classify": 400, "judge": 200}
    assert s["wall_s"] == 240.0
    assert s["judge_backends"] == ["nebius_job"]


def test_sweep_summary_empty_is_honest():
    s = data_mod.sweep_summary([])
    assert s["avg_delta"] is None and s["tokens"] == {} and s["wall_s"] is None


def test_verify_commands_reference_run_id():
    cmds = data_mod.verify_commands("swzz01")
    assert "'\"run_id\": \"swzz01\"'" in cmds
    assert "logs/runs/swzz01.json" in cmds


# ─── pages via AppTest ──────────────────────────────────────────────
@pytest.fixture()
def offline(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    import supertaco.settings as settings_mod

    monkeypatch.setattr(
        settings_mod.settings, "token_factory_base_url", "http://127.0.0.1:9", raising=True
    )


@pytest.fixture()
def at_app(offline, monkeypatch):
    monkeypatch.delenv("SUPERTACO_SHOWCASE", raising=False)
    at = AppTest.from_file(str(DASHBOARD), default_timeout=60)
    at.run()
    assert not at.exception, f"landing crashed: {[e.value for e in at.exception]}"
    return at


def test_landing_renders_hero_ctas_and_empty_state(at_app):
    assert any("▶️ Run a demo" == b.label for b in at_app.button)
    assert any("📊 See latest results" == b.label for b in at_app.button)
    md = "\n".join(m.value for m in at_app.markdown)
    assert "How it works" in md  # six-stage pipeline section
    assert "What it can do" in md  # capabilities grid
    assert "Under the hood" in md  # stack facts
    assert any("No run artifacts found yet" in i.value for i in at_app.info)


def test_landing_content_is_informative_not_just_navigation():
    hero = landing._hero_html()
    assert "fine-tuning sandbox" in hero  # the pitch lives on the page itself
    assert len(landing._PIPELINE) == 6  # plan → materialize → train → heal → judge → receipt
    assert len(landing._CAPABILITIES) == 6
    stages = {stage for stage, _ in landing._PIPELINE}
    assert {"1 · Plan", "4 · Heal", "6 · Receipt"} <= stages
    for title, body in landing._CAPABILITIES:
        assert len(body) > 60  # each capability actually explains itself
    stack = landing._stack_table_html()
    assert "Nebius" in stack and "Colab T4" in stack and "Schema-v1" in stack
    card = landing._card_html("1 · Plan", "x" * 200, body_html=False)
    assert "&" not in card  # bodies are escaped by default
    raw = landing._card_html("1 · Plan", "<strong>real</strong>", body_html=True)
    assert "<strong>real</strong>" in raw


def test_landing_run_demo_button_navigates(at_app):
    btn = next(b for b in at_app.button if b.label == "▶️ Run a demo")
    btn.click().run()
    assert not at_app.exception, [e.value for e in at_app.exception]
    assert at_app.session_state["nav_page"] == "🛰️ Live Run"
    assert any("Fine-tune on Colab" in b.label for b in at_app.button)


def test_landing_evidence_strip_from_artifacts(at_app, tmp_path, monkeypatch):
    runs = tmp_path / "logs" / "runs"
    runs.mkdir(parents=True)
    (runs / "a.json").write_text(json.dumps(_artifact("swzz01", delta=2.5)), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    at_app.run()
    assert not at_app.exception, [e.value for e in at_app.exception]
    md = "\n".join(m.value for m in at_app.markdown)
    assert "Latest evidence" in md
    assert "swzz01" in md  # best-run line cites the artifact


def test_results_page_lists_runs_and_verify(at_app, tmp_path, monkeypatch):
    runs = tmp_path / "logs" / "runs"
    runs.mkdir(parents=True)
    (runs / "a.json").write_text(json.dumps(_artifact("swzz01", delta=2.5)), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    at_app.session_state["nav_page"] = "📊 Results"
    at_app.run()
    assert not at_app.exception, [e.value for e in at_app.exception]
    labels = [m.label for m in at_app.metric]
    assert "Runs" in labels and "Avg Δ" in labels and "Improved" in labels
    df = next(d for d in at_app.dataframe if "run" in d.value.columns)
    assert "swzz01" in list(df.value["run"])
    assert any("Verify this run" in e.label for e in at_app.expander)


def test_results_page_empty_state(at_app):
    at_app.session_state["nav_page"] = "📊 Results"
    at_app.run()
    assert not at_app.exception, [e.value for e in at_app.exception]
    assert any("No run artifacts yet" in i.value for i in at_app.info)
