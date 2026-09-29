"""SuperTaco Dashboard - Frontend for AI-supervised fine-tuning sandbox.

Streamlit app that lets users submit fine-tune jobs, monitor training,
diagnose failures, and evaluate before/after model comparison.
Reads settings directly from .env file without requiring package installation.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict

import plotly.graph_objects as go
import streamlit as st
import yaml

from supertaco.errors import ConfigurationError
from supertaco.runner import Event, make_llm
from supertaco.runner import run as runner_run


# ─── Load settings directly from .env (no package import needed) ────────────
def _load_env():
    """Load .env file manually if present, otherwise fall back to os.environ."""
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if env_path.exists():
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    key = key.strip()
                    value = value.strip()
                    if key and key not in os.environ:
                        os.environ[key] = value


_load_env()

# Read settings from environment (matching SandboxTuneSettings field names)
_SANDBOXTUNE_ENV = os.getenv("SANDBOXTUNE_ENV", "dev")
_NEBIUS_PROJECT_ID = os.getenv("NEBIUS_PROJECT_ID", "")
_NEBIUS_API_KEY = os.getenv("NEBIUS_API_KEY", "")
_TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
_TOKEN_FACTORY_BASE_URL = os.getenv(
    "TOKEN_FACTORY_BASE_URL", "https://api.tokenfactory.nebius.com/v1"
)


def get_settings():
    """Return a settings namespace with the same fields as SandboxTuneSettings."""

    class _Settings:
        nebius_api_key: str = _NEBIUS_API_KEY
        nebius_project_id: str = _NEBIUS_PROJECT_ID
        tavily_api_key: str = _TAVILY_API_KEY
        token_factory_base_url: str = _TOKEN_FACTORY_BASE_URL
        environment: str = _SANDBOXTUNE_ENV

    return _Settings()


# ─── Main settings object (module-level, from .env) ───────────────────
_settings = get_settings()


# ─── Session State Initialisation ────────────────────────────────
def init_session2():
    """Initialise session-state variables if not already set."""
    defaults = {
        "config": {},
        "llm_calls": [],
        "eval_results": None,
        "events": [],
        "run_result": None,
        "gpu_result": None,
        "gpu_lines": [],
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def reset_session2():
    """Reset all session-state variables."""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    init_session2()


# ─── UI Components ────────────────────────────────────────────────
def render_sidebar() -> Dict[str, any]:
    """Render the sidebar with controls and settings."""
    st.sidebar.title("⚙️ Controls")

    calls = st.session_state.get("llm_calls") or []
    fallbacks = [c for c in calls if c.get("mode") != "real"]
    if fallbacks:
        st.sidebar.error(
            f"⚠️ Nemotron fallback: {len(fallbacks)}/{len(calls)} call(s) used heuristics. "
            f"First error: {fallbacks[0].get('error')}"
        )
    elif calls:
        st.sidebar.success(f"✅ Nemotron: {len(calls)} real call(s)")

    st.sidebar.markdown("### Environment")
    env = st.sidebar.selectbox(
        "Environment",
        ["dev", "prod"],
        index=0 if _settings.environment != "prod" else 1,
        help="prod blocks dry-run-only bypasses",
    )
    st.session_state["env"] = env

    st.sidebar.markdown("### API Keys (from .env)")
    st.sidebar.caption(f"NEBIUS_PROJECT_ID: {_settings.nebius_project_id}")
    st.sidebar.caption("NEBIUS_API_KEY: loaded from .env on import")
    st.sidebar.caption("TAVILY_API_KEY: loaded from .env on import")

    st.sidebar.markdown("### Model Cost Routing")
    st.sidebar.caption(
        "Start at nano → escalate to super on low confidence → ultra only when playbook misses"
    )

    st.sidebar.markdown("### Max Retries")
    st.session_state["max_retries"] = st.sidebar.number_input(
        "Max retries per run",
        min_value=1,
        max_value=10,
        value=st.session_state.get("max_retries", 3),
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Quick Start")
    demo_options = {
        "Demo: NaN Loss": {
            "learning_rate": 0.1,
            "batch_size": 8,
            "lora_r": 8,
            "lora_alpha": 16,
            "num_epochs": 3,
        },
        "Demo: OOM": {
            "learning_rate": 5e-5,
            "batch_size": 64,
            "lora_r": 8,
            "lora_alpha": 16,
            "num_epochs": 3,
        },
        "Demo: Loss Divergence": {
            "learning_rate": 0.1,
            "batch_size": 8,
            "lora_r": 8,
            "lora_alpha": 16,
            "num_epochs": 10,
        },
        "Demo: Loss Plateau": {
            "learning_rate": 1e-6,
            "batch_size": 8,
            "lora_r": 8,
            "lora_alpha": 16,
            "num_epochs": 1,
        },
        "Demo: Eval Regression": {
            "learning_rate": 1e-4,
            "batch_size": 8,
            "lora_r": 64,
            "lora_alpha": 32,
            "num_epochs": 10,
        },
    }
    selected_demo = st.sidebar.selectbox("Load demo config", list(demo_options.keys()))
    if st.sidebar.button("Load Demo Config"):
        demo = demo_options[selected_demo]
        st.session_state["config"] = demo
        st.sidebar.success(f"Loaded {selected_demo} config")

    st.sidebar.markdown("### Broken Fixtures")
    from pathlib import Path as _Path

    runs_dir = _Path(__file__).resolve().parents[2] / "configs" / "runs"
    fixtures = sorted(p for p in runs_dir.glob("*.yaml") if "_patched" not in p.name)
    if fixtures:
        chosen = st.sidebar.selectbox(
            "Fixture",
            [f.name for f in fixtures],
            format_func=lambda n: n.split("_", 2)[-1].rsplit("_", 1)[0],
        )
        if st.sidebar.button("Load Fixture"):
            import yaml as _yaml

            try:
                cfg = _yaml.safe_load((runs_dir / chosen).read_text(encoding="utf-8-sig"))
            except _yaml.YAMLError as exc:
                st.sidebar.error(f"Could not parse {chosen}: {exc}")
            else:
                if not isinstance(cfg, dict):
                    st.sidebar.error(f"{chosen} is not a config mapping")
                else:
                    st.session_state["config"] = cfg
                    st.sidebar.success(f"Loaded {chosen}")

    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Reset Session", use_container_width=True):
        reset_session2()
        st.rerun()

    return {"settings": _settings, "env": env}


def render_config_editor(config: dict) -> dict:
    """Render an editable YAML config panel."""
    st.markdown("### 📝 Training Config")
    with st.expander("View / Edit YAML", expanded=True):
        yaml_str = yaml.dump(config, sort_keys=True, allow_unicode=True)
        edited = st.text_area(
            "YAML config",
            value=yaml_str,
            height=300,
            help="Edit the training config keys. Changes affect the next launch.",
        )
    try:
        parsed = yaml.safe_load(edited)
    except Exception as e:
        st.error(f"YAML parse error: {e}")
        parsed = config
    return parsed


def render_eval_panel() -> None:
    """Before/after eval over the fixed 5-prompt suite (real judge)."""
    st.markdown("### 📊 Before / After Evaluation")
    result = st.session_state.get("run_result")
    if result is None or not result.success:
        st.info("Run a job to completion first — eval needs a final config.")
        return

    if st.button(
        "🧪 Run Eval Suite (5 prompts, real Nemotron judge)",
        use_container_width=True,
        key="eval_suite_btn",
    ):
        from supertaco.eval.harness import DEFAULT_PROMPTS, build_responses, run_eval_suite

        with st.spinner("Scoring with Nemotron judge..."):
            base_responses, ft_responses = build_responses(result.final_config)
            eval_llm = make_llm()  # fresh client -> fresh circuit breaker
            suite_results = run_eval_suite(
                DEFAULT_PROMPTS, base_responses, ft_responses, llm=eval_llm
            )
            st.session_state["eval_results"] = suite_results

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


def _render_events(events: list) -> str:
    """Build the full timeline markdown from collected events."""
    if not events:
        return "No decisions yet — launch a job to see the loop."
    lines = []
    for ev in events:
        d = ev.data
        if ev.type == "job_launched":
            lines.append(
                f"**Attempt {d['attempt']}** · 🚀 dry-run launch "
                f"({len(d['payload'].get('config', {}))} config keys)"
            )
        elif ev.type == "failure_detected":
            lines.append(f"**Attempt {d['attempt']}** · ❌ detected `{d['failure_key']}`")
        elif ev.type == "classified":
            mark = "⚠️ diverges from playbook" if d["diverged"] else "agrees with playbook"
            lines.append(
                f"**Attempt {d['attempt']}** · 🤖 Nemotron nano → "
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
                f"{k}: {before.get(k)!r}→{after[k]!r}"
                for k in after
                if before.get(k) != after.get(k)
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


def _loss_figure(events: list):
    """Per-attempt loss series parsed from generated logs (spec 5.5)."""
    import math

    fig = go.Figure()
    for ev in events:
        if ev.type != "logs_produced":
            continue
        points = [None if math.isnan(v) else v for v in ev.data["loss_points"]]
        fig.add_trace(go.Scatter(y=points, mode="lines", name=f"attempt {ev.data['attempt']}"))
    fig.update_layout(
        title="Training loss (dry-run, per attempt)",
        xaxis_title="Step",
        yaxis_title="Loss",
        hovermode="x unified",
        height=300,
    )
    return fig


def _llm_rows(calls: list) -> list:
    return [
        {
            "model": c.get("model_key"),
            "id": c.get("model_id"),
            "mode": c.get("mode"),
            "tokens": c.get("tokens"),
            "latency_s": c.get("latency"),
            "error": (c.get("error") or "")[:60],
        }
        for c in calls
    ]


def execute_launch(
    config: dict,
    max_retries: int,
    timeline_slot,
    logs_slot,
    chart_slot,
    dry_run: bool = True,
):
    """Run the pipeline, streaming events into the status placeholders."""
    events: list[Event] = []

    def on_event(ev: Event) -> None:
        events.append(ev)
        st.session_state["events"] = events  # survive a mid-run crash (spec 6)
        timeline_slot.markdown(_render_events(events))
        if ev.type == "logs_produced":
            logs_slot.code(ev.data["text"], language="log")
            chart_slot.plotly_chart(_loss_figure(events), use_container_width=True)

    result = runner_run(
        config,
        max_retries=max_retries,
        on_event=on_event,
        llm=make_llm(),
        dry_run=dry_run,
    )
    return events, result


def _gpu_figure(loss_points: list):
    """Loss curve for a real Colab run (NaN-safe, same style as dry-run chart)."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            y=[None if v != v else v for v in loss_points],
            mode="lines",
            name="colab T4",
        )
    )
    fig.update_layout(
        title="Training loss (Colab T4, real GPU)",
        xaxis_title="Step",
        yaxis_title="Loss",
        height=300,
    )
    return fig


def render_gpu_panel() -> None:
    """Free Colab T4 training driven from the dashboard (colab/train_lora.py)."""
    st.markdown("### 🏋 Real GPU Training (Google Colab, free T4)")
    st.caption(
        "Provisions a Colab T4, runs colab/train_lora.py, streams loss lines, "
        "and pulls back lora_adapter.zip (one-time auth: `colab.exe usage`)."
    )

    if st.button("🏋 Train on Colab (T4)", use_container_width=True, key="gpu_train_btn"):
        from supertaco.agent.simlogs import extract_loss_points
        from supertaco.gpu.colab import run_training

        st.session_state["gpu_result"] = None
        st.session_state["gpu_lines"] = []
        live = st.empty()
        chart = st.empty()
        loss_points: list[float] = []
        lines: list[str] = []

        def on_line(ln: str) -> None:
            lines.append(ln)
            st.session_state["gpu_lines"] = list(lines)
            live.markdown(f"`{ln[:200]}`")
            pts = extract_loss_points(ln)
            if pts:
                loss_points.extend(pts)
                chart.plotly_chart(_gpu_figure(loss_points), width="stretch", key="gpu_live_chart")

        repo = Path(__file__).resolve().parents[2]
        try:
            outcome = run_training(
                repo / "colab" / "train_lora.py",
                output_dir=repo / "colab" / "output",
                on_line=on_line,
            )
            st.session_state["gpu_result"] = {
                "success": outcome.success,
                "error": outcome.error,
                "artifact": str(outcome.artifact) if outcome.artifact else None,
                "loss_points": loss_points,
            }
        except Exception as exc:  # spec 6: dashboard never shows a traceback
            st.session_state["gpu_result"] = {
                "success": False,
                "error": str(exc),
                "artifact": None,
                "loss_points": loss_points,
            }

    result = st.session_state.get("gpu_result")
    if result is None:
        return
    if result.get("success") and result.get("artifact"):
        artifact = Path(result["artifact"])
        st.success(f"✅ Training complete — artifact: {artifact.name}")
        if artifact.exists():
            st.download_button(
                "⬇️ Download lora_adapter.zip",
                data=artifact.read_bytes(),
                file_name=artifact.name,
                mime="application/zip",
                key="gpu_download_btn",
            )
        if result.get("loss_points"):
            st.plotly_chart(
                _gpu_figure(result["loss_points"]), width="stretch", key="gpu_final_chart"
            )
    else:
        st.error(f"❌ {result.get('error') or 'Colab run failed'}")
    lines = st.session_state.get("gpu_lines") or []
    if lines:
        with st.expander(f"📜 Colab log ({len(lines)} lines)"):
            st.code("\n".join(lines[-400:]), language="log")


# ─── Main App ─────────────────────────────────────────────────────
def main():
    init_session2()
    render_sidebar()

    st.title("🚀 SuperTune — AI-supervised fine-tuning sandbox")
    st.caption(
        "Submit a dataset + goal → agent writes config → launches on Nebius → "
        "monitors → auto-heals failures → evaluates → deploys endpoint"
    )

    # ── Left Column: Job Launcher ──────────────────────────────────
    with st.container(border=True):
        st.markdown("### 🚦 Launch Job")
        cfg = render_config_editor(st.session_state["config"])

        c1, c2 = st.columns(2)
        with c1:
            dry_run_on = st.checkbox(
                "Dry-run (Nebius GPU jobs stay dry-run until Gate 1)", value=True
            )
        with c2:
            st.text_input("Job label", value=f"run-{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}")

        run_clicked = st.button("▶ Launch Job", type="primary", use_container_width=True)
        cfg_for_run = cfg

    # ── Right Column: Monitor & Metrics ────────────────────────────
    with st.container(border=True):
        st.markdown("### 📡 Job Status & Logs")
        timeline_slot = st.empty()
        logs_slot = st.empty()
        chart_slot = st.empty()

        if run_clicked:
            st.session_state["run_result"] = None  # never trust a stale result
            st.session_state["llm_calls"] = []  # stale calls must not outlive a failed run
            st.session_state["eval_results"] = None  # stale eval must not score a newer run
            st.session_state["events"] = []  # stale timeline must not outlive a failed run
            try:
                events, result = execute_launch(
                    cfg_for_run,
                    st.session_state.get("max_retries", 3),
                    timeline_slot,
                    logs_slot,
                    chart_slot,
                    dry_run=dry_run_on,
                )
                st.session_state["events"] = events
                st.session_state["run_result"] = result
                st.session_state["llm_calls"] = result.llm_calls
                if not result.success:
                    st.error(f"🛑 {result.error}")
                else:
                    st.toast("✅ Run healed", icon="✅")
            except ConfigurationError as exc:
                st.error(f"❌ {exc}")
                timeline_slot.markdown(_render_events(st.session_state.get("events", [])))
            except Exception as exc:  # spec 6: never a Streamlit traceback
                st.error(f"❌ Run crashed: {exc}")
                timeline_slot.markdown(_render_events(st.session_state.get("events", [])))
        elif st.session_state.get("events"):
            timeline_slot.markdown(_render_events(st.session_state["events"]))
            latest_logs = next(
                (
                    e.data["text"]
                    for e in reversed(st.session_state["events"])
                    if e.type == "logs_produced"
                ),
                "",
            )
            logs_slot.code(latest_logs or "—", language="log")
            chart_slot.plotly_chart(
                _loss_figure(st.session_state["events"]), use_container_width=True
            )
        else:
            timeline_slot.info("No run yet — load a config and press Launch.")

        calls = st.session_state.get("llm_calls") or []
        if calls:
            with st.expander(f"🧠 LLM Calls ({len(calls)})"):
                st.dataframe(_llm_rows(calls), use_container_width=True, hide_index=True)

    # ── Bottom Row: Evaluation ──────────────────────────────────────
    st.markdown("---")
    render_eval_panel()

    # ── Bottom Row: Real GPU training ──────────────────────────────
    st.markdown("---")
    render_gpu_panel()


if __name__ == "__main__":
    main()
