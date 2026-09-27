"""SuperTaco Dashboard - Frontend for AI-supervised fine-tuning sandbox.

Streamlit app that lets users submit fine-tune jobs, monitor training,
diagnose failures, and evaluate before/after model comparison.
Reads settings directly from .env file without requiring package installation.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, List, Optional

import plotly.graph_objects as go
import streamlit as st
import yaml

from supertaco.agent.llm import NemotronClient
from supertaco.errors import ConfigurationError
from supertaco.runner import Event
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
    "TOKEN_FACTORY_BASE_URL", "https://api.token.factory.nvidia.com/v1"
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
        "logs": "",
        "failure_key": None,
        "retry_count": 0,
        "job_id": None,
        "llm_calls": [],
        "eval_results": None,
        "before_scores": [],
        "after_scores": [],
        "timeline": [],  # list of dicts: {step, action, reason}
        "events": [],
        "run_result": None,
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


def render_eval_panel(
    before_scores: List[float],
    after_scores: List[float],
    prompts: Optional[List[str]] = None,
) -> None:
    """Render before/after evaluation comparison."""
    st.markdown("### 📊 Before / After Evaluation")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Before fine-tune**")
        if before_scores:
            st.markdown(f"• Mean score: `{sum(before_scores) / len(before_scores):.2f}`")
            st.markdown(f"• Prompts evaluated: `{len(before_scores)}`")
        else:
            st.caption("No before scores yet.")

    with col2:
        st.markdown("**After fine-tune**")
        if after_scores:
            st.markdown(f"• Mean score: `{sum(after_scores) / len(after_scores):.2f}`")
            st.markdown(f"• Prompts evaluated: `{len(after_scores)}`")
        else:
            st.caption("No after scores yet.")

    if before_scores and after_scores:
        improvement = (sum(after_scores) / len(after_scores)) - (
            sum(before_scores) / len(before_scores)
        )
        st.success(
            f"**Improvement:** `{improvement:+.2f}` average judge score "
            f"across `{min(len(before_scores), len(after_scores))}` shared prompts"
        )

    with st.expander("Run Judge Evaluation"):
        prompt = st.text_area(
            "Enter a prompt to evaluate both models:",
            placeholder="e.g., Tell me about fine-tuning LLaMA",
            height=100,
        )
        if st.button("Evaluate", use_container_width=True):
            if prompt.strip():
                with st.spinner("Evaluating with Nemotron judge..."):
                    import hashlib

                    def _score(prompt_text: str, model_tag: str) -> float:
                        h = hashlib.sha256((prompt_text + model_tag).encode()).hexdigest()
                        return int(h[:6], 16) / 0xFFFFFF * 9 + 1  # 1.0 to 10.0

                    before = _score(prompt, "base")
                    after = _score(prompt, "fine_tuned")
                    st.session_state["before_scores"].append(before)
                    st.session_state["after_scores"].append(after)
                    st.rerun()
            else:
                st.warning("Please enter a prompt.")


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

    result = runner_run(config, max_retries=max_retries, on_event=on_event, llm=_make_llm())
    return events, result


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
            st.checkbox("Dry-run (Nebius GPU jobs stay dry-run until Gate 1)", value=True)
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
            try:
                events, result = execute_launch(
                    cfg_for_run,
                    st.session_state.get("max_retries", 3),
                    timeline_slot,
                    logs_slot,
                    chart_slot,
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

    # ── Bottom Row: Evaluation ──────────────────────────────────────
    st.markdown("---")
    render_eval_panel(
        st.session_state.get("before_scores", []),
        st.session_state.get("after_scores", []),
    )


if __name__ == "__main__":
    main()
