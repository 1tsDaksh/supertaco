"""SuperTaco Dashboard - Frontend for AI-supervised fine-tuning sandbox.

Streamlit app that lets users submit fine-tune jobs, monitor training,
diagnose failures, and evaluate before/after model comparison.
Reads settings directly from .env file without requiring package installation.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

import plotly.graph_objects as go
import streamlit as st
import yaml

from supertaco.errors import ConfigurationError
from supertaco.loop import Event, extract_loss_points, make_llm, run_training_loop


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
        "cfg_path": "configs/defaults/colab_t4.yaml",
        "knobs": {},
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
    st.sidebar.markdown("### Training Config")
    repo = Path(__file__).resolve().parents[2]
    cfg_path = st.sidebar.text_input(
        "Config YAML path", value=st.session_state.get("cfg_path", "configs/defaults/colab_t4.yaml")
    )
    st.session_state["cfg_path"] = cfg_path
    if st.sidebar.button("Load config"):
        target = Path(cfg_path) if Path(cfg_path).is_absolute() else repo / cfg_path
        try:
            loaded = yaml.safe_load(target.read_text(encoding="utf-8-sig"))
        except Exception as exc:
            st.sidebar.error(f"Could not load {cfg_path}: {exc}")
        else:
            if not isinstance(loaded, dict):
                st.sidebar.error(f"{cfg_path} is not a config mapping")
            else:
                st.session_state["config"] = loaded
                st.sidebar.success(f"Loaded {cfg_path}")

    from supertaco.gpu.transport import MODEL_WHITELIST

    st.sidebar.markdown("### Model (T4-viable)")
    # Plan 3c adjustment: surface a non-whitelist configured model instead of replacing it.
    # NOTE: no key= for this selectbox — rejection of invalid configured models depends on
    # `index` participating in the no-key element id (adding key= would let stale widget
    # state override the configured model and silently clobber pre-launch validation).
    options = list(MODEL_WHITELIST)
    current_model = (st.session_state.get("config") or {}).get("model")
    if isinstance(current_model, str) and current_model and current_model not in options:
        options.append(current_model)
    st.session_state["model_sel"] = st.sidebar.selectbox(
        "model",
        options,
        index=options.index(current_model) if current_model in options else 0,
        help="Whitelisted HF models that fit a free T4",
    )
    st.sidebar.markdown("### Hyperparameters")
    knobs = st.session_state.setdefault("knobs", {})
    knobs["learning_rate"] = st.sidebar.number_input(
        "learning_rate", value=float(knobs.get("learning_rate", 2e-4)), format="%.6f"
    )
    knobs["batch_size"] = st.sidebar.number_input(
        "batch_size", min_value=1, max_value=64, value=int(knobs.get("batch_size", 4))
    )
    knobs["lora_r"] = st.sidebar.number_input(
        "lora_r", min_value=4, max_value=64, value=int(knobs.get("lora_r", 8))
    )
    knobs["lora_alpha"] = st.sidebar.number_input(
        "lora_alpha", min_value=4, max_value=128, value=int(knobs.get("lora_alpha", 16))
    )
    knobs["num_epochs"] = st.sidebar.number_input(
        "num_epochs", min_value=1, max_value=10, value=int(knobs.get("num_epochs", 1))
    )

    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Reset Session", use_container_width=True):
        reset_session2()
        st.rerun()

    return {"settings": _settings}


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


def render_report() -> None:
    """Full attempt ledger + judge table + final config + artifact (spec D5)."""
    st.markdown("### 📋 Training Report")
    result = st.session_state.get("run_result")
    if result is None:
        st.info("Run the supervisor loop to see attempts, judge scores and config changes.")
        return

    if result.attempt_ledger:
        st.markdown(
            f"**Attempts used:** {result.attempts} — "
            + ("✅ passed the judge" if result.success else f"❌ {result.error}")
        )
        rows: dict[str, list] = {
            "attempt": [],
            "failure": [],
            "verdict": [],
            "config changes": [],
            "loss points": [],
        }
        for rec in result.attempt_ledger:
            rows["attempt"].append(rec.attempt)
            rows["failure"].append(rec.failure_key or "—")
            rows["verdict"].append(rec.verdict or "—")
            if rec.config_after:
                changed = ", ".join(
                    f"{k}: {rec.config_before.get(k)!r}->{rec.config_after[k]!r}"
                    for k in rec.config_after
                    if rec.config_before.get(k) != rec.config_after[k]
                )
            else:
                changed = "—"
            rows["config changes"].append(changed)
            rows["loss points"].append(len(rec.loss_points))
        st.dataframe(rows, use_container_width=True, hide_index=True)

    results = result.eval_results
    if not results:
        st.caption("No judge evaluation — the run never reached a healthy attempt.")
    else:
        base_avg = sum(results["base_scores"]) / len(results["base_scores"])
        ft_avg = sum(results["fine_tuned_scores"]) / len(results["fine_tuned_scores"])
        col1, col2, col3 = st.columns(3)
        col1.metric("Judge: base", f"{base_avg:.2f}")
        col2.metric("Judge: fine-tuned", f"{ft_avg:.2f}")
        col3.metric("Improvement", f"{ft_avg - base_avg:+.2f}")
        if results["regression_flagged"]:
            st.error("🚨 EVAL REGRESSION: fine-tuned is >10% below baseline.")
        else:
            st.caption("✅ Regression check passed (fine-tuned ≥ 90% of base).")
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

    if result.success:
        st.markdown("**Final config**")
        st.code(yaml.dump(result.final_config, sort_keys=True), language="yaml")
        if result.artifact and Path(result.artifact).exists():
            artifact = Path(result.artifact)
            st.download_button(
                f"⬇️ Download {artifact.name}",
                data=artifact.read_bytes(),
                file_name=artifact.name,
                mime="application/zip",
                key="report_download_btn",
            )


def _render_events(events: list) -> str:
    """Build the full timeline markdown from collected events."""
    if not events:
        return "No decisions yet - launch a job to see the loop."
    lines = []
    for ev in events:
        d = ev.data
        if ev.type == "job_launched":
            payload = d.get("payload", {})
            lines.append(
                f"**Attempt {d['attempt']}** · 🚀 launch `{payload.get('model', '?')}` on T4 "
                f"({len(payload.get('config', {}))} config keys)"
            )
        elif ev.type == "failure_detected":
            lines.append(f"**Attempt {d['attempt']}** · ❌ detected `{d['failure_key']}`")
        elif ev.type == "classified":
            mark = "⚠️ diverges from playbook" if d["diverged"] else "agrees with playbook"
            lines.append(
                f"**Attempt {d['attempt']}** · 🤖 Nemotron → "
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
                f"{k}: {before.get(k)!r}->{after[k]!r}" for k in after if before.get(k) != after[k]
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
        title="Training loss (per attempt)",
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


def execute_launch(config, max_retries, timeline_slot, logs_slot, chart_slot):
    """Run the real loop, streaming events + live loss into the placeholders."""
    from supertaco.gpu.transport import ColabTransport

    events: list[Event] = []
    live_points: list[float] = []

    def on_line(ln: str) -> None:
        logs_slot.code(ln[-500:], language="log")
        pts = extract_loss_points(ln)
        if pts:
            before = len(live_points)
            live_points.extend(p for p in pts if p == p)
            if len(live_points) > before:
                chart_slot.plotly_chart(_gpu_figure(live_points), width="stretch")

    def on_event(ev: Event) -> None:
        events.append(ev)
        st.session_state["events"] = events  # survive a mid-run crash (spec 6)
        timeline_slot.markdown(_render_events(events))
        if ev.type == "logs_produced":
            logs_slot.code(ev.data["text"], language="log")
            chart_slot.plotly_chart(_loss_figure(events), use_container_width=True)

    result = run_training_loop(
        config,
        transport=ColabTransport(),
        on_event=on_event,
        on_line=on_line,
        llm=make_llm(),
        max_retries=max_retries,
        runs_dir="configs/runs",
    )
    return events, result


def _gpu_figure(loss_points: list):
    """Loss curve for a real Colab run (NaN-safe, same style as the live loss chart)."""
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
        st.markdown("### 🚦 Fine-tune on Colab")
        cfg = render_config_editor(st.session_state["config"])
        knobs = st.session_state.get("knobs") or {}
        model_sel = st.session_state.get("model_sel")
        cfg_for_run = {**cfg, **knobs}
        if model_sel:
            cfg_for_run["model"] = model_sel
        run_clicked = st.button(
            "🚀 Fine-tune on Colab (supervisor loop)", type="primary", use_container_width=True
        )

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
                )
                st.session_state["events"] = events
                st.session_state["run_result"] = result
                st.session_state["llm_calls"] = result.llm_calls
                if not result.success:
                    st.error(f"🛑 {result.error}")
                else:
                    st.toast("✅ Run complete", icon="✅")
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

    # ── Bottom Row: Training report ────────────────────────────────
    st.markdown("---")
    render_report()


if __name__ == "__main__":
    main()
