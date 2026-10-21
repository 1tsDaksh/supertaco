"""SuperTaco Dashboard - Frontend for AI-supervised fine-tuning sandbox.

Streamlit app that lets users submit fine-tune jobs, monitor training,
diagnose failures, and evaluate before/after model comparison.
Reads settings directly from .env file without requiring package installation.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import plotly.graph_objects as go  # type: ignore[import-untyped]  # plotly ships no stubs
import streamlit as st
import yaml
from streamlit import config
from streamlit.errors import StreamlitSecretNotFoundError

from supertaco.errors import ConfigurationError
from supertaco.eval.harness import PROMPT_SUITE_SIZE
from supertaco.loop import Event, extract_loss_points, make_llm, run_training_loop
from supertaco.planner import materialize_dataset, plan_run, plan_to_config, revise_run_config
from supertaco.ui.secrets_bridge import inject_secrets

st.set_page_config(page_title="SuperTaco", page_icon="🌮")


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

# Community Cloud: no .env file — the Token Factory key arrives via st.secrets and must
# reach supertaco.settings (env name NEBIUS_API_KEY). Rebuild the settings singleton only
# when a key was actually injected; make_llm()/judge read it at call time, and local runs
# (no secrets) keep the original instance so the offline test seam stays intact.
if inject_secrets(st.secrets):
    import supertaco.settings as _settings_mod

    _settings_mod.settings = _settings_mod.SuperTacoSettings()

# Read settings from environment (matching SuperTacoSettings field names)
_SUPERTACO_ENV = os.getenv("SUPERTACO_ENV") or os.getenv("SANDBOXTUNE_ENV") or "dev"
_NEBIUS_PROJECT_ID = os.getenv("NEBIUS_PROJECT_ID", "")
_NEBIUS_API_KEY = os.getenv("NEBIUS_API_KEY", "")
_TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
_TOKEN_FACTORY_BASE_URL = os.getenv(
    "TOKEN_FACTORY_BASE_URL", "https://api.tokenfactory.nebius.com/v1"
)


def _showcase_enabled() -> bool:
    """SUPERTACO_SHOWCASE from the env, else st.secrets (Community Cloud has no env UI)."""
    raw = os.getenv("SUPERTACO_SHOWCASE")
    if raw is None:
        try:
            raw = st.secrets["SUPERTACO_SHOWCASE"]
        except StreamlitSecretNotFoundError:
            # Streamlit raises this for both a missing secrets file and a
            # malformed one; a configured-but-broken file must fail loud.
            # Must precede the FileNotFoundError clause: it subclasses it.
            if any(Path(p).exists() for p in config.get_option("secrets.files")):
                raise
            raw = None
        except (KeyError, FileNotFoundError):
            raw = None
    return str(raw or "").strip().lower() in {"1", "true", "yes"}


# Showcase mode: training disabled on the hosted demo; planning stays live.
_SHOWCASE = _showcase_enabled()


def get_settings():
    """Return a settings namespace with the same fields as SuperTacoSettings."""

    class _Settings:
        nebius_api_key: str = _NEBIUS_API_KEY
        nebius_project_id: str = _NEBIUS_PROJECT_ID
        tavily_api_key: str = _TAVILY_API_KEY
        token_factory_base_url: str = _TOKEN_FACTORY_BASE_URL
        environment: str = _SUPERTACO_ENV

    return _Settings()


# ─── Main settings object (module-level, from .env) ───────────────────
_settings = get_settings()


# ─── Session State Initialisation ────────────────────────────────
def _default_config(cfg_path: str = "configs/defaults/colab_t4.yaml") -> dict:
    """Shipped T4 default so the YAML editor starts filled; {} on any error."""
    repo = Path(__file__).resolve().parents[2]
    target = Path(cfg_path) if Path(cfg_path).is_absolute() else repo / cfg_path
    try:
        loaded = yaml.safe_load(target.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def init_session2():
    """Initialise session-state variables if not already set."""
    defaults = {
        "cfg_path": "configs/defaults/colab_t4.yaml",
        "knobs": {},
        "llm_calls": [],
        "eval_results": None,
        "events": [],
        "run_result": None,
        "prev_run_result": None,
        "run_active": False,
        "gpu_result": None,
        "gpu_lines": [],
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val
    if "config" not in st.session_state:
        st.session_state["config"] = _default_config()


def reset_session2():
    """Reset all session-state variables."""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    init_session2()


# ─── UI Components ────────────────────────────────────────────────
def _mask_key(key: str) -> str:
    """Mask an API key for sidebar display: tvly-****abcd (short keys fully masked)."""
    if len(key) < 12:
        return "****"
    return f"{key[:5]}****{key[-4:]}"


def render_sidebar() -> Dict[str, Any]:
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

    st.sidebar.markdown("### Model Cost Routing")
    st.sidebar.caption(
        "Start at nano → escalate to super on low confidence → ultra only when playbook misses"
    )
    if _TAVILY_API_KEY:
        st.sidebar.caption(f"🔎 Tavily citations: on ({_mask_key(_TAVILY_API_KEY)})")
    else:
        st.sidebar.caption("🔎 Tavily citations: off (no TAVILY_API_KEY)")

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


def render_goal_panel() -> None:
    """Goal textarea -> Nemotron plan (config, dataset, eval prompts) -> session state (G7)."""
    st.markdown("### 🎯 Goal")
    goal = st.text_area(
        "What should the model learn?",
        height=80,
        key="goal_text",
        placeholder="e.g. Answer like a pirate chef: warm, nautical, precise about cooking times",
    )
    if not st.button("✨ Plan with Nemotron", use_container_width=True):
        return
    if not goal.strip():
        st.warning("Enter a goal first.")
        return
    with st.spinner("Nemotron is drafting your run plan..."):
        llm = make_llm()
        plan = materialize_dataset(plan_run(goal, llm), goal, llm)
    st.session_state["plan_goal"] = goal
    st.session_state["plan_dataset"] = plan.dataset.model_dump(exclude_none=True)
    st.session_state["plan_prompts"] = list(plan.eval_prompts)
    cfg = plan_to_config(plan)
    from supertaco.gpu.transport import MODEL_WHITELIST

    # Planner canonicalizes to HF ids; the dashboard's config convention is
    # whitelist keys (selectbox options, configs/defaults/colab_t4.yaml).
    cfg["model"] = next((k for k, v in MODEL_WHITELIST.items() if v == cfg["model"]), cfg["model"])
    st.session_state["config"] = {k: v for k, v in cfg.items() if k != "dataset"}
    st.session_state["knobs"] = {
        "learning_rate": plan.learning_rate,
        "batch_size": plan.batch_size,
        "lora_r": plan.lora_r,
        "lora_alpha": plan.lora_alpha,
        "num_epochs": plan.num_epochs,
    }
    if plan.plan_mode == "fallback":
        st.warning(
            f'Planned for: "{st.session_state["plan_goal"]}" — '
            "Planner fell back to shipped defaults (Nemotron reply unparseable) - "
            "review everything before running."
        )
    else:
        st.success(
            f'Planned for: "{st.session_state["plan_goal"]}" — '
            "Plan ready - review config, dataset and eval prompts, then Run."
        )


def render_report() -> None:
    """Full attempt ledger + judge table + final config + artifact (spec D5)."""
    st.markdown("### 📋 Training Report")
    result = st.session_state.get("run_result")
    if result is None:
        if st.session_state.get("run_active"):
            st.info("Run in progress - logs stream below...")
        else:
            st.info("Run the supervisor loop to see attempts, judge scores and config changes.")
        return

    if result.attempt_ledger:
        if result.success and result.attempts == 1:
            outcome = "✅ succeeded on first attempt - no heal needed"
        elif result.success:
            outcome = "✅ passed the judge"
        else:
            outcome = f"❌ {result.error}"
        st.markdown(f"**Attempts used:** {result.attempts} — {outcome}")
        rows: dict[str, list] = {
            "attempt": [],
            "failure": [],
            "verdict": [],
            "config changes": [],
            "loss points": [],
            "sources": [],
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
            rows["sources"].append(
                ", ".join(c.get("title", "") for c in rec.citations).strip(", ") or "—"
            )
        st.dataframe(rows, use_container_width=True, hide_index=True)

    results = result.eval_results
    if not results:
        st.caption("No judge evaluation — the run never reached a healthy attempt.")
    else:
        _render_eval_summary(results)

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
            line = f"**Attempt {d['attempt']}** · 🧩 model reasoning: {reason}"
            cites = d.get("citations") or []
            if cites:
                line += "\n   ↳ sources: " + ", ".join(f"[{c['title']}]({c['url']})" for c in cites)
            lines.append(line)
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
            if d["attempts"] > 1:
                lines.append(f"✅ **Healed** after {d['attempts']} attempt(s)")
            else:
                lines.append("✅ **Succeeded** on first attempt - no heal needed")
        elif ev.type == "run_failed":
            detail = f" · attempts {d['attempts']}" if d.get("attempts") is not None else ""
            last = f" · last `{d['path']}`" if d.get("path") else ""
            line = f"🛑 **Failed**: {d['error']}{detail}{last}"
            cit_ev = next(
                (
                    e
                    for e in reversed(events)
                    if e.type == "citations_found"
                    and e.data.get("failure_key") == d.get("failure_key")
                    and e.data.get("citations")
                ),
                None,
            )
            if cit_ev is not None:
                line += "\n   ↳ sources: " + ", ".join(
                    f"[{c['title']}]({c['url']})" for c in cit_ev.data["citations"]
                )
            lines.append(line)
    return "\n\n".join(lines)


def _render_timeline(slot, events: list) -> None:
    """Render the event timeline plus an advisory caption once a patch is proposed."""
    slot.markdown(_render_events(events))
    if any(ev.type == "patch_proposed" for ev in events):
        st.caption(
            "Nemotron analysis is advisory - the applied patch is the deterministic playbook fix."
        )


def _render_eval_summary(results: dict) -> None:
    """Judge metrics, regression/backend/mode warnings, and the per-prompt table."""
    base_avg = sum(results["base_scores"]) / len(results["base_scores"])
    ft_avg = sum(results["fine_tuned_scores"]) / len(results["fine_tuned_scores"])
    col1, col2, col3 = st.columns(3)
    col1.metric("Judge: base", f"{base_avg:.2f}")
    col2.metric("Judge: fine-tuned", f"{ft_avg:.2f}")
    col3.metric("Improvement", f"{ft_avg - base_avg:+.2f}")
    prev = st.session_state.get("prev_run_result")
    prev_eval = prev.eval_results if prev else None
    prev_scores = (prev_eval or {}).get("fine_tuned_scores") or []
    prev_modes = (prev_eval or {}).get("modes") or []
    if (
        prev_scores
        and results["fine_tuned_scores"]
        and all("hash" not in m and "fallback" not in m for m in results["modes"])
        and all("hash" not in m and "fallback" not in m for m in prev_modes)
    ):
        prev_ft = sum(prev_scores) / len(prev_scores)
        st.caption(
            f"Before feedback: fine-tuned avg {prev_ft:.2f} -> after: {ft_avg:.2f} "
            f"({ft_avg - prev_ft:+.2f})"
        )
    if results["regression_flagged"]:
        st.error("🚨 EVAL REGRESSION: fine-tuned is >10% below baseline.")
    else:
        st.caption("✅ Regression check passed (fine-tuned ≥ 90% of base).")
    backend = results.get("judge_backend", "direct")
    if backend == "nebius_job":
        st.success("✓ Judged on Nebius AI Cloud — eval ran as a serverless job")
    elif backend == "direct_fallback":
        st.warning("⚠ Nebius judge unavailable — fell back to direct Token Factory judging")
    modes = results["modes"]
    if any("fallback" in m for m in modes):
        st.warning(
            "⚠ Judge HTTP failure (transport/circuit) — affected scores did not "
            "come from a live Nemotron reply."
        )
    if any("hash" in m for m in modes):
        st.warning(
            "⚠ Judge reply unparseable after retry — affected scores are "
            "deterministic hash placeholders."
        )
        st.caption("† hash-mode score — deterministic placeholder (no parseable Nemotron reply).")
    st.dataframe(
        {
            "prompt": results["prompts"],
            "base": results["base_scores"],
            "fine_tuned": results["fine_tuned_scores"],
            "judge": [f"{m} †" if "hash" in m else m for m in modes],
        },
        use_container_width=True,
        hide_index=True,
    )


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


def _load_showcase_runs() -> list[dict]:
    """Committed schema-v1 artifacts from examples/runs (SUPERTACO_EXAMPLES_RUNS overrides)."""
    override = os.getenv("SUPERTACO_EXAMPLES_RUNS")
    if override:
        runs_dir = Path(override)
    else:
        runs_dir = Path(__file__).resolve().parents[2] / "examples" / "runs"
    artifacts = []
    for path in sorted(runs_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            st.warning(f"Skipped unreadable recorded run `{path.name}`: {exc}")
            continue
        if not isinstance(data, dict) or data.get("schema") != 1:
            continue  # non-run files (e.g. *.timing.json) are not recordings
        artifacts.append(data)
    return artifacts


def render_showcase_runs() -> None:
    """Browse recorded sweep artifacts: metrics, judge summary, timeline (WS5)."""
    artifacts = _load_showcase_runs()
    if not artifacts:
        st.caption("No recorded runs shipped with this deployment.")
        return
    for data in artifacts:
        ev = data.get("eval") or {}
        title = data.get("goal") or data.get("run_id", "recorded run")
        with st.container(border=True):
            st.markdown(f"#### 📁 {title}")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Run", str(data.get("run_id", "?")))
            m2.metric("Attempts", data.get("attempts", 0))
            m3.metric("Healed", "yes" if data.get("healed") else "no")
            m4.metric(
                "Judge delta",
                f"{ev['delta']:+.2f}" if isinstance(ev.get("delta"), (int, float)) else "—",
            )
            if ev and "base_scores" in ev and "modes" in ev:
                _render_eval_summary({**ev, "judge_backend": data.get("judge_backend") or "direct"})
            events = [
                SimpleNamespace(type=e.get("type", "?"), data=e.get("data") or {})
                for e in data.get("events", [])
                if isinstance(e, dict)
            ]
            st.markdown(_render_events(events))
            with st.expander("🧠 LLM calls"):
                st.dataframe(
                    _llm_rows(data.get("llm_calls") or []),
                    use_container_width=True,
                    hide_index=True,
                )


def execute_launch(
    config, max_retries, timeline_slot, logs_slot, chart_slot, prompts=None, judge_backend="direct"
):
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
        _render_timeline(timeline_slot, events)
        if ev.type == "logs_produced":
            logs_slot.code(ev.data["text"], language="log")
            chart_slot.plotly_chart(_loss_figure(events), use_container_width=True)

    st.session_state["run_active"] = True
    try:
        result = run_training_loop(
            config,
            transport=ColabTransport(),
            on_event=on_event,
            on_line=on_line,
            llm=make_llm(),
            max_retries=max_retries,
            runs_dir="configs/runs",
            prompts=prompts,
            judge_backend=judge_backend,
            goal=(st.session_state.get("plan_goal") or None),
        )
    finally:
        st.session_state["run_active"] = False
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

    st.title("🚀 SuperTaco — AI-supervised fine-tuning sandbox")
    st.caption(
        "Type a goal → Nemotron plans config, data and eval prompts → you review → "
        "free Colab T4 → self-healing loop → Nemotron judge before/after"
    )
    if _SHOWCASE:
        st.info(
            "🧪 **Showcase mode** — live training is disabled on this deployment; "
            "browse recorded runs below. Planning is live."
        )
    render_goal_panel()
    render_sidebar()

    # ── Left Column: Job Launcher ──────────────────────────────────
    with st.container(border=True):
        st.markdown("### 🚦 Fine-tune on Colab")
        cfg = render_config_editor(st.session_state["config"])
        ds = st.session_state.get("plan_dataset")
        if ds:
            if ds.get("kind") == "synthetic":
                label = f"synthetic - {len(ds.get('rows') or [])} generated rows"
            else:
                label = f"huggingface - {ds.get('hf_id')} ({ds.get('max_rows')} rows)"
            if ds.get("note"):
                label += f" | {ds['note']}"
            st.caption(f"Dataset: {label}")
        if "plan_prompts" in st.session_state:
            edited_prompts = st.text_area(
                "Eval prompts (one per line - the Nemotron judge scores these)",
                value="\n".join(st.session_state["plan_prompts"]),
                height=110,
            )
            st.session_state["plan_prompts"] = [p for p in edited_prompts.splitlines() if p.strip()]
            n_prompts = len(st.session_state["plan_prompts"])
            if n_prompts == 0:
                st.caption("Prompts cleared - the judge will use the default 5-prompt suite.")
            elif n_prompts != PROMPT_SUITE_SIZE:
                st.warning(
                    f"Expected exactly {PROMPT_SUITE_SIZE} eval prompts ({n_prompts} submitted) "
                    f"- the judge scores at most {PROMPT_SUITE_SIZE}."
                )
        knobs = st.session_state.get("knobs") or {}
        model_sel = st.session_state.get("model_sel")
        cfg_for_run = {**cfg, **knobs}
        if model_sel:
            cfg_for_run["model"] = model_sel
        if st.session_state.get("plan_dataset"):
            cfg_for_run["dataset"] = dict(st.session_state["plan_dataset"])
        st.checkbox(
            "Judge on Nebius AI Cloud (serverless job)",
            value=True,
            key="judge_on_nebius",
            disabled=_SHOWCASE,
            help=(
                "Runs the Nemotron eval as a Nebius Serverless Job; falls back "
                "to direct Token Factory judging on infra failure."
            ),
        )
        run_clicked = st.button(
            "🚀 Fine-tune on Colab (supervisor loop)",
            type="primary",
            use_container_width=True,
            disabled=_SHOWCASE,
        )

        feedback_clicked = False
        if st.session_state.get("run_result") is not None:
            feedback = st.text_area(
                "Feedback on the result",
                key="feedback_text",
                height=68,
                placeholder='e.g. "be more concise", "pirate voice too subtle"',
            )
            feedback_clicked = st.button(
                "🔁 Re-run with feedback", use_container_width=True, disabled=_SHOWCASE
            )
        if feedback_clicked:
            goal = (st.session_state.get("plan_goal") or "").strip()
            if not feedback.strip():
                st.warning("Describe what to change first.")
            elif not goal:
                st.warning("Feedback needs a goal-planned run - plan a goal first.")
            else:
                with st.spinner("Nemotron is revising the plan from your feedback..."):
                    llm = make_llm()
                    prior_cfg = {**cfg, **knobs}
                    if model_sel:
                        prior_cfg["model"] = model_sel
                    if st.session_state.get("plan_dataset"):
                        prior_cfg["dataset"] = dict(st.session_state["plan_dataset"])
                    prior_cfg["eval_prompts"] = st.session_state.get("plan_prompts") or []
                    new_cfg, revised = revise_run_config(feedback.strip(), prior_cfg, goal, llm)
                from supertaco.gpu.transport import MODEL_WHITELIST

                st.session_state["config"] = {
                    k: v
                    for k, v in new_cfg.items()
                    if k not in ("dataset", "goal", "eval_prompts", "revision_note")
                }
                st.session_state["config"]["model"] = next(
                    (
                        k
                        for k, v in MODEL_WHITELIST.items()
                        if v == st.session_state["config"]["model"]
                    ),
                    st.session_state["config"]["model"],
                )
                st.session_state["knobs"] = {
                    k: getattr(revised, k)
                    for k in ("learning_rate", "batch_size", "lora_r", "lora_alpha", "num_epochs")
                }
                st.session_state["plan_dataset"] = revised.dataset.model_dump(exclude_none=True)
                st.session_state["plan_prompts"] = list(revised.eval_prompts)
                st.session_state["prev_run_result"] = st.session_state["run_result"]
                if revised.plan_mode == "fallback":
                    st.warning(
                        f'Revision for: "{feedback.strip()}" - kept the previous plan '
                        "(Nemotron reply unparseable)."
                    )
                else:
                    st.success(
                        f'Revised with feedback: "{feedback.strip()}"'
                        + (f" - {revised.revision_note}" if revised.revision_note else "")
                    )
                cfg_for_run = {
                    **st.session_state["config"],
                    **(st.session_state.get("knobs") or {}),
                }
                if st.session_state.get("plan_dataset"):
                    cfg_for_run["dataset"] = dict(st.session_state["plan_dataset"])
                run_clicked = True
        elif run_clicked:
            st.session_state["prev_run_result"] = None

    # ── Right Column: Monitor & Metrics ────────────────────────────
    with st.container(border=True):
        st.markdown("### 📡 Job Status & Logs")
        timeline_slot = st.empty()
        logs_slot = st.empty()
        chart_slot = st.empty()

        if run_clicked and _SHOWCASE:
            st.warning("Training is disabled in showcase mode.")
            run_clicked = False
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
                    prompts=st.session_state.get("plan_prompts") or None,
                    judge_backend=(
                        "nebius_job" if st.session_state.get("judge_on_nebius", True) else "direct"
                    ),
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
                _render_timeline(timeline_slot, st.session_state.get("events", []))
            except Exception as exc:  # spec 6: never a Streamlit traceback
                st.error(f"❌ Run crashed: {exc}")
                _render_timeline(timeline_slot, st.session_state.get("events", []))
        elif st.session_state.get("events"):
            _render_timeline(timeline_slot, st.session_state["events"])
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
    if _SHOWCASE:
        render_showcase_runs()


if __name__ == "__main__":
    main()
