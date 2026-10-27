"""Live Run page — setup & launch (left) + live activity monitor (right).

Split workspace per the redesign: the left column holds everything needed to
configure and launch a run; the right column shows what is happening right now —
phase timeline, an auto-scrolling terminal with full history, the loss chart and
the event feed. Patchable collaborators (loop, planner, transport) are always
reached through their module objects so tests can monkeypatch between runs.
"""

from __future__ import annotations

import os
from pathlib import Path

import plotly.graph_objects as go  # type: ignore[import-untyped]  # plotly ships no stubs
import streamlit as st
import yaml

import supertaco.loop as loop_mod
import supertaco.planner as planner_mod
from supertaco.errors import ConfigurationError
from supertaco.eval.harness import PROMPT_SUITE_SIZE
from supertaco.loop import Event, extract_loss_points
from supertaco.ui import components, report, runtime
from supertaco.ui.theme import TOKENS

_RUN_BTN_LABEL = "🚀 Fine-tune on Colab (supervisor loop)"


def _style_fig(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=TOKENS["bg_card"],
        plot_bgcolor=TOKENS["bg_inset"],
        font_color=TOKENS["text"],
        height=300,
    )
    return fig


def render_setup_panel() -> None:
    """Launch controls (formerly the sidebar): retries, config path, model, knobs."""
    tavily = os.getenv("TAVILY_API_KEY", "")
    st.markdown("#### ⚙️ Setup")
    calls = st.session_state.get("llm_calls") or []
    fallbacks = [c for c in calls if c.get("mode") != "real"]
    if fallbacks:
        st.error(
            f"⚠️ Nemotron fallback: {len(fallbacks)}/{len(calls)} call(s) used heuristics. "
            f"First error: {fallbacks[0].get('error')}"
        )
    elif calls:
        st.success(f"✅ Nemotron: {len(calls)} real call(s)")
    st.caption(
        "Start at nano → escalate to super on low confidence → ultra only when playbook misses"
    )
    if tavily:
        st.caption(f"🔎 Tavily citations: on ({runtime.mask_key(tavily)})")
    else:
        st.caption("🔎 Tavily citations: off (no TAVILY_API_KEY)")
    st.session_state["max_retries"] = st.number_input(
        "Max retries per run",
        min_value=1,
        max_value=10,
        value=int(st.session_state.get("max_retries", 3)),
    )
    st.markdown("---")
    st.markdown("##### Training Config")
    repo = Path(__file__).resolve().parents[3]
    cfg_path = st.text_input(
        "Config YAML path", value=st.session_state.get("cfg_path", "configs/defaults/colab_t4.yaml")
    )
    st.session_state["cfg_path"] = cfg_path
    if st.button("Load config"):
        target = Path(cfg_path) if Path(cfg_path).is_absolute() else repo / cfg_path
        try:
            loaded = yaml.safe_load(target.read_text(encoding="utf-8-sig"))
        except Exception as exc:
            st.error(f"Could not load {cfg_path}: {exc}")
        else:
            if not isinstance(loaded, dict):
                st.error(f"{cfg_path} is not a config mapping")
            else:
                st.session_state["config"] = loaded
                st.success(f"Loaded {cfg_path}")

    from supertaco.gpu.transport import MODEL_WHITELIST

    st.markdown("##### Model (T4-viable)")
    # Plan 3c adjustment: surface a non-whitelist configured model instead of replacing it.
    # NOTE: no key= for this selectbox — rejection of invalid configured models depends on
    # `index` participating in the no-key element id (adding key= would let stale widget
    # state override the configured model and silently clobber pre-launch validation).
    options = list(MODEL_WHITELIST)
    current_model = (st.session_state.get("config") or {}).get("model")
    if isinstance(current_model, str) and current_model and current_model not in options:
        options.append(current_model)
    st.session_state["model_sel"] = st.selectbox(
        "model",
        options,
        index=options.index(current_model) if current_model in options else 0,
        help="Whitelisted HF models that fit a free T4",
    )
    st.markdown("##### Hyperparameters")
    knobs = st.session_state.setdefault("knobs", {})
    k1, k2 = st.columns(2)
    knobs["learning_rate"] = k1.number_input(
        "learning_rate", value=float(knobs.get("learning_rate", 2e-4)), format="%.6f"
    )
    knobs["batch_size"] = k2.number_input(
        "batch_size", min_value=1, max_value=64, value=int(knobs.get("batch_size", 4))
    )
    knobs["lora_r"] = k1.number_input(
        "lora_r", min_value=4, max_value=64, value=int(knobs.get("lora_r", 8))
    )
    knobs["lora_alpha"] = k2.number_input(
        "lora_alpha", min_value=4, max_value=128, value=int(knobs.get("lora_alpha", 16))
    )
    knobs["num_epochs"] = st.number_input(
        "num_epochs", min_value=1, max_value=10, value=int(knobs.get("num_epochs", 1))
    )
    if st.button("🔄 Reset Session", width="stretch"):
        from supertaco.ui.dashboard import reset_session2

        reset_session2()
        st.rerun()


def render_goal_panel() -> None:
    """Goal textarea -> Nemotron plan (config, dataset, eval prompts) -> session state (G7)."""
    st.markdown("#### 🎯 Goal")
    goal = st.text_area(
        "What should the model learn?",
        height=80,
        key="goal_text",
        placeholder="e.g. Answer like a pirate chef: warm, nautical, precise about cooking times",
    )
    if not st.button("✨ Plan with Nemotron", width="stretch"):
        return
    if not goal.strip():
        st.warning("Enter a goal first.")
        return
    with st.spinner("Nemotron is drafting your run plan..."):
        llm = loop_mod.make_llm()
        plan = planner_mod.materialize_dataset(planner_mod.plan_run(goal, llm), goal, llm)
    st.session_state["plan_goal"] = goal
    st.session_state["plan_dataset"] = plan.dataset.model_dump(exclude_none=True)
    st.session_state["plan_prompts"] = list(plan.eval_prompts)
    st.session_state["plan_mode"] = plan.plan_mode
    cfg = planner_mod.plan_to_config(plan)
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


def render_config_editor(config: dict) -> dict:
    """Render an editable YAML config panel."""
    with st.expander("📝 View / Edit config YAML", expanded=False):
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
    return parsed


def _loss_figure(events: list):
    """Per-attempt loss series parsed from generated logs (spec 5.5)."""
    import math

    fig = go.Figure()
    for ev in events:
        if ev.type != "logs_produced":
            continue
        points = [None if math.isnan(v) else v for v in ev.data["loss_points"]]
        fig.add_trace(go.Scatter(y=points, mode="lines", name=f"attempt {ev.data['attempt']}"))
    fig.update_layout(title="Training loss (per attempt)", xaxis_title="Step", yaxis_title="Loss")
    return _style_fig(fig)


def _gpu_figure(loss_points: list):
    """Loss curve for a real Colab run (NaN-safe)."""
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
    )
    return _style_fig(fig)


def execute_launch(
    config, max_retries, timeline_slot, logs_slot, chart_slot, prompts=None, judge_backend="direct"
):
    """Run the real loop, streaming events + the auto-scrolled terminal into slots."""
    from supertaco.gpu.transport import ColabTransport

    events: list[Event] = []
    live_points: list[float] = []
    term_lines: list[str] = []

    def on_line(ln: str) -> None:
        term_lines.append(ln[-500:])
        st.session_state["term_lines"] = list(term_lines)  # pinned terminal mirror
        logs_slot.html(components.build_terminal_html(term_lines))
        pts = extract_loss_points(ln)
        if pts:
            before = len(live_points)
            live_points.extend(p for p in pts if p == p)
            if len(live_points) > before:
                chart_slot.plotly_chart(_gpu_figure(live_points), width="stretch")

    def on_event(ev: Event) -> None:
        events.append(ev)
        st.session_state["events"] = events  # survive a mid-run crash (spec 6)
        report.render_timeline(timeline_slot, events)
        if ev.type == "logs_produced":
            text = ev.data["text"]
            term_lines.extend(text.splitlines() or [""])
            st.session_state["term_lines"] = list(term_lines)
            logs_slot.html(components.build_terminal_html(term_lines))
            chart_slot.plotly_chart(_loss_figure(events), width="stretch")

    st.session_state["run_active"] = True
    try:
        result = loop_mod.run_training_loop(
            config,
            transport=ColabTransport(),
            on_event=on_event,
            on_line=on_line,
            llm=loop_mod.make_llm(),
            max_retries=max_retries,
            runs_dir="configs/runs",
            prompts=prompts,
            judge_backend=judge_backend,
            goal=(st.session_state.get("plan_goal") or None),
        )
    finally:
        st.session_state["run_active"] = False
    return events, result


def _run_status(events: list, run_active: bool, run_result) -> str:
    if run_active:
        return "running"
    if run_result is not None:
        return "ok" if run_result.success else "failed"
    if events:
        last = events[-1].type
        if last == "run_failed":
            return "failed"
    return "idle"


def render_live_run() -> None:
    """Split workspace: setup & launch left, live activity right."""
    showcase = runtime.showcase_enabled()
    events = st.session_state.get("events") or []
    run_active = bool(st.session_state.get("run_active"))
    run_result = st.session_state.get("run_result")

    head_l, head_r = st.columns([6, 1])
    head_l.markdown("## 🛰️ Live Run")
    with head_r:
        st.markdown("")  # vertical alignment spacer
        components.render_status_chip(_run_status(events, run_active, run_result))
    components.render_phase_timeline(events)
    if showcase:
        st.info(
            "🧪 **Showcase mode** — live training is disabled on this deployment; "
            "browse recorded runs on the Landing page. Planning is live."
        )

    left, right = st.columns([5, 7])

    # ── Left: setup & launch ────────────────────────────────────────
    with left:
        with st.container(border=True):
            render_goal_panel()
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
                st.session_state["plan_prompts"] = [
                    p for p in edited_prompts.splitlines() if p.strip()
                ]
                n_prompts = len(st.session_state["plan_prompts"])
                if n_prompts == 0:
                    st.caption("Prompts cleared - the judge will use the default 5-prompt suite.")
                elif n_prompts != PROMPT_SUITE_SIZE:
                    st.warning(
                        f"Expected exactly {PROMPT_SUITE_SIZE} eval prompts "
                        f"({n_prompts} submitted) "
                        f"- the judge scores at most {PROMPT_SUITE_SIZE}."
                    )
            render_setup_panel()  # knobs/model/retries must exist before the merge below
            knobs = st.session_state.get("knobs") or {}
            model_sel = st.session_state.get("model_sel")
            cfg_for_run = {**cfg, **knobs}
            if model_sel:
                cfg_for_run["model"] = model_sel
            if st.session_state.get("plan_dataset"):
                cfg_for_run["dataset"] = dict(st.session_state["plan_dataset"])
            plan_goal = st.session_state.get("plan_goal")
            if plan_goal:
                plan_mode = (
                    "fallback" if st.session_state.get("plan_mode") == "fallback" else "real"
                )
                components.render_status_chip(
                    "recorded" if plan_mode == "fallback" else "ok",
                    label=f"plan: {plan_mode}",
                )
            st.checkbox(
                "Judge on Nebius AI Cloud (serverless job)",
                value=True,
                key="judge_on_nebius",
                disabled=showcase,
                help=(
                    "Runs the Nemotron eval as a Nebius Serverless Job; falls back "
                    "to direct Token Factory judging on infra failure."
                ),
            )
            run_clicked = st.button(
                _RUN_BTN_LABEL,
                type="primary",
                width="stretch",
                disabled=showcase,
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
                    "🔁 Re-run with feedback", width="stretch", disabled=showcase
                )
            if feedback_clicked:
                goal = (st.session_state.get("plan_goal") or "").strip()
                if not feedback.strip():
                    st.warning("Describe what to change first.")
                elif not goal:
                    st.warning("Feedback needs a goal-planned run - plan a goal first.")
                else:
                    with st.spinner("Nemotron is revising the plan from your feedback..."):
                        llm = loop_mod.make_llm()
                        prior_cfg = {**cfg, **knobs}
                        if model_sel:
                            prior_cfg["model"] = model_sel
                        if st.session_state.get("plan_dataset"):
                            prior_cfg["dataset"] = dict(st.session_state["plan_dataset"])
                        prior_cfg["eval_prompts"] = st.session_state.get("plan_prompts") or []
                        new_cfg, revised = planner_mod.revise_run_config(
                            feedback.strip(), prior_cfg, goal, llm
                        )
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
                        for k in (
                            "learning_rate",
                            "batch_size",
                            "lora_r",
                            "lora_alpha",
                            "num_epochs",
                        )
                    }
                    st.session_state["plan_dataset"] = revised.dataset.model_dump(exclude_none=True)
                    st.session_state["plan_prompts"] = list(revised.eval_prompts)
                    st.session_state["plan_mode"] = revised.plan_mode
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

    # ── Right: live activity ────────────────────────────────────────
    with right:
        with st.container(border=True):
            st.markdown("#### 🖥️ Activity")
            timeline_slot = st.empty()
            term_slot = st.empty()
            chart_slot = st.empty()

            if run_clicked and showcase:
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
                        term_slot,
                        chart_slot,
                        prompts=st.session_state.get("plan_prompts") or None,
                        judge_backend=(
                            "nebius_job"
                            if st.session_state.get("judge_on_nebius", True)
                            else "direct"
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
                    report.render_timeline(timeline_slot, st.session_state.get("events", []))
                except Exception as exc:  # spec 6: never a Streamlit traceback
                    st.error(f"❌ Run crashed: {exc}")
                    report.render_timeline(timeline_slot, st.session_state.get("events", []))
            elif st.session_state.get("events"):
                report.render_timeline(timeline_slot, st.session_state["events"])
                latest_logs = next(
                    (
                        e.data["text"]
                        for e in reversed(st.session_state["events"])
                        if e.type == "logs_produced"
                    ),
                    "",
                )
                term_slot.html(components.build_terminal_html(latest_logs.splitlines() or ["—"]))
                chart_slot.plotly_chart(_loss_figure(st.session_state["events"]), width="stretch")
            else:
                timeline_slot.info("No run yet — describe a goal and press Launch.")

            calls = st.session_state.get("llm_calls") or []
            if calls:
                with st.expander(f"🧠 LLM Calls ({len(calls)})"):
                    st.dataframe(report.llm_rows(calls), width="stretch", hide_index=True)

    st.markdown("---")
    report.render_report()
