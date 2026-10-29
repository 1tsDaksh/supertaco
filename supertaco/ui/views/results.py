"""Results page — sweep receipts and run history.

Summary strip, per-run table with delta badges, and a deep-dive pane for any
selected artifact: facts, judge detail, event receipts and copy-paste verify
commands. Every value comes from schema-v1 artifacts (see ui/data.py).
"""

from __future__ import annotations

from types import SimpleNamespace

import streamlit as st

from supertaco.ui import components, report, runtime
from supertaco.ui import data as data_mod


def render_results() -> None:
    showcase = runtime.showcase_enabled()
    st.markdown("## 📊 Results")
    artifacts, warnings = data_mod.load_collection(showcase)
    for msg in warnings:
        st.warning(msg)

    if not artifacts:
        st.info("No run artifacts yet — run a demo from the Live page to generate results.")
        return

    summary = data_mod.sweep_summary(artifacts)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Runs", f"{summary['successful']}/{summary['runs']} ok")
    avg = summary.get("avg_delta")
    m2.metric("Avg Δ", f"{avg:+.2f}" if isinstance(avg, float) else "—")
    m3.metric("Improved", str(summary["improved"]))
    wall = summary.get("wall_s")
    m4.metric("Wall clock", f"{wall / 60:.1f} min" if isinstance(wall, (int, float)) else "—")

    rows = [r for r in (data_mod.eval_row(a) for a in artifacts) if r]
    if rows:
        st.markdown("### Sweep table")
        st.dataframe(
            {
                "run": [r["run_id"] for r in rows],
                "goal": [r["goal"] for r in rows],
                "attempts": [r["attempts"] for r in rows],
                "healed": ["yes" if r["healed"] else "no" for r in rows],
                "base": [
                    f"{r['base']:.2f}" if isinstance(r["base"], (int, float)) else "—" for r in rows
                ],
                "ft": [
                    f"{r['ft']:.2f}" if isinstance(r["ft"], (int, float)) else "—" for r in rows
                ],
                "Δ": [f"{r['delta']:+.2f}" for r in rows],
            },
            width="stretch",
            hide_index=True,
        )

    st.markdown("### Run detail")
    by_label = {report.artifact_run_label(a): a for a in artifacts}
    choice = st.selectbox("Run", list(by_label))
    art = by_label[choice]
    with st.container(border=True):
        ev = art.get("eval") or {}
        c1, c2, c3 = st.columns(3)
        c1.metric("Attempts", art.get("attempts", 0))
        c2.metric("Healed", "yes" if art.get("healed") else "no")
        c3.metric(
            "Judge delta",
            f"{ev['delta']:+.2f}" if isinstance(ev.get("delta"), (int, float)) else "—",
        )
        st.caption(
            f"run `{art.get('run_id', '?')}` · judge `{art.get('judge_backend') or 'direct'}` · "
            f"plan `{art.get('plan_mode') or 'real'}`"
        )
        if ev and "base_scores" in ev and "modes" in ev:
            report.render_eval_summary(
                {**ev, "judge_backend": art.get("judge_backend") or "direct"}
            )
        events = [
            SimpleNamespace(type=e.get("type", "?"), data=e.get("data") or {})
            for e in art.get("events", [])
            if isinstance(e, dict)
        ]
        components.render_phase_timeline(events)
        healed = bool(art.get("healed"))
        with st.expander("🩹 Heal timeline" if healed else "🕓 Attempt timeline", expanded=healed):
            st.markdown(report.render_events_markdown(events))
        with st.expander("🔎 Verify this run"):
            st.code(data_mod.verify_commands(str(art.get("run_id", "?"))), language="bash")
        with st.expander("🧠 LLM calls"):
            st.dataframe(
                report.llm_rows(art.get("llm_calls") or []),
                width="stretch",
                hide_index=True,
            )
