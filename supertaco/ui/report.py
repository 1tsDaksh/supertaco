"""Shared report rendering for dashboard pages (Live Run, Results, showcase).

Moved out of the old monolithic dashboard so multiple pages can render the same
attempt ledger, judge summary and event timeline. Functions here take plain data
(session state or schema-v1 artifacts) and write Streamlit elements.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import streamlit as st

from supertaco.ui import data as data_mod


def render_events_markdown(events: list) -> str:
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


def render_timeline(slot, events: list) -> None:
    """Render the event timeline plus an advisory caption once a patch is proposed."""
    slot.markdown(render_events_markdown(events))
    if any(ev.type == "patch_proposed" for ev in events):
        st.caption(
            "Nemotron analysis is advisory - the applied patch is the deterministic playbook fix."
        )


def render_eval_summary(results: dict) -> None:
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
        width="stretch",
        hide_index=True,
    )


def llm_rows(calls: list) -> list:
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
        if isinstance(c, dict)
    ]


def render_showcase_runs() -> None:
    """Browse recorded sweep artifacts: metrics, judge summary, timeline."""
    artifacts, warnings = data_mod.load_artifacts(data_mod.examples_runs_dir())
    for msg in warnings:
        st.warning(msg)
    if not artifacts:
        st.caption("No recorded runs shipped with this deployment.")
        return
    for art in artifacts:
        ev = art.get("eval") or {}
        title = art.get("goal") or art.get("run_id", "recorded run")
        with st.container(border=True):
            st.markdown(f"#### 📁 {title}")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Run", str(art.get("run_id", "?")))
            m2.metric("Attempts", art.get("attempts", 0))
            m3.metric("Healed", "yes" if art.get("healed") else "no")
            m4.metric(
                "Judge delta",
                f"{ev['delta']:+.2f}" if isinstance(ev.get("delta"), (int, float)) else "—",
            )
            if ev and "base_scores" in ev and "modes" in ev:
                render_eval_summary({**ev, "judge_backend": art.get("judge_backend") or "direct"})
            events = [
                SimpleNamespace(type=e.get("type", "?"), data=e.get("data") or {})
                for e in art.get("events", [])
                if isinstance(e, dict)
            ]
            st.markdown(render_events_markdown(events))
            with st.expander("🧠 LLM calls"):
                st.dataframe(
                    llm_rows(art.get("llm_calls") or []),
                    width="stretch",
                    hide_index=True,
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
        st.dataframe(rows, width="stretch", hide_index=True)

    results = result.eval_results
    if not results:
        st.caption("No judge evaluation — the run never reached a healthy attempt.")
    else:
        render_eval_summary(results)

    if result.success:
        st.markdown("**Final config**")
        import yaml

        st.code(yaml.dump(result.final_config, sort_keys=True), language="yaml")
        from pathlib import Path

        if result.artifact and Path(result.artifact).exists():
            artifact = Path(result.artifact)
            st.download_button(
                f"⬇️ Download {artifact.name}",
                data=artifact.read_bytes(),
                file_name=artifact.name,
                mime="application/zip",
                key="report_download_btn",
            )


def artifact_run_label(art: dict[str, Any]) -> str:
    """Selectbox label for an artifact: run_id + short goal."""
    goal = str(art.get("goal") or "").strip()
    short = goal if len(goal) <= 48 else goal[:45] + "…"
    return f"{art.get('run_id', '?')} — {short}" if short else str(art.get("run_id", "?"))
