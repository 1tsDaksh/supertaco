"""Landing page — the front door.

A cold visitor should understand SuperTaco in ~30 seconds: what it is, proof it
works (live numbers from run artifacts, never fabricated), and one obvious next
action. No judge-bait — a real product front door.
"""

from __future__ import annotations

import streamlit as st

from supertaco.ui import components, report, runtime
from supertaco.ui import data as data_mod
from supertaco.ui.theme import TOKENS, esc

_HOW_STEPS = (
    (
        "Describe a goal",
        'Plain English — "JSON-only extractor", "kids\' science explainer", '
        '"formal email tone". No configs, no YAML required to start.',
    ),
    (
        "The planner LLM writes the run",
        "Dataset source, model, eval prompts and hyperparameters become a reviewable "
        "plan — flagged real or fallback, nothing hidden.",
    ),
    (
        "A judge LLM supervises training",
        "Base vs fine-tuned scored side by side on your eval prompts; failures heal "
        "themselves; every receipt (tokens, attempts, configs) is saved.",
    ),
)


def _hero_html() -> str:
    return f"""
<div class="stx-hero">
  <div style="font-size:34px;font-weight:700;color:{TOKENS['text_bright']};
       font-family:{TOKENS['sans']};letter-spacing:-.02em">🌮 SuperTaco</div>
  <div style="color:{TOKENS['text']};margin:10px auto 18px auto;max-width:620px;font-size:15px">
    AI-supervised fine-tuning sandbox — describe a goal, an AI agent plans the run,
    a judge LLM supervises training and scores before vs after.
  </div>
</div>
"""


def _step_card_html(number: int, title: str, body: str) -> str:
    return f"""
<div class="stx-card" style="height:100%">
  <div style="color:{TOKENS['accent']};font-weight:700;font-size:15px;
       font-family:{TOKENS['mono']}">{number}</div>
  <div style="color:{TOKENS['text_bright']};font-weight:600;margin:6px 0">{esc(title)}</div>
  <div style="color:{TOKENS['text']};font-size:13px;line-height:1.5">{esc(body)}</div>
</div>"""


def _evidence_strip_html(summary: dict) -> str:
    avg = summary.get("avg_delta")
    avg_txt = f"{avg:+.2f}" if isinstance(avg, float) else "—"
    tokens = summary.get("tokens") or {}
    cells = [
        (str(summary.get("successful", 0)) + "/" + str(summary.get("runs", 0)), "successful"),
        (str(summary.get("improved", 0)), "improved"),
        (avg_txt, "avg Δ judge pts"),
        (str(summary.get("regressed", 0)), "regressed"),
    ]
    for key in ("classify", "judge", "patch"):
        if tokens.get(key):
            cells.append((f"{tokens[key] / 1000:.1f}k", f"{key} tokens"))
    inner = "".join(
        f'<div style="flex:1;text-align:center">'
        f'<div style="color:{TOKENS["text_bright"]};font-weight:700;font-size:20px;'
        f'font-family:{TOKENS["mono"]}">{esc(value)}</div>'
        f'<div style="color:{TOKENS["text_dim"]};font-size:10px;text-transform:uppercase;'
        f'letter-spacing:.06em">{esc(label)}</div></div>'
        for value, label in cells
    )
    return (
        f'<div class="stx-card" style="display:flex;gap:10px;align-items:center">'
        f"{inner}</div>"
    )


def _nav_to(page: str) -> None:
    st.session_state["nav_page"] = page


def render_landing() -> None:
    showcase = runtime.showcase_enabled()
    if showcase:
        st.info(
            "🧪 **Showcase mode** — live training is disabled on this deployment; "
            "browse the recorded runs below. Planning is live from the Live Run page."
        )
    st.html(_hero_html())
    c1, c2 = st.columns(2)
    c1.button(
        "▶️ Run a demo",
        width="stretch",
        type="primary",
        on_click=_nav_to,
        args=("🛰️ Live Run",),
    )
    c2.button(
        "📊 See latest results",
        width="stretch",
        on_click=_nav_to,
        args=("📊 Results",),
    )

    st.markdown("### How it works")
    cols = st.columns(3)
    for idx, (title, body) in enumerate(_HOW_STEPS):
        with cols[idx]:
            st.html(_step_card_html(idx + 1, title, body))

    artifacts, warnings = data_mod.load_collection(showcase)
    for msg in warnings:
        st.warning(msg)
    summary = data_mod.sweep_summary(artifacts)
    if artifacts:
        st.markdown("### Latest evidence")
        st.html(_evidence_strip_html(summary))
        best = max(
            (a for a in artifacts if isinstance((a.get("eval") or {}).get("delta"), (int, float))),
            key=lambda a: a["eval"]["delta"],
            default=None,
        )
        if best:
            ev = best["eval"]
            st.markdown(
                f"Best run: `{best.get('run_id', '?')}` — {best.get('goal', '')} "
                f"(base {ev['base_avg']:.2f} → ft {ev['ft_avg']:.2f})"
            )
            components.render_delta_badge(ev["delta"])
    else:
        st.info("No run artifacts found yet — run a demo to generate evidence.")

    st.caption("Every number traces to a schema-v1 artifact in `logs/runs/`; verify with:")
    st.code(
        "grep -c '\"run_id\": \"<run_id>\"' logs/llm_calls.jsonl && cat logs/runs/<run_id>.json"
    )

    if showcase:
        st.markdown("### 📁 Recorded runs")
        report.render_showcase_runs()
