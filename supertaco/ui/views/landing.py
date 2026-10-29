"""Landing page — the product's front door.

A cold visitor should leave this page *informed*: what SuperTaco is, exactly how
a run works end to end, what it can do, what it runs on, and proof it works
(live numbers from run artifacts, never fabricated). Navigation buttons are a
convenience, not the content. No judge-bait — a real product page.
"""

from __future__ import annotations

import streamlit as st

from supertaco.ui import components, report, runtime
from supertaco.ui import data as data_mod
from supertaco.ui.theme import TOKENS, esc

_PITCH = (
    "SuperTaco is an AI-supervised fine-tuning sandbox. You describe the behavior "
    "you want in plain English — “JSON-only extractor”, “kids' science explainer” — "
    "and an agent pipeline plans the run, trains on a free Colab T4, heals its own "
    "failures, and proves the before/after with a judge LLM. Every decision is "
    "receipted: configs, attempts, scores and token spend land in auditable "
    "artifacts instead of a black box."
)

# (stage, title, what happens)
_PIPELINE = (
    (
        "1 · Plan",
        "Nemotron (via Nebius Token Factory) drafts the run: model, LoRA "
        "hyperparameters, dataset plan and eval prompts. Plans are labeled "
        "<strong>real</strong> or <strong>fallback</strong> — shipped defaults "
        "are never passed off as AI plans.",
    ),
    (
        "2 · Materialize",
        "The dataset becomes real rows: Hugging Face datasets are honored "
        "exactly, or the LLM generates synthetic rows, with an alpaca slice as "
        "the last resort. The eval prompt suite is sized to the judge harness.",
    ),
    (
        "3 · Train",
        "A LoRA fine-tune runs on a free Colab T4 through the colab-cli "
        "transport. Loss streams live to the dashboard terminal while steps land.",
    ),
    (
        "4 · Heal",
        "On failure, Nemotron classifies the logs (cost-routed nano → super → "
        "ultra), a deterministic playbook applies the patch, and Tavily "
        "citations back the reasoning. The loop relaunches until healthy or the "
        "retry budget is exhausted — failures are data, not dead ends.",
    ),
    (
        "5 · Judge",
        "Base vs fine-tuned are scored per eval prompt by the judge LLM — as a "
        "Nebius Serverless Job or direct API. Unparseable replies and HTTP "
        "failures are flagged per cell, never silently averaged away.",
    ),
    (
        "6 · Receipt",
        "Each run writes a schema-v1 artifact (config, attempts, judge scores, "
        "token totals) plus a per-run audit trail in `logs/llm_calls.jsonl`. "
        "Judge regressions (>10% below base) are flagged on the spot.",
    ),
)

# (title, body) — what the product can do beyond one happy-path run
_CAPABILITIES = (
    (
        "Self-healing training",
        "OOM, NaN loss, connection drops and unknown crashes are classified, "
        "patched from a deterministic playbook and retried — unattended.",
    ),
    (
        "Feedback loop",
        "“Be more concise” becomes a revised plan and a re-run, with the "
        "before/after fine-tuned delta shown only when both judges were real.",
    ),
    (
        "Transparent planning",
        "Every plan carries a real/fallback badge; every config change is "
        "diffed in the attempt ledger with before → after values.",
    ),
    (
        "Token audit",
        "Classify / judge / patch token totals per run and per sweep — metered "
        "spend reported honestly, no fabricated dollar figures.",
    ),
    (
        "Receipts you can verify",
        "Any number on the Results page is one `grep` away from the audit log "
        "and one `cat` away from its full artifact.",
    ),
    (
        "Showcase mode",
        "A hosted demo can disable training server-side while keeping planning "
        "live and recorded runs browsable — safe to put on the internet.",
    ),
)

# (component, what it is) — the stack, for people who read footers
_STACK = (
    ("Models", "Qwen2.5-0.5B / 1.5B-Instruct (T4-viable whitelist), LoRA adapters"),
    ("Training", "Free Colab T4 via colab-cli transport, streamed loss"),
    ("Planner / classifier / judge", "Nemotron via Nebius Token Factory, cost-routed"),
    ("Judge backend", "Nebius Serverless Jobs, with direct-API fallback"),
    ("Citations", "Tavily search, surfaced next to every patch proposal"),
    ("Artifacts", "Schema-v1 JSON in logs/runs + append-only llm_calls.jsonl"),
    ("Dashboard", "Streamlit multi-page app (this UI), 350+ test offline suite"),
)


def _hero_html() -> str:
    return f"""
<div class="stx-hero">
  <div style="font-size:34px;font-weight:700;color:{TOKENS['text_bright']};
       font-family:{TOKENS['sans']};letter-spacing:-.02em">🌮 SuperTaco</div>
  <div style="color:{TOKENS['text']};margin:10px auto 6px auto;max-width:680px;font-size:15px;
       line-height:1.6">{esc(_PITCH)}</div>
</div>
"""


def _card_html(
    heading: str, body: str, accent: str | None = None, *, body_html: bool = False
) -> str:
    color = accent or TOKENS["text_bright"]
    body_markup = body if body_html else esc(body)
    return f"""
<div class="stx-card" style="height:100%">
  <div style="color:{color};font-weight:600;margin-bottom:6px">{esc(heading)}</div>
  <div style="color:{TOKENS['text']};font-size:13px;line-height:1.55">{body_markup}</div>
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


def _stack_table_html() -> str:
    rows = "".join(
        f"<tr><td style='color:{TOKENS['text_dim']};white-space:nowrap;padding:4px 14px 4px 0;'"
        f">{esc(k)}</td>"
        f"<td style='color:{TOKENS['text']};padding:4px 0'>{esc(v)}</td></tr>"
        for k, v in _STACK
    )
    return (
        f"<table style='border-collapse:collapse;margin:4px 0 8px 0'>{rows}</table>"
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
    st.caption("One run, six stages — from a sentence to a receipted before/after score.")
    cols = st.columns(3)
    for idx, (stage, body) in enumerate(_PIPELINE):
        with cols[idx % 3]:
            st.html(_card_html(stage, body, accent=TOKENS["accent"], body_html=True))

    st.markdown("### What it can do")
    cols = st.columns(3)
    for idx, (title, body) in enumerate(_CAPABILITIES):
        with cols[idx % 3]:
            st.html(_card_html(title, body))

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

    st.markdown("### Under the hood")
    st.html(_stack_table_html())

    st.caption("Every number traces to a schema-v1 artifact in `logs/runs/`; verify with:")
    st.code(
        "grep -c '\"run_id\": \"<run_id>\"' logs/llm_calls.jsonl && cat logs/runs/<run_id>.json"
    )

    if showcase:
        st.markdown("### 📁 Recorded runs")
        report.render_showcase_runs()
