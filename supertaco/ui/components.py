"""Reusable Midnight Ops components for the SuperTaco dashboard.

Pure builders return HTML strings (unit-testable without Streamlit); ``render_*``
helpers emit them via :func:`streamlit.html`. The terminal pins its scroll to the
latest line on every rerun so live output is always in view, while keeping full
history scrollable (per the redesign: "auto-scrolling so it shows everything").
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import streamlit as st

from supertaco.ui.theme import STATUS_COLORS, TOKENS, esc

PHASES: tuple[str, ...] = ("plan", "data", "train", "eval", "report")

_TRAIN_EVENTS = {
    "logs_produced",
    "failure_detected",
    "classified",
    "patch_proposed",
    "patch_written",
}


def _event_type(ev: object) -> str:
    return str(getattr(ev, "type", ""))


def phase_states(events: Sequence[object]) -> dict[str, str]:
    """Map collected loop events to per-phase states: done/active/pending/failed.

    ``events`` items need only a ``.type`` attribute (Event instances in the app,
    SimpleNamespace stand-ins in showcase recordings).
    """
    types = {_event_type(e) for e in events}
    failed = "run_failed" in types
    succeeded = "run_succeeded" in types
    launched = "job_launched" in types
    trained = bool(types & _TRAIN_EVENTS)
    states = dict.fromkeys(PHASES, "pending")
    if failed:
        if not launched:
            states["plan"] = "failed"
        elif not trained:
            states.update({"plan": "done", "data": "done", "train": "failed"})
        else:
            states.update(
                {
                    "plan": "done",
                    "data": "done",
                    "train": "done",
                    "eval": "failed",
                    "report": "failed",
                }
            )
    elif succeeded:
        states.update(dict.fromkeys(PHASES, "done"))
    elif trained:
        states.update({"plan": "done", "data": "done", "train": "done", "eval": "active"})
    elif launched:
        states.update({"plan": "done", "data": "done", "train": "active"})
    return states


def build_phase_timeline_html(states: Mapping[str, str]) -> str:
    """Render the 5-step phase strip as HTML (pure, testable)."""
    marks = {"done": "✓", "active": "●", "pending": "○", "failed": "✕"}
    steps = "".join(
        f'<div class="stx-step {esc(state)}">{esc(phase)} {marks.get(state, "?")}</div>'
        for phase, state in states.items()
    )
    return f'<div style="display:flex;gap:8px;margin-bottom:12px">{steps}</div>'


def render_phase_timeline(events: Sequence[object]) -> None:
    st.html(build_phase_timeline_html(phase_states(events)))


TERMINAL_CSS = """
body { margin: 0; background: %(inset)s; }
.term {
  background: %(inset)s;
  border: 1px solid %(border)s;
  border-radius: 8px;
  font-family: %(mono)s;
  font-size: 12px;
  line-height: 1.55;
  padding: 10px 12px;
  height: %(height)spx;
  overflow-y: auto;
  white-space: pre-wrap;
  word-break: break-word;
  color: #a5b4fc;
  box-sizing: border-box;
}
""" % {
    "inset": TOKENS["bg_inset"],
    "border": TOKENS["border"],
    "mono": TOKENS["mono"],
    "height": "%(height)s",  # placeholder resolved by the builder
}


def build_terminal_html(lines: Iterable[str], height: int = 340) -> str:
    """Self-contained terminal document (for components.v1.html).

    Full history is kept ("shows everything") and the container pins its scroll
    to the latest line on every render, so live output is always in view
    without the user scrolling.
    """
    body = "\n".join(esc(line) for line in lines) or '<span class="t-dim">—</span>'
    css = TERMINAL_CSS % {"height": str(int(height))}
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8"><style>{css}</style></head>
<body><div class="term" id="term">{body}</div>
<script>
  var el = document.getElementById('term');
  el.scrollTop = el.scrollHeight;
</script></body></html>"""


def render_terminal(lines: Iterable[str], height: int = 340) -> None:
    import streamlit.components.v1 as components

    components.html(build_terminal_html(lines, height=height), height=height + 12)


def build_status_chip(state: str, label: str | None = None) -> str:
    """Small pill describing the run state: idle/running/ok/failed/recorded."""
    default_labels = {
        "idle": "idle",
        "running": "running",
        "ok": "✓ complete",
        "failed": "✕ failed",
        "recorded": "◷ recorded run",
    }
    text = label if label is not None else default_labels.get(state, state)
    color = STATUS_COLORS.get("active" if state == "running" else state, TOKENS["info"])
    if state in {"ok", "recorded", "idle"}:
        color = {
            "ok": TOKENS["accent_soft"],
            "recorded": TOKENS["info"],
            "idle": TOKENS["text_dim"],
        }[state]
    if state == "failed":
        color = TOKENS["danger"]
    return (
        f'<span class="stx-chip {esc(state)}" style="border-color:{color}55;'
        f'color:{color}">{esc(text)}</span>'
    )


def render_status_chip(state: str, label: str | None = None) -> None:
    st.html(build_status_chip(state, label))


def delta_class(delta: float) -> str:
    if delta > 0:
        return "up"
    if delta < 0:
        return "down"
    return "flat"


def build_delta_badge_html(delta: float) -> str:
    colors = {"up": TOKENS["accent_soft"], "down": TOKENS["danger"], "flat": TOKENS["warn"]}
    cls = delta_class(delta)
    sign = "+" if delta > 0 else ""
    return (
        f'<span class="stx-chip {cls}" style="border-color:{colors[cls]}66;'
        f'background:{colors[cls]}18;color:{colors[cls]};font-family:{TOKENS["mono"]}">'
        f"{sign}{delta:.2f}</span>"
    )


def render_delta_badge(delta: float) -> None:
    st.html(build_delta_badge_html(delta))
