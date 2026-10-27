"""Midnight Ops design system — tokens and global CSS for the SuperTaco dashboard.

Single source of visual truth for the redesigned multi-page app. Every page calls
:func:`inject_theme` (the app shell does it once per rerun, which covers all pages).
Pure-string builders live here so tests can assert on emitted CSS/HTML without a
running Streamlit session.
"""

from __future__ import annotations

import html

import streamlit as st

# ─── Tokens ────────────────────────────────────────────────────────────────
TOKENS: dict[str, str] = {
    "bg": "#0b1220",
    "bg_card": "#131f38",
    "bg_inset": "#050b18",
    "border": "#233252",
    "text": "#cbd5e1",
    "text_bright": "#f8fafc",
    "text_dim": "#64748b",
    "accent": "#22c55e",
    "accent_soft": "#4ade80",
    "warn": "#fbbf24",
    "warn_bg": "#422006",
    "danger": "#ef4444",
    "info": "#7dd3fc",
    "mono": "ui-monospace, 'JetBrains Mono', Menlo, Consolas, monospace",
    "sans": "Inter, ui-sans-serif, system-ui, 'Segoe UI', sans-serif",
}

STATUS_COLORS = {
    "done": TOKENS["accent_soft"],
    "active": TOKENS["warn"],
    "pending": TOKENS["text_dim"],
    "failed": TOKENS["danger"],
}


def build_theme_css() -> str:
    """Global Midnight Ops overrides for Streamlit's DOM."""
    t = TOKENS
    return f"""
<style>
:root {{
  --stx-bg: {t['bg']};
  --stx-card: {t['bg_card']};
  --stx-inset: {t['bg_inset']};
  --stx-border: {t['border']};
  --stx-text: {t['text']};
  --stx-bright: {t['text_bright']};
  --stx-dim: {t['text_dim']};
  --stx-accent: {t['accent']};
  --stx-accent-soft: {t['accent_soft']};
  --stx-warn: {t['warn']};
  --stx-danger: {t['danger']};
  --stx-info: {t['info']};
  --stx-mono: {t['mono']};
  --stx-sans: {t['sans']};
}}
.stApp {{
  background: var(--stx-bg);
  color: var(--stx-text);
  font-family: var(--stx-sans);
}}
[data-testid="stSidebar"] {{
  background: #0d1526;
  border-right: 1px solid var(--stx-border);
}}
[data-testid="stSidebar"] * {{ color: var(--stx-text); }}
h1, h2, h3, h4 {{ color: var(--stx-bright) !important; font-family: var(--stx-sans); }}
p, li, label, .stMarkdown {{ color: var(--stx-text); }}
[data-testid="stMetric"] {{
  background: var(--stx-card);
  border: 1px solid var(--stx-border);
  border-radius: 8px;
  padding: 12px 14px;
}}
[data-testid="stMetricValue"] {{ color: var(--stx-bright); font-family: var(--stx-mono); }}
[data-testid="stMetricLabel"] {{ color: var(--stx-dim); }}
[data-testid="stMetricDelta"] {{ font-family: var(--stx-mono); }}
[data-testid="stDataFrame"], [data-testid="stTable"] {{
  border: 1px solid var(--stx-border);
  border-radius: 8px;
}}
.stButton > button {{
  background: var(--stx-accent);
  color: #052e16;
  border: none;
  border-radius: 6px;
  font-weight: 600;
}}
.stButton > button:hover {{ background: var(--stx-accent-soft); color: #052e16; }}
.stButton > button:disabled {{ background: #1e293b; color: var(--stx-dim); }}
.stTextArea textarea, .stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb] {{
  background: var(--stx-inset);
  color: var(--stx-text);
  border: 1px solid var(--stx-border);
  border-radius: 6px;
  font-family: var(--stx-mono);
}}
[data-testid="stExpander"] {{
  background: var(--stx-card);
  border: 1px solid var(--stx-border);
  border-radius: 8px;
}}
[data-testid="stCodeBlock"], pre {{
  background: var(--stx-inset) !important;
  border: 1px solid var(--stx-border);
  border-radius: 6px;
}}
code {{ font-family: var(--stx-mono); color: var(--stx-info); }}
[data-testid="stContainer"][class*="border"] {{
  background: var(--stx-card);
  border: 1px solid var(--stx-border) !important;
  border-radius: 10px;
  padding: 12px;
}}
/* SuperTaco custom component classes (see components.py) */
.stx-card {{
  background: var(--stx-card);
  border: 1px solid var(--stx-border);
  border-radius: 10px;
  padding: 14px;
  margin-bottom: 12px;
}}
.stx-label {{
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: .08em;
  color: var(--stx-dim);
  margin-bottom: 6px;
}}
.stx-terminal {{
  background: var(--stx-inset);
  border: 1px solid var(--stx-border);
  border-radius: 8px;
  font-family: var(--stx-mono);
  font-size: 12px;
  line-height: 1.55;
  padding: 10px 12px;
  height: 340px;
  overflow-y: auto;
  white-space: pre-wrap;
  word-break: break-word;
  color: #a5b4fc;
}}
.stx-terminal .t-dim {{ color: var(--stx-dim); }}
.stx-step {{
  flex: 1;
  text-align: center;
  padding: 8px 4px;
  border-radius: 6px;
  border: 1px solid var(--stx-border);
  background: var(--stx-card);
  font-family: var(--stx-mono);
  font-size: 12px;
}}
.stx-step.done {{ border-color: #166534; background: #052e16; color: var(--stx-accent-soft); }}
.stx-step.active {{ border-color: #854d0e; background: #422006; color: var(--stx-warn); }}
.stx-step.pending {{ color: var(--stx-dim); }}
.stx-step.failed {{ border-color: #7f1d1d; background: #450a0a; color: var(--stx-danger); }}
.stx-chip {{
  display: inline-block;
  border-radius: 999px;
  padding: 2px 12px;
  font-size: 12px;
  font-weight: 600;
  font-family: var(--stx-mono);
  border: 1px solid var(--stx-border);
}}
.stx-chip.idle {{ color: var(--stx-dim); }}
.stx-chip.running {{ color: var(--stx-warn); border-color: #854d0e; background: #42200666; }}
.stx-chip.ok {{ color: var(--stx-accent-soft); border-color: #166534; background: #052e1666; }}
.stx-chip.failed {{ color: var(--stx-danger); border-color: #7f1d1d; background: #450a0a66; }}
.stx-chip.recorded {{ color: var(--stx-info); border-color: #155e75; background: #082f4966; }}
.stx-hero {{
  background: radial-gradient(ellipse at 30% 20%, #16223f, var(--stx-bg));
  border: 1px solid var(--stx-border);
  border-radius: 14px;
  padding: 36px 28px;
  text-align: center;
  margin-bottom: 16px;
}}
</style>
"""


def inject_theme() -> None:
    """Inject the Midnight Ops stylesheet once per rerun (idempotent in-session)."""
    st.markdown(build_theme_css(), unsafe_allow_html=True)


def esc(text: object) -> str:
    """HTML-escape arbitrary text for embedding in component markup."""
    return html.escape(str(text), quote=True)
