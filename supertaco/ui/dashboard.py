"""SuperTaco Dashboard — app shell for the multi-page Midnight Ops redesign.

The shell owns everything global: page config, .env/secrets bootstrap, session
state init, the design system, and ``st.navigation`` across three pages —
Landing (front door), Live Run (split workspace), Results (sweep receipts).
Page bodies live in ``supertaco/ui/pages`` and render views from
``supertaco/ui/views`` so they stay testable outside Streamlit.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import streamlit as st
import yaml

st.set_page_config(
    page_title="SuperTaco",
    page_icon="🌮",
    layout="wide",
    initial_sidebar_state="expanded",
)

from supertaco.ui.runtime import (  # noqa: E402  (after set_page_config)  # isort: skip
    bootstrap_secrets,
    load_env,
    mask_key as _mask_key,  # noqa: F401  (re-exported for tests)
    showcase_enabled as _showcase_enabled,
)
from supertaco.ui.theme import inject_theme  # noqa: E402  # isort: skip

load_env()
bootstrap_secrets()

# Read settings from environment (matching SuperTacoSettings field names)
_SUPERTACO_ENV = os.getenv("SUPERTACO_ENV") or os.getenv("SANDBOXTUNE_ENV") or "dev"
_NEBIUS_PROJECT_ID = os.getenv("NEBIUS_PROJECT_ID", "")
_NEBIUS_API_KEY = os.getenv("NEBIUS_API_KEY", "")
_TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
_TOKEN_FACTORY_BASE_URL = os.getenv(
    "TOKEN_FACTORY_BASE_URL", "https://api.tokenfactory.nebius.com/v1"
)

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
        "live_step": 1,
        "live_run_token": None,
        "run_published_token": None,
        "run_error": None,
        "term_lines": [],
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val
    if "config" not in st.session_state:
        st.session_state["config"] = _default_config()


def reset_session2():
    """Reset all session-state variables."""
    from supertaco.ui import live_state

    live_state.clear_finished()  # a fresh session must not re-adopt old runs
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    init_session2()


# ─── Main App ─────────────────────────────────────────────────────
_PAGES: Dict[str, Any] = {}  # filled in main() to defer view imports


def main():
    init_session2()
    inject_theme()

    # In-app routing (rather than st.navigation): AppTest cannot execute
    # st.navigation pages, and routing keeps the shell — theme, session state,
    # secrets — running ahead of every page render.
    from supertaco.ui.views import landing, live_run, results

    _PAGES.clear()
    _PAGES.update(
        {
            "🏠 Landing": landing.render_landing,
            "🛰️ Live Run": live_run.render_live_run,
            "📊 Results": results.render_results,
        }
    )
    st.sidebar.markdown("### 🌮 SuperTaco")
    st.sidebar.caption("Pages")
    choice = st.sidebar.radio(
        "Pages",
        list(_PAGES),
        key="nav_page",
        label_visibility="collapsed",
    )
    st.sidebar.caption(f"env: {_SUPERTACO_ENV} · showcase: {'on' if _SHOWCASE else 'off'}")
    _PAGES[choice]()


if __name__ == "__main__":
    main()
