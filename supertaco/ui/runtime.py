"""Runtime environment for the SuperTaco dashboard (shared by shell and views).

Kept separate from ``dashboard.py`` so view modules can import these helpers
without importing the app entrypoint (which must stay side-effect-light and is
re-executed by Streamlit/AppTest). Showcase detection is a *function* re-evaluated
per script execution: tests flip ``SUPERTACO_SHOWCASE`` between AppTest runs.
"""

from __future__ import annotations

import os
from pathlib import Path

from streamlit import config
from streamlit.errors import StreamlitSecretNotFoundError


def load_env() -> None:
    """Load the repo-root .env if present, without overriding real env vars."""
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if not env_path.exists():
        return
    with open(env_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                key, value = key.strip(), value.strip()
                if key and key not in os.environ:
                    os.environ[key] = value


def bootstrap_secrets() -> None:
    """Community Cloud: Token Factory key via st.secrets -> supertaco.settings.

    Rebuilds the settings singleton only when a key was actually injected; local
    runs (no secrets) keep the original instance so the offline test seam stays
    intact. NOTE: tests snapshot supertaco.settings.settings (conftest does)
    before running with st.secrets keys — this assignment bypasses
    monkeypatch's undo.
    """
    import streamlit as st

    from supertaco.ui.secrets_bridge import inject_secrets

    if inject_secrets(st.secrets):
        import supertaco.settings as settings_mod

        settings_mod.settings = settings_mod.SuperTacoSettings()


def showcase_enabled() -> bool:
    """SUPERTACO_SHOWCASE from the env, else st.secrets (Community Cloud has no env UI)."""
    import streamlit as st

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


def mask_key(key: str) -> str:
    """Mask an API key for display: tvly-****abcd (short keys fully masked)."""
    if len(key) < 12:
        return "****"
    return f"{key[:5]}****{key[-4:]}"
