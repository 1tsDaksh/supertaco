"""Streamlit Community Cloud secrets -> process env (showcase deployments)."""

import os
from typing import Any


def inject_secrets(secrets: Any) -> bool:
    """Inject the Token Factory key from st.secrets as NEBIUS_API_KEY.

    NEBIUS_API_KEY is the env name SuperTacoSettings reads (settings.py:53-56) and
    the sole credential make_llm() uses — the same Nebius/Token Factory key.
    Returns True only when a key was newly written. Missing or malformed secrets
    never raise (local runs have no secrets file).
    """
    try:
        key = secrets["TOKEN_FACTORY_API_KEY"]
    except Exception:
        return False
    if not key or "NEBIUS_API_KEY" in os.environ:
        return False
    os.environ["NEBIUS_API_KEY"] = str(key)
    return True
