"""st.secrets → NEBIUS_API_KEY injection for Community Cloud (never raises)."""

import os

from supertaco.ui.secrets_bridge import inject_secrets


def test_injects_when_key_present(monkeypatch):
    monkeypatch.delenv("NEBIUS_API_KEY", raising=False)
    assert inject_secrets({"TOKEN_FACTORY_API_KEY": "tv-123"}) is True
    assert os.environ["NEBIUS_API_KEY"] == "tv-123"


def test_no_crash_when_secrets_missing(monkeypatch):
    monkeypatch.delenv("NEBIUS_API_KEY", raising=False)

    class _NoSecrets:
        def __getitem__(self, key):
            raise FileNotFoundError("no secrets.toml")

    assert inject_secrets(_NoSecrets()) is False
    assert inject_secrets(object()) is False  # not subscriptable at all
    assert "NEBIUS_API_KEY" not in os.environ


def test_existing_env_key_not_overwritten(monkeypatch):
    monkeypatch.setenv("NEBIUS_API_KEY", "existing")
    assert inject_secrets({"TOKEN_FACTORY_API_KEY": "tv-123"}) is False
    assert os.environ["NEBIUS_API_KEY"] == "existing"


def test_empty_key_ignored(monkeypatch):
    monkeypatch.delenv("NEBIUS_API_KEY", raising=False)
    assert inject_secrets({"TOKEN_FACTORY_API_KEY": ""}) is False
    assert "NEBIUS_API_KEY" not in os.environ
