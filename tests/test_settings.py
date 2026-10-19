"""Settings env resolution: SUPERTACO_ENV preferred, legacy SANDBOXTUNE_ENV honored."""

from supertaco.settings import SuperTacoSettings


def test_legacy_sandboxtune_env_still_resolves(monkeypatch):
    monkeypatch.delenv("SUPERTACO_ENV", raising=False)
    monkeypatch.setenv("SANDBOXTUNE_ENV", "prod")
    assert SuperTacoSettings().environment == "prod"


def test_supertaco_env_wins_over_legacy(monkeypatch):
    monkeypatch.setenv("SUPERTACO_ENV", "prod")
    monkeypatch.setenv("SANDBOXTUNE_ENV", "dev")
    assert SuperTacoSettings().environment == "prod"
