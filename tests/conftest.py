import pytest

import supertaco.agent.citations as cit_module
import supertaco.agent.llm as llm_module
import supertaco.eval.nebius_backend as nb_module
import supertaco.runlog as rl_module
import supertaco.settings as settings_mod
from supertaco.ui import live_state


@pytest.fixture(autouse=True)
def _isolate_audit_logs(monkeypatch, tmp_path):
    monkeypatch.setattr(llm_module, "_logs_dir", lambda: tmp_path / "logs")
    monkeypatch.setattr(nb_module, "_logs_dir", lambda: tmp_path / "logs")
    monkeypatch.setattr(cit_module, "_logs_dir", lambda: tmp_path / "logs")
    monkeypatch.setattr(cit_module, "client_from_settings", lambda call_log=None: None)
    monkeypatch.setattr(rl_module, "_logs_dir", lambda: tmp_path / "logs")
    # snapshot singleton: dashboard rebuilds it when st.secrets injects a key
    monkeypatch.setattr(settings_mod, "settings", settings_mod.settings)


@pytest.fixture(autouse=True)
def _isolate_live_runs():
    """A background run from one test must never be adopted by the next session."""
    live_state.clear_all()
    yield
    live_state.clear_all()


@pytest.fixture(autouse=True)
def _no_fragment_poll(monkeypatch):
    """AppTest cannot follow ``st.fragment(run_every=...)`` timers.

    The browser polls with a fragment so the script run finishes (and stale
    widgets from the previous step are retired). Under AppTest that timer is
    inert and its merged element tree trips over unkeyed widget ids, so the
    suite exercises the equivalent plain-rerun fallback.
    """
    from supertaco.ui.views import live_run

    monkeypatch.setattr(live_run, "_POLL_WITH_FRAGMENT", False)
