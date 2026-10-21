import pytest

import supertaco.agent.citations as cit_module
import supertaco.agent.llm as llm_module
import supertaco.eval.nebius_backend as nb_module
import supertaco.runlog as rl_module
import supertaco.settings as settings_mod


@pytest.fixture(autouse=True)
def _isolate_audit_logs(monkeypatch, tmp_path):
    monkeypatch.setattr(llm_module, "_logs_dir", lambda: tmp_path / "logs")
    monkeypatch.setattr(nb_module, "_logs_dir", lambda: tmp_path / "logs")
    monkeypatch.setattr(cit_module, "_logs_dir", lambda: tmp_path / "logs")
    monkeypatch.setattr(cit_module, "client_from_settings", lambda call_log=None: None)
    monkeypatch.setattr(rl_module, "_logs_dir", lambda: tmp_path / "logs")
    # snapshot singleton: dashboard rebuilds it when st.secrets injects a key
    monkeypatch.setattr(settings_mod, "settings", settings_mod.settings)
