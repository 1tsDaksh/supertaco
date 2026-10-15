import pytest

import supertaco.agent.llm as llm_module
import supertaco.eval.nebius_backend as nb_module


@pytest.fixture(autouse=True)
def _isolate_audit_logs(monkeypatch, tmp_path):
    monkeypatch.setattr(llm_module, "_logs_dir", lambda: tmp_path / "logs")
    monkeypatch.setattr(nb_module, "_logs_dir", lambda: tmp_path / "logs")
