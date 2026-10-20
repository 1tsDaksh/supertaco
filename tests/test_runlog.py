"""Run identity resolution + per-run artifact writes (hermetic via conftest seam)."""

import json

import pytest

import supertaco.runlog as runlog


def test_current_run_id_none_by_default(monkeypatch):
    monkeypatch.delenv("SUPERTACO_RUN_ID", raising=False)
    assert runlog.current_run_id() is None


def test_current_run_id_reads_env(monkeypatch):
    monkeypatch.setenv("SUPERTACO_RUN_ID", "envrun12")
    assert runlog.current_run_id() == "envrun12"


def test_contextvar_beats_env_and_resets(monkeypatch):
    monkeypatch.setenv("SUPERTACO_RUN_ID", "envrun12")
    token = runlog.set_run_id("ctxrun34")
    try:
        assert runlog.current_run_id() == "ctxrun34"
        assert runlog.resolve_run_id() == "ctxrun34"
    finally:
        runlog.reset_run_id(token)
    assert runlog.current_run_id() == "envrun12"


def test_resolve_priority_and_generated_format(monkeypatch):
    monkeypatch.delenv("SUPERTACO_RUN_ID", raising=False)
    assert runlog.resolve_run_id("explicit1") == "explicit1"
    generated = runlog.resolve_run_id()
    assert len(generated) == 8
    int(generated, 16)  # hex only


def test_write_run_artifact_writes_json_through_seam(tmp_path):
    path = runlog.write_run_artifact("abc123ef", {"schema": 1, "success": True})
    assert path == tmp_path / "logs" / "runs" / "abc123ef.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data == {"schema": 1, "success": True}


def test_write_run_artifact_oserror_warns_and_never_raises(monkeypatch, tmp_path):
    p = tmp_path / "logs"
    p.write_text("x", encoding="utf-8")  # logs is a FILE -> mkdir fails
    monkeypatch.setattr(runlog, "_logs_dir", lambda: p)
    with pytest.warns(UserWarning, match="run artifact write failed"):
        path = runlog.write_run_artifact("deadbeef", {"schema": 1})
    assert path == p / "runs" / "deadbeef.json"
