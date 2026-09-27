"""CLI smoke: runs the shared runner and prints events."""

from pathlib import Path

import pytest
import yaml

from supertaco import cli


def test_run_config_success_prints_events(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)  # configs/runs + logs land in tmp
    fixture = next(
        p
        for p in (Path(__file__).resolve().parents[1] / "configs" / "runs").glob("*NAN_LOSS*")
        if "_patched" not in p.name
    )

    class FakeLLM:
        call_log = []

        def classify_failure(self, logs):
            return "NAN_LOSS"

        def propose_patch(self, key, config, logs):
            return {"fix": "x", "reason": "y"}

    monkeypatch.setattr(cli, "_make_llm", lambda: FakeLLM())

    cli.run_config(str(fixture), dry_run=True)
    out = capsys.readouterr().out
    assert "[run_succeeded]" in out
    assert "[patch_written]" in out
    assert "Success" in out
    written = list((tmp_path / "configs" / "runs").glob("*_patched.yaml"))
    assert len(written) == 1
    patched = yaml.safe_load(written[0].read_text(encoding="utf-8"))
    assert patched["learning_rate"] == 0.01


def test_missing_config_exits_cleanly(tmp_path):
    with pytest.raises(SystemExit) as exc:
        cli.run_config(str(tmp_path / "nope.yaml"))
    assert "Error loading" in str(exc.value)


def test_config_without_learning_rate_exits_cleanly(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "_make_llm", lambda: object())
    cfg = tmp_path / "bad.yaml"
    cfg.write_text("batch_size: 8\n", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        cli.run_config(str(cfg))
    assert "learning_rate" in str(exc.value)


def test_gate1_real_job_prints_failure_and_exits_one(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)  # keep runs_dir out of the repo
    fixture = next(
        p
        for p in (Path(__file__).resolve().parents[1] / "configs" / "runs").glob("*NAN_LOSS*")
        if "_patched" not in p.name
    )
    monkeypatch.setattr(cli, "_make_llm", lambda: object())
    with pytest.raises(SystemExit) as exc:
        cli.run_config(str(fixture), dry_run=False)
    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert "[run_failed]" in out
    assert "Gate 1" in out
    assert "Run failed" in out
