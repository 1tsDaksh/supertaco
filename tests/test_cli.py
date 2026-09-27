"""CLI smoke: runs the shared runner and prints events."""

from pathlib import Path

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
