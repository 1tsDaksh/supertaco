"""Colab adapter: lifecycle order, failure handling, auth/setup errors."""

import pytest

from supertaco.gpu.colab import ColabError, run_training

FAKE_CLI = """
import os, pathlib, sys

args = sys.argv[1:]
cmd = args[0] if args else ""
log = os.environ.get("FAKE_COLAB_LOG")
if log:
    with open(log, "a", encoding="utf-8") as fh:
        fh.write(" ".join(args) + "\\n")

if cmd == "new" and os.environ.get("FAKE_COLAB_AUTH"):
    print("To authorize colab-cli, visit this URL: https://example.invalid/auth")
    print("Enter the authorization code: ")
    sys.exit(1)

if cmd == "new":
    print("session assigned")

if cmd == "exec":
    print("step 4 loss 2.1000")
    if os.environ.get("FAKE_COLAB_FAIL") == "exec":
        print("boom: RuntimeError in cell")
        sys.exit(3)
    print("step 8 loss 1.5000")
    print("training finished")

if cmd == "download":
    if os.environ.get("FAKE_COLAB_FAIL") == "download":
        print("download failed: no such file")
        sys.exit(4)
    pathlib.Path(args[-1]).write_bytes(b"PK\\x03\\x04fakezip")
    print("downloaded")

if cmd == "stop":
    print("session terminated")

sys.exit(0)
"""


@pytest.fixture
def fake_cli(tmp_path, monkeypatch):
    script = tmp_path / "fake_colab.py"
    script.write_text(FAKE_CLI, encoding="utf-8")
    import sys

    log = tmp_path / "calls.log"
    monkeypatch.setenv("FAKE_COLAB_LOG", str(log))
    monkeypatch.delenv("FAKE_COLAB_FAIL", raising=False)
    monkeypatch.delenv("FAKE_COLAB_AUTH", raising=False)
    return [sys.executable, str(script)], log


def test_success_downloads_artifact_and_orders_calls(fake_cli, tmp_path):
    cli, log = fake_cli
    script = tmp_path / "train.py"
    script.write_text("print('hi')", encoding="utf-8")
    seen = []

    outcome = run_training(
        script,
        cli=cli,
        output_dir=tmp_path / "out",
        on_line=seen.append,
        provision_deadline_s=30,
        exec_deadline_s=30,
        download_deadline_s=30,
    )

    assert outcome.success and outcome.error is None
    assert outcome.artifact is not None and outcome.artifact.exists()
    assert outcome.artifact.read_bytes().startswith(b"PK")
    calls = [ln.split()[0] for ln in log.read_text(encoding="utf-8").splitlines()]
    assert calls == ["new", "exec", "download", "stop"]
    assert any("step 4 loss 2.1000" == ln for ln in seen)


def test_exec_failure_reports_error_and_still_stops(fake_cli, tmp_path, monkeypatch):
    cli, log = fake_cli
    monkeypatch.setenv("FAKE_COLAB_FAIL", "exec")
    script = tmp_path / "train.py"
    script.write_text("print('hi')", encoding="utf-8")

    outcome = run_training(
        script,
        cli=cli,
        output_dir=tmp_path / "out",
        provision_deadline_s=30,
        exec_deadline_s=30,
        download_deadline_s=30,
    )

    assert not outcome.success and outcome.artifact is None
    assert "boom" in outcome.error
    assert outcome.logs
    assert any("boom: RuntimeError in cell" in ln for ln in outcome.logs)
    assert any("step 4 loss 2.1000" in ln for ln in outcome.logs)
    calls = [ln.split()[0] for ln in log.read_text(encoding="utf-8").splitlines()]
    assert calls == ["new", "exec", "stop"]


def test_download_failure_reports_error_and_still_stops(fake_cli, tmp_path, monkeypatch):
    cli, log = fake_cli
    monkeypatch.setenv("FAKE_COLAB_FAIL", "download")
    script = tmp_path / "train.py"
    script.write_text("print('hi')", encoding="utf-8")

    outcome = run_training(
        script,
        cli=cli,
        output_dir=tmp_path / "out",
        provision_deadline_s=30,
        exec_deadline_s=30,
        download_deadline_s=30,
    )

    assert not outcome.success and outcome.artifact is None
    assert "download" in outcome.error
    assert outcome.logs
    assert any("download failed: no such file" in ln for ln in outcome.logs)
    assert any("training finished" in ln for ln in outcome.logs)
    calls = [ln.split()[0] for ln in log.read_text(encoding="utf-8").splitlines()]
    assert calls == ["new", "exec", "download", "stop"]


def test_auth_prompt_raises_actionable_error(fake_cli, tmp_path, monkeypatch):
    cli, _log = fake_cli
    monkeypatch.setenv("FAKE_COLAB_AUTH", "1")
    script = tmp_path / "train.py"
    script.write_text("print('hi')", encoding="utf-8")

    with pytest.raises(ColabError, match="authenticated"):
        run_training(
            script,
            cli=cli,
            output_dir=tmp_path / "out",
            provision_deadline_s=30,
            exec_deadline_s=30,
            download_deadline_s=30,
        )


def test_missing_cli_gives_install_hint(tmp_path):
    script = tmp_path / "train.py"
    script.write_text("print('hi')", encoding="utf-8")

    with pytest.raises(ColabError, match="not found"):
        run_training(
            script,
            cli=str(tmp_path / "nope" / "colab.exe"),
            output_dir=tmp_path / "out",
        )


def test_missing_script_raises(tmp_path, fake_cli):
    cli, _log = fake_cli

    with pytest.raises(ColabError, match="not found"):
        run_training(
            tmp_path / "missing.py",
            cli=cli,
            output_dir=tmp_path / "out",
            provision_deadline_s=30,
        )


def test_outcome_carries_collected_logs(fake_cli, tmp_path):
    cli, _log = fake_cli
    script = tmp_path / "t.py"
    script.write_text("print('x')", encoding="utf-8")
    outcome = run_training(
        script,
        cli=cli,
        output_dir=tmp_path / "out",
        provision_deadline_s=30,
        exec_deadline_s=30,
        download_deadline_s=30,
    )
    assert outcome.success is True
    assert any("step 4 loss" in ln for ln in outcome.logs)
    assert any("session assigned" in ln for ln in outcome.logs)
    assert any("downloaded" in ln for ln in outcome.logs)
