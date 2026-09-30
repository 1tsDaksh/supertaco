"""CLI: run executes the real loop with an injected transport (tests fake it)."""

import json

import pytest
import yaml

from supertaco import cli
from supertaco.eval.harness import RESPONSES_END, RESPONSES_START
from supertaco.loop import AttemptResult

HEALTHY_LOG = (
    "config: {'model': 'Qwen/Qwen2.5-0.5B-Instruct'}\n"
    "step 4 loss 1.9218\n"
    "step 8 loss 1.6969\n"
    "training finished\n"
)


def _responses_block(prompts):
    payload = {
        "base": {p: f"base answer to {p}" for p in prompts},
        "fine_tuned": {p: f"ft answer to {p}" for p in prompts},
    }
    return f"\n{RESPONSES_START}\n{json.dumps(payload)}\n{RESPONSES_END}"


class FakeTransport:
    def __init__(self, logs=HEALTHY_LOG, error=None):
        self.logs = logs
        self.error = error
        self.calls = 0

    def run_attempt(self, config, prompts, on_line=None):
        self.calls += 1
        text = self.logs
        if self.error is None:
            text += _responses_block(prompts)
        return AttemptResult(logs=text, artifact="adapter.zip", error=self.error)


class FakeLLM:
    """Healthy judge: base 5/10, ft 9/10 on alternating _call slots."""

    def __init__(self):
        self.call_log = []
        self._n = 0

    def classify_failure(self, logs):
        return "NAN_LOSS"

    def propose_patch(self, failure_key, config, logs):
        return {"fix": f"fix-{failure_key}", "reason": "because"}

    def _call(self, model_key, prompt, **kw):
        self.call_log.append(
            {"model_key": model_key, "mode": "real", "tokens": 1, "latency": 0.01, "error": None}
        )
        is_base = self._n % 2 == 0
        self._n += 1
        return "5/10" if is_base else "9/10"


def _write_cfg(tmp_path, **overrides):
    cfg = {"model": "qwen2.5-0.5b", "learning_rate": 2e-4, "batch_size": 8}
    cfg.update(overrides)
    path = tmp_path / "ok.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


def test_run_config_success_prints_events(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)  # patched configs land in tmp cwd
    transport = FakeTransport()
    monkeypatch.setattr(cli, "_make_transport", lambda: transport)

    cli.run_config(str(_write_cfg(tmp_path)), llm=FakeLLM())
    out = capsys.readouterr().out
    assert "[run_started]" in out
    assert "[run_succeeded]" in out
    assert "Success after 1 attempt(s)" in out
    assert transport.calls == 1


def test_missing_config_exits_cleanly(tmp_path):
    with pytest.raises(SystemExit) as exc:
        cli.run_config(str(tmp_path / "nope.yaml"))
    assert "Error loading" in str(exc.value)


def test_config_without_learning_rate_exits_cleanly(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "_make_transport", lambda: FakeTransport())
    bad = tmp_path / "bad.yaml"
    bad.write_text("batch_size: 8\n", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        cli.run_config(str(bad), llm=FakeLLM())
    assert exc.value.code == 1
    assert "learning_rate" in capsys.readouterr().out


def test_transport_failure_exits_one(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    transport = FakeTransport(logs="", error="provisioning timeout")
    monkeypatch.setattr(cli, "_make_transport", lambda: transport)
    with pytest.raises(SystemExit) as exc:
        cli.run_config(str(_write_cfg(tmp_path)), llm=FakeLLM())
    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert "[run_failed]" in out
    assert "Run failed:" in out
    assert transport.calls >= 1


def test_dry_run_flag_is_gone(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["run", "x.yaml", "--dry-run"])
    assert exc.value.code == 2
    assert "unrecognized arguments" in capsys.readouterr().err


def test_transport_kwarg_used_directly(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    transport = FakeTransport()
    cli.run_config(str(_write_cfg(tmp_path)), llm=FakeLLM(), transport=transport)
    assert transport.calls == 1
    assert "Success after 1 attempt(s)" in capsys.readouterr().out


def test_non_mapping_config_exits_one(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "_make_transport", lambda: FakeTransport())
    bad = tmp_path / "bad.yaml"
    bad.write_text("- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        cli.run_config(str(bad), llm=FakeLLM())
    assert exc.value.code == 1
    assert "mapping" in capsys.readouterr().out
