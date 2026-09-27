"""Real-call path with fallback: modes, circuit breaker, JSONL audit."""

import json

from supertaco.agent.llm import NemotronClient


def _client() -> NemotronClient:
    return NemotronClient(base_url="http://token-factory.invalid/v1", api_key="k", timeout=1)


def test_real_call_records_mode_and_tokens(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    client = _client()
    monkeypatch.setattr(
        NemotronClient,
        "_http_call",
        lambda self, model_id, prompt, max_tokens, temperature: ("7", 42),
    )
    text = client.classify_failure("step 1 loss NaN")
    assert text == "7"
    entry = client.call_log[-1]
    assert entry["mode"] == "real"
    assert entry["tokens"] == 42
    assert entry["error"] is None
    lines = (tmp_path / "logs" / "llm_calls.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[-1])["mode"] == "real"


def test_failed_call_falls_back_and_opens_circuit(monkeypatch):
    client = _client()
    calls = []

    def boom(self, model_id, prompt, max_tokens, temperature):
        calls.append(model_id)
        raise RuntimeError("connection refused")

    monkeypatch.setattr(NemotronClient, "_http_call", boom)
    first = client.classify_failure("loss is NaN")
    assert first.strip() == "NAN_LOSS"  # heuristic simulation
    assert client.call_log[-1]["mode"] == "fallback"
    assert "connection refused" in client.call_log[-1]["error"]
    assert client._circuit_open is True

    second = client.classify_failure("CUDA out of memory")
    assert second.strip() == "OOM"
    assert client.call_log[-1]["mode"] == "fallback"
    assert "circuit_open" in client.call_log[-1]["error"]
    assert len(calls) == 1  # second call skipped HTTP


def test_jsonl_write_failure_never_kills_run(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    client = _client()
    monkeypatch.setattr(
        NemotronClient,
        "_http_call",
        lambda self, model_id, prompt, max_tokens, temperature: ("1", 1),
    )
    (tmp_path / "logs").write_text("", encoding="utf-8")  # logs is a FILE -> mkdir fails
    text = client.classify_failure("x")
    assert text == "1"
    assert client.call_log[-1].get("jsonl_written") is False
