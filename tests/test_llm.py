"""Real-call path with fallback: modes, circuit breaker, JSONL audit."""

import json

import pytest

from supertaco.agent import llm as llm_module
from supertaco.agent.llm import NemotronClient
from supertaco.data import generate_dataset


@pytest.fixture(autouse=True)
def _isolate_logs(monkeypatch, tmp_path):
    """Never write the audit log to the real repo logs/ during tests."""
    monkeypatch.setattr(llm_module, "_logs_dir", lambda: tmp_path / "logs")


def _client() -> NemotronClient:
    return NemotronClient(base_url="http://token-factory.invalid/v1", api_key="k", timeout=1)


class _Clock:
    """Deterministic stand-in for the llm module's time source."""

    def __init__(self) -> None:
        self.now = 1000.0

    def monotonic(self) -> float:
        return self.now

    def time(self) -> float:
        return self.now


class _StatusError(Exception):
    """Stand-in for an openai APIStatusError that carries an HTTP status."""

    def __init__(self, status_code: int, message: str = "http error") -> None:
        super().__init__(message)
        self.status_code = status_code


def _scripted_http(*results):
    """Build a _http_call replacement that replays results (raise or return)."""
    queue = list(results)
    calls: list[int] = []

    def fake(self, model_id, prompt, max_tokens, temperature, timeout):
        calls.append(1)
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    return fake, calls


def test_logs_dir_defaults_to_repo_root(monkeypatch):
    monkeypatch.undo()  # drop the autouse seam patch to see the real default
    from supertaco.paths import repo_root

    assert llm_module._logs_dir() == repo_root() / "logs"


def test_real_call_records_mode_and_tokens(monkeypatch, tmp_path):
    client = _client()
    seen = {}

    def fake(self, model_id, prompt, max_tokens, temperature, timeout):
        seen["timeout"] = timeout
        return "7", 42

    monkeypatch.setattr(NemotronClient, "_http_call", fake)
    text = client.classify_failure("step 1 loss NaN")
    assert text == "7"
    assert seen["timeout"] == 1  # no per-call override -> client default
    entry = client.call_log[-1]
    assert entry["mode"] == "real"
    assert entry["tokens"] == 42
    assert entry["error"] is None
    assert entry["jsonl_written"] is True
    lines = (tmp_path / "logs" / "llm_calls.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[-1])["mode"] == "real"


def test_transient_failure_falls_back_without_opening_breaker(monkeypatch):
    client = _client()
    fake, calls = _scripted_http(TimeoutError("timed out"), ("OOM", 5))
    monkeypatch.setattr(NemotronClient, "_http_call", fake)

    first = client.classify_failure("loss is NaN")
    assert first.strip() == "NAN_LOSS"  # heuristic simulation
    assert client.call_log[-1]["mode"] == "fallback"
    assert "timed out" in client.call_log[-1]["error"]
    assert client._circuit_open is False

    second = client.classify_failure("CUDA out of memory")
    assert second.strip() == "OOM"
    assert client.call_log[-1]["mode"] == "real"
    assert client.call_log[-1]["tokens"] == 5
    assert len(calls) == 2  # one strike must not short-circuit later calls


def test_breaker_opens_after_two_consecutive_failures(monkeypatch):
    client = _client()
    fake, calls = _scripted_http(TimeoutError("timed out"), TimeoutError("timed out"), ("OOM", 5))
    monkeypatch.setattr(NemotronClient, "_http_call", fake)

    client.classify_failure("loss is NaN")
    client.classify_failure("loss is NaN")
    assert client._circuit_open is True

    third = client.classify_failure("CUDA out of memory")
    assert third.strip() == "OOM"
    assert client.call_log[-1]["mode"] == "fallback"
    assert client.call_log[-1]["tokens"] == 0
    assert "circuit_open" in client.call_log[-1]["error"]
    assert len(calls) == 2  # third call skipped HTTP


def test_success_resets_the_failure_streak(monkeypatch):
    client = _client()
    fake, calls = _scripted_http(
        TimeoutError("timed out"),
        ("7", 1),
        TimeoutError("timed out"),
        ("8", 1),
    )
    monkeypatch.setattr(NemotronClient, "_http_call", fake)

    client.classify_failure("loss is NaN")
    client.classify_failure("loss is NaN")
    client.classify_failure("loss is NaN")
    client.classify_failure("loss is NaN")
    assert len(calls) == 4  # every call still reached HTTP
    assert client._circuit_open is False


def test_breaker_recovers_after_cooldown(monkeypatch):
    clock = _Clock()
    monkeypatch.setattr(llm_module, "time", clock)
    client = _client()
    fake, calls = _scripted_http(
        TimeoutError("timed out"), TimeoutError("timed out"), ("7", 1), ("8", 1)
    )
    monkeypatch.setattr(NemotronClient, "_http_call", fake)

    client.classify_failure("loss is NaN")
    client.classify_failure("loss is NaN")
    assert client._circuit_open is True
    client.classify_failure("CUDA out of memory")
    assert len(calls) == 2  # open breaker short-circuits

    clock.now += NemotronClient.COOLDOWN_SECONDS + 1
    assert client._circuit_open is False  # half-open: probe allowed

    assert client.classify_failure("loss is NaN").strip() == "7"
    assert client.call_log[-1]["mode"] == "real"
    assert client.classify_failure("loss is NaN").strip() == "8"
    assert len(calls) == 4
    assert client._circuit_open is False


def test_failed_probe_reopens_the_breaker(monkeypatch):
    clock = _Clock()
    monkeypatch.setattr(llm_module, "time", clock)
    client = _client()
    fake, calls = _scripted_http(
        TimeoutError("timed out"), TimeoutError("timed out"), TimeoutError("timed out")
    )
    monkeypatch.setattr(NemotronClient, "_http_call", fake)

    client.classify_failure("loss is NaN")
    client.classify_failure("loss is NaN")
    clock.now += NemotronClient.COOLDOWN_SECONDS + 1
    client.classify_failure("loss is NaN")  # probe fails -> re-open
    assert client._circuit_open is True

    client.classify_failure("loss is NaN")
    assert len(calls) == 3  # only the probe reached HTTP


@pytest.mark.parametrize("status", [401, 404])
def test_non_retryable_error_trips_breaker_immediately(monkeypatch, status):
    client = _client()

    def auth_boom(self, model_id, prompt, max_tokens, temperature, timeout):
        raise _StatusError(status, "auth failed")

    monkeypatch.setattr(NemotronClient, "_http_call", auth_boom)
    client.classify_failure("loss is NaN")
    assert client._circuit_open is True


def test_jsonl_write_failure_never_kills_run(monkeypatch, tmp_path):
    client = _client()
    monkeypatch.setattr(
        NemotronClient,
        "_http_call",
        lambda self, model_id, prompt, max_tokens, temperature, timeout: ("1", 1),
    )
    (tmp_path / "logs").write_text("", encoding="utf-8")  # logs is a FILE -> mkdir fails
    text = client.classify_failure("x")
    assert text == "1"
    assert client.call_log[-1].get("jsonl_written") is False


def test_plan_goal_prompt_and_return(monkeypatch, tmp_path):
    client = _client()
    seen = {}

    def fake(self, model_id, prompt, max_tokens, temperature, timeout):
        seen["model_id"] = model_id
        seen["prompt"] = prompt
        return '{"model": "qwen2.5-0.5b"}', 10

    monkeypatch.setattr(NemotronClient, "_http_call", fake)
    text = client.plan_goal("pirate chef")
    assert '"model": "qwen2.5-0.5b"' in text
    assert seen["model_id"] == "nvidia/nemotron-3-super-120b-a12b"
    assert "pirate chef" in seen["prompt"]
    assert "eval_prompts" in seen["prompt"]
    assert client.call_log[-1]["mode"] == "real"


def test_revise_plan_prompt_carries_goal_plan_and_feedback(monkeypatch):
    client = _client()
    seen = {}

    def fake(self, model_id, prompt, max_tokens, temperature, timeout):
        seen["prompt"] = prompt
        return '{"note": "ok"}', 10

    monkeypatch.setattr(NemotronClient, "_http_call", fake)
    out = client.revise_plan("pirate chef", '{"batch_size": 4}', "be more concise")
    assert out == '{"note": "ok"}'
    assert "pirate chef" in seen["prompt"]
    assert "be more concise" in seen["prompt"]
    assert '"batch_size": 4' in seen["prompt"]
    assert "eval_prompts" in seen["prompt"] and "note" in seen["prompt"]
    assert client.call_log[-1]["model_key"] == "patch"


def test_generate_rows_prompt_and_return(monkeypatch, tmp_path):
    client = _client()
    seen = {}

    def fake(self, model_id, prompt, max_tokens, temperature, timeout):
        seen["prompt"] = prompt
        seen["max_tokens"] = max_tokens
        return '[{"instruction": "x", "input": "", "output": "y"}]', 10

    monkeypatch.setattr(NemotronClient, "_http_call", fake)
    text = client.generate_rows("goal", 5, ["old one"])
    assert json.loads(text) == [{"instruction": "x", "input": "", "output": "y"}]
    assert "exactly 5" in seen["prompt"]
    assert "old one" in seen["prompt"]
    assert seen["max_tokens"] == 8192
    assert client.call_log[-1]["model_key"] == "patch"


def test_generate_rows_passes_long_timeout(monkeypatch):
    client = _client()
    seen = {}

    def fake(self, model_id, prompt, max_tokens, temperature, timeout):
        seen["timeout"] = timeout
        return "[]", 1

    monkeypatch.setattr(NemotronClient, "_http_call", fake)
    client.generate_rows("goal", 25, [])
    assert seen["timeout"] == 180.0


def test_rows_call_falls_back_to_empty_array(monkeypatch):
    client = _client()
    fake, _calls = _scripted_http(TimeoutError("timed out"), TimeoutError("timed out"))
    monkeypatch.setattr(NemotronClient, "_http_call", fake)

    first = client.generate_rows("goal", 25, [])
    assert json.loads(first) == []
    assert client.call_log[-1]["mode"] == "fallback"
    assert client._circuit_open is False

    second = client.generate_rows("goal", 25, [])
    assert json.loads(second) == []
    assert client._circuit_open is True

    third = client.generate_rows("goal", 25, [])
    assert json.loads(third) == []
    assert "circuit_open" in client.call_log[-1]["error"]


def test_patch_call_falls_back_to_patch_json(monkeypatch):
    client = _client()
    fake, _calls = _scripted_http(TimeoutError("timed out"))
    monkeypatch.setattr(NemotronClient, "_http_call", fake)

    result = client.propose_patch("LOSS_PLATEAU", {"learning_rate": 0.001}, "loss stuck")
    assert result == {
        "fix": "lower_learning_rate",
        "reason": "model suggests LR reduction",
    }
    assert client.call_log[-1]["mode"] == "fallback"


def test_generation_completes_with_one_flaky_chunk(monkeypatch):
    client = _client()
    attempts = {"n": 0}

    def flaky(self, model_id, prompt, max_tokens, temperature, timeout):
        attempts["n"] += 1
        if attempts["n"] == 2:
            raise TimeoutError("APITimeoutError: request timed out")
        count = int(prompt.split("Write exactly ")[1].split()[0])
        rows = [
            {"instruction": f"chunk {attempts['n']} row {i}", "input": "", "output": f"answer {i}"}
            for i in range(count)
        ]
        return json.dumps(rows), 100

    monkeypatch.setattr(NemotronClient, "_http_call", flaky)
    rows = generate_dataset("pirate chef", client, target=50, chunk=25, floor=40, max_calls=6)

    assert len(rows) == 50
    assert attempts["n"] == 3  # flaky chunk skipped, later chunks still ran
    assert [entry["mode"] for entry in client.call_log] == ["real", "fallback", "real"]
