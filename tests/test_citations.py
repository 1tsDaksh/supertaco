"""CitationClient: parsing, labeled degradation, audit seam, settings wiring."""

import json

from pydantic import SecretStr

from supertaco.agent.citations import Citation, CitationClient, client_from_settings

RAW = [
    {
        "title": "HF Trainer docs",
        "url": "https://huggingface.co/docs/transformers",
        "content": "gradient checkpointing " + "x" * 500,
    },
    {"title": "PyTorch OOM", "url": "https://pytorch.org/docs", "content": "out of memory"},
    {"title": "no url", "content": "must be dropped"},
]


def test_search_parses_results_truncates_content_and_drops_urlless():
    seen = {}

    def fake(query, max_results=2):
        seen["query"] = query
        seen["max_results"] = max_results
        return RAW

    client = CitationClient("tvly-test", search=fake)
    out = client.search_citations("OOM: CUDA OOM in traceback pytorch LoRA fine-tuning")
    assert seen["query"].startswith("OOM:")
    assert seen["max_results"] == 2
    assert [c.title for c in out] == ["HF Trainer docs", "PyTorch OOM"]
    assert len(out[0].content) == 300
    assert out[0].url == "https://huggingface.co/docs/transformers"
    assert isinstance(out[0], Citation)


def test_raising_search_returns_empty_and_fallback_audit_entry():
    call_log = []

    def boom(query, max_results=2):
        raise RuntimeError("tavily 429")

    client = CitationClient("tvly-test", search=boom, call_log=call_log)
    out = client.search_citations("q")
    assert out == []
    assert len(call_log) == 1
    entry = call_log[0]
    assert entry["model_key"] == "tavily"
    assert entry["model_id"] == "tavily-search"
    assert entry["mode"] == "fallback"
    assert entry["tokens"] == 0
    assert "RuntimeError" in entry["error"]
    assert entry["jsonl_written"] is True


def test_real_search_audits_real_mode_with_titles_preview():
    call_log = []
    client = CitationClient("tvly-test", search=lambda q, max_results=2: RAW, call_log=call_log)
    client.search_citations("some query")
    entry = call_log[0]
    assert entry["mode"] == "real"
    assert entry["error"] is None
    assert entry["response_preview"] == "HF Trainer docs; PyTorch OOM"


def test_audit_entry_appended_to_jsonl_via_seam():
    import supertaco.agent.citations as cit

    client = CitationClient("tvly-test", search=lambda q, max_results=2: RAW, call_log=[])
    client.search_citations("q")
    lines = (cit._logs_dir() / "llm_calls.jsonl").read_text(encoding="utf-8").splitlines()
    entry = json.loads(lines[-1])
    assert entry["model_key"] == "tavily"
    assert entry["mode"] == "real"


def test_client_from_settings_returns_none_when_key_empty(monkeypatch):
    from supertaco.settings import settings

    monkeypatch.setattr(settings, "tavily_api_key", SecretStr(""))
    # imported by name -> original function, not conftest's module-attr stub
    assert client_from_settings([]) is None


def test_client_from_settings_builds_client_with_key(monkeypatch):
    from supertaco.settings import settings

    monkeypatch.setattr(settings, "tavily_api_key", SecretStr("tvly-x"))
    out = client_from_settings([])
    assert isinstance(out, CitationClient)
    assert out.call_log is not None


def test_jsonl_write_failure_never_raises_and_flags_entry(monkeypatch, tmp_path):
    import supertaco.agent.citations as cit

    (tmp_path / "logs").write_text("", encoding="utf-8")  # logs is a FILE -> mkdir fails
    monkeypatch.setattr(cit, "_logs_dir", lambda: tmp_path / "logs")
    call_log = []
    client = CitationClient("tvly-test", search=lambda q, max_results=2: RAW, call_log=call_log)
    out = client.search_citations("q")
    assert [c.title for c in out] == ["HF Trainer docs", "PyTorch OOM"]
    assert call_log[-1]["jsonl_written"] is False


def test_max_results_caps_chatty_search_at_two():
    def chatty(query, max_results=2):
        return [
            {"title": f"doc {i}", "url": f"https://ex.example/{i}", "content": "c"}
            for i in range(5)
        ]

    client = CitationClient("tvly-test", search=chatty)
    out = client.search_citations("q")
    assert len(out) == 2
    assert [c.title for c in out] == ["doc 0", "doc 1"]


def test_null_fields_do_not_render_none_strings():
    def odd(query, max_results=2):
        return [
            {"title": None, "url": None, "content": None},
            {"title": None, "url": "https://x", "content": None},
        ]

    client = CitationClient("tvly-test", search=odd)
    out = client.search_citations("q")
    assert len(out) == 1
    assert out[0].title == "untitled"
    assert out[0].url == "https://x"
    assert out[0].content == ""


def test_record_stamps_contextvar_run_id():
    import supertaco.agent.citations as cit
    import supertaco.runlog as runlog

    call_log = []
    client = CitationClient("tvly-test", search=lambda q, max_results=2: RAW, call_log=call_log)
    token = runlog.set_run_id("citrun77")
    try:
        client.search_citations("q")
    finally:
        runlog.reset_run_id(token)
    assert call_log[0]["run_id"] == "citrun77"
    line = (cit._logs_dir() / "llm_calls.jsonl").read_text(encoding="utf-8").strip()
    assert json.loads(line)["run_id"] == "citrun77"


def test_record_run_id_null_without_context_or_env(monkeypatch):
    import supertaco.agent.citations as cit

    monkeypatch.delenv("SUPERTACO_RUN_ID", raising=False)
    call_log = []
    client = CitationClient("tvly-test", search=lambda q, max_results=2: RAW, call_log=call_log)
    client.search_citations("q")
    assert call_log[0]["run_id"] is None
    line = (cit._logs_dir() / "llm_calls.jsonl").read_text(encoding="utf-8").strip()
    assert json.loads(line)["run_id"] is None
