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
