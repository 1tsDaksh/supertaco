"""Tavily citation search for playbook failures (spec 2026-10-16-tavily-citations-design)."""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path
from typing import Any, Callable, Optional

from pydantic import BaseModel

from supertaco.paths import repo_root
from supertaco.runlog import current_run_id


def _logs_dir() -> Path:
    """Audit-log directory, anchored to the repo root (cwd-independent). Test seam."""
    return repo_root() / "logs"


class Citation(BaseModel):
    """One Tavily search result (top-2 per lookup)."""

    title: str
    url: str
    content: str = ""


class CitationClient:
    """Tavily search with NemotronClient-style labeled degradation.

    Never raises: any failure yields a mode="fallback" audit entry and an
    empty citation list, exactly like the judge's direct_fallback path.
    """

    def __init__(
        self,
        api_key: str,
        *,
        search: Optional[Callable[..., Any]] = None,
        call_log: Optional[list[dict]] = None,
    ) -> None:
        self.api_key = api_key
        self._search = search
        self.call_log: list[dict] = call_log if call_log is not None else []

    def _default_search(self, query: str, max_results: int) -> list[dict]:
        """Real Tavily SDK call. Imported lazily so tests never need the package."""
        from tavily import TavilyClient  # type: ignore[import-untyped]  # tavily ships no stubs

        client = TavilyClient(api_key=self.api_key)
        reply = client.search(query=query, max_results=max_results)
        return reply.get("results", []) if isinstance(reply, dict) else []

    def search_citations(self, query: str, *, max_results: int = 2) -> list[Citation]:
        """Return up to max_results citations; on any failure return []."""
        start = time.monotonic()
        try:
            if self._search is not None:
                raw = self._search(query, max_results=max_results)
            else:
                raw = self._default_search(query, max_results)
            citations = [
                Citation(
                    title=(str(item.get("title") or "")).strip() or "untitled",
                    url=(str(item.get("url") or "")).strip(),
                    content=(str(item.get("content") or ""))[:300],
                )
                for item in (raw or [])
                if isinstance(item, dict)
            ]
            citations = [c for c in citations if c.url][:max_results]
            self._record(
                query, citations, mode="real", latency=time.monotonic() - start, error=None
            )
            return citations
        except Exception as exc:
            self._record(
                query,
                [],
                mode="fallback",
                latency=time.monotonic() - start,
                error=f"{type(exc).__name__}: {exc}",
            )
            return []

    def _record(
        self,
        query: str,
        citations: list[Citation],
        *,
        mode: str,
        latency: float,
        error: Optional[str],
    ) -> None:
        """Append an audit entry to the injected call_log and logs/llm_calls.jsonl."""
        entry = {
            "model_key": "tavily",
            "model_id": "tavily-search",
            "mode": mode,
            "tokens": 0,
            "latency": round(latency, 3),
            "error": error,
            "timestamp": time.time(),
            "response_preview": "; ".join(c.title for c in citations)[:200] or query[:200],
            "run_id": current_run_id(),
        }
        self.call_log.append(entry)
        try:
            log_dir = _logs_dir()
            log_dir.mkdir(exist_ok=True)
            with open(log_dir / "llm_calls.jsonl", "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry) + "\n")
            entry["jsonl_written"] = True
        except OSError as exc:
            entry["jsonl_written"] = False
            warnings.warn(f"llm_calls.jsonl write failed: {exc}", stacklevel=2)


def client_from_settings(call_log: Optional[list[dict]] = None) -> Optional[CitationClient]:
    """Build a client from TAVILY_API_KEY, or None when the key is absent.

    None (rather than a failing client) keeps runs at zero Tavily noise when
    no key is configured. Test seam: conftest replaces this with a None-returning
    stub so the suite never builds a real client.
    """
    from supertaco.settings import settings

    key = settings.tavily_api_key.get_secret_value()
    if not key:
        return None
    return CitationClient(key, call_log=call_log)
