"""Tavily web search (free tier: 1,000 credits/month, 1 credit per basic search) with a disk cache.

- Cache: JSON file per (query, params, and today's date when `recent_days` is set). Re-running a
  question while developing costs zero credits; tomorrow's "latest news" query still refetches.
- Never raises for unavailability: a missing key, exhausted quota (Tavily 432/433), rate limit
  (429) or network error returns a WebSearchResponse with `error` set, so the research agent
  continues with filings only and records a gap.
- recent_days -> topic="news" (Tavily then includes published dates) + start_date filter, and
  UNDATED results are dropped: if we can't tell when something was published, it can't support
  "latest news" (in practice these were profile pages, one of them for the wrong company).
- tavily.com's own "knowledge" profile pages are excluded: they're summaries, not reporting.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any


EXCLUDED_DOMAINS = ("tavily.com",)


@dataclass(frozen=True)
class WebResult:
    title: str
    url: str
    snippet: str
    published_date: str | None
    score: float | None


@dataclass
class WebSearchResponse:
    query: str
    results: list[WebResult] = field(default_factory=list)
    error: str | None = None  # set when search was unavailable; results then empty
    from_cache: bool = False


class WebSearch:
    def __init__(
        self,
        api_key: str | None,
        cache_dir: str | Path = "data/cache/tavily",
        *,
        client: Any = None,  # anything with .search(query, **params) -> dict; tests pass a fake
        today: Callable[[], date] = date.today,
        snippet_chars: int = 600,
    ):
        self.api_key = (api_key or "").strip() or None
        self.cache_dir = Path(cache_dir)
        self.client = client
        self._today = today
        self.snippet_chars = snippet_chars
        self.api_calls = 0
        self.cache_hits = 0

    def _params(self, max_results: int, recent_days: int | None) -> dict[str, Any]:
        params: dict[str, Any] = {"max_results": max_results, "search_depth": "basic",  # basic = 1 credit
                                  "exclude_domains": list(EXCLUDED_DOMAINS)}
        if recent_days:
            params["topic"] = "news"
            params["start_date"] = (self._today() - timedelta(days=recent_days)).isoformat()
        return params

    def _cache_path(self, query: str, params: dict) -> Path:
        key = json.dumps({"q": query.strip().lower(), **params}, sort_keys=True)
        return self.cache_dir / f"{hashlib.sha256(key.encode()).hexdigest()[:24]}.json"

    def search(self, query: str, max_results: int = 3, recent_days: int | None = None) -> WebSearchResponse:
        params = self._params(max_results, recent_days)
        path = self._cache_path(query, params)
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            self.cache_hits += 1
            return WebSearchResponse(query, [WebResult(**r) for r in data["results"]], None, True)

        if not self.api_key:
            return WebSearchResponse(query, error="TAVILY_API_KEY is not set; web search skipped")
        if self.client is None:  # created only when a real search is needed
            from tavily import TavilyClient

            self.client = TavilyClient(api_key=self.api_key)
        try:
            raw = self.client.search(query, **params)
            self.api_calls += 1
        except Exception as e:  # quota (432/433), 429, invalid key, timeout, network
            self.api_calls += 1
            return WebSearchResponse(query, error=f"web search unavailable ({type(e).__name__}: {str(e)[:150]})")

        results = [
            WebResult(
                title=(r.get("title") or "").strip() or r.get("url", ""),
                url=r.get("url", ""),
                snippet=" ".join((r.get("content") or "").split())[: self.snippet_chars],
                published_date=r.get("published_date") or None,
                score=r.get("score"),
            )
            for r in raw.get("results", [])
            if r.get("url") and not any(d in r["url"] for d in EXCLUDED_DOMAINS)
        ]
        if recent_days:  # news: an undated page can't support "recent"
            results = [r for r in results if r.published_date]
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"query": query, "params": params, "results": [asdict(r) for r in results]},
                                   indent=1), encoding="utf-8")
        return WebSearchResponse(query, results)
