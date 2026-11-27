"""Research tools: payload functions + thin @tool wrappers.

Same pattern as agents/tools.py (Phase 2): payloads are plain Python returning dicts (unit-tested,
reused by MCP in Phase 5). The @tool wrappers only supply the name, description and argument
schema the LLM sees: research.py binds them to the model, then runs the payload functions itself.

Every chunk/page a tool returns gets a short id and a Source record in ResearchRunContext.sources.
That dict is the ONLY place Source objects come from: the LLM can cite ids, but can't create sources.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Annotated, Any

from langchain.tools import ToolRuntime, tool

from finsight.rag.retriever import FilingRetriever
from finsight.schemas.research import Source
from finsight.tools.web_search import WebSearch


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def web_source_id(url: str) -> str:
    return "web-" + hashlib.sha1(url.encode()).hexdigest()[:8]


# Per-question caps. These are cost policy for the free tiers (Tavily: 1,000 credits/month;
# every result also costs prompt tokens), so they are constants, not settings.
FILING_RESULTS_PER_SEARCH = 5
WEB_RESULTS_PER_SEARCH = 3
MAX_FILING_SEARCHES = 3
MAX_WEB_SEARCHES = 2


@dataclass
class ResearchDeps:
    """Long-lived resources (shared across questions). None = that source is unavailable."""

    retriever: FilingRetriever | None
    web: WebSearch | None


@dataclass
class ResearchRunContext:
    """Per-question state used by the payload functions."""

    deps: ResearchDeps
    sources: dict[str, Source] = field(default_factory=dict)  # source id -> what a tool returned
    evidence: dict[str, str] = field(default_factory=dict)  # source id -> text the LLM is shown
    filing_searches: int = 0
    web_searches: int = 0
    gaps: list[str] = field(default_factory=list)


# ---- payloads -------------------------------------------------------------------------------------


def search_filings_payload(run: ResearchRunContext, query: str, ticker: str | None = None,
                           section: str | None = None) -> dict[str, Any]:
    deps = run.deps
    if deps.retriever is None:
        return {"error": "filing search unavailable (no 10-K index)", "results": []}
    if run.filing_searches >= MAX_FILING_SEARCHES:
        return {"error": f"limit of {MAX_FILING_SEARCHES} filing searches per question reached", "results": []}
    run.filing_searches += 1
    hits = deps.retriever.search(query, ticker=ticker, section=section, k=FILING_RESULTS_PER_SEARCH)
    if ticker and not hits:
        msg = f"no 10-K ingested for {ticker.upper()} (run: python -m finsight.rag.ingest {ticker.upper()})"
        if msg not in run.gaps:
            run.gaps.append(msg)
    results = []
    for h in hits:
        sid = h.chunk_id
        run.sources.setdefault(sid, Source(
            id=sid, type="filing", title=f"{h.company} 10-K FY{h.fiscal_year}, {h.section}",
            url=h.url, ticker=h.ticker, form="10-K", fiscal_year=h.fiscal_year, section=h.section,
            retrieved_at=_now(),
        ))
        run.evidence.setdefault(sid, h.text)
        results.append({"source_id": sid, "ticker": h.ticker, "section": h.section,
                        "fiscal_year": h.fiscal_year, "score": h.score, "text": h.text})
    return {"query": query, "ticker": ticker, "section": section, "results": results}


def web_search_payload(run: ResearchRunContext, query: str, recent_days: int | None = None) -> dict[str, Any]:
    deps = run.deps
    if deps.web is None:
        return {"error": "web search unavailable", "results": []}
    if run.web_searches >= MAX_WEB_SEARCHES:
        return {"error": f"limit of {MAX_WEB_SEARCHES} web searches per question reached", "results": []}
    run.web_searches += 1
    resp = deps.web.search(query, max_results=WEB_RESULTS_PER_SEARCH, recent_days=recent_days)
    if resp.error:
        if resp.error not in run.gaps:
            run.gaps.append(resp.error)
        return {"query": query, "error": resp.error, "results": []}
    results = []
    for r in resp.results:
        sid = web_source_id(r.url)
        run.sources.setdefault(sid, Source(
            id=sid, type="web", title=r.title, url=r.url, published_date=r.published_date, retrieved_at=_now(),
        ))
        run.evidence.setdefault(sid, r.snippet)
        results.append({"source_id": sid, "title": r.title, "url": r.url,
                        "published_date": r.published_date, "snippet": r.snippet})
    return {"query": query, "results": results}


# ---- LLM-facing tools -----------------------------------------------------------------------------


@tool
def search_filings(
    query: Annotated[str, "What to look for, in plain words, e.g. 'supply chain and manufacturing risks'"],
    runtime: ToolRuntime[ResearchRunContext],
    ticker: Annotated[str | None, "Company ticker to restrict to, e.g. 'NVDA'. Strongly recommended."] = None,
    section: Annotated[str | None, "Optional 10-K section: '1' Business, '1A' Risk Factors, '7' MD&A, '7A' Market Risk"] = None,
) -> dict:
    """Search the companies' latest annual reports (SEC 10-K filings) for relevant passages.

    Use for durable facts: business model, products, competition, risk factors, strategy, and
    management's discussion of results. Returns passages with source_ids to cite.
    """
    return search_filings_payload(runtime.context, query, ticker, section)


@tool
def web_search(
    query: Annotated[str, "Web search query, e.g. 'Tesla news'"],
    runtime: ToolRuntime[ResearchRunContext],
    recent_days: Annotated[int | None, "Only results from the last N days (use for news, e.g. 30)"] = None,
) -> dict:
    """Search the web for recent news and events not found in annual reports.

    Use for: latest news, recent announcements, events after the last 10-K. Returns results with
    source_ids to cite. Limited to 2 searches per question, so make each query specific.
    """
    return web_search_payload(runtime.context, query, recent_days)


RESEARCH_TOOLS = [search_filings, web_search]
PAYLOADS = {"search_filings": search_filings_payload, "web_search": web_search_payload}
