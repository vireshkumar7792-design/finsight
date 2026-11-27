"""Research tools: 10-K search and Tavily web search. Every result is recorded as a Source."""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

from pydantic import BaseModel, Field

from finsight.rag.ingest import LocalEmbedder, open_collection
from finsight.rag.retriever import search_filings
from finsight.schemas import Source

# Per-question limits: they protect the free tiers (Tavily: 1,000 credits/month) and the prompt size.
MAX_FILING_SEARCHES = 3
MAX_WEB_SEARCHES = 2
FILING_RESULTS_PER_SEARCH = 5
WEB_RESULTS_PER_SEARCH = 3
SNIPPET_CHARS = 600
EXCLUDED_DOMAINS = ["tavily.com"]  # Tavily's own profile pages are summaries, not reporting

# =========================
# Where research comes from
# =========================


@dataclass
class ResearchSources:
    collection: object | None  # Chroma collection of 10-K chunks (None = no filings)
    embedder: object | None
    tavily_api_key: str | None
    tavily_cache_dir: str


def open_research_sources(settings) -> ResearchSources:
    embedder = LocalEmbedder(settings.embedding_model)
    collection = open_collection(settings.chroma_path, embedder.name)
    api_key = settings.tavily_api_key.get_secret_value() if settings.tavily_api_key else None
    return ResearchSources(collection, embedder, api_key, settings.tavily_cache_dir)


# =========================
# Source registry (one per question)
# =========================


@dataclass
class FoundSources:
    """Everything the tools returned for one question. The LLM may only cite ids found here."""

    sources: dict[str, Source] = field(default_factory=dict)  # id -> Source
    texts: dict[str, str] = field(default_factory=dict)  # id -> text shown to the writer LLM
    gaps: list[str] = field(default_factory=list)  # what we could not search, and why

    def add(self, source: Source, text: str) -> None:
        self.sources.setdefault(source.id, source)
        self.texts.setdefault(source.id, text)

    def add_gap(self, gap: str) -> None:
        if gap not in self.gaps:
            self.gaps.append(gap)


# =========================
# Tool descriptions for the LLM
# =========================
# The LLM only chooses the searches. Code runs them, so it can enforce the limits above.


class SearchFilings(BaseModel):
    """Search the companies' latest annual reports (10-K) for business, competition, risks and strategy."""

    query: str = Field(description="What to look for, e.g. 'supply chain risks'")
    ticker: str | None = Field(default=None, description="Company ticker, e.g. 'NVDA'. Always give it.")
    section: str | None = Field(default=None, description="'1' Business, '1A' Risk Factors, '7' MD&A")


class WebSearch(BaseModel):
    """Search the web for recent news and events that are not in the annual reports."""

    query: str = Field(description="Use the company NAME, e.g. 'Tesla news' (tickers are ambiguous)")
    recent_days: int | None = Field(default=None, description="Only results from the last N days, e.g. 30")


# =========================
# Running the searches
# =========================


def run_filing_search(research: ResearchSources, found: FoundSources, query: str,
                      ticker: str | None = None, section: str | None = None) -> int:
    """Search the 10-Ks and record each chunk as a Source. Returns the number of results."""
    if research.collection is None:
        found.add_gap("filing search unavailable (no 10-K index)")
        return 0
    hits = search_filings(research.collection, research.embedder, query, ticker, section,
                          k=FILING_RESULTS_PER_SEARCH)
    if ticker and not hits:
        found.add_gap(f"no 10-K ingested for {ticker.upper()} (run: finsight ingest {ticker.upper()})")
    for hit in hits:
        title = f"{hit['company']} 10-K FY{hit['fiscal_year']}, {hit['section']}"
        found.add(Source(id=hit["id"], type="filing", title=title, url=hit["url"], section=hit["section"]), hit["text"])
    return len(hits)


def run_web_search(research: ResearchSources, found: FoundSources, query: str,
                   recent_days: int | None = None) -> int:
    """Search the web and record each page as a Source. Returns the number of results."""
    results, error = tavily_search(query, research.tavily_api_key, research.tavily_cache_dir, recent_days)
    if error:
        found.add_gap(error)
        return 0
    for result in results:
        source_id = "web-" + hashlib.sha1(result["url"].encode()).hexdigest()[:8]
        source = Source(id=source_id, type="web", title=result["title"], url=result["url"],
                        published_date=result["published_date"])
        found.add(source, result["snippet"])
    return len(results)


# =========================
# Tavily with a disk cache
# =========================


def tavily_search(query: str, api_key: str | None, cache_dir: str, recent_days: int | None = None):
    """Returns (results, error). Never raises: a failed search becomes a gap, not a crash."""
    params = {"max_results": WEB_RESULTS_PER_SEARCH, "search_depth": "basic", "exclude_domains": EXCLUDED_DOMAINS}
    if recent_days:
        params["topic"] = "news"  # news results come with a published date
        params["start_date"] = (date.today() - timedelta(days=recent_days)).isoformat()

    cache_key = json.dumps({"q": query.strip().lower(), **params}, sort_keys=True)
    path = Path(cache_dir) / f"{hashlib.sha256(cache_key.encode()).hexdigest()[:24]}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))["results"], None
    if not api_key:
        return [], "TAVILY_API_KEY is not set; web search skipped"

    try:
        raw = call_tavily(api_key, query, params)
    except Exception as error:  # quota used up, rate limit, bad key, network...
        return [], f"web search unavailable ({type(error).__name__}: {str(error)[:120]})"

    results = []
    for item in raw.get("results", []):
        url = item.get("url") or ""
        if not url or "tavily.com" in url:
            continue
        if recent_days and not item.get("published_date"):
            continue  # an undated page can't support a claim about recent news
        results.append({
            "title": (item.get("title") or url).strip(),
            "url": url,
            "snippet": " ".join((item.get("content") or "").split())[:SNIPPET_CHARS],
            "published_date": item.get("published_date"),
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"query": query, "params": params, "results": results}, indent=1), encoding="utf-8")
    return results, None


def call_tavily(api_key: str, query: str, params: dict) -> dict:
    """The only network call for web search (tests replace this function)."""
    from tavily import TavilyClient

    return TavilyClient(api_key=api_key).search(query, **params)
