"""Research agent: 10-K RAG + web search -> grounded ResearchFindings.

Pipeline (2 LLM calls in the normal case):
  1. TOOL SELECTION  (agent model, tool calling)   the LLM picks searches; code caps them.
                     If it calls no tools or fails, a deterministic default plan runs instead.
  2. SYNTHESIS       (report model, strict structured output) -> ResearchDraft(insights, gaps)
                     retry once; then two-step fallback (free-text notes -> structure the notes);
                     if everything fails: findings with no insights and the error in `gaps`.
  3. POST-CHECK      (code) drop insights citing ids we never retrieved; build ResearchFindings
                     with Source objects from the registry; its validator guarantees every
                     remaining citation resolves.
Why not create_agent here? A ReAct loop would spend a 3rd call on a free-text summary nobody
reads. Expected LLM calls: 2 (best), up to 5 (worst: retry + two-step fallback).
Failures that were recovered from go to ResearchRun.errors (shown in the graph's error list);
informational notes go to ResearchRun.log.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.runtime import Runtime

from finsight.agents.research_tools import PAYLOADS, RESEARCH_TOOLS, ResearchDeps, ResearchRunContext
from finsight.agents.structured import message_text, structured_invoke
from finsight.config import Settings, get_settings
from finsight.context import PortfolioContext
from finsight.rag.embeddings import LocalEmbedder
from finsight.rag.retriever import FilingRetriever
from finsight.schemas.findings import Finding
from finsight.schemas.research import Insight, ResearchDraft, ResearchFindings
from finsight.state import FinSightState
from finsight.tools.web_search import WebSearch

MAX_EVIDENCE_CHARS = 9000  # ~2.3K tokens: keeps the synthesis call well inside 8K tokens/min
NEWS_WORDS = re.compile(r"\b(news|latest|recent|recently|today|this (week|month)|announce\w*|lately)\b", re.I)
_TICKER = re.compile(r"\b[A-Z]{2,5}(?:\.[A-Z])?\b")
_NOT_TICKERS = {"ETF", "USD", "CEO", "CFO", "AI", "US", "SEC", "IPO", "EPS", "GDP", "VAR", "HHI", "MD", "PE"}
COMPANY_NAMES = {"apple": "AAPL", "nvidia": "NVDA", "tesla": "TSLA", "microsoft": "MSFT", "amazon": "AMZN",
                 "google": "GOOGL", "alphabet": "GOOGL", "jpmorgan": "JPM", "johnson & johnson": "JNJ",
                 "coinbase": "COIN"}

TOOL_PROMPT = """You plan research for a portfolio analyst. Call the tools needed to answer the \
question, ALL IN ONE TURN (parallel tool calls), then stop.

- search_filings: durable facts from the latest 10-K (business, competition, risk factors, \
strategy, MD&A). Always pass `ticker`; pass `section` when obvious ('1A' for risks, '1' for \
business/competition, '7' for results discussion).
- web_search: recent news/events only (pass recent_days, e.g. 30). Max 2 per question. Search by COMPANY NAME, e.g. "Tesla news", not the bare ticker: tickers are ambiguous on the web.
Use 1-3 calls total. Do not answer the question yourself."""

WRITER_PROMPT = """You write grounded research notes for a portfolio analyst, using ONLY the \
evidence provided. Each evidence item has a source_id.

Rules:
- Each insight is ONE sentence stating something the cited evidence actually says.
- source_ids: copy the exact source_id(s) of the evidence supporting the claim. Never invent ids.
- Do not add facts, numbers or dates that aren't in the evidence. No buy/sell recommendations.
- confidence: high = stated directly; medium = clearly implied; low = partially supported.
- 3-8 insights, most important first. Put parts of the question the evidence can't answer in `gaps`."""


@dataclass
class ResearchRun:
    findings: ResearchFindings | None = None  # set at the end of run_research
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    llm_calls: int = 0
    dropped: list[str] = field(default_factory=list)  # insights removed by the post-check
    synthesis_mode: str = "structured"  # structured | two_step | failed | no_evidence
    errors: list[str] = field(default_factory=list)  # LLM failures (a fallback was used)
    log: list[str] = field(default_factory=list)  # informational notes


def resolve_tickers(question: str, tickers: list[str] | None, holdings: list[str] | None) -> list[str]:
    """Explicit tickers > tickers/company names in the question > the user's holdings (top 3)."""
    if tickers:
        return [t.upper() for t in tickers]
    found = [t for t in _TICKER.findall(question) if t not in _NOT_TICKERS]
    low = question.lower()
    found += [t for name, t in COMPANY_NAMES.items() if name in low]
    words = {w.strip(".,?!'s").upper() for w in question.split()}
    found += [t for t in (holdings or []) if t in words]
    found = list(dict.fromkeys(found))
    return found or list(holdings or [])[:3]


ALLOWED_ARGS = {"search_filings": {"query", "ticker", "section"}, "web_search": {"query", "recent_days"}}


def clean_args(name: str, args: dict) -> dict:
    """Free models sometimes invent arguments or send ''. Keep only known, non-empty ones."""
    out = {k: v for k, v in args.items() if k in ALLOWED_ARGS[name] and v not in (None, "")}
    if "recent_days" in out:
        try:
            out["recent_days"] = max(1, min(int(out["recent_days"]), 365))
        except (TypeError, ValueError):
            out.pop("recent_days")
    out.setdefault("query", "")
    return out


def default_plan(question: str, tickers: list[str]) -> list[tuple[str, dict]]:
    plan = [("search_filings", {"query": question, "ticker": t}) for t in tickers[:2]]
    if NEWS_WORDS.search(question) or not tickers:
        plan.append(("web_search", {"query": question, "recent_days": 30}))
    return plan


def _select_tools(llm: BaseChatModel, question: str, tickers: list[str], available: set[str] | None,
                  run: ResearchRun) -> list[tuple[str, dict]]:
    info = (f"Question: {question}\nTickers in focus: {', '.join(tickers) or 'none identified'}\n"
            f"10-Ks available for: {', '.join(sorted(available)) if available else 'unknown'}\n"
            f"Today: {date.today().isoformat()}")
    run.llm_calls += 1
    try:
        msg = llm.bind_tools(RESEARCH_TOOLS).invoke([SystemMessage(content=TOOL_PROMPT), HumanMessage(content=info)])
        calls = [(tc["name"], dict(tc["args"])) for tc in msg.tool_calls if tc["name"] in PAYLOADS]
    except Exception as e:
        run.errors.append(f"tool selection failed ({type(e).__name__}); using default plan")
        return default_plan(question, tickers)
    if not calls:
        run.log.append("model called no tools; using default plan")
        return default_plan(question, tickers)
    return calls


def _evidence(ctx: ResearchRunContext, budget: int = MAX_EVIDENCE_CHARS) -> tuple[list[dict], int]:
    items, used, skipped = [], 0, 0
    for sid, src in ctx.sources.items():
        text = ctx.evidence[sid]
        if used + len(text) > budget:
            skipped += 1
            continue
        used += len(text)
        label = src.section if src.type == "filing" else (src.published_date or "web")
        items.append({"source_id": sid, "type": src.type, "from": f"{src.title} ({label})", "text": text})
    return items, skipped


def _synthesize(llm: BaseChatModel, question: str, evidence: list[dict], run: ResearchRun) -> ResearchDraft | None:
    body = f"Question: {question}\n\nEvidence (JSON):\n{json.dumps(evidence, separators=(',', ':'))}"
    messages = [SystemMessage(content=WRITER_PROMPT), HumanMessage(content=body)]
    for attempt in range(2):  # structured, then one retry
        run.llm_calls += 1
        try:
            return structured_invoke(llm, ResearchDraft, messages)
        except Exception as e:
            run.errors.append(f"structured synthesis attempt {attempt + 1} failed: {str(e)[:120]}")
    # Two-step fallback: free-text notes with [source_id] citations, then structure the notes.
    run.synthesis_mode = "two_step"
    try:
        run.llm_calls += 1
        notes = message_text(llm.invoke(messages + [HumanMessage(
            content="Write plain-text bullet notes instead. End each bullet with its [source_id].")]))
        run.llm_calls += 1
        return structured_invoke(llm, ResearchDraft, [
            SystemMessage(content=WRITER_PROMPT),
            HumanMessage(content=f"Question: {question}\nValid source_ids: {[e['source_id'] for e in evidence]}\n\n"
                                 f"Convert these notes into the structured format:\n{notes}"),
        ])
    except Exception as e:
        run.errors.append(f"two-step synthesis failed: {str(e)[:120]}")
        run.synthesis_mode = "failed"
        return None


def post_check(draft: ResearchDraft, ctx: ResearchRunContext, run: ResearchRun) -> list[Insight]:
    """Keep only citations of retrieved sources; drop insights left with none."""
    kept = []
    for d in draft.insights:
        valid = [sid for sid in dict.fromkeys(d.source_ids) if sid in ctx.sources]
        invalid = [sid for sid in d.source_ids if sid not in ctx.sources]
        if invalid:
            run.log.append(f"removed unknown source id(s) {invalid} from: {d.claim[:80]}")
        if not valid:
            run.dropped.append(d.claim)
            continue
        kept.append(Insight(claim=d.claim, source_ids=valid, confidence=d.confidence, category=d.category))
    return kept


def run_research(question: str, deps: ResearchDeps, tool_llm: BaseChatModel, writer_llm: BaseChatModel, *,
                 tickers: list[str] | None = None, holdings: list[str] | None = None) -> ResearchRun:
    focus = resolve_tickers(question, tickers, holdings)
    ctx = ResearchRunContext(deps)
    run = ResearchRun()
    available = deps.retriever.available_tickers() if deps.retriever else None
    for t in focus:
        if available is not None and t not in available:
            ctx.gaps.append(f"no 10-K ingested for {t} (run: python -m finsight.rag.ingest {t})")

    if deps.retriever is None and deps.web is None:  # nothing to search: don't spend an LLM call
        ctx.gaps.append("no research sources configured (no 10-K index, web search disabled)")
        plan: list[tuple[str, dict]] = []
    else:
        plan = _select_tools(tool_llm, question, focus, available, run)
    for name, args in plan:
        out = PAYLOADS[name](ctx, **clean_args(name, args))
        run.tool_calls.append({"name": name, "args": args, "results": len(out.get("results", [])),
                               "error": out.get("error")})

    evidence, skipped = _evidence(ctx)
    if skipped:
        run.log.append(f"{skipped} retrieved source(s) left out to fit the prompt budget")
    insights: list[Insight] = []
    gaps = list(dict.fromkeys(ctx.gaps))
    if not evidence:
        run.synthesis_mode = "no_evidence"
        gaps.append("no relevant filing passages or web results were found")
    else:
        draft = _synthesize(writer_llm, question, evidence, run)
        if draft is None:
            gaps.append("research synthesis failed on the free model; sources were retrieved but not summarized")
        else:
            insights = post_check(draft, ctx, run)
            gaps += [g for g in draft.gaps if g not in gaps]
            if run.dropped:
                gaps.append(f"{len(run.dropped)} claim(s) removed because they cited no retrieved source")

    cited = list(dict.fromkeys(sid for ins in insights for sid in ins.source_ids))
    run.findings = ResearchFindings(
        tickers=focus, question=question, insights=insights,
        sources=[ctx.sources[sid] for sid in cited], gaps=gaps, as_of=date.today().isoformat(),
    )
    return run


# ---- graph node (Phase 3 integration) ---------------------------------------------------------------


def finding_from_research(run: ResearchRun) -> Finding:
    """Wrap ResearchFindings as a graph Finding: data holds the full cited findings."""
    f = run.findings
    lines = [f"{ins.claim} [{', '.join(ins.source_ids)}]" for ins in f.insights[:4]]
    summary = (f"{len(f.insights)} cited insight(s) from {len(f.sources)} source(s). " + " ".join(lines)
               if f.insights else "No cited insights. Gaps: " + "; ".join(f.gaps[:3]))
    return Finding(
        agent="research", summary=summary, data={"research:findings": f.model_dump()},
        tools_called=[{"name": c["name"], "args": c["args"]} for c in run.tool_calls],
    )


def make_research_node(tool_llm: BaseChatModel, writer_llm: BaseChatModel, deps_provider):
    """Graph node. deps_provider() returns ResearchDeps (lazy: loading the embedder takes seconds)."""

    def research_node(state: FinSightState, runtime: Runtime[PortfolioContext]) -> dict[str, Any]:
        holdings: list[str] = []
        if runtime.context is not None:  # focus on the user's stocks, largest first
            w = runtime.context.weights
            equities = [h.ticker for h in runtime.context.portfolio.holdings if h.asset_type.value == "equity"]
            holdings = sorted(equities, key=lambda t: -w.get(t, 0))
        try:
            deps = deps_provider()
            run = run_research(state["question"], deps, tool_llm, writer_llm, holdings=holdings)
        except Exception as e:  # never crash the graph
            return {"research_findings": None, "completed_agents": ["research"],
                    "errors": [f"research agent failed: {type(e).__name__}: {str(e)[:200]}"]}
        finding = finding_from_research(run)
        return {
            "research_findings": finding.model_dump(),
            "completed_agents": ["research"],
            "tool_calls": [{"agent": "research", "name": c["name"], "args": c["args"]} for c in run.tool_calls],
            "llm_calls": run.llm_calls,
            "errors": [f"research: {m}" for m in run.errors],
        }

    return research_node


def research_deps_from_settings(settings: Settings | None = None) -> ResearchDeps:
    """Production deps: local embedder + persistent Chroma + Tavily (free tier, disk-cached)."""
    from finsight.rag.store import open_collection  # imports chromadb (slow): only when research runs

    s = settings or get_settings()
    embedder = LocalEmbedder(s.embedding_model)
    retriever = FilingRetriever(open_collection(s.chroma_path, embedder.name), embedder)
    key = s.tavily_api_key.get_secret_value() if s.tavily_api_key else None
    return ResearchDeps(retriever=retriever, web=WebSearch(key, s.tavily_cache_dir))
