"""Research agent: 10-K search + web news -> insights that each cite a source the tools really returned.

Two LLM calls: (1) choose the searches, (2) write insights from the evidence. Code runs the
searches in between and, at the end, drops every claim that cites an unknown source id.
"""

import json
import re
from datetime import date

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.runtime import Runtime

from finsight.agents.common import ask_structured, short_error
from finsight.schemas import Insight, ResearchDraft
from finsight.state import FinSightState, RunContext
from finsight.tools.research_tools import (
    MAX_FILING_SEARCHES,
    MAX_WEB_SEARCHES,
    FoundSources,
    SearchFilings,
    WebSearch,
    open_research_sources,
    run_filing_search,
    run_web_search,
)

MAX_EVIDENCE_CHARS = 9000  # about 2,300 tokens, so the writer call stays inside 8K tokens/minute

# =========================
# Prompts
# =========================

SEARCH_PROMPT = """You plan research for a portfolio analyst. Call the search tools needed to
answer the question, ALL IN ONE TURN, then stop. Use 1-3 calls in total.
- SearchFilings: facts from the latest 10-K. Always give `ticker`. Give `section` when
  obvious: '1A' for risks, '1' for business and competition, '7' for results.
- WebSearch: recent news only (give recent_days, e.g. 30). Search by company NAME.
Do not answer the question yourself."""

WRITER_PROMPT = """You write research notes for a portfolio analyst using ONLY the evidence given.
Each evidence item has a source_id.
- Each insight is ONE sentence saying something the cited evidence actually says.
- source_ids: copy the exact source_id(s) behind the claim. Never invent ids.
- Don't add facts, numbers or dates that aren't in the evidence. No buy/sell advice.
- 3-8 insights, most important first. Put what the evidence can't answer in `gaps`."""

# =========================
# Tools
# =========================

TOOLS = [SearchFilings, WebSearch]

# =========================
# Node
# =========================


def research_node(state: FinSightState, runtime: Runtime[RunContext]) -> dict:
    context = runtime.context
    portfolio = context.portfolio
    stocks = [h.ticker for h in portfolio.portfolio.holdings if h.asset_type == "equity"]
    stocks.sort(key=lambda ticker: portfolio.weights[ticker], reverse=True)
    tickers = pick_tickers(state["question"], [], stocks)

    update = {"completed_agents": state["completed_agents"] + ["research"]}
    try:
        if context.research is None:
            context.research = open_research_sources(context.settings)  # opened only when needed
        result = run_research(state["question"], tickers, context.research, context.agent_llm, context.report_llm)
    except Exception as error:  # e.g. the 10-K index can't be opened: report without research
        update["errors"] = state["errors"] + [f"research failed: {short_error(error)}"]
        return update

    finding = {"summary": research_summary(result), "data": {"research": result["findings"]}}
    tool_calls = [{"agent": "research", **call} for call in result["tool_calls"]]
    update["findings"] = {**state["findings"], "research": finding}
    update["tool_calls"] = state["tool_calls"] + tool_calls
    update["errors"] = state["errors"] + result["errors"]
    return update


# =========================
# The research steps
# =========================


def run_research(question, tickers, research, search_llm, writer_llm) -> dict:
    """Used by the graph node and by `finsight research`."""
    found = FoundSources()
    errors = []

    searches = choose_searches(question, tickers, search_llm, errors)
    tool_calls = []
    for name, args in searches:
        if name == "SearchFilings":
            count = run_filing_search(research, found, **args)
        else:
            count = run_web_search(research, found, **args)
        tool_calls.append({"name": name, "args": args, "results": count})

    insights, writer_gaps = write_insights(question, found, writer_llm, errors)
    insights, dropped = keep_grounded(insights, found)

    gaps = found.gaps + writer_gaps
    if dropped:
        gaps.append(f"{dropped} claim(s) removed because they cited no retrieved source")
    cited_ids = []
    for insight in insights:
        for source_id in insight.source_ids:
            if source_id not in cited_ids:
                cited_ids.append(source_id)

    findings = {
        "tickers": tickers,
        "as_of": date.today().isoformat(),
        "insights": [insight.model_dump() for insight in insights],
        "sources": [found.sources[source_id].model_dump() for source_id in cited_ids],
        "gaps": gaps,
    }
    return {"findings": findings, "tool_calls": tool_calls, "errors": errors}


def choose_searches(question, tickers, search_llm, errors) -> list[tuple[str, dict]]:
    """LLM call 1 (try, retry once). Fallback: a simple default plan without the LLM."""
    info = f"Question: {question}\nTickers in focus: {', '.join(tickers) or 'none'}\nToday: {date.today()}"
    messages = [SystemMessage(content=SEARCH_PROMPT), HumanMessage(content=info)]
    searches = []
    for attempt in range(2):
        try:
            answer = search_llm.bind_tools(TOOLS).invoke(messages)
            searches = [(call["name"], clean_args(call["name"], call["args"])) for call in answer.tool_calls
                        if call["name"] in ("SearchFilings", "WebSearch")]
            break
        except Exception as error:
            errors.append(f"research search planning attempt {attempt + 1} failed: {short_error(error)}")

    if not searches:
        searches = default_searches(question, tickers)
    filing_searches = [search for search in searches if search[0] == "SearchFilings"]
    web_searches = [search for search in searches if search[0] == "WebSearch"]
    return filing_searches[:MAX_FILING_SEARCHES] + web_searches[:MAX_WEB_SEARCHES]


def write_insights(question, found: FoundSources, writer_llm, errors) -> tuple[list[Insight], list[str]]:
    """LLM call 2 (try, retry once). Fallback: no insights, with the reason as a gap."""
    evidence = []
    used_chars = 0
    for source_id, source in found.sources.items():
        text = found.texts[source_id]
        if used_chars + len(text) > MAX_EVIDENCE_CHARS:
            continue
        used_chars += len(text)
        evidence.append({"source_id": source_id, "from": source.title, "text": text})
    if not evidence:
        return [], ["no relevant filing passages or web results were found"]

    question_text = f"Question: {question}\n\nEvidence (JSON):\n{json.dumps(evidence)}"
    messages = [SystemMessage(content=WRITER_PROMPT), HumanMessage(content=question_text)]
    try:
        draft = ask_structured(writer_llm, ResearchDraft, messages)
    except Exception as error:
        errors.append(f"research writer failed twice: {short_error(error)}")
        return [], ["sources were found but could not be summarized (writer model failed)"]
    return draft.insights, draft.gaps


def keep_grounded(insights: list[Insight], found: FoundSources) -> tuple[list[Insight], int]:
    """Grounding in code: remove unknown source ids, and drop claims left with none."""
    kept = []
    dropped = 0
    for insight in insights:
        valid_ids = [source_id for source_id in dict.fromkeys(insight.source_ids) if source_id in found.sources]
        if valid_ids:
            kept.append(Insight(claim=insight.claim, source_ids=valid_ids, confidence=insight.confidence))
        else:
            dropped += 1
    return kept, dropped


# =========================
# Helpers
# =========================

NOT_TICKERS = {"ETF", "USD", "CEO", "CFO", "AI", "US", "SEC", "IPO", "EPS", "GDP", "VAR", "HHI", "MD", "PE", "OK"}
COMPANY_NAMES = {"apple": "AAPL", "nvidia": "NVDA", "tesla": "TSLA", "microsoft": "MSFT", "amazon": "AMZN",
                 "google": "GOOGL", "alphabet": "GOOGL", "jpmorgan": "JPM", "coinbase": "COIN"}
NEWS_WORDS = ["news", "latest", "recent", "today", "this week", "this month", "announce"]


def pick_tickers(question: str, explicit: list[str], holdings: list[str]) -> list[str]:
    """Given tickers > tickers or company names in the question > the 3 largest stock holdings."""
    if explicit:
        return [ticker.upper() for ticker in explicit]
    found = []
    for word in re.findall(r"\b[A-Z]{2,5}\b", question):
        if word not in NOT_TICKERS and word not in found:
            found.append(word)
    for name, ticker in COMPANY_NAMES.items():
        if name in question.lower() and ticker not in found:
            found.append(ticker)
    return found or holdings[:3]


def clean_args(name: str, args: dict) -> dict:
    """Free models sometimes send empty or made-up arguments. Keep only known, non-empty ones."""
    allowed = ["query", "ticker", "section"] if name == "SearchFilings" else ["query", "recent_days"]
    cleaned = {key: value for key, value in args.items() if key in allowed and value not in (None, "")}
    cleaned.setdefault("query", "")
    if "recent_days" in cleaned:
        try:
            cleaned["recent_days"] = min(max(int(cleaned["recent_days"]), 1), 365)
        except (TypeError, ValueError):
            del cleaned["recent_days"]
    return cleaned


def default_searches(question: str, tickers: list[str]) -> list[tuple[str, dict]]:
    searches = [("SearchFilings", {"query": question, "ticker": ticker}) for ticker in tickers[:2]]
    if not tickers or any(word in question.lower() for word in NEWS_WORDS):
        searches.append(("WebSearch", {"query": question, "recent_days": 30}))
    return searches


def research_summary(result: dict) -> str:
    findings = result["findings"]
    if not findings["insights"]:
        return "No cited insights. Gaps: " + "; ".join(findings["gaps"][:3])
    claims = [f"{insight['claim']} [{', '.join(insight['source_ids'])}]" for insight in findings["insights"][:4]]
    return f"{len(findings['insights'])} cited insight(s). " + " ".join(claims)
