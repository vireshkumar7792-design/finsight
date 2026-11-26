"""Research agent: STUB until Phase 4 (10-K RAG via Chroma + Tavily web search).

It makes no LLM call and no web request. It returns a clearly labeled placeholder Finding so
the graph's shape (routing, state keys, report input) is final before Phase 4 fills it in.
"""

from __future__ import annotations

import re
from typing import Any

from langgraph.runtime import Runtime

from finsight.context import PortfolioContext
from finsight.schemas.findings import Finding
from finsight.state import FinSightState

_TICKER = re.compile(r"\b[A-Z]{2,5}(?:\.[A-Z])?\b")


def requested_tickers(question: str, holdings: list[str]) -> list[str]:
    """Tickers mentioned in the question: ALL-CAPS symbols, plus held tickers in any case."""
    found = list(dict.fromkeys(_TICKER.findall(question)))
    words = {w.strip(".,?!").upper() for w in question.split()}
    found += [t for t in holdings if t in words and t not in found]
    return found


def research_node(state: FinSightState, runtime: Runtime[PortfolioContext]) -> dict[str, Any]:
    holdings = runtime.context.portfolio.tickers if runtime.context else []
    tickers = requested_tickers(state["question"], holdings)
    finding = Finding(
        agent="research",
        summary=(
            "[STUB] The research agent is not implemented yet (Phase 4: SEC 10-K retrieval and web "
            "search). No company or market research was performed"
            + (f" for {', '.join(tickers)}." if tickers else ".")
        ),
        data={"research:stub": {"status": "not_implemented", "requested_tickers": tickers}},
        is_stub=True,
    )
    return {"research_findings": finding.model_dump(), "completed_agents": ["research"]}
