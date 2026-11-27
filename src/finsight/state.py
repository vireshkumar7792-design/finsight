"""Shared graph state (FinSightState) and its reducers.

REDUCERS: when a node returns {"key": value}, LangGraph merges it into state.
  - No reducer  -> overwrite (last write wins). Used for single values: plan, report, ...
  - add_messages -> append (and update by message id). Conversation memory across questions.
  - accumulate  -> append lists; an explicit None RESETS to []. Per-question logs use this, so
                   a new question on the same thread can start clean while messages persist.
  - add_or_reset -> integer counter; None resets to 0.

WHAT IS NOT IN STATE: PortfolioContext. It holds DataFrames and would bloat (or fail to
serialize into) every checkpoint. It is passed per run via `context=` (LangGraph runtime
context), like Phase 2. Tradeoff: on resume the context must be rebuilt, so we store the
CSV path (`portfolio_source`) in state and reload it. The rebuilt context may hold NEWER
market data than the findings already saved, so `data_as_of` (date of the prices a question
started with) is stored too, and graph.resume_thread() never mixes dates: on a different day it
re-runs the question on the new data.
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.graph.message import add_messages

MAX_ITERATIONS = 6  # supervisor visits per question before forcing the report


def accumulate(old: list | None, new: list | None) -> list:
    """List reducer: append `new`; `None` means reset to empty."""
    if new is None:
        return []
    return [*(old or []), *new]


def add_or_reset(old: int | None, new: int | None) -> int:
    """Counter reducer: add `new`; `None` means reset to 0."""
    if new is None:
        return 0
    return (old or 0) + new


class FinSightState(TypedDict, total=False):
    # Conversation memory (persists across questions on a thread): questions + final summaries.
    messages: Annotated[list[AnyMessage], add_messages]
    # The current question and where the portfolio came from (used to rebuild context on resume).
    question: str
    portfolio_source: str
    data_as_of: str | None  # market-data date (YYYY-MM-DD) this question's findings are based on
    # Supervisor's plan for this question: {"agents": [...], "reason": str, "source": "llm"|"fallback"}
    plan: dict[str, Any] | None
    next_agent: str | None
    iteration: int
    # Specialist outputs: Finding.model_dump() dicts (plain dicts checkpoint cleanly).
    portfolio_findings: dict[str, Any] | None
    risk_findings: dict[str, Any] | None
    research_findings: dict[str, Any] | None
    # Final output: Report.model_dump() plus how it was produced.
    report: dict[str, Any] | None
    report_mode: str | None  # "structured" | "plain_text_fallback" | "deterministic_fallback"
    # Per-question accumulators (reset with None at the start of each question).
    completed_agents: Annotated[list[str], accumulate]
    errors: Annotated[list[str], accumulate]
    route_log: Annotated[list[dict[str, Any]], accumulate]  # [{"to": ..., "reason": ...}]
    tool_calls: Annotated[list[dict[str, Any]], accumulate]  # [{"agent", "name", "args"}]
    llm_calls: Annotated[int, add_or_reset]


FINDING_KEYS = {"portfolio": "portfolio_findings", "risk": "risk_findings", "research": "research_findings"}


def new_question_input(question: str, portfolio_source: str, data_as_of: str | None = None) -> dict[str, Any]:
    """Graph input for a new question: append it to the conversation, reset per-question fields."""
    return {
        "messages": [HumanMessage(content=question)],
        "question": question,
        "portfolio_source": portfolio_source,
        "data_as_of": data_as_of,
        "plan": None,
        "next_agent": None,
        "iteration": 0,
        "portfolio_findings": None,
        "risk_findings": None,
        "research_findings": None,
        "report": None,
        "report_mode": None,
        "completed_agents": None,  # None -> reducer resets to []
        "errors": None,
        "route_log": None,
        "tool_calls": None,
        "llm_calls": None,
    }


def prior_conversation(state: FinSightState, max_turns: int = 2) -> str:
    """Compact text of earlier Q&A on this thread (excluding the current question)."""
    msgs = list(state.get("messages") or [])
    if msgs and isinstance(msgs[-1], HumanMessage):
        msgs = msgs[:-1]  # drop the current question
    msgs = msgs[-2 * max_turns :]
    lines = [f"{'User' if isinstance(m, HumanMessage) else 'FinSight'}: {m.content}" for m in msgs]
    return "\n".join(lines) if lines else "(none)"
