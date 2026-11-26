"""Supervisor: decides which specialists a question needs, via structured output on the router LLM.

Plan once, execute deterministically:
  1st visit for a question -> ONE LLM call returns RouteDecision(next, then, reason): the plan.
  later visits             -> no LLM call: go to the next planned agent not yet completed,
                              or to "report" when the plan is done.
The routing decision is still LLM-driven (the LLM chose the plan); code only walks it. A
classic supervisor re-asks the LLM after every specialist, which costs 1-2 extra calls per
question on a free tier with little benefit for questions this direct.

Guard rails (free models fail more often):
  - router exception / unparseable output -> fallback: straight to report with what we have
  - "finish" or "report" -> report (every question ends with a report)
  - an agent already completed is never re-planned
  - more than MAX_ITERATIONS supervisor visits -> report
"""

from __future__ import annotations

from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field

from finsight.agents.structured import structured_invoke
from finsight.state import MAX_ITERATIONS, FinSightState, prior_conversation

SPECIALISTS = ("portfolio", "risk", "research")
ROUTES = (*SPECIALISTS, "report")


class RouteDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    next: Literal["portfolio", "risk", "research", "report", "finish"] = Field(
        description="First specialist to run, or 'report' if no specialist is needed"
    )
    then: list[Literal["portfolio", "risk", "research"]] = Field(
        description="Other specialists also needed after `next`, in order. Empty list if none."
    )
    reason: str = Field(description="One short sentence explaining the choice")


ROUTER_PROMPT = """You route questions about a user's investment portfolio to specialist analysts.

Specialists:
- portfolio: holdings, total value, position sizes/weights, gains and losses (P&L, performance), \
allocation by sector or asset type, details of one holding.
- risk: volatility, beta, Sharpe ratio, Value at Risk, drawdown, concentration, correlations, \
how risky or how diversified the portfolio is.
- research: news, fundamentals, business outlook or filings for a specific company or market.

Choose the minimum set of specialists that fully covers the question:
- set `next` to the first one and list any others in `then` (no duplicates);
- diversification needs both portfolio (allocation) and risk (correlation, concentration);
- use `next: "report"` with empty `then` only if no specialist is needed (greetings, or the \
answer is already in the earlier conversation).
Return only the structured decision."""


def plan_from_decision(d: RouteDecision, completed: list[str]) -> dict[str, Any]:
    """Turn a RouteDecision into an ordered list of agents still to run."""
    if d.next in ("report", "finish"):
        agents: list[str] = []
    else:
        agents = list(dict.fromkeys([d.next, *d.then]))  # dedupe, keep order
    agents = [a for a in agents if a not in completed]
    return {"agents": agents, "reason": d.reason, "source": "llm"}


def route_with_llm(llm: BaseChatModel, state: FinSightState) -> RouteDecision:
    messages = [
        SystemMessage(content=ROUTER_PROMPT),
        HumanMessage(
            content=f"Earlier conversation:\n{prior_conversation(state)}\n\nQuestion: {state['question']}"
        ),
    ]
    return structured_invoke(llm, RouteDecision, messages)


def make_supervisor_node(llm: BaseChatModel):
    def supervisor(state: FinSightState) -> dict[str, Any]:
        iteration = (state.get("iteration") or 0) + 1
        completed = list(state.get("completed_agents") or [])
        update: dict[str, Any] = {"iteration": iteration}

        if iteration > MAX_ITERATIONS:
            reason = f"max iterations ({MAX_ITERATIONS}) reached; reporting with what we have"
            return {**update, "next_agent": "report", "errors": [reason],
                    "route_log": [{"to": "report", "reason": reason, "by": "guard"}]}

        plan = state.get("plan")
        by = "plan"
        if plan is None:  # first visit for this question: the one LLM routing call
            update["llm_calls"] = 1
            by = "llm"
            try:
                plan = plan_from_decision(route_with_llm(llm, state), completed)
            except Exception as e:  # StructuredOutputError, provider errors, anything
                msg = f"router failed ({type(e).__name__}: {str(e)[:150]}); fallback to report"
                plan = {"agents": [], "reason": msg, "source": "fallback"}
                update["errors"] = [msg]
                by = "fallback"
            update["plan"] = plan

        pending = [a for a in plan["agents"] if a not in completed and a in SPECIALISTS]
        if pending:
            nxt = pending[0]
            reason = plan["reason"] if by != "plan" else f"next in plan (remaining: {', '.join(pending)})"
        else:
            nxt = "report"
            reason = plan["reason"] if (by != "plan" and not plan["agents"]) else "all planned specialists done"
        return {**update, "next_agent": nxt, "route_log": [{"to": nxt, "reason": reason, "by": by}]}

    return supervisor


def route_after_supervisor(state: FinSightState) -> str:
    """Conditional edge: where the graph goes after the supervisor. Anything unknown -> report."""
    nxt = state.get("next_agent")
    return nxt if nxt in ROUTES else "report"
