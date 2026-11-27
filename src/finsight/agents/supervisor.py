"""Supervisor: ONE LLM call per question makes a plan; after that, code just follows the plan."""

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.runtime import Runtime

from finsight.agents.common import ask_structured, short_error
from finsight.schemas import RoutePlan
from finsight.state import FinSightState, RunContext, earlier_conversation

MAX_SUPERVISOR_VISITS = 6  # safety net: never loop forever

# =========================
# Prompt
# =========================

PROMPT = """You route questions about a user's investment portfolio to specialist analysts.

Analysts:
- portfolio: holdings, total value, position sizes, gains and losses (performance),
  allocation by sector or asset type, details of one holding.
- risk: volatility, beta, Sharpe ratio, Value at Risk, drawdown, concentration,
  correlations, how risky or how diversified the portfolio is.
- research: news, business outlook, competition or 10-K filings for a specific company.

Pick the smallest list of analysts that fully answers the question, in a sensible order.
Diversification needs both portfolio and risk. Use an empty list only when no analyst is
needed (a greeting, or the answer is already in the earlier conversation)."""

# =========================
# Node
# =========================


def supervisor_node(state: FinSightState, runtime: Runtime[RunContext]) -> dict:
    visits = state["supervisor_visits"] + 1
    update = {"supervisor_visits": visits}

    if visits == 1:
        # First visit for this question: the one LLM call.
        plan, reason, decided_by, errors = make_plan(state, runtime.context.router_llm)
        update["plan"] = plan
        update["plan_reason"] = reason
        update["errors"] = errors
    else:
        plan = state["plan"]
        reason = "next agent in the plan"
        decided_by = "plan"

    remaining = [agent for agent in plan if agent not in state["completed_agents"]]
    if visits > MAX_SUPERVISOR_VISITS:
        next_agent = "report"
        reason = f"safety stop after {MAX_SUPERVISOR_VISITS} steps"
    elif remaining:
        next_agent = remaining[0]
    else:
        next_agent = "report"
        if decided_by == "plan":
            reason = "all planned agents are done"

    update["next_agent"] = next_agent
    update["route_log"] = state["route_log"] + [{"to": next_agent, "by": decided_by, "reason": reason}]
    return update


# =========================
# Helpers
# =========================


def make_plan(state: FinSightState, router_llm) -> tuple[list[str], str, str, list[str]]:
    """Returns (plan, reason, decided_by, errors). Falls back to keywords if the LLM fails."""
    messages = [
        SystemMessage(content=PROMPT),
        HumanMessage(content=f"Earlier conversation:\n{earlier_conversation(state)}\n\nQuestion: {state['question']}"),
    ]
    try:
        route_plan = ask_structured(router_llm, RoutePlan, messages)
    except Exception as error:
        plan = keyword_plan(state["question"])
        errors = state["errors"] + [f"supervisor LLM failed ({short_error(error)}); used keyword routing"]
        return plan, "keyword fallback", "fallback", errors
    plan = list(dict.fromkeys(route_plan.agents))  # drop duplicates, keep order
    return plan, route_plan.reason, "llm", state["errors"]


PORTFOLIO_WORDS = ["sector", "allocation", "holding", "position", "value", "worth", "gain", "loss",
                   "p&l", "perform", "diversif", "weight", "exposure"]
RISK_WORDS = ["risk", "volatil", "beta", "sharpe", "drawdown", "diversif", "correlat", "concentrat"]
RESEARCH_WORDS = ["news", "research", "10-k", "filing", "competition", "outlook", "business"]


def keyword_plan(question: str) -> list[str]:
    """Simple routing by keywords, used only when the router LLM fails twice."""
    text = question.lower()
    plan = []
    if any(word in text for word in PORTFOLIO_WORDS):
        plan.append("portfolio")
    if any(word in text for word in RISK_WORDS):
        plan.append("risk")
    if any(word in text for word in RESEARCH_WORDS):
        plan.append("research")
    return plan or ["portfolio"]
