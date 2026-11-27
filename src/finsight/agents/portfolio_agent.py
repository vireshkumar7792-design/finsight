"""Portfolio agent: holdings, value, weights, profit/loss and allocation."""

from langgraph.runtime import Runtime

from finsight.agents.common import run_tool_agent
from finsight.state import FinSightState, RunContext
from finsight.tools.portfolio_tools import (
    allocation_payload,
    get_allocation,
    get_portfolio_summary,
    get_position_detail,
    summary_payload,
)

# =========================
# Prompt
# =========================

PROMPT = """You are the Portfolio analyst in a team. You cover holdings, total value, weights,
gains and losses (P&L) and allocation by sector or asset type. Other analysts cover risk and
company research, so ignore those parts of the question."""

# =========================
# Tools
# =========================

TOOLS = [get_portfolio_summary, get_allocation, get_position_detail]

# Used if the LLM fails or calls no tools: (tool name, args, payload function).
DEFAULT_TOOLS = [
    ("get_portfolio_summary", {}, summary_payload),
    ("get_allocation", {"by": "sector"}, allocation_payload),
]

# =========================
# Node
# =========================


def portfolio_node(state: FinSightState, runtime: Runtime[RunContext]) -> dict:
    return run_tool_agent("portfolio", PROMPT, TOOLS, DEFAULT_TOOLS, state, runtime.context)
