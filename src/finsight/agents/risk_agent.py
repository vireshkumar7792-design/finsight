"""Risk agent: volatility, Sharpe, beta, VaR, drawdown, concentration and correlations."""

from langgraph.runtime import Runtime

from finsight.agents.common import run_tool_agent
from finsight.state import FinSightState, RunContext
from finsight.tools.portfolio_tools import (
    correlations_payload,
    get_correlations,
    get_position_detail,
    get_risk_metrics,
    risk_payload,
)

# =========================
# Prompt
# =========================

PROMPT = """You are the Risk analyst in a team. You cover volatility, Sharpe ratio, beta,
Value at Risk, max drawdown, concentration (HHI, largest position) and how holdings move
together (correlation, diversification). Compare with the SPY benchmark figures the tools give.
Other analysts cover holdings/P&L and company research, so ignore those parts of the question."""

# =========================
# Tools
# =========================

TOOLS = [get_risk_metrics, get_correlations, get_position_detail]

# Used if the LLM fails or calls no tools: (tool name, args, payload function).
DEFAULT_TOOLS = [
    ("get_risk_metrics", {}, risk_payload),
    ("get_correlations", {}, correlations_payload),
]

# =========================
# Node
# =========================


def risk_node(state: FinSightState, runtime: Runtime[RunContext]) -> dict:
    return run_tool_agent("risk", PROMPT, TOOLS, DEFAULT_TOOLS, state, runtime.context)
