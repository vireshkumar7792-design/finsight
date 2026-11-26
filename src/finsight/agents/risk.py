"""Risk agent: volatility, Sharpe, beta, VaR, drawdown, concentration and correlations."""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel

from finsight.agents.specialist import make_specialist_node
from finsight.agents.tools import (
    correlations_payload,
    get_correlations,
    get_position_detail,
    get_risk_metrics,
    risk_payload,
)

RISK_TOOLS = [get_risk_metrics, get_correlations, get_position_detail]

RISK_PROMPT = """You are the Risk specialist in a team of analysts. You cover volatility, Sharpe \
ratio, beta, Value at Risk, max drawdown, concentration (HHI, largest position) and how holdings \
correlate (diversification). Compare against the benchmark figures the tools provide. Other \
specialists cover holdings/P&L and company research; ignore those parts of the question."""

RISK_DEFAULTS = [
    ("get_risk_metrics", {}, risk_payload),
    ("get_correlations", {}, correlations_payload),
]


def make_risk_node(model: BaseChatModel):
    return make_specialist_node("risk", model, RISK_TOOLS, RISK_PROMPT, RISK_DEFAULTS)
