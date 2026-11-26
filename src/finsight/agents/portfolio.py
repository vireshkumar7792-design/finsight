"""Portfolio agent: holdings, value, weights, P&L and allocation, via the Phase 2 tools."""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel

from finsight.agents.specialist import make_specialist_node
from finsight.agents.tools import (
    allocation_payload,
    get_allocation,
    get_portfolio_summary,
    get_position_detail,
    summary_payload,
)

PORTFOLIO_TOOLS = [get_portfolio_summary, get_allocation, get_position_detail]

PORTFOLIO_PROMPT = """You are the Portfolio specialist in a team of analysts. You cover holdings, \
total value, weights, gains/losses (P&L) and allocation by sector or asset type. Other specialists \
cover risk and company research; ignore those parts of the question."""

PORTFOLIO_DEFAULTS = [
    ("get_portfolio_summary", {}, summary_payload),
    ("get_allocation", {"by": "sector"}, lambda ctx: allocation_payload(ctx, "sector")),
]


def make_portfolio_node(model: BaseChatModel):
    return make_specialist_node("portfolio", model, PORTFOLIO_TOOLS, PORTFOLIO_PROMPT, PORTFOLIO_DEFAULTS)
