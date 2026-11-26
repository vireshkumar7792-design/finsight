"""Typed outputs of the deterministic tools.

Agents receive these instead of loose dicts. In Phase 6 the number-consistency guardrail
compares figures quoted in the report against these exact fields.
Convention: all ratios are fractions (0.25 == 25%), all money is USD.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class PositionMetrics(BaseModel):
    """Valuation and P&L for one holding at a given price."""

    model_config = ConfigDict(frozen=True)

    ticker: str
    quantity: float
    avg_buy_price: float
    current_price: float
    cost_basis: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float  # fraction of cost basis
    weight: float  # fraction of total portfolio market value


class PortfolioSummary(BaseModel):
    """Portfolio-level totals plus the per-position breakdown."""

    model_config = ConfigDict(frozen=True)

    total_cost_basis: float
    total_market_value: float
    total_unrealized_pnl: float
    total_unrealized_pnl_pct: float
    positions: list[PositionMetrics]
