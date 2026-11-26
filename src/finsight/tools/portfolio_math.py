"""Portfolio math: market value, weights, unrealized P&L, sector/asset-type allocation.

Pure functions: prices and sectors are passed in, never fetched here. That keeps every
function deterministic and testable with hand-computed numbers.
Convention: ratios are fractions (0.25 == 25%).
"""

from __future__ import annotations

from collections.abc import Mapping

from finsight.schemas.metrics import PortfolioSummary, PositionMetrics
from finsight.schemas.portfolio import Portfolio


def _require_prices(portfolio: Portfolio, prices: Mapping[str, float]) -> None:
    missing = [t for t in portfolio.tickers if t not in prices]
    if missing:
        raise ValueError(f"missing current price for: {', '.join(missing)}")
    bad = [t for t in portfolio.tickers if not prices[t] > 0]
    if bad:
        raise ValueError(f"non-positive price for: {', '.join(bad)}")


def market_values(portfolio: Portfolio, prices: Mapping[str, float]) -> dict[str, float]:
    """Market value per ticker: quantity * current price."""
    _require_prices(portfolio, prices)
    return {h.ticker: h.quantity * prices[h.ticker] for h in portfolio.holdings}


def portfolio_weights(portfolio: Portfolio, prices: Mapping[str, float]) -> dict[str, float]:
    """Each position's share of total market value. Sums to 1."""
    values = market_values(portfolio, prices)
    total = sum(values.values())
    return {t: v / total for t, v in values.items()}


def portfolio_summary(portfolio: Portfolio, prices: Mapping[str, float]) -> PortfolioSummary:
    """Per-position valuation and unrealized P&L, plus portfolio totals."""
    weights = portfolio_weights(portfolio, prices)
    positions = []
    for h in portfolio.holdings:
        price = prices[h.ticker]
        mv = h.quantity * price
        pnl = mv - h.cost_basis
        positions.append(
            PositionMetrics(
                ticker=h.ticker,
                quantity=h.quantity,
                avg_buy_price=h.avg_buy_price,
                current_price=price,
                cost_basis=h.cost_basis,
                market_value=mv,
                unrealized_pnl=pnl,
                unrealized_pnl_pct=pnl / h.cost_basis,
                weight=weights[h.ticker],
            )
        )
    total_cost = sum(p.cost_basis for p in positions)
    total_mv = sum(p.market_value for p in positions)
    total_pnl = total_mv - total_cost
    return PortfolioSummary(
        total_cost_basis=total_cost,
        total_market_value=total_mv,
        total_unrealized_pnl=total_pnl,
        total_unrealized_pnl_pct=total_pnl / total_cost,
        positions=positions,
    )


def allocation_by_group(weights: Mapping[str, float], groups: Mapping[str, str]) -> dict[str, float]:
    """Sum weights by group label (e.g. sector). Tickers without a label go to "Unknown".

    Result is sorted largest group first.
    """
    totals: dict[str, float] = {}
    for ticker, w in weights.items():
        label = groups.get(ticker) or "Unknown"
        totals[label] = totals.get(label, 0.0) + w
    return dict(sorted(totals.items(), key=lambda kv: kv[1], reverse=True))


def asset_type_allocation(portfolio: Portfolio, prices: Mapping[str, float]) -> dict[str, float]:
    """Weight by asset_type (equity/etf/bond/crypto/cash)."""
    groups = {h.ticker: h.asset_type.value for h in portfolio.holdings}
    return allocation_by_group(portfolio_weights(portfolio, prices), groups)


def sector_allocation(
    portfolio: Portfolio, prices: Mapping[str, float], sectors: Mapping[str, str]
) -> dict[str, float]:
    """Weight by sector. `sectors` maps ticker -> sector (from market_data.get_sectors)."""
    return allocation_by_group(portfolio_weights(portfolio, prices), sectors)
