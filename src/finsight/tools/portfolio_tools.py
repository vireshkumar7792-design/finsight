"""Tools for the Portfolio and Risk agents. They return finished, rounded, unit-labeled numbers.

Field names say the unit: _usd = US dollars, _pct = percent (12.5 means 12.5%), others are ratios.
"""

import json
from typing import Literal

import numpy as np
from langchain.tools import ToolRuntime, tool

from finsight.portfolio import calculations
from finsight.portfolio.context import BENCHMARK_TICKER, PortfolioContext
from finsight.state import RunContext

# =========================
# Rounding helpers
# =========================


def usd(value: float) -> float:
    return round(float(value), 2)


def pct(fraction: float) -> float:
    return round(float(fraction) * 100, 2)


def ratio(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 3)


def try_metric(function, *args) -> float | None:
    """Some metrics are undefined (e.g. Sharpe with zero volatility): return None for those."""
    try:
        return function(*args)
    except ValueError:
        return None


def risk_numbers(returns, prices, benchmark_returns) -> dict:
    """Volatility, Sharpe, beta, VaR and max drawdown for one return series."""
    return {
        "annualized_volatility_pct": pct(calculations.annualized_volatility(returns)),
        "sharpe_ratio": ratio(try_metric(calculations.sharpe_ratio, returns)),
        f"beta_vs_{BENCHMARK_TICKER}": ratio(try_metric(calculations.beta, returns, benchmark_returns)),
        "var_1day_95_pct": pct(calculations.historical_var(returns)),
        "max_drawdown_pct": pct(calculations.max_drawdown(prices)),
    }


# =========================
# Payloads (plain functions, tested directly)
# =========================


def summary_payload(context: PortfolioContext) -> dict:
    positions = calculations.position_details(context.portfolio, context.prices)
    totals = calculations.portfolio_totals(positions)
    by_pnl = sorted(positions, key=lambda position: position["unrealized_pnl"])
    by_weight = sorted(positions, key=lambda position: position["weight"], reverse=True)
    return {
        "as_of": context.as_of,
        "total_market_value_usd": usd(totals["total_market_value"]),
        "total_cost_basis_usd": usd(totals["total_cost_basis"]),
        "total_unrealized_pnl_usd": usd(totals["total_unrealized_pnl"]),
        "total_unrealized_pnl_pct": pct(totals["total_unrealized_pnl_pct"]),
        "number_of_positions": len(positions),
        "worst_position": position_row(by_pnl[0]),
        "best_position": position_row(by_pnl[-1]),
        "positions_by_weight": [position_row(position) for position in by_weight],
        "note": "P&L is price-only (dividends not included).",
    }


def position_row(position: dict) -> dict:
    return {
        "ticker": position["ticker"],
        "asset_type": position["asset_type"],
        "market_value_usd": usd(position["market_value"]),
        "weight_pct": pct(position["weight"]),
        "unrealized_pnl_usd": usd(position["unrealized_pnl"]),
        "unrealized_pnl_pct": pct(position["unrealized_pnl_pct"]),
    }


def allocation_payload(context: PortfolioContext, by: str) -> dict:
    if by == "sector":
        labels = context.sectors
    else:
        labels = {holding.ticker: holding.asset_type for holding in context.portfolio.holdings}
    groups = calculations.allocation(context.weights, labels)

    rows = []
    for group, weight in groups.items():
        tickers = [ticker for ticker in context.portfolio.tickers if labels.get(ticker, "Unknown") == group]
        rows.append({"group": group, "weight_pct": pct(weight), "tickers": tickers})
    result = {"as_of": context.as_of, "by": by, "allocation": rows}
    if by == "sector":
        result["note"] = "ETFs are not split into their underlying sectors; they show as 'ETF / Fund'."
    return result


def risk_payload(context: PortfolioContext) -> dict:
    portfolio_returns = context.portfolio_returns
    growth_of_one_dollar = (1 + portfolio_returns).cumprod()
    portfolio_risk = risk_numbers(portfolio_returns, growth_of_one_dollar, context.benchmark_returns)
    positions = calculations.position_details(context.portfolio, context.prices)
    total_value = calculations.portfolio_totals(positions)["total_market_value"]
    portfolio_risk["var_1day_95_usd"] = usd(calculations.historical_var(portfolio_returns) * total_value)

    hhi = calculations.herfindahl_index(context.weights)
    largest = max(context.weights, key=context.weights.get)
    return {
        "as_of": context.as_of,
        "lookback": f"1 year, {len(portfolio_returns)} trading days",
        "portfolio": portfolio_risk,
        "benchmark": {
            "ticker": BENCHMARK_TICKER,
            **risk_numbers(context.benchmark_returns, context.benchmark, context.benchmark_returns),
        },
        "concentration": {
            "herfindahl_index": ratio(hhi),
            "effective_number_of_positions": round(1 / hhi, 2),
            "number_of_positions": len(context.weights),
            "largest_position": {"ticker": largest, "weight_pct": pct(context.weights[largest])},
        },
        "method": f"Historical 1-day VaR at 95%; risk-free rate {pct(calculations.RISK_FREE_RATE)}%; "
        "portfolio returns assume today's weights were held all year.",
    }


def correlations_payload(context: PortfolioContext) -> dict:
    tickers = context.portfolio.market_tickers  # cash never moves, so it is left out
    if len(tickers) < 2:
        return {"error": "need at least 2 non-cash holdings to compute correlations"}
    matrix = calculations.correlation_matrix(context.returns[tickers])
    pairs = calculations.correlation_pairs(matrix)
    pair_rows = [{"pair": [first, second], "correlation": ratio(value)} for first, second, value in pairs]
    averages = calculations.average_correlation_by_ticker(matrix)
    return {
        "as_of": context.as_of,
        "basis": "Correlation of daily returns over 1 year (-1 to 1).",
        "average_pairwise_correlation": ratio(np.mean([value for _, _, value in pairs])),
        "most_correlated_pairs": pair_rows[:3],
        "least_correlated_pairs": list(reversed(pair_rows[-3:])),
        "average_correlation_with_others": {ticker: ratio(value) for ticker, value in averages.items()},
    }


def position_payload(context: PortfolioContext, ticker: str) -> dict:
    ticker = ticker.strip().upper()
    if ticker not in context.portfolio.tickers:
        return {"error": f"{ticker} is not in the portfolio", "holdings": context.portfolio.tickers}
    positions = calculations.position_details(context.portfolio, context.prices)
    position = next(row for row in positions if row["ticker"] == ticker)
    result = {
        "as_of": context.as_of,
        "sector": context.sectors.get(ticker, "Unknown"),
        "quantity": position["quantity"],
        "avg_buy_price_usd": usd(position["avg_buy_price"]),
        "current_price_usd": usd(position["current_price"]),
        **position_row(position),
    }
    if position["asset_type"] == "cash":
        result["risk"] = "Cash has a fixed $1 price and no market risk."
    else:
        result["risk"] = risk_numbers(context.returns[ticker], context.history[ticker], context.benchmark_returns)
    return result


# =========================
# LLM tools
# =========================
# The docstring is what the model reads to decide which tool to call.
# `runtime` is filled in by LangChain and hidden from the model, so it can't invent data.


@tool
def get_portfolio_summary(runtime: ToolRuntime[RunContext]) -> str:
    """Total value, cost and profit/loss, every position's weight and P&L, and the best/worst position."""
    return json.dumps(summary_payload(runtime.context.portfolio))


@tool
def get_allocation(by: Literal["sector", "asset_type"], runtime: ToolRuntime[RunContext]) -> str:
    """Portfolio weight per sector, or per asset type (equity/etf/bond/crypto/cash), largest first."""
    return json.dumps(allocation_payload(runtime.context.portfolio, by))


@tool
def get_risk_metrics(runtime: ToolRuntime[RunContext]) -> str:
    """Volatility, Sharpe, beta, 1-day VaR and max drawdown vs SPY, plus concentration (HHI, largest position)."""
    return json.dumps(risk_payload(runtime.context.portfolio))


@tool
def get_correlations(runtime: ToolRuntime[RunContext]) -> str:
    """How holdings move together: most/least correlated pairs and each holding's average correlation."""
    return json.dumps(correlations_payload(runtime.context.portfolio))


@tool
def get_position_detail(ticker: str, runtime: ToolRuntime[RunContext]) -> str:
    """One holding in detail (e.g. ticker='AAPL'): value, weight, P&L and its own risk numbers."""
    return json.dumps(position_payload(runtime.context.portfolio, ticker))
