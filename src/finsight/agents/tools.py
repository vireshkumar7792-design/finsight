"""LangChain tools that expose Phase 1 calculations to an LLM.

Two layers:
  *_payload(ctx, ...) -> dict  Plain Python. Calls tools/ math, rounds, labels units. Unit-tested
                               directly and reused by the MCP servers in Phase 5.
  @tool wrappers               What the LLM sees: a name, a docstring and an argument schema.
                               The PortfolioContext arrives via ToolRuntime, which LangChain
                               injects at call time and HIDES from the LLM's schema, so the
                               model cannot (and need not) pass portfolio data as arguments.

Output conventions (the LLM must not convert units itself, since that is math):
  *_usd  US dollars, 2 decimals       *_pct  percent, 2 decimals (12.5 means 12.5%)
  ratios (Sharpe, beta, HHI, correlation) unitless, 3 decimals
"""

from __future__ import annotations

import json
from typing import Annotated, Any, Literal

from langchain.tools import ToolRuntime, tool

from finsight.context import PortfolioContext
from finsight.tools import portfolio_math, risk_math

UNITS_NOTE = "_usd = US dollars; _pct = percent (12.5 means 12.5%); ratios are unitless."


# ---- formatting helpers -----------------------------------------------------------------


def _usd(x: float) -> float:
    return round(float(x), 2)


def _pct(fraction: float) -> float:
    return round(float(fraction) * 100, 2)


def _ratio(x: float) -> float:
    return round(float(x), 3)


def _dump(payload: dict[str, Any]) -> str:
    return json.dumps(payload, separators=(",", ":"))  # compact: fewer tokens


def _safe(fn, *args) -> float | None:
    """Run a risk function; None when the metric is undefined (e.g. zero volatility)."""
    try:
        return fn(*args)
    except ValueError:
        return None


def _risk_block(ctx: PortfolioContext, returns, prices_or_wealth) -> dict[str, Any]:
    """Volatility / Sharpe / beta / VaR / drawdown for one return series."""
    vol = risk_math.annualized_volatility(returns)
    sharpe = _safe(risk_math.sharpe_ratio, returns, ctx.risk_free_rate)
    beta = _safe(risk_math.beta, returns, ctx.benchmark_returns)
    return {
        "annualized_volatility_pct": _pct(vol),
        "sharpe_ratio": None if sharpe is None else _ratio(sharpe),
        f"beta_vs_{ctx.benchmark_ticker}": None if beta is None else _ratio(beta),
        "var_1day_pct": _pct(risk_math.historical_var(returns, ctx.var_confidence)),
        "max_drawdown_pct": _pct(risk_math.max_drawdown(prices_or_wealth)),
    }


def _method(ctx: PortfolioContext) -> dict[str, Any]:
    return {
        "lookback": ctx.history_period,
        "daily_observations": int(len(ctx.portfolio_returns)),
        "risk_free_rate_pct": _pct(ctx.risk_free_rate),
        "var_confidence_pct": _pct(ctx.var_confidence),
        "notes": "Historical 1-day VaR; portfolio series assumes current weights held constant.",
    }


# ---- payloads (plain Python, unit-tested) -----------------------------------------------


def summary_payload(ctx: PortfolioContext) -> dict[str, Any]:
    s = ctx.summary
    types = {h.ticker: h.asset_type.value for h in ctx.portfolio.holdings}
    positions = sorted(s.positions, key=lambda p: p.weight, reverse=True)
    by_pnl = sorted(s.positions, key=lambda p: p.unrealized_pnl)
    worst, best = by_pnl[0], by_pnl[-1]
    return {
        "as_of": ctx.as_of,
        "units": UNITS_NOTE,
        "total_market_value_usd": _usd(s.total_market_value),
        "total_cost_basis_usd": _usd(s.total_cost_basis),
        "total_unrealized_pnl_usd": _usd(s.total_unrealized_pnl),
        "total_unrealized_pnl_pct": _pct(s.total_unrealized_pnl_pct),
        "n_positions": len(positions),
        "n_positions_with_loss": sum(p.unrealized_pnl < 0 for p in positions),
        "worst_position_by_pnl": {"ticker": worst.ticker, "unrealized_pnl_usd": _usd(worst.unrealized_pnl),
                                  "unrealized_pnl_pct": _pct(worst.unrealized_pnl_pct)},
        "best_position_by_pnl": {"ticker": best.ticker, "unrealized_pnl_usd": _usd(best.unrealized_pnl),
                                 "unrealized_pnl_pct": _pct(best.unrealized_pnl_pct)},
        "positions_by_weight": [
            {
                "ticker": p.ticker,
                "asset_type": types[p.ticker],
                "quantity": p.quantity,
                "avg_buy_price_usd": _usd(p.avg_buy_price),
                "current_price_usd": _usd(p.current_price),
                "market_value_usd": _usd(p.market_value),
                "weight_pct": _pct(p.weight),
                "unrealized_pnl_usd": _usd(p.unrealized_pnl),
                "unrealized_pnl_pct": _pct(p.unrealized_pnl_pct),
            }
            for p in positions
        ],
        "note": "P&L is price-only (excludes dividends received).",
    }


def allocation_payload(ctx: PortfolioContext, by: Literal["sector", "asset_type"]) -> dict[str, Any]:
    if by == "sector":
        groups = ctx.sectors
    elif by == "asset_type":
        groups = {h.ticker: h.asset_type.value for h in ctx.portfolio.holdings}
    else:
        return {"error": f"unknown grouping {by!r}; use 'sector' or 'asset_type'"}
    alloc = portfolio_math.allocation_by_group(ctx.weights, groups)
    members: dict[str, list[str]] = {}
    for t in ctx.portfolio.tickers:
        members.setdefault(groups.get(t) or "Unknown", []).append(t)
    out = {
        "as_of": ctx.as_of,
        "by": by,
        "units": UNITS_NOTE,
        "allocation": [{"group": g, "weight_pct": _pct(w), "tickers": members[g]} for g, w in alloc.items()],
    }
    if by == "sector":
        out["note"] = "Funds (ETFs) are not looked through to their underlying sectors; they appear as 'ETF / Fund'."
    return out


def risk_payload(ctx: PortfolioContext) -> dict[str, Any]:
    pr = ctx.portfolio_returns
    portfolio_risk = _risk_block(ctx, pr, (1 + pr).cumprod())
    var_frac = risk_math.historical_var(pr, ctx.var_confidence)
    portfolio_risk["var_1day_usd"] = _usd(var_frac * ctx.summary.total_market_value)
    top_ticker, top_weight = risk_math.max_weight(ctx.weights)
    return {
        "as_of": ctx.as_of,
        "units": UNITS_NOTE,
        "portfolio": portfolio_risk,
        "benchmark": {"ticker": ctx.benchmark_ticker, **_risk_block(ctx, ctx.benchmark_returns, ctx.benchmark)},
        "concentration": {
            "herfindahl_index": _ratio(risk_math.herfindahl_index(ctx.weights)),
            "effective_number_of_positions": round(risk_math.effective_number_of_positions(ctx.weights), 2),
            "n_positions": len(ctx.weights),
            "largest_position": {"ticker": top_ticker, "weight_pct": _pct(top_weight)},
        },
        "method": _method(ctx),
    }


def correlations_payload(ctx: PortfolioContext, top_n: int = 3) -> dict[str, Any]:
    market = ctx.portfolio.market_tickers
    excluded = [t for t in ctx.portfolio.tickers if t not in market]
    if len(market) < 2:
        return {"error": "need at least 2 non-cash holdings to compute correlations", "excluded_cash": excluded}
    corr = risk_math.correlation_matrix(ctx.returns[market])
    pairs = risk_math.correlation_pairs(corr)
    top_n = max(1, min(top_n, len(pairs)))

    def fmt(p):
        return {"pair": [p[0], p[1]], "correlation": _ratio(p[2])}

    avg = risk_math.average_pairwise_correlation(corr)
    return {
        "as_of": ctx.as_of,
        "lookback": ctx.history_period,
        "basis": "Pearson correlation of daily returns (-1 to 1).",
        "average_pairwise_correlation": None if avg is None else _ratio(avg),
        "most_correlated_pairs": [fmt(p) for p in pairs[:top_n]],
        "least_correlated_pairs": [fmt(p) for p in reversed(pairs[-top_n:])],
        "avg_correlation_with_other_holdings": {
            t: _ratio(v) for t, v in risk_math.average_correlation_by_asset(corr).items()
        },  # lowest first = best diversifiers
        "n_pairs": len(pairs),
        "excluded_cash": excluded,
    }


def position_payload(ctx: PortfolioContext, ticker: str) -> dict[str, Any]:
    h = ctx.portfolio.get(ticker)
    if h is None:
        return {"error": f"{ticker.upper()!r} is not in the portfolio", "holdings": ctx.portfolio.tickers}
    p = next(x for x in ctx.summary.positions if x.ticker == h.ticker)
    out: dict[str, Any] = {
        "as_of": ctx.as_of,
        "units": UNITS_NOTE,
        "ticker": h.ticker,
        "asset_type": h.asset_type.value,
        "sector": ctx.sectors.get(h.ticker, "Unknown"),
        "buy_date": h.buy_date.isoformat() if h.buy_date else None,
        "quantity": h.quantity,
        "avg_buy_price_usd": _usd(h.avg_buy_price),
        "current_price_usd": _usd(p.current_price),
        "cost_basis_usd": _usd(p.cost_basis),
        "market_value_usd": _usd(p.market_value),
        "weight_pct": _pct(p.weight),
        "unrealized_pnl_usd": _usd(p.unrealized_pnl),
        "unrealized_pnl_pct": _pct(p.unrealized_pnl_pct),
    }
    if h.is_cash:
        out["risk"] = "Cash: fixed $1 price, no market risk."
    else:
        out["risk"] = _risk_block(ctx, ctx.returns[h.ticker], ctx.history[h.ticker])
    return out


# ---- LLM-facing tools -------------------------------------------------------------------
# The docstring IS the tool description the model reads when deciding what to call.
# Say when to use it, what it returns, and the units. Keep it short and specific.


@tool
def get_portfolio_summary(runtime: ToolRuntime[PortfolioContext]) -> str:
    """Get total portfolio value, cost basis and unrealized profit/loss, plus every position's
    market value, weight and P&L (sorted by weight), and the best/worst positions by P&L.

    Use for: overall value, gains/losses, which holdings made or lost money, position sizes.
    """
    return _dump(summary_payload(runtime.context))


@tool
def get_allocation(
    by: Annotated[Literal["sector", "asset_type"], "Group by 'sector' or by 'asset_type' (equity/etf/bond/crypto/cash)."],
    runtime: ToolRuntime[PortfolioContext],
) -> str:
    """Get the portfolio's allocation (weight in percent) grouped by sector or by asset type,
    largest group first, with the tickers in each group.

    Use for: sector exposure, asset mix (stocks vs bonds vs cash), diversification across groups.
    """
    return _dump(allocation_payload(runtime.context, by))


@tool
def get_risk_metrics(runtime: ToolRuntime[PortfolioContext]) -> str:
    """Get portfolio risk metrics over the lookback period, side by side with the benchmark (SPY):
    annualized volatility, Sharpe ratio, beta, 1-day historical VaR (percent and USD), max drawdown;
    plus concentration: Herfindahl index (HHI), effective number of positions, largest position.

    Use for: how risky the portfolio is, risk vs the market, concentration, diversification.
    """
    return _dump(risk_payload(runtime.context))


@tool
def get_correlations(
    runtime: ToolRuntime[PortfolioContext],
    top_n: Annotated[int, "How many most- and least-correlated pairs to return (default 3)."] = 3,
) -> str:
    """Get how holdings move together (correlation of daily returns): the portfolio's average
    pairwise correlation, the most- and least-correlated pairs, and each holding's average
    correlation with all other holdings (lowest = diversifies the rest). Cash is excluded.

    Use for: diversification, whether holdings are redundant, which positions hedge each other.
    """
    return _dump(correlations_payload(runtime.context, top_n))


@tool
def get_position_detail(
    ticker: Annotated[str, "Ticker symbol of a holding in the portfolio, e.g. 'AAPL'."],
    runtime: ToolRuntime[PortfolioContext],
) -> str:
    """Get one holding in detail: sector, asset type, buy date, cost, value, weight, P&L, and its
    own volatility, Sharpe, beta, VaR and max drawdown.

    Use for: questions about a specific ticker. Returns an error listing valid tickers if not held.
    """
    return _dump(position_payload(runtime.context, ticker))


ALL_TOOLS = [get_portfolio_summary, get_allocation, get_risk_metrics, get_correlations, get_position_detail]
