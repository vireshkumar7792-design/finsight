"""Portfolio math and risk math. Plain functions: data goes in, numbers come out, no network.

Conventions: ratios are fractions (0.25 = 25%), returns are daily, losses are positive numbers.
"""

import numpy as np
import pandas as pd

from finsight.schemas import Portfolio

TRADING_DAYS_PER_YEAR = 252
RISK_FREE_RATE = 0.04  # annual
VAR_CONFIDENCE = 0.95


# =========================
# Portfolio math
# =========================


def position_details(portfolio: Portfolio, prices: dict[str, float]) -> list[dict]:
    """Value, weight and unrealized P&L for every holding."""
    market_values = {}
    for holding in portfolio.holdings:
        market_values[holding.ticker] = holding.quantity * prices[holding.ticker]
    total_value = sum(market_values.values())

    positions = []
    for holding in portfolio.holdings:
        market_value = market_values[holding.ticker]
        pnl = market_value - holding.cost_basis
        positions.append({
            "ticker": holding.ticker,
            "asset_type": holding.asset_type,
            "quantity": holding.quantity,
            "avg_buy_price": holding.avg_buy_price,
            "current_price": prices[holding.ticker],
            "cost_basis": holding.cost_basis,
            "market_value": market_value,
            "unrealized_pnl": pnl,
            "unrealized_pnl_pct": pnl / holding.cost_basis,
            "weight": market_value / total_value,
        })
    return positions


def portfolio_totals(positions: list[dict]) -> dict:
    total_cost = sum(position["cost_basis"] for position in positions)
    total_value = sum(position["market_value"] for position in positions)
    total_pnl = total_value - total_cost
    return {
        "total_cost_basis": total_cost,
        "total_market_value": total_value,
        "total_unrealized_pnl": total_pnl,
        "total_unrealized_pnl_pct": total_pnl / total_cost,
    }


def allocation(weights: dict[str, float], labels: dict[str, str]) -> dict[str, float]:
    """Add up weights per label (sector or asset type). Largest group first."""
    totals = {}
    for ticker, weight in weights.items():
        label = labels.get(ticker) or "Unknown"
        totals[label] = totals.get(label, 0.0) + weight
    return dict(sorted(totals.items(), key=lambda item: item[1], reverse=True))


# =========================
# Returns
# =========================


def daily_returns(prices):
    """Simple returns: today / yesterday - 1. The first day has no return, so it is dropped."""
    return prices.pct_change().iloc[1:]


def portfolio_returns(returns: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    """Weighted sum of daily returns, assuming today's weights were held all year."""
    weight_series = pd.Series(weights)
    aligned_returns = returns[weight_series.index].dropna()
    return (aligned_returns * weight_series).sum(axis=1)


# =========================
# Risk math
# =========================


def annualized_volatility(returns: pd.Series) -> float:
    """Standard deviation of daily returns, scaled to a year by sqrt(252)."""
    return float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = RISK_FREE_RATE) -> float:
    """(Mean daily return - daily risk-free rate) / daily std, scaled to a year by sqrt(252)."""
    daily_std = returns.std(ddof=1)
    if daily_std == 0:
        raise ValueError("returns have zero volatility, so Sharpe is undefined")
    daily_excess = returns.mean() - risk_free_rate / TRADING_DAYS_PER_YEAR
    return float(daily_excess / daily_std * np.sqrt(TRADING_DAYS_PER_YEAR))


def beta(asset_returns: pd.Series, market_returns: pd.Series) -> float:
    """Cov(asset, market) / Var(market), using only the dates both series have."""
    both = pd.concat([asset_returns, market_returns], axis=1, join="inner").dropna()
    asset = both.iloc[:, 0]
    market = both.iloc[:, 1]
    market_variance = market.var(ddof=1)
    if market_variance == 0:
        raise ValueError("market returns have zero variance, so beta is undefined")
    return float(asset.cov(market) / market_variance)


def historical_var(returns: pd.Series, confidence: float = VAR_CONFIDENCE) -> float:
    """One-day Value at Risk: the 5th-percentile daily return, as a positive loss.

    0.03 means: on 95% of days, the loss was no worse than 3%.
    """
    worst_percentile = np.quantile(returns.dropna().to_numpy(), 1 - confidence)
    return float(-worst_percentile) + 0.0  # "+ 0.0" turns -0.0 into 0.0


def max_drawdown(prices: pd.Series) -> float:
    """Biggest fall from a previous peak, as a positive fraction (0.5 = fell 50%)."""
    running_peak = prices.cummax()
    drawdowns = prices / running_peak - 1
    return float(-drawdowns.min()) + 0.0


def herfindahl_index(weights: dict[str, float]) -> float:
    """Concentration = sum of squared weights. 1/N if equal weights, 1.0 if one position."""
    return float(sum(weight**2 for weight in weights.values()))


# =========================
# Correlation
# =========================


def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation of daily returns. Cash never moves, so its correlations are NaN."""
    return returns.dropna().corr()


def correlation_pairs(correlations: pd.DataFrame) -> list[tuple[str, str, float]]:
    """Every pair of different tickers once, most correlated first. NaN pairs are skipped."""
    tickers = list(correlations.columns)
    pairs = []
    for i, first in enumerate(tickers):
        for second in tickers[i + 1:]:
            value = correlations.loc[first, second]
            if not np.isnan(value):
                pairs.append((first, second, float(value)))
    pairs.sort(key=lambda pair: pair[2], reverse=True)
    return pairs


def average_correlation_by_ticker(correlations: pd.DataFrame) -> dict[str, float]:
    """Each ticker's mean correlation with the others, lowest (best diversifier) first."""
    averages = {}
    for ticker in correlations.columns:
        others = correlations.loc[ticker].drop(ticker).dropna()
        if len(others) > 0:
            averages[ticker] = float(others.mean())
    return dict(sorted(averages.items(), key=lambda item: item[1]))
