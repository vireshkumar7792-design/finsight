"""Risk math: volatility, beta, Sharpe, historical VaR, max drawdown, concentration (HHI), correlation.

Pure numpy/pandas functions over return or price series; data is passed in, never fetched.

Conventions (stated once, used everywhere):
- Returns are simple periodic returns as fractions (0.01 == +1%), daily by default.
- Annualization uses `periods_per_year` (252 trading days by default).
- Sample statistics use ddof=1.
- Losses (VaR, max drawdown) are returned as POSITIVE magnitudes: 0.25 means a 25% loss.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def _clean(returns: pd.Series, min_obs: int = 2) -> pd.Series:
    r = pd.Series(returns, dtype=float).dropna()
    if len(r) < min_obs:
        raise ValueError(f"need at least {min_obs} return observations, got {len(r)}")
    return r


def simple_returns(prices: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Period-over-period simple returns: p_t / p_{t-1} - 1. Drops the first (undefined) row."""
    return prices.pct_change().iloc[1:]


def annualized_volatility(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    """Sample standard deviation of returns, scaled by sqrt(periods_per_year)."""
    r = _clean(returns)
    return float(r.std(ddof=1) * np.sqrt(periods_per_year))


def sharpe_ratio(
    returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = TRADING_DAYS
) -> float:
    """Annualized Sharpe: mean excess return / std of returns * sqrt(periods_per_year).

    `risk_free_rate` is ANNUAL and is converted to a per-period rate.
    """
    r = _clean(returns)
    sd = r.std(ddof=1)
    if sd == 0:
        raise ValueError("returns have zero volatility; Sharpe ratio is undefined")
    excess_mean = r.mean() - risk_free_rate / periods_per_year
    return float(excess_mean / sd * np.sqrt(periods_per_year))


def beta(asset_returns: pd.Series, market_returns: pd.Series) -> float:
    """Cov(asset, market) / Var(market), on dates both series have."""
    joined = pd.concat([asset_returns, market_returns], axis=1, join="inner").dropna()
    if len(joined) < 2:
        raise ValueError(f"need at least 2 overlapping observations, got {len(joined)}")
    a, m = joined.iloc[:, 0], joined.iloc[:, 1]
    var_m = m.var(ddof=1)
    if var_m == 0:
        raise ValueError("market returns have zero variance; beta is undefined")
    return float(a.cov(m, ddof=1) / var_m)


def historical_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """One-period historical Value at Risk as a positive loss fraction.

    The (1 - confidence) quantile of returns using linear interpolation (numpy default),
    negated. 0.03 means: on (confidence)% of periods, the loss was no worse than 3%.
    """
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    r = _clean(returns)
    return float(-np.quantile(r.to_numpy(), 1 - confidence))


def max_drawdown(prices: pd.Series) -> float:
    """Largest peak-to-trough decline of a price (or wealth) series, as a positive fraction."""
    p = _clean(prices, min_obs=1)
    if (p <= 0).any():
        raise ValueError("prices must be positive")
    drawdowns = p / p.cummax() - 1
    return float(-drawdowns.min())


def herfindahl_index(weights: Mapping[str, float]) -> float:
    """Concentration: sum of squared weights. 1/N for equal weights, 1.0 for a single position."""
    w = np.array(list(weights.values()), dtype=float)
    if len(w) == 0:
        raise ValueError("weights are empty")
    if not np.isclose(w.sum(), 1.0):
        raise ValueError(f"weights must sum to 1, got {w.sum():.6f}")
    return float((w**2).sum())


def effective_number_of_positions(weights: Mapping[str, float]) -> float:
    """1 / HHI: how many equal-weight positions the portfolio 'behaves like'."""
    return 1.0 / herfindahl_index(weights)


def max_weight(weights: Mapping[str, float]) -> tuple[str, float]:
    """The largest position as (ticker, weight)."""
    if not weights:
        raise ValueError("weights are empty")
    ticker = max(weights, key=weights.__getitem__)
    return ticker, float(weights[ticker])


def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation between columns (tickers) of a returns DataFrame."""
    if returns.shape[1] < 2:
        raise ValueError("need at least 2 assets for a correlation matrix")
    return returns.dropna().corr()


def portfolio_returns(returns: pd.DataFrame, weights: Mapping[str, float]) -> pd.Series:
    """Weighted sum of asset returns, assuming constant weights (rebalanced each period).

    This is the standard simplification for risk estimates from current weights.
    """
    missing = [t for t in weights if t not in returns.columns]
    if missing:
        raise ValueError(f"no return history for: {', '.join(missing)}")
    w = pd.Series(weights, dtype=float)
    return returns[w.index].dropna().mul(w, axis=1).sum(axis=1)
