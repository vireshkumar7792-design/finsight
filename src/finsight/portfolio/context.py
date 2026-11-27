"""PortfolioContext: the CSV plus its market data, loaded once per run and shared by all tools."""

from dataclasses import dataclass

import pandas as pd

from finsight.config import Settings, get_settings
from finsight.portfolio import calculations, market
from finsight.portfolio.loader import load_portfolio_csv
from finsight.schemas import CASH_PRICE, Portfolio

BENCHMARK_TICKER = "SPY"
CASH_SECTOR = "Cash"


# =========================
# The context object
# =========================


@dataclass
class PortfolioContext:
    csv_path: str
    portfolio: Portfolio
    history: pd.DataFrame  # daily closes, one column per holding (cash is always 1.0)
    benchmark: pd.Series  # daily closes of SPY
    sectors: dict[str, str]
    as_of: str  # date of the latest close, e.g. "2026-11-27"
    prices: dict[str, float]  # latest close per holding
    weights: dict[str, float]
    returns: pd.DataFrame  # daily returns per holding
    benchmark_returns: pd.Series
    portfolio_returns: pd.Series


# =========================
# Loading
# =========================


def load_context(csv_path: str, settings: Settings | None = None) -> PortfolioContext:
    """Validate the CSV, download (or read cached) market data, and pre-compute returns."""
    settings = settings or get_settings()
    portfolio = load_portfolio_csv(csv_path)

    # One download for all holdings plus the benchmark. dict.fromkeys removes duplicates
    # (a portfolio can hold SPY itself) and keeps the order.
    tickers_to_download = list(dict.fromkeys(portfolio.market_tickers + [BENCHMARK_TICKER]))
    downloaded = market.get_price_history(tickers_to_download, settings.market_cache_dir)

    history = build_history(portfolio, downloaded)
    sectors = build_sectors(portfolio, settings.market_cache_dir)

    prices = {}
    for ticker in history.columns:
        prices[ticker] = float(history[ticker].dropna().iloc[-1])

    positions = calculations.position_details(portfolio, prices)
    weights = {position["ticker"]: position["weight"] for position in positions}

    returns = calculations.daily_returns(history)
    benchmark = downloaded[BENCHMARK_TICKER]

    return PortfolioContext(
        csv_path=str(csv_path),
        portfolio=portfolio,
        history=history,
        benchmark=benchmark,
        sectors=sectors,
        as_of=history.index.max().date().isoformat(),
        prices=prices,
        weights=weights,
        returns=returns,
        benchmark_returns=calculations.daily_returns(benchmark),
        portfolio_returns=calculations.portfolio_returns(returns, weights),
    )


# =========================
# Helpers
# =========================


def build_history(portfolio: Portfolio, downloaded: pd.DataFrame) -> pd.DataFrame:
    """Price table in CSV order. Cash is never downloaded: it gets a flat $1 line."""
    history = pd.DataFrame(index=downloaded.index)
    for holding in portfolio.holdings:
        if holding.is_cash:
            history[holding.ticker] = CASH_PRICE
        else:
            history[holding.ticker] = downloaded[holding.ticker]
    return history


def build_sectors(portfolio: Portfolio, cache_dir: str) -> dict[str, str]:
    sectors = {}
    if portfolio.market_tickers:
        sectors = market.get_sectors(portfolio.market_tickers, cache_dir)
    for holding in portfolio.holdings:
        if holding.is_cash:
            sectors[holding.ticker] = CASH_SECTOR
    return sectors
