"""PortfolioContext: everything an agent run needs, loaded once and cached.

Built once per run (CSV + one batched price download + sector lookup), then handed to every
tool call via LangChain's runtime context, so tools never re-download and the LLM never
has to pass portfolio data around as arguments.

Fetchers are injectable: production uses a same-day disk cache over yfinance
(tools/market_cache.py), tests pass fakes.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import cached_property
from os import PathLike

import pandas as pd

from finsight.config import Settings, get_settings
from finsight.schemas.metrics import PortfolioSummary
from finsight.schemas.portfolio import CASH_PRICE, Portfolio
from finsight.tools import portfolio_math, risk_math
from finsight.tools.csv_loader import load_portfolio_csv
from finsight.tools.market_cache import MarketDataCache

CASH_SECTOR_LABEL = "Cash"

HistoryFetcher = Callable[[Sequence[str], str], pd.DataFrame]
SectorFetcher = Callable[[Sequence[str]], dict[str, str]]


@dataclass(frozen=True, eq=False)  # eq=False: DataFrame fields can't be compared/hashed by value
class PortfolioContext:
    portfolio: Portfolio
    history: pd.DataFrame  # adjusted daily closes, one column per holding (cash = constant 1.0)
    benchmark: pd.Series  # adjusted daily closes of the benchmark (e.g. SPY)
    sectors: dict[str, str]  # ticker -> sector label
    benchmark_ticker: str
    history_period: str
    risk_free_rate: float  # annual fraction
    var_confidence: float

    # ---- construction -------------------------------------------------------------------

    @classmethod
    def load(
        cls,
        portfolio: Portfolio,
        settings: Settings | None = None,
        *,
        fetch_history: HistoryFetcher | None = None,
        fetch_sectors: SectorFetcher | None = None,
    ) -> PortfolioContext:
        """Fetch market data for `portfolio` (one batched history download incl. benchmark).

        By default data comes from the same-day disk cache over yfinance (delete
        data/cache/market/ to force a fresh download). Tests pass their own fetchers.
        """
        s = settings or get_settings()
        if fetch_history is None or fetch_sectors is None:
            cache = MarketDataCache(s.market_cache_dir)
            fetch_history = fetch_history or cache.get_price_history
            fetch_sectors = fetch_sectors or cache.get_sectors
        bench = s.benchmark_ticker.upper()
        market = portfolio.market_tickers  # cash is never sent to yfinance

        raw = fetch_history([*market, bench], s.history_period)
        history = raw[market].copy() if market else pd.DataFrame(index=raw.index)
        for h in portfolio.holdings:
            if h.is_cash:
                history[h.ticker] = CASH_PRICE  # constant price series -> zero returns
        history = history[portfolio.tickers]  # CSV order

        sectors = dict(fetch_sectors(market)) if market else {}
        sectors.update({h.ticker: CASH_SECTOR_LABEL for h in portfolio.holdings if h.is_cash})

        return cls(
            portfolio=portfolio,
            history=history,
            benchmark=raw[bench],
            sectors=sectors,
            benchmark_ticker=bench,
            history_period=s.history_period,
            risk_free_rate=s.risk_free_rate,
            var_confidence=s.var_confidence,
        )

    @classmethod
    def from_csv(cls, path: str | PathLike[str], settings: Settings | None = None, **fetchers) -> PortfolioContext:
        return cls.load(load_portfolio_csv(path), settings, **fetchers)

    # ---- derived data (computed on first use, then cached) ---------------------------

    @cached_property
    def as_of(self) -> str:
        """Date of the latest close used for 'current' prices (YYYY-MM-DD)."""
        return pd.Timestamp(self.history.index.max()).date().isoformat()

    @cached_property
    def prices(self) -> dict[str, float]:
        """Latest available close per holding."""
        return {t: float(self.history[t].dropna().iloc[-1]) for t in self.history.columns}

    @cached_property
    def weights(self) -> dict[str, float]:
        return portfolio_math.portfolio_weights(self.portfolio, self.prices)

    @cached_property
    def summary(self) -> PortfolioSummary:
        return portfolio_math.portfolio_summary(self.portfolio, self.prices)

    @cached_property
    def returns(self) -> pd.DataFrame:
        """Daily simple returns per holding."""
        return risk_math.simple_returns(self.history)

    @cached_property
    def benchmark_returns(self) -> pd.Series:
        return risk_math.simple_returns(self.benchmark)

    @cached_property
    def portfolio_returns(self) -> pd.Series:
        """Daily returns of the portfolio at current weights."""
        return risk_math.portfolio_returns(self.returns, self.weights)
