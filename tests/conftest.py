"""Shared pytest fixtures."""

from pathlib import Path

import pandas as pd
import pytest

from finsight.config import Settings
from finsight.context import PortfolioContext
from finsight.schemas.portfolio import Holding, Portfolio

SAMPLES_DIR = Path(__file__).resolve().parents[1] / "data" / "samples"


@pytest.fixture
def samples_dir() -> Path:
    return SAMPLES_DIR


@pytest.fixture
def two_stock_portfolio() -> Portfolio:
    """AAPL 10 @ $100 and MSFT 5 @ $200: cost basis $1,000 each, $2,000 total."""
    return Portfolio(
        holdings=[
            Holding(ticker="AAPL", quantity=10, avg_buy_price=100, asset_type="equity"),
            Holding(ticker="MSFT", quantity=5, avg_buy_price=200, asset_type="equity"),
        ]
    )


@pytest.fixture
def two_stock_prices() -> dict[str, float]:
    """AAPL now $150 (+50%), MSFT now $180 (-10%)."""
    return {"AAPL": 150.0, "MSFT": 180.0}


# ---- fake market data for PortfolioContext (offline) ------------------------------------
#
# Holdings: AAA 10 @ $100 (equity), BBB 20 @ $50 (etf), CASH $400.
# Last closes: AAA 120, BBB 45  ->  MV: AAA 1200, BBB 900, CASH 400 = 2500
# Weights: 48% / 36% / 16%.  Cost: 1000 + 1000 + 400 = 2400.  P&L: +200, -100, 0 = +100 (+4.17%)
FAKE_CLOSES = {
    "AAA": [100.0, 110.0, 99.0, 120.0],
    "BBB": [50.0, 50.0, 55.0, 45.0],
    "SPY": [400.0, 404.0, 396.0, 408.0],
}
FAKE_SECTORS = {"AAA": "Technology", "BBB": "ETF / Fund"}


class FakeMarket:
    """Records what the context asked for, returns canned data shaped like market_data output."""

    def __init__(self):
        self.history_calls: list[tuple[list[str], str]] = []
        self.sector_calls: list[list[str]] = []

    def history(self, tickers, period):
        tickers = list(tickers)
        self.history_calls.append((tickers, period))
        idx = pd.date_range("2024-01-01", periods=4, name="Date")
        return pd.DataFrame({t: FAKE_CLOSES[t] for t in tickers}, index=idx)

    def sectors(self, tickers):
        self.sector_calls.append(list(tickers))
        return {t: FAKE_SECTORS[t] for t in tickers}


@pytest.fixture
def fake_portfolio() -> Portfolio:
    return Portfolio(
        holdings=[
            Holding(ticker="AAA", quantity=10, avg_buy_price=100, asset_type="equity"),
            Holding(ticker="BBB", quantity=20, avg_buy_price=50, asset_type="etf"),
            Holding(ticker="CASH", quantity=400, avg_buy_price=1, asset_type="cash"),
        ]
    )


@pytest.fixture
def fake_market() -> FakeMarket:
    return FakeMarket()


@pytest.fixture
def test_settings() -> Settings:
    """Settings with dummy keys: never reads the real .env values."""
    return Settings(groq_api_key="test-groq", openrouter_api_key="test-or")


@pytest.fixture
def fake_ctx(fake_portfolio, fake_market, test_settings) -> PortfolioContext:
    return PortfolioContext.load(
        fake_portfolio, test_settings, fetch_history=fake_market.history, fetch_sectors=fake_market.sectors
    )
