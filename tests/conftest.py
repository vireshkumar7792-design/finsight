"""Shared pytest fixtures."""

from pathlib import Path

import pytest

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
