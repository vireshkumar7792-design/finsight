"""Holding / Portfolio validation rules."""

from datetime import date, timedelta

import pytest
from pydantic import ValidationError

from finsight.schemas.portfolio import AssetType, Holding, Portfolio


def test_holding_normalizes_and_defaults():
    h = Holding(ticker=" brk.b ", quantity="2.5", avg_buy_price="400", asset_type="ETF", currency="usd")
    assert h.ticker == "BRK.B"
    assert h.quantity == 2.5
    assert h.asset_type is AssetType.ETF
    assert h.currency == "USD"
    assert h.buy_date is None


def test_holding_defaults_asset_type_equity_and_usd():
    h = Holding(ticker="AAPL", quantity=1, avg_buy_price=1)
    assert h.asset_type is AssetType.EQUITY
    assert h.currency == "USD"


def test_cost_basis():
    assert Holding(ticker="AAPL", quantity=10, avg_buy_price=12.5).cost_basis == 125.0


@pytest.mark.parametrize(
    "field, value",
    [
        ("quantity", 0),
        ("quantity", -1),
        ("avg_buy_price", 0),
        ("ticker", "AAPL$"),
        ("ticker", "123"),
        ("asset_type", "stock"),
        ("currency", "EUR"),
        ("buy_date", date.today() + timedelta(days=1)),
    ],
)
def test_holding_rejects_bad_values(field, value):
    data = {"ticker": "AAPL", "quantity": 1, "avg_buy_price": 100, field: value}
    with pytest.raises(ValidationError):
        Holding(**data)


def test_holding_rejects_unknown_field():
    with pytest.raises(ValidationError):
        Holding(ticker="AAPL", quantity=1, avg_buy_price=1, sector="Tech")


def test_portfolio_rejects_duplicates_and_empty():
    h = Holding(ticker="AAPL", quantity=1, avg_buy_price=1)
    with pytest.raises(ValidationError, match="duplicate tickers: AAPL"):
        Portfolio(holdings=[h, h])
    with pytest.raises(ValidationError):
        Portfolio(holdings=[])


def test_portfolio_tickers(two_stock_portfolio):
    assert two_stock_portfolio.tickers == ["AAPL", "MSFT"]


def test_cash_must_be_priced_at_one():
    h = Holding(ticker="CASH", quantity=2500, avg_buy_price=1, asset_type="cash")
    assert h.is_cash and h.cost_basis == 2500
    with pytest.raises(ValidationError, match="cash must have avg_buy_price 1.0"):
        Holding(ticker="CASH", quantity=2500, avg_buy_price=1.05, asset_type="cash")


def test_market_tickers_and_get():
    p = Portfolio(holdings=[
        Holding(ticker="AAPL", quantity=1, avg_buy_price=1),
        Holding(ticker="CASH", quantity=10, avg_buy_price=1, asset_type="cash"),
    ])
    assert p.market_tickers == ["AAPL"]
    assert p.get(" aapl ").ticker == "AAPL"
    assert p.get("MSFT") is None
