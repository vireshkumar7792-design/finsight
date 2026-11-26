"""PortfolioContext: loads once, handles cash, never sends cash to the market data API."""

import pandas as pd
import pytest

from finsight.context import CASH_SECTOR_LABEL, PortfolioContext
from finsight.schemas.portfolio import Holding, Portfolio


def test_one_batched_download_without_cash(fake_ctx, fake_market):
    assert fake_market.history_calls == [(["AAA", "BBB", "SPY"], "1y")]
    assert fake_market.sector_calls == [["AAA", "BBB"]]


def test_cash_is_constant_one_dollar(fake_ctx):
    assert (fake_ctx.history["CASH"] == 1.0).all()
    assert fake_ctx.prices["CASH"] == 1.0
    assert (fake_ctx.returns["CASH"] == 0.0).all()
    assert fake_ctx.sectors["CASH"] == CASH_SECTOR_LABEL


def test_prices_weights_and_summary(fake_ctx):
    # See conftest: MV 1200 / 900 / 400 = 2500
    assert fake_ctx.prices == {"AAA": 120.0, "BBB": 45.0, "CASH": 1.0}
    assert fake_ctx.weights == pytest.approx({"AAA": 0.48, "BBB": 0.36, "CASH": 0.16})
    assert fake_ctx.summary.total_market_value == pytest.approx(2500)
    assert list(fake_ctx.history.columns) == ["AAA", "BBB", "CASH"]  # CSV order
    assert fake_ctx.as_of == "2024-01-04"


def test_portfolio_returns_day_one(fake_ctx):
    # Day 1: AAA +10%, BBB 0%, CASH 0% -> 0.48*0.10 = 0.048
    assert fake_ctx.portfolio_returns.iloc[0] == pytest.approx(0.048)


def test_benchmark_is_loaded(fake_ctx):
    assert fake_ctx.benchmark_ticker == "SPY"
    assert fake_ctx.benchmark.iloc[-1] == 408.0


def test_all_cash_portfolio(fake_market, test_settings):
    p = Portfolio(holdings=[Holding(ticker="CASH", quantity=1000, avg_buy_price=1, asset_type="cash")])
    ctx = PortfolioContext.load(p, test_settings, fetch_history=fake_market.history, fetch_sectors=fake_market.sectors)
    assert fake_market.history_calls == [(["SPY"], "1y")]
    assert fake_market.sector_calls == []
    assert ctx.weights == {"CASH": 1.0}
    assert isinstance(ctx.history, pd.DataFrame) and len(ctx.history) == 4


def test_from_csv(tmp_path, fake_market, test_settings):
    f = tmp_path / "p.csv"
    f.write_text("ticker,quantity,avg_buy_price,asset_type\nAAA,1,100,equity\n")
    ctx = PortfolioContext.from_csv(f, test_settings, fetch_history=fake_market.history, fetch_sectors=fake_market.sectors)
    assert ctx.portfolio.tickers == ["AAA"]
