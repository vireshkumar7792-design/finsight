"""Market data cache and PortfolioContext, using the fake market from conftest.py."""

import json
from datetime import date, timedelta
from pathlib import Path

import pytest

from finsight.portfolio import market
from finsight.portfolio.context import load_context

# =========================
# Market data cache
# =========================


def test_price_history_downloads_once_per_day(fake_market, tmp_path):
    first = market.get_price_history(["AAA", "BBB"], str(tmp_path))
    second = market.get_price_history(["AAA", "BBB"], str(tmp_path))
    assert fake_market["closes"] == 1
    assert second.equals(first)
    assert second.index.dtype.kind == "M"  # dates come back as dates, not text


def test_sectors_are_cached_but_unknown_is_not(fake_market, tmp_path):
    market.get_sectors(["AAA", "ZZZ"], str(tmp_path))  # ZZZ has no fake sector -> "Unknown"
    sectors = market.get_sectors(["AAA", "ZZZ"], str(tmp_path))
    assert sectors == {"AAA": "Technology", "ZZZ": "Unknown"}
    # AAA was fetched once; ZZZ was asked again because "Unknown" is never cached.
    assert fake_market["sectors"] == 3


def test_old_sector_entries_are_refreshed(fake_market, tmp_path):
    old_day = (date.today() - timedelta(days=31)).isoformat()
    cache_file = Path(tmp_path) / "sectors.json"
    cache_file.write_text(json.dumps({"AAA": {"sector": "Old", "fetched": old_day}}))
    assert market.get_sectors(["AAA"], str(tmp_path)) == {"AAA": "Technology"}


# =========================
# PortfolioContext
# =========================


def test_context_prices_weights_and_cash(fake_market, fake_csv, test_settings):
    context = load_context(fake_csv, test_settings)

    assert list(context.history.columns) == ["AAA", "BBB", "CASH"]
    assert context.prices == {"AAA": 120.0, "BBB": 45.0, "CASH": 1.0}
    # Worth 1200 / 900 / 400 out of 2500.
    assert context.weights["AAA"] == pytest.approx(0.48)
    assert context.weights["BBB"] == pytest.approx(0.36)
    assert context.weights["CASH"] == pytest.approx(0.16)
    assert context.sectors == {"AAA": "Technology", "BBB": "ETF / Fund", "CASH": "Cash"}
    assert context.as_of == "2024-01-04"


def test_context_returns(fake_market, fake_csv, test_settings):
    context = load_context(fake_csv, test_settings)
    assert (context.returns["CASH"] == 0).all()
    # Day 1: AAA +10%, BBB 0%, cash 0% -> 0.48 * 0.10 = 0.048
    assert context.portfolio_returns.iloc[0] == pytest.approx(0.048)
    # SPY 400 -> 404 = +1%
    assert context.benchmark_returns.iloc[0] == pytest.approx(0.01)
