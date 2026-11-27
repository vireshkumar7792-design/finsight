"""Same-day market data cache (offline: fake fetchers and a fake 'today')."""

from datetime import date, timedelta

import pandas as pd

from finsight.config import Settings
from finsight.context import PortfolioContext
from finsight.tools import market_data as md
from finsight.tools.market_cache import SECTOR_TTL_DAYS, MarketDataCache


class Day:
    def __init__(self, d=date(2026, 11, 26)):
        self.d = d

    def __call__(self):
        return self.d


def history(tickers, period):
    idx = pd.date_range("2024-01-01", periods=3, name="Date")
    return pd.DataFrame({t: [1.5 + i, 2.25 + i, 3.125 + i] for i, t in enumerate(tickers)}, index=idx)


def test_history_roundtrip_is_exact(tmp_path):
    c = MarketDataCache(tmp_path, today=Day(), fetch_history=history)
    first = c.get_price_history(["AAA", "BBB"], "1y")
    again = c.get_price_history(["AAA", "BBB"], "1y")
    pd.testing.assert_frame_equal(first, again, check_freq=False)
    assert c.history_downloads == 1
    assert isinstance(again.index, pd.DatetimeIndex) and again.index.name == "Date"


def test_history_key_ignores_order_but_respects_columns(tmp_path):
    c = MarketDataCache(tmp_path, today=Day(), fetch_history=history)
    c.get_price_history(["AAA", "BBB"], "1y")
    out = c.get_price_history(["bbb", "aaa"], "1y")  # same set, different order/case -> cache hit
    assert c.history_downloads == 1
    assert list(out.columns) == ["BBB", "AAA"]  # caller's order


def test_history_redownloads_next_day_or_new_period(tmp_path):
    day = Day()
    c = MarketDataCache(tmp_path, today=day, fetch_history=history)
    c.get_price_history(["AAA"], "1y")
    c.get_price_history(["AAA"], "6mo")
    assert c.history_downloads == 2
    day.d += timedelta(days=1)
    c.get_price_history(["AAA"], "1y")
    assert c.history_downloads == 3


def test_sectors_cached_with_ttl_and_unknown_not_cached(tmp_path):
    calls = []

    def sectors(ts):
        calls.append(list(ts))
        return {t: ("Unknown" if t == "ZZZ" else "Technology") for t in ts}

    day = Day()
    c = MarketDataCache(tmp_path, today=day, fetch_sectors=sectors)
    assert c.get_sectors(["AAA", "ZZZ"]) == {"AAA": "Technology", "ZZZ": "Unknown"}
    assert c.get_sectors(["AAA", "ZZZ"]) == {"AAA": "Technology", "ZZZ": "Unknown"}
    assert calls == [["AAA", "ZZZ"], ["ZZZ"]]  # AAA cached; Unknown retried
    day.d += timedelta(days=SECTOR_TTL_DAYS + 1)
    c.get_sectors(["AAA"])
    assert calls[-1] == ["AAA"]  # expired -> looked up again


def test_context_uses_cache_by_default(tmp_path, monkeypatch, fake_portfolio):
    downloads = []

    def fake_download(tickers, period):  # lowest-level network call in market_data
        downloads.append(list(tickers))
        df = history(tickers, period)
        df.columns = pd.MultiIndex.from_product([["Close"], df.columns], names=["Price", "Ticker"])
        return df

    monkeypatch.setattr(md, "_download", fake_download)
    monkeypatch.setattr(md, "_ticker_info", lambda t: {"quoteType": "EQUITY", "sector": "Technology"})
    s = Settings(groq_api_key="k", openrouter_api_key="k", market_cache_dir=str(tmp_path))
    a = PortfolioContext.load(fake_portfolio, s)
    b = PortfolioContext.load(fake_portfolio, s)
    assert len(downloads) == 1  # second load served from the same-day cache
    assert a.prices == b.prices and a.as_of == b.as_of


def test_clear(tmp_path):
    c = MarketDataCache(tmp_path, today=Day(), fetch_history=history)
    c.get_price_history(["AAA"], "1y")
    c.clear()
    assert not list(tmp_path.glob("*.json"))
    c.get_price_history(["AAA"], "1y")
    assert c.history_downloads == 2  # cleared -> downloaded again
