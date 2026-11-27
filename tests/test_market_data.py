"""market_data tests. Offline tests monkeypatch the two raw yfinance wrappers with fake data
shaped exactly like real yfinance 1.x output (MultiIndex columns: Price x Ticker).
The `network` test hits Yahoo for real; run it with: uv run pytest -m network
"""

import pandas as pd
import pytest

from finsight.tools import market_data as md


def fake_download(closes: dict[str, list[float]]):
    """Build a yfinance-shaped DataFrame from {ticker: [closes]}."""
    idx = pd.date_range("2024-01-01", periods=len(next(iter(closes.values()))), name="Date")
    frames = {}
    for t, vals in closes.items():
        for field in ("Close", "High", "Low", "Open", "Volume"):
            frames[(field, t)] = vals
    df = pd.DataFrame(frames, index=idx)
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=["Price", "Ticker"])
    return lambda tickers, period: df


def test_price_history_reshapes_to_close_columns(monkeypatch):
    monkeypatch.setattr(md, "_download", fake_download({"AAPL": [1.0, 2.0], "SPY": [3.0, 4.0]}))
    h = md.get_price_history(["spy", "AAPL", "aapl"])  # normalized + deduped, order kept
    assert list(h.columns) == ["SPY", "AAPL"]
    assert h["AAPL"].tolist() == [1.0, 2.0]


def test_unknown_ticker_raises(monkeypatch):
    monkeypatch.setattr(md, "_download", fake_download({"AAPL": [1.0, 2.0], "ZZZZ": [None, None]}))
    with pytest.raises(ValueError, match="no price data for: ZZZZ"):
        md.get_price_history(["AAPL", "ZZZZ"])


def test_empty_download_raises(monkeypatch):
    monkeypatch.setattr(md, "_download", lambda tickers, period: pd.DataFrame())
    with pytest.raises(ValueError, match="no price data returned"):
        md.get_price_history(["AAPL"])


def test_no_tickers_raises():
    with pytest.raises(ValueError, match="no tickers"):
        md.get_price_history([])


def test_sectors(monkeypatch):
    infos = {
        "AAPL": {"quoteType": "EQUITY", "sector": "Technology"},
        "SPY": {"quoteType": "ETF"},
        "XYZ": {},
    }

    def fake_info(t):
        if t == "BOOM":
            raise ConnectionError("network down")
        return infos[t]

    monkeypatch.setattr(md, "_ticker_info", fake_info)
    assert md.get_sectors(["AAPL", "SPY", "XYZ", "BOOM"]) == {
        "AAPL": "Technology",
        "SPY": md.ETF_SECTOR_LABEL,
        "XYZ": "Unknown",
        "BOOM": "Unknown",
    }


@pytest.mark.network
def test_live_yfinance_smoke():
    """Guards against yfinance changing its output shape under us."""
    history = md.get_price_history(["AAPL", "SPY"], "5d")
    assert list(history.columns) == ["AAPL", "SPY"] and (history.dropna() > 0).all().all()
    assert md.get_sectors(["AAPL", "SPY"]) == {"AAPL": "Technology", "SPY": md.ETF_SECTOR_LABEL}
