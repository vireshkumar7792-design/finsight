"""Market data access via yfinance: current prices, price history, sector info.

This is the ONLY module that touches the network. The raw yfinance calls are isolated in
two tiny functions (`_download`, `_ticker_info`) so tests can monkeypatch them and exercise
the reshaping/validation logic offline.

yfinance notes (v1.x): `yf.download` returns MultiIndex columns (Price, Ticker) even for a
single ticker; with auto_adjust=True, "Close" is split- and dividend-adjusted.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
import yfinance as yf

ETF_SECTOR_LABEL = "ETF / Fund"


def _download(tickers: list[str], period: str) -> pd.DataFrame:
    return yf.download(tickers, period=period, auto_adjust=True, progress=False, threads=True)


def _ticker_info(ticker: str) -> dict:
    return yf.Ticker(ticker).info or {}


def _normalize(tickers: Sequence[str]) -> list[str]:
    out = list(dict.fromkeys(t.strip().upper() for t in tickers))  # dedupe, keep order
    if not out:
        raise ValueError("no tickers given")
    return out


def get_price_history(tickers: Sequence[str], period: str = "1y") -> pd.DataFrame:
    """Adjusted daily close prices: index = date, one column per ticker.

    Raises ValueError if any ticker returned no data (e.g. a typo or delisted symbol).
    """
    tickers = _normalize(tickers)
    raw = _download(tickers, period)
    if raw is None or raw.empty or "Close" not in raw.columns.get_level_values(0):
        raise ValueError(f"no price data returned for {', '.join(tickers)}")
    close = raw["Close"]
    missing = [t for t in tickers if t not in close.columns or close[t].dropna().empty]
    if missing:
        raise ValueError(f"no price data for: {', '.join(missing)}")
    return close[tickers].sort_index()


def get_current_prices(tickers: Sequence[str]) -> dict[str, float]:
    """Latest available adjusted close per ticker (one batched request, last 5 days).

    Using the last close rather than a live quote keeps results reproducible within a day
    and covers weekends/holidays.
    """
    history = get_price_history(tickers, period="5d")
    return {t: float(history[t].dropna().iloc[-1]) for t in history.columns}


def get_sectors(tickers: Sequence[str]) -> dict[str, str]:
    """Sector per ticker. ETFs/funds have no sector -> ETF_SECTOR_LABEL; unknown -> "Unknown"."""
    out: dict[str, str] = {}
    for t in _normalize(tickers):
        try:
            info = _ticker_info(t)
        except Exception:  # network/parse errors shouldn't kill the whole analysis
            info = {}
        if info.get("quoteType") in {"ETF", "MUTUALFUND"}:
            out[t] = ETF_SECTOR_LABEL
        else:
            out[t] = info.get("sector") or "Unknown"
    return out
