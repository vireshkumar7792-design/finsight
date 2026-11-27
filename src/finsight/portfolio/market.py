"""Market data from yfinance (prices and sectors), cached on disk so each day downloads once."""

import hashlib
import io
import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

HISTORY_PERIOD = "1y"
ETF_SECTOR = "ETF / Fund"
UNKNOWN_SECTOR = "Unknown"
SECTOR_CACHE_DAYS = 30


# =========================
# Downloads (the only network code)
# =========================


def download_closes(tickers: list[str], period: str = HISTORY_PERIOD) -> pd.DataFrame:
    """Daily closing prices, adjusted for splits and dividends. One column per ticker."""
    raw = yf.download(tickers, period=period, auto_adjust=True, progress=False)
    if raw is None or raw.empty:
        raise ValueError(f"no price data returned for: {', '.join(tickers)}")
    closes = raw["Close"]
    missing = [ticker for ticker in tickers if ticker not in closes.columns or closes[ticker].dropna().empty]
    if missing:
        raise ValueError(f"no price data for: {', '.join(missing)} (typo or delisted?)")
    return closes[tickers].sort_index()


def download_sector(ticker: str) -> str:
    try:
        info = yf.Ticker(ticker).info or {}
    except Exception:
        # A failed lookup should not stop the analysis; "Unknown" is shown to the user.
        return UNKNOWN_SECTOR
    if info.get("quoteType") in ("ETF", "MUTUALFUND"):
        return ETF_SECTOR
    return info.get("sector") or UNKNOWN_SECTOR


# =========================
# Price history (cached per day)
# =========================


def get_price_history(tickers: list[str], cache_dir: str) -> pd.DataFrame:
    """Same-day cache: every run on the same day sees exactly the same prices."""
    path = history_cache_path(tickers, cache_dir)
    if path.exists():
        history = pd.read_json(io.StringIO(path.read_text(encoding="utf-8")), orient="split")
        history.index = pd.to_datetime(history.index)
        # read_json turns 100.0 into the integer 100; prices must stay floats.
        return history[tickers].astype(float)

    history = download_closes(tickers)
    path.parent.mkdir(parents=True, exist_ok=True)
    # JSON, not pickle: loading a pickle file can run code.
    path.write_text(history.to_json(orient="split", date_format="iso"), encoding="utf-8")
    return history


def history_cache_path(tickers: list[str], cache_dir: str) -> Path:
    # A short hash keeps the file name small even for a 30-ticker portfolio.
    ticker_key = hashlib.sha256(",".join(sorted(tickers)).encode()).hexdigest()[:16]
    file_name = f"history_{date.today().isoformat()}_{HISTORY_PERIOD}_{ticker_key}.json"
    return Path(cache_dir) / file_name


# =========================
# Sectors (cached for 30 days)
# =========================


def get_sectors(tickers: list[str], cache_dir: str) -> dict[str, str]:
    path = Path(cache_dir) / "sectors.json"
    cache = {}
    if path.exists():
        cache = json.loads(path.read_text(encoding="utf-8"))

    oldest_allowed = (date.today() - timedelta(days=SECTOR_CACHE_DAYS)).isoformat()
    sectors = {}
    cache_changed = False
    for ticker in tickers:
        entry = cache.get(ticker)
        if entry and entry["fetched"] >= oldest_allowed:
            sectors[ticker] = entry["sector"]
            continue
        sector = download_sector(ticker)
        sectors[ticker] = sector
        # Don't remember "Unknown": it is usually a temporary network failure.
        if sector != UNKNOWN_SECTOR:
            cache[ticker] = {"sector": sector, "fetched": date.today().isoformat()}
            cache_changed = True

    if cache_changed:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cache, indent=1, sort_keys=True), encoding="utf-8")
    return sectors
