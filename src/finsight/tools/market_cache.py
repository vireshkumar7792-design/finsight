"""On-disk cache for market data, so repeated runs don't hammer Yahoo Finance.

- Price history: cached per calendar day, keyed by (date, period, tickers). Every run on the
  same day sees identical prices (reproducible, and same-day resumes stay consistent).
  Intraday moves are ignored until the next day: fine for portfolio analysis.
- Sectors: cached per ticker for SECTOR_TTL_DAYS. "Unknown" (usually a failed lookup) is not
  cached, so a transient network error isn't remembered.
Stored as JSON (never pickle: loading a pickle can execute code).
"""

from __future__ import annotations

import hashlib
import io
import json
from collections.abc import Callable, Sequence
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from finsight.tools import market_data

SECTOR_TTL_DAYS = 30
Today = Callable[[], date]


class MarketDataCache:
    def __init__(
        self,
        cache_dir: str | Path,
        *,
        today: Today = date.today,
        fetch_history: Callable[[Sequence[str], str], pd.DataFrame] = market_data.get_price_history,
        fetch_sectors: Callable[[Sequence[str]], dict[str, str]] = market_data.get_sectors,
    ):
        self.dir = Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._today = today
        self._fetch_history = fetch_history
        self._fetch_sectors = fetch_sectors
        self.sectors_path = self.dir / "sectors.json"
        self.history_downloads = 0  # for tests/diagnostics

    # ---- price history ----------------------------------------------------------------------

    def _history_path(self, tickers: list[str], period: str) -> Path:
        key = hashlib.sha256(",".join(sorted(tickers)).encode()).hexdigest()[:16]
        return self.dir / f"history_{self._today().isoformat()}_{period}_{key}.json"

    def get_price_history(self, tickers: Sequence[str], period: str = "1y") -> pd.DataFrame:
        tickers = list(dict.fromkeys(t.strip().upper() for t in tickers))
        path = self._history_path(tickers, period)
        if path.exists():
            df = pd.read_json(io.StringIO(path.read_text(encoding="utf-8")), orient="split")
            df.index = pd.to_datetime(df.index)
            df.index.name = "Date"
            return df[tickers]
        df = self._fetch_history(tickers, period)
        self.history_downloads += 1
        path.write_text(df.to_json(orient="split", date_format="iso"), encoding="utf-8")
        return df

    # ---- sectors ---------------------------------------------------------------------------------

    def get_sectors(self, tickers: Sequence[str]) -> dict[str, str]:
        tickers = list(dict.fromkeys(t.strip().upper() for t in tickers))
        store: dict[str, dict[str, str]] = {}
        if self.sectors_path.exists():
            store = json.loads(self.sectors_path.read_text(encoding="utf-8"))
        cutoff = (self._today() - timedelta(days=SECTOR_TTL_DAYS)).isoformat()
        fresh = {t: e["sector"] for t, e in store.items() if e["fetched"] >= cutoff}

        missing = [t for t in tickers if t not in fresh]
        if missing:
            fetched = self._fetch_sectors(missing)
            for t, sector in fetched.items():
                fresh[t] = sector
                if sector != "Unknown":
                    store[t] = {"sector": sector, "fetched": self._today().isoformat()}
            self.sectors_path.write_text(json.dumps(store, indent=1, sort_keys=True), encoding="utf-8")
        return {t: fresh.get(t, "Unknown") for t in tickers}

    def clear(self) -> None:
        for f in self.dir.glob("*.json"):
            f.unlink()
