"""SEC EDGAR client: ticker -> CIK -> latest 10-K -> primary document (HTML) on disk.

SEC rules (sec.gov/os/accessing-edgar-data): declare a User-Agent with contact info, and stay
under 10 requests/second. The User-Agent comes from SEC_USER_AGENT in .env (a SecretStr in
Settings); it is only ever placed in the request header, never printed, logged or stored.

Requests per new ticker: ~2 (submissions JSON + primary document); the ticker->CIK map is one
shared file cached on disk for a week.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nodash}/{doc}"
TICKER_MAP_TTL_SECONDS = 7 * 24 * 3600


class MissingUserAgentError(RuntimeError):
    """SEC_USER_AGENT is required by the SEC for automated access."""


class FilingNotFoundError(LookupError):
    pass


@dataclass(frozen=True)
class FilingRef:
    ticker: str
    cik: int
    company: str
    form: str
    accession: str  # with dashes: 0001045810-26-000021
    primary_document: str
    filing_date: str  # YYYY-MM-DD
    report_date: str  # period end, YYYY-MM-DD
    url: str

    @property
    def fiscal_year(self) -> int:
        """Fiscal year = year of the period end date (NVDA's FY2026 ends Jan 2026, AAPL's Sept)."""
        return int(self.report_date[:4])


class EdgarClient:
    def __init__(
        self,
        user_agent: str | None,
        data_dir: str | Path = "data/filings",
        *,
        requests_per_second: float = 4.0,  # SEC allows 10/s; we stay well below
        http: httpx.Client | None = None,
    ):
        if not user_agent or not user_agent.strip():
            raise MissingUserAgentError(
                "SEC_USER_AGENT is not set in .env. The SEC requires a User-Agent with contact "
                'info, e.g. SEC_USER_AGENT="FinSight research you@example.com".'
            )
        self.dir = Path(data_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._http = http or httpx.Client(timeout=30.0, follow_redirects=True)
        self._headers = {"User-Agent": user_agent.strip(), "Accept-Encoding": "gzip, deflate"}
        self._min_interval = 1.0 / requests_per_second
        self._last = 0.0
        self.requests_made = 0

    # ---- HTTP with throttle --------------------------------------------------------------------

    def _get(self, url: str) -> httpx.Response:
        wait = self._min_interval - (time.monotonic() - self._last)  # stay under the SEC's rate limit
        if wait > 0:
            time.sleep(wait)
        self._last = time.monotonic()
        resp = self._http.get(url, headers=self._headers)
        self.requests_made += 1
        resp.raise_for_status()
        return resp

    # ---- lookups -------------------------------------------------------------------------------

    def _ticker_map(self) -> dict[str, dict]:
        path = self.dir / "company_tickers.json"
        if not path.exists() or time.time() - path.stat().st_mtime > TICKER_MAP_TTL_SECONDS:
            path.write_text(self._get(TICKERS_URL).text, encoding="utf-8")
        raw = json.loads(path.read_text(encoding="utf-8"))  # {"0": {"cik_str", "ticker", "title"}, ...}
        return {row["ticker"].upper(): row for row in raw.values()}

    def lookup_cik(self, ticker: str) -> tuple[int, str]:
        row = self._ticker_map().get(ticker.strip().upper().replace(".", "-"))
        if row is None:
            raise FilingNotFoundError(f"{ticker}: not in SEC's ticker list (ETFs and funds don't file 10-Ks)")
        return int(row["cik_str"]), row["title"]

    def latest_filing(self, ticker: str, form: str = "10-K") -> FilingRef:
        cik, company = self.lookup_cik(ticker)
        recent = self._get(SUBMISSIONS_URL.format(cik=cik)).json()["filings"]["recent"]
        for i, f in enumerate(recent["form"]):  # newest first; exact match skips "10-K/A"
            if f == form:
                acc, doc = recent["accessionNumber"][i], recent["primaryDocument"][i]
                return FilingRef(
                    ticker=ticker.upper(), cik=cik, company=company, form=form, accession=acc,
                    primary_document=doc, filing_date=recent["filingDate"][i],
                    report_date=recent["reportDate"][i] or recent["filingDate"][i],
                    url=ARCHIVE_URL.format(cik=cik, acc_nodash=acc.replace("-", ""), doc=doc),
                )
        raise FilingNotFoundError(f"{ticker}: no {form} in recent filings")

    # ---- download --------------------------------------------------------------------------------

    def download(self, ref: FilingRef) -> Path:
        """Save the primary document (skip if already on disk) plus a small JSON of its metadata."""
        folder = self.dir / ref.ticker
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{ref.accession}_{ref.primary_document}"
        if not path.exists():
            path.write_bytes(self._get(ref.url).content)
        path.with_suffix(".json").write_text(json.dumps(asdict(ref), indent=1), encoding="utf-8")
        return path
