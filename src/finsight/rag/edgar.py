"""Download a company's latest 10-K from SEC EDGAR (free). The SEC needs a User-Agent with contact info."""

import json
import time
from pathlib import Path

import httpx

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
DOCUMENT_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{document}"

SECONDS_BETWEEN_REQUESTS = 0.25  # 4 requests/second; the SEC allows 10
TICKER_LIST_MAX_AGE_SECONDS = 7 * 24 * 3600

last_request_time = 0.0


# =========================
# HTTP
# =========================


def sec_get(url: str, user_agent: str) -> httpx.Response:
    """GET with a small pause between requests. The User-Agent goes only into the header."""
    global last_request_time
    wait = SECONDS_BETWEEN_REQUESTS - (time.monotonic() - last_request_time)
    if wait > 0:
        time.sleep(wait)
    last_request_time = time.monotonic()
    response = httpx.get(url, headers={"User-Agent": user_agent}, timeout=30, follow_redirects=True)
    response.raise_for_status()
    return response


# =========================
# Find the filing
# =========================


def lookup_company(ticker: str, user_agent: str, filings_dir: str) -> tuple[int, str]:
    """Ticker -> (CIK number, company name). The SEC's ticker list is cached for a week."""
    path = Path(filings_dir) / "company_tickers.json"
    is_old = path.exists() and time.time() - path.stat().st_mtime > TICKER_LIST_MAX_AGE_SECONDS
    if not path.exists() or is_old:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(sec_get(TICKERS_URL, user_agent).text, encoding="utf-8")

    companies = json.loads(path.read_text(encoding="utf-8"))
    wanted = ticker.upper().replace(".", "-")  # the SEC writes BRK.B as BRK-B
    for company in companies.values():
        if company["ticker"].upper() == wanted:
            return int(company["cik_str"]), company["title"]
    raise LookupError(f"{ticker}: not in the SEC's company list (ETFs and funds don't file 10-Ks)")


def latest_10k(ticker: str, user_agent: str, filings_dir: str) -> dict:
    """Facts about the newest 10-K: company, dates, accession number and document URL."""
    cik, company = lookup_company(ticker, user_agent, filings_dir)
    recent = sec_get(SUBMISSIONS_URL.format(cik=cik), user_agent).json()["filings"]["recent"]
    for i, form in enumerate(recent["form"]):  # newest first
        if form != "10-K":  # exact match skips amendments ("10-K/A")
            continue
        accession = recent["accessionNumber"][i]
        document = recent["primaryDocument"][i]
        report_date = recent["reportDate"][i] or recent["filingDate"][i]
        return {
            "ticker": ticker.upper(),
            "company": company,
            "form": "10-K",
            "fiscal_year": int(report_date[:4]),  # year the fiscal period ended
            "filing_date": recent["filingDate"][i],
            "accession": accession,
            "url": DOCUMENT_URL.format(cik=cik, accession=accession.replace("-", ""), document=document),
        }
    raise LookupError(f"{ticker}: no 10-K in its recent filings")


# =========================
# Download
# =========================


def download_10k(ticker: str, user_agent: str, filings_dir: str) -> tuple[str, dict]:
    """Returns (HTML text, filing facts). The HTML is saved once and reused afterwards."""
    filing = latest_10k(ticker, user_agent, filings_dir)
    path = Path(filings_dir) / filing["ticker"] / f"{filing['accession']}.html"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(sec_get(filing["url"], user_agent).content)
    return path.read_text(encoding="utf-8", errors="replace"), filing
