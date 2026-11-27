"""SEC EDGAR client against a mocked HTTP transport (offline)."""

import json

import httpx
import pytest

from finsight.rag.edgar import EdgarClient, FilingNotFoundError, MissingUserAgentError
from finsight.rag.ingest import ingest_ticker

UA = "FinSight test contact@example.com"
TICKERS = {"0": {"cik_str": 1045810, "ticker": "NVDA", "title": "NVIDIA CORP"},
           "1": {"cik_str": 1067983, "ticker": "BRK-B", "title": "BERKSHIRE HATHAWAY INC"}}
SUBMISSIONS = {"filings": {"recent": {
    "form": ["8-K", "10-K/A", "10-K", "10-Q", "10-K"],
    "accessionNumber": ["a-1", "0001045810-26-000099", "0001045810-26-000021", "q", "0001045810-25-000023"],
    "primaryDocument": ["x.htm", "amend.htm", "nvda-20260125.htm", "q.htm", "nvda-20250126.htm"],
    "filingDate": ["2026-05-01", "2026-04-01", "2026-02-25", "2025-11-20", "2025-02-26"],
    "reportDate": ["", "2026-01-25", "2026-01-25", "2025-10-26", "2025-01-26"],
}}}


def client(tmp_path, seen):
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        url = str(request.url)
        if url.endswith("company_tickers.json"):
            return httpx.Response(200, json=TICKERS)
        if "submissions/CIK0001045810.json" in url:
            return httpx.Response(200, json=SUBMISSIONS)
        if url.endswith("nvda-20260125.htm"):
            return httpx.Response(200, content=b"<html>10-K body</html>")
        return httpx.Response(404)

    return EdgarClient(UA, tmp_path, requests_per_second=1000, http=httpx.Client(transport=httpx.MockTransport(handler)))


def test_latest_10k_skips_amendments_and_builds_url(tmp_path):
    seen = []
    ref = client(tmp_path, seen).latest_filing("nvda")
    assert (ref.ticker, ref.cik, ref.accession, ref.fiscal_year) == ("NVDA", 1045810, "0001045810-26-000021", 2026)
    assert ref.url == "https://www.sec.gov/Archives/edgar/data/1045810/000104581026000021/nvda-20260125.htm"
    assert all(r.headers["User-Agent"] == UA for r in seen)  # SEC requirement, sent on every request


def test_download_is_cached_on_disk(tmp_path):
    seen = []
    ec = client(tmp_path, seen)
    ref = ec.latest_filing("NVDA")
    path = ec.download(ref)
    assert path.read_bytes() == b"<html>10-K body</html>"
    assert json.loads(path.with_suffix(".json").read_text())["accession"] == ref.accession
    n = len(seen)
    ec.download(ref)
    assert len(seen) == n  # second download: no request


def test_ticker_map_cached_and_share_classes(tmp_path):
    seen = []
    ec = client(tmp_path, seen)
    assert ec.lookup_cik("BRK.B") == (1067983, "BERKSHIRE HATHAWAY INC")  # SEC uses BRK-B
    ec.lookup_cik("NVDA")
    assert sum(str(r.url).endswith("company_tickers.json") for r in seen) == 1


def test_unknown_ticker_explains_funds(tmp_path):
    with pytest.raises(FilingNotFoundError, match="ETFs and funds"):
        client(tmp_path, []).lookup_cik("VTI")


@pytest.mark.parametrize("ua", [None, "", "   "])
def test_missing_user_agent_is_refused(tmp_path, ua):
    with pytest.raises(MissingUserAgentError, match="SEC_USER_AGENT is not set"):
        EdgarClient(ua, tmp_path)


def test_user_agent_never_in_errors(tmp_path):
    with pytest.raises(FilingNotFoundError) as e:
        client(tmp_path, []).lookup_cik("ZZZZ")
    assert "contact@example.com" not in str(e.value)


def test_http_error_skips_only_that_ticker(tmp_path):
    def handler(request):
        if str(request.url).endswith("company_tickers.json"):
            return httpx.Response(200, json=TICKERS)
        return httpx.Response(503)  # SEC temporarily unavailable

    ec = EdgarClient(UA, tmp_path, requests_per_second=1000, http=httpx.Client(transport=httpx.MockTransport(handler)))
    r = ingest_ticker("NVDA", ec, collection=None, embedder=None)  # fails before touching the store
    assert r.status == "error" and "SEC request failed" in r.message and "503" in r.message
    assert "contact@example.com" not in r.message
