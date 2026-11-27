"""Shared test fixtures: fake settings and fake market data, so tests never touch the network."""

from pathlib import Path

import pandas as pd
import pytest

from finsight.config import Settings
from finsight.portfolio import market
from finsight.portfolio.context import load_context
from finsight.rag.ingest import ingest_html, open_vector_store
from tests.fakes import HashEmbedder

SAMPLES_DIR = Path(__file__).resolve().parents[1] / "data" / "samples"

# A tiny fake market. Last closes: AAA 120, BBB 45.
FAKE_CLOSES = {
    "AAA": [100.0, 110.0, 99.0, 120.0],
    "BBB": [50.0, 50.0, 55.0, 45.0],
    "SPY": [400.0, 404.0, 396.0, 408.0],
}
FAKE_SECTORS = {"AAA": "Technology", "BBB": "ETF / Fund"}

# AAA 10 @ $100, BBB 20 @ $50, $400 cash.
# Market value: 1200 + 900 + 400 = 2500 -> weights 48% / 36% / 16%.
# Cost: 1000 + 1000 + 400 = 2400 -> P&L +200, -100, 0 = +100 (+4.1667%).
FAKE_CSV = """ticker,quantity,avg_buy_price,buy_date,asset_type,currency
AAA,10,100,2024-01-02,equity,USD
BBB,20,50,,etf,USD
CASH,400,1,,cash,USD
"""


@pytest.fixture
def test_settings(tmp_path) -> Settings:
    """Dummy keys and temporary folders: the real .env is never read."""
    return Settings(
        groq_api_key="test-groq",
        openrouter_api_key="test-openrouter",
        market_cache_dir=str(tmp_path / "market"),
        tavily_cache_dir=str(tmp_path / "tavily"),
        checkpoint_path=str(tmp_path / "checkpoints.sqlite"),
        chroma_path=str(tmp_path / "chroma"),
        filings_dir=str(tmp_path / "filings"),
    )


@pytest.fixture
def fake_market(monkeypatch):
    """Replace the two yfinance download functions. Records how often they are called."""
    calls = {"closes": 0, "sectors": 0}

    def fake_download_closes(tickers, period="1y"):
        calls["closes"] += 1
        dates = pd.date_range("2024-01-01", periods=4)
        return pd.DataFrame({ticker: FAKE_CLOSES[ticker] for ticker in tickers}, index=dates)

    def fake_download_sector(ticker):
        calls["sectors"] += 1
        return FAKE_SECTORS.get(ticker, "Unknown")

    monkeypatch.setattr(market, "download_closes", fake_download_closes)
    monkeypatch.setattr(market, "download_sector", fake_download_sector)
    return calls


@pytest.fixture
def fake_csv(tmp_path) -> str:
    path = tmp_path / "fake.csv"
    path.write_text(FAKE_CSV, encoding="utf-8")
    return str(path)


@pytest.fixture
def portfolio_context(fake_market, fake_csv, test_settings):
    """The fake portfolio loaded with fake market data."""
    return load_context(fake_csv, test_settings)


# A tiny fake 10-K, ingested twice under two different companies (for the ticker filter).
MINI_10K = (Path(__file__).parent / "fixtures" / "mini_10k.html").read_text(encoding="utf-8")


def fake_filing(ticker: str) -> dict:
    return {"ticker": ticker, "company": f"{ticker} Corp", "form": "10-K", "fiscal_year": 2025,
            "filing_date": "2025-02-01", "accession": f"{ticker}-0001", "url": f"https://example.com/{ticker}"}


@pytest.fixture
def filings_index(tmp_path):
    """A real Chroma store on disk, filled using the fake embedder."""
    vector_store = open_vector_store(str(tmp_path / "chroma"), HashEmbedder())
    ingest_html(MINI_10K, fake_filing("ACME"), vector_store)
    ingest_html(MINI_10K, fake_filing("OTHER"), vector_store)
    return vector_store
