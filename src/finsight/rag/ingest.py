"""Ingest the latest 10-K per ticker into Chroma: download -> clean text -> sections -> chunks -> embed.

    uv run python -m finsight.rag.ingest NVDA AAPL TSLA
    uv run python -m finsight.rag.ingest --from-samples      # equities in data/samples/*.csv

Idempotent: chunk ids are deterministic (NVDA-10K-FY2026-1A-007) and we upsert, so re-ingesting
never duplicates. If the stored ids for a filing already match, embedding is skipped entirely
(embedding is the slow part). ETFs, bond funds and cash are skipped: funds file N-CSR/N-PORT
reports, not 10-Ks, and their holdings aren't one company's business anyway.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from finsight.rag.edgar import EdgarClient, FilingNotFoundError, FilingRef, MissingUserAgentError
from finsight.rag.parse import FULL_DOCUMENT, html_to_text, make_chunks, split_sections
from finsight.tools.csv_loader import load_portfolio_csv


@dataclass
class IngestResult:
    ticker: str
    status: str  # "ingested" | "already_ingested" | "error"
    chunks: int = 0
    sections: list[str] = field(default_factory=list)
    sections_detected: bool = True
    fiscal_year: int | None = None
    filing_date: str | None = None
    message: str = ""


def base_metadata(ref: FilingRef) -> dict:
    """Chunk metadata (Chroma accepts only str/int/float/bool values: no None)."""
    return {
        "ticker": ref.ticker, "company": ref.company, "form": ref.form, "fiscal_year": ref.fiscal_year,
        "filing_date": ref.filing_date, "report_date": ref.report_date, "accession": ref.accession, "url": ref.url,
    }


def ingest_html(html: str | bytes, meta: dict, collection, embedder, *, force: bool = False,
                batch_size: int = 64) -> IngestResult:
    """Parse, chunk, embed and upsert one filing. Pure w.r.t. the network (testable offline)."""
    sections, detected = split_sections(html_to_text(html))
    chunks = make_chunks(sections, meta)
    result = IngestResult(meta["ticker"], "ingested", len(chunks), [s.key for s in sections], detected,
                          meta["fiscal_year"], meta.get("filing_date"))
    if not detected:
        result.message = "section headings not detected; indexed as one 'Full document' section"
    if not chunks:
        result.status, result.message = "error", "no text extracted"
        return result

    new_ids = {c.chunk_id for c in chunks}
    existing = set(collection.get(where={"accession": meta["accession"]}, include=[])["ids"])
    if existing == new_ids and not force:
        result.status = "already_ingested"
        return result
    stale = sorted(existing - new_ids)  # e.g. the parser changed and produces fewer chunks now
    if stale:
        collection.delete(ids=stale)

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        collection.upsert(
            ids=[c.chunk_id for c in batch],
            documents=[c.text for c in batch],
            metadatas=[c.metadata for c in batch],
            embeddings=embedder.embed_documents([c.text for c in batch]),
        )
    return result


def ingest_ticker(ticker: str, edgar: EdgarClient, collection, embedder, *, force: bool = False) -> IngestResult:
    """Download + ingest one ticker. A missing filing or an HTTP/network error skips this ticker
    (status "error") so the other tickers in the batch still get ingested."""
    try:
        ref = edgar.latest_filing(ticker)
        path = edgar.download(ref)
    except FilingNotFoundError as e:
        return IngestResult(ticker.upper(), "error", message=str(e))
    except httpx.HTTPError as e:  # message holds the URL and status, never the User-Agent header
        return IngestResult(ticker.upper(), "error", message=f"SEC request failed: {type(e).__name__}: {e}")
    return ingest_html(path.read_bytes(), base_metadata(ref), collection, embedder, force=force)


def sample_equity_tickers(samples_dir: str | Path = "data/samples") -> list[str]:
    """Equity tickers across all sample CSVs (ETFs, bonds, cash, crypto don't file 10-Ks)."""
    tickers: list[str] = []
    for csv in sorted(Path(samples_dir).glob("*.csv")):
        for h in load_portfolio_csv(csv).holdings:
            if h.asset_type.value == "equity" and h.ticker not in tickers:
                tickers.append(h.ticker)
    return tickers


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    from finsight.config import get_settings
    from finsight.rag.embeddings import LocalEmbedder
    from finsight.rag.store import open_collection

    p = argparse.ArgumentParser(prog="finsight-ingest", description="Ingest latest 10-K filings into Chroma.")
    p.add_argument("tickers", nargs="*")
    p.add_argument("--from-samples", action="store_true", help="ingest equities from data/samples/*.csv")
    p.add_argument("--force", action="store_true", help="re-embed even if already ingested")
    args = p.parse_args(argv)
    s = get_settings()
    tickers = [t.upper() for t in args.tickers] + (sample_equity_tickers() if args.from_samples else [])
    tickers = list(dict.fromkeys(tickers))
    if not tickers:
        p.error("give tickers or --from-samples")

    try:
        ua = s.sec_user_agent.get_secret_value() if s.sec_user_agent else None
        edgar = EdgarClient(ua, s.filings_dir)
    except MissingUserAgentError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    embedder = LocalEmbedder(s.embedding_model)
    collection = open_collection(s.chroma_path, embedder.name)

    print(f"Ingesting {len(tickers)} ticker(s): {', '.join(tickers)}  (embeddings: {embedder.name}, local)")
    failures = 0
    for t in tickers:
        r = ingest_ticker(t, edgar, collection, embedder, force=args.force)
        failures += r.status == "error"
        secs = ",".join(k for k in r.sections if k != FULL_DOCUMENT) or "-"
        print(f"  {r.ticker:<6} {r.status:<17} FY{r.fiscal_year or '?':<5} filed {r.filing_date or '?':<10} "
              f"chunks={r.chunks:<4} sections={secs} {('WARNING: ' + r.message) if r.message else ''}")
    print(f"Done. Collection now holds {collection.count()} chunks. SEC requests made: {edgar.requests_made}.")
    return 1 if failures == len(tickers) else 0


if __name__ == "__main__":
    raise SystemExit(main())
