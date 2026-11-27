"""Ingest 10-Ks into Chroma: download -> parse -> chunk -> embed (locally) -> store. Safe to re-run."""

from pathlib import Path

import httpx

from finsight.portfolio.loader import load_portfolio_csv
from finsight.rag.edgar import download_10k
from finsight.rag.parse import html_to_text, make_chunks, split_sections

COLLECTION_NAME = "sec_filings"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


# =========================
# Embeddings (local, free)
# =========================


def make_embeddings(model_name: str):
    """A local sentence-transformers model (~130 MB, downloaded once). No embedding API."""
    from langchain_huggingface import HuggingFaceEmbeddings  # slow import (torch)

    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
        # bge models were trained with this prefix on questions (not on documents).
        query_encode_kwargs={"normalize_embeddings": True, "prompt": BGE_QUERY_PREFIX},
    )


# =========================
# Vector store
# =========================


def open_vector_store(chroma_path: str, embeddings):
    """Open (or create) the 10-K store. `embeddings` must have a `model_name`."""
    import chromadb  # slow import: only when research or ingest runs
    from langchain_chroma import Chroma

    Path(chroma_path).mkdir(parents=True, exist_ok=True)
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=chroma_path,
        client_settings=chromadb.config.Settings(anonymized_telemetry=False),
        collection_configuration={"hnsw": {"space": "cosine"}},
        collection_metadata={"embedding_model": embeddings.model_name},
    )
    # Vectors from two different models can't be compared, so refuse a mismatch.
    # LangChain has no public getter for collection metadata, hence `_collection`.
    built_with = (vector_store._collection.metadata or {}).get("embedding_model")
    if built_with and built_with != embeddings.model_name:
        raise ValueError(f"the 10-K index was built with {built_with}, not {embeddings.model_name}")
    return vector_store


# =========================
# Ingest
# =========================


def ingest_html(html: str, filing: dict, vector_store, force: bool = False) -> dict:
    """Parse, chunk, embed and store one filing. Returns a small status dict."""
    sections = split_sections(html_to_text(html))
    chunks = make_chunks(sections, filing)
    result = {"ticker": filing["ticker"], "fiscal_year": filing["fiscal_year"], "chunks": len(chunks),
              "sections": [section.metadata["section_key"] for section in sections]}
    if not chunks:
        return {**result, "status": "error: no text found"}

    new_ids = [chunk.id for chunk in chunks]
    old_ids = vector_store.get(where={"accession": filing["accession"]}, include=[])["ids"]
    if set(old_ids) == set(new_ids) and not force:
        return {**result, "status": "already ingested"}  # skip the slow embedding step

    leftover_ids = [chunk_id for chunk_id in old_ids if chunk_id not in new_ids]
    if leftover_ids:
        vector_store.delete(ids=leftover_ids)
    vector_store.add_documents(chunks)  # Chroma upserts by Document.id, so nothing duplicates
    return {**result, "status": "ingested"}


def ingest_ticker(ticker: str, user_agent: str, settings, vector_store, force: bool = False) -> dict:
    """Download and ingest one ticker. Errors are returned, so other tickers still run."""
    try:
        html, filing = download_10k(ticker, user_agent, settings.filings_dir)
    except (LookupError, httpx.HTTPError) as error:
        return {"ticker": ticker.upper(), "status": f"error: {error}", "chunks": 0, "sections": []}
    return ingest_html(html, filing, vector_store, force)


def sample_equity_tickers(samples_dir: str = "data/samples") -> list[str]:
    """Stocks in the sample CSVs. ETFs, bonds and cash don't file 10-Ks."""
    tickers = []
    for csv_path in sorted(Path(samples_dir).glob("*.csv")):
        for holding in load_portfolio_csv(csv_path).holdings:
            if holding.asset_type == "equity" and holding.ticker not in tickers:
                tickers.append(holding.ticker)
    return tickers
