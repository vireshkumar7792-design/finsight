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


class LocalEmbedder:
    """Turns text into 384 numbers so similar meanings end up close together.

    The model (~130 MB) is downloaded once and only loaded on first use, because loading is slow.
    """

    def __init__(self, model_name: str):
        self.name = model_name
        self.model = None

    def load(self):
        if self.model is None:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(self.name, device="cpu")
        return self.model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.load().encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> list[float]:
        # bge models were trained with this prefix on questions (not on documents).
        return self.load().encode([BGE_QUERY_PREFIX + text], normalize_embeddings=True)[0].tolist()


# =========================
# Chroma collection
# =========================


def open_collection(chroma_path: str, embedder_name: str):
    """Open (or create) the 10-K collection. We pass our own vectors, so Chroma never embeds."""
    import chromadb  # slow import: only when research or ingest runs
    from chromadb.config import Settings as ChromaSettings

    Path(chroma_path).mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=chroma_path, settings=ChromaSettings(anonymized_telemetry=False))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=None,
        configuration={"hnsw": {"space": "cosine"}},
        metadata={"embedding_model": embedder_name},
    )
    # Vectors from two different models can't be compared, so refuse a mismatch.
    built_with = (collection.metadata or {}).get("embedding_model")
    if built_with and built_with != embedder_name:
        raise ValueError(f"the 10-K index was built with {built_with}, not {embedder_name}")
    return collection


# =========================
# Ingest
# =========================


def ingest_html(html: str, filing: dict, collection, embedder, force: bool = False) -> dict:
    """Parse, chunk, embed and store one filing. Returns a small status dict."""
    sections = split_sections(html_to_text(html))
    chunks = make_chunks(sections, filing)
    result = {"ticker": filing["ticker"], "fiscal_year": filing["fiscal_year"],
              "chunks": len(chunks), "sections": [section["key"] for section in sections]}
    if not chunks:
        return {**result, "status": "error: no text found"}

    new_ids = [chunk["id"] for chunk in chunks]
    old_ids = collection.get(where={"accession": filing["accession"]}, include=[])["ids"]
    if set(old_ids) == set(new_ids) and not force:
        return {**result, "status": "already ingested"}  # skip the slow embedding step

    leftover_ids = [chunk_id for chunk_id in old_ids if chunk_id not in new_ids]
    if leftover_ids:
        collection.delete(ids=leftover_ids)
    for start in range(0, len(chunks), 64):  # in batches, to keep memory small
        batch = chunks[start:start + 64]
        texts = [chunk["text"] for chunk in batch]
        collection.upsert(  # upsert = insert, or overwrite the same id
            ids=[chunk["id"] for chunk in batch],
            documents=texts,
            metadatas=[chunk["metadata"] for chunk in batch],
            embeddings=embedder.embed_documents(texts),
        )
    return {**result, "status": "ingested"}


def ingest_ticker(ticker: str, user_agent: str, settings, collection, embedder, force: bool = False) -> dict:
    """Download and ingest one ticker. Errors are returned, so other tickers still run."""
    try:
        html, filing = download_10k(ticker, user_agent, settings.filings_dir)
    except (LookupError, httpx.HTTPError) as error:
        return {"ticker": ticker.upper(), "status": f"error: {error}", "chunks": 0, "sections": []}
    return ingest_html(html, filing, collection, embedder, force)


def sample_equity_tickers(samples_dir: str = "data/samples") -> list[str]:
    """Stocks in the sample CSVs. ETFs, bonds and cash don't file 10-Ks."""
    tickers = []
    for csv_path in sorted(Path(samples_dir).glob("*.csv")):
        for holding in load_portfolio_csv(csv_path).holdings:
            if holding.asset_type == "equity" and holding.ticker not in tickers:
                tickers.append(holding.ticker)
    return tickers
