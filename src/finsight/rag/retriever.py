"""Search the 10-K chunks in Chroma by meaning, filtered by ticker and section."""

import re

from finsight.rag.parse import SECTION_TITLES

MAX_CHUNK_CHARS = 1200  # trim each result so 5 of them fit in a free-tier prompt


# =========================
# Search
# =========================


def search_filings(collection, embedder, query: str, ticker: str | None = None,
                   section: str | None = None, k: int = 5) -> list[dict]:
    """Most similar chunks first.

    The filters are applied BEFORE ranking. Without the ticker filter, Apple's supply-chain
    paragraph could be returned (and cited) for a question about NVIDIA.
    """
    filters = []
    if ticker:
        filters.append({"ticker": ticker.strip().upper()})
    section_key = normalize_section(section)
    if section_key:
        filters.append({"section_key": section_key})

    if len(filters) == 0:
        where = None
    elif len(filters) == 1:
        where = filters[0]
    else:
        where = {"$and": filters}

    found = collection.query(
        query_embeddings=[embedder.embed_query(query)],
        n_results=k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    hits = []
    for chunk_id, text, metadata, distance in zip(
        found["ids"][0], found["documents"][0], found["metadatas"][0], found["distances"][0]
    ):
        hits.append({
            "id": chunk_id,
            "text": trim(text),
            "score": round(1 - float(distance), 4),  # cosine similarity: higher = closer
            "ticker": metadata["ticker"],
            "company": metadata["company"],
            "section": metadata["section"],
            "fiscal_year": metadata["fiscal_year"],
            "url": metadata["url"],
        })
    return hits


def indexed_tickers(collection) -> set[str]:
    metadatas = collection.get(include=["metadatas"])["metadatas"] or []
    return {metadata["ticker"] for metadata in metadatas}


# =========================
# Helpers
# =========================


def normalize_section(section: str | None) -> str | None:
    """'1A', 'Item 1A' or 'Risk Factors' -> '1A'. Anything unknown -> None (no filter)."""
    if not section or not section.strip():
        return None
    match = re.match(r"^(?:item\s*)?(1[0-6]|[1-9])([abc])?\b", section.strip(), re.IGNORECASE)
    if match:
        return (match.group(1) + (match.group(2) or "")).upper()
    for key, title in SECTION_TITLES.items():
        if section.strip().lower() in title.lower():
            return key
    return None


def trim(text: str, max_chars: int = MAX_CHUNK_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + " ..."
