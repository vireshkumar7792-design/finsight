"""Search the 10-K chunks in Chroma by meaning, filtered by ticker and section."""

import re

from langchain_core.documents import Document

from finsight.rag.parse import SECTION_TITLES

# =========================
# Search
# =========================


def search_filings(vector_store, query: str, ticker: str | None = None,
                   section: str | None = None, k: int = 5) -> list[Document]:
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
    return vector_store.similarity_search(query, k=k, filter=where)


def indexed_tickers(vector_store) -> set[str]:
    metadatas = vector_store.get(include=["metadatas"])["metadatas"] or []
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
