"""Semantic search over ingested 10-K chunks, with metadata filters and trimmed text.

WHY FILTER: similarity alone ranks by meaning, not by company. "NVDA supply chain risks" is
genuinely similar to AAPL's supply-chain paragraph, and without a filter the LLM would cite an
Apple chunk as NVIDIA's risk. Chroma applies `where` filters BEFORE ranking, so only the
requested ticker/section compete.
Text is trimmed (default 1,200 chars) so k results stay affordable in a free-tier prompt.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from finsight.rag.parse import SECTION_TITLES


@dataclass(frozen=True)
class ChunkHit:
    chunk_id: str
    text: str
    score: float  # cosine similarity, higher = closer (1 - cosine distance)
    ticker: str
    company: str
    section: str
    section_key: str
    fiscal_year: int
    filing_date: str
    url: str


def normalize_section(section: str | None) -> str | None:
    """'1A', 'Item 1A', 'item 1a. risk factors', 'Risk Factors' -> '1A'. None/'' -> None."""
    if not section or not section.strip():
        return None
    s = section.strip()
    m = re.match(r"^(?:item\s*)?(1[0-6]|[1-9])([abc])?\b", s, re.IGNORECASE)
    if m:
        return (m.group(1) + (m.group(2) or "")).upper()
    for key, title in SECTION_TITLES.items():
        if s.lower() in title.lower() or title.lower() in s.lower():
            return key
    return None


def trim(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars].rsplit(" ", 1)[0]
    return cut + " ..."


class FilingRetriever:
    def __init__(self, collection, embedder, max_chars: int = 1200):
        self.collection = collection
        self.embedder = embedder
        self.max_chars = max_chars

    def available_tickers(self) -> set[str]:
        metas = self.collection.get(include=["metadatas"])["metadatas"] or []
        return {m["ticker"] for m in metas}

    def search(self, query: str, ticker: str | None = None, section: str | None = None, k: int = 5) -> list[ChunkHit]:
        conds = []
        if ticker:
            conds.append({"ticker": ticker.strip().upper()})
        if (key := normalize_section(section)) is not None:
            conds.append({"section_key": key})
        where = None if not conds else conds[0] if len(conds) == 1 else {"$and": conds}

        res = self.collection.query(
            query_embeddings=[self.embedder.embed_query(query)],
            n_results=k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        for cid, doc, meta, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]):
            hits.append(ChunkHit(
                chunk_id=cid, text=trim(doc, self.max_chars), score=round(1.0 - float(dist), 4),
                ticker=meta["ticker"], company=meta.get("company", ""), section=meta["section"],
                section_key=meta["section_key"], fiscal_year=int(meta["fiscal_year"]),
                filing_date=meta.get("filing_date", ""), url=meta.get("url", ""),
            ))
        return hits
