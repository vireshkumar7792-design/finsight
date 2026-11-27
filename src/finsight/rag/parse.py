"""10-K HTML -> clean text -> sections (Item 1, 1A, 7, ...) -> overlapping chunks with metadata.

SECTION DETECTION
Every "Item" heading appears twice: in the table of contents and where the section really starts.
We find every line that *starts* like "Item 1A. Risk Factors" (mid-sentence references such as
"see Item 7" don't match), then for each item keep the occurrence followed by the most text: a
TOC entry is followed by a few dozen characters, the real section by thousands.
If detection fails (fewer than 2 items, or neither 1A nor 7 found), the whole document becomes
one "Full document" section: retrieval still works, section filters don't, and ingest warns.

CHUNKING
~220 words (~300 tokens) per chunk with 40 words of overlap. Smaller chunks match questions more
precisely; bigger chunks carry more context but cost more of the free-tier token budget when k
of them go into a prompt. 300 tokens also fits in bge-small's 512-token window, so nothing is
silently truncated at embedding time. Overlap keeps a sentence that straddles a boundary intact
in at least one chunk.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from bs4 import BeautifulSoup

SECTION_TITLES = {
    "1": "Business", "1A": "Risk Factors", "1B": "Unresolved Staff Comments", "1C": "Cybersecurity",
    "2": "Properties", "3": "Legal Proceedings", "4": "Mine Safety Disclosures",
    "5": "Market for Registrant's Common Equity", "6": "[Reserved]",
    "7": "Management's Discussion and Analysis", "7A": "Quantitative and Qualitative Disclosures About Market Risk",
    "8": "Financial Statements and Supplementary Data", "9": "Changes in and Disagreements with Accountants",
    "9A": "Controls and Procedures", "9B": "Other Information", "9C": "Foreign Jurisdictions that Prevent Inspections",
    "10": "Directors, Executive Officers and Corporate Governance", "11": "Executive Compensation",
    "12": "Security Ownership", "13": "Certain Relationships and Related Transactions",
    "14": "Principal Accountant Fees and Services", "15": "Exhibits and Financial Statement Schedules",
    "16": "Form 10-K Summary",
}
FULL_DOCUMENT = "FULL"

# Sections worth retrieving for research. Item 8 (financial tables) and exhibits chunk badly and
# numbers are our tools' job; governance items rarely answer research questions.
INDEXED_SECTIONS = {"1", "1A", "1C", "2", "3", "7", "7A", FULL_DOCUMENT}

_HEADING = re.compile(
    r"^\s*item\s*(1[0-6]|[1-9])([abc])?\s*[\.\:\-–—]?\s*(.{0,120})$", re.IGNORECASE
)
_BLOCK_TAGS = ["p", "div", "br", "tr", "li", "h1", "h2", "h3", "h4", "h5", "h6", "table", "section"]


def section_label(key: str) -> str:
    return "Full document" if key == FULL_DOCUMENT else f"Item {key} {SECTION_TITLES.get(key, '')}".strip()


def html_to_text(html: str | bytes) -> str:
    """Visible text of a 10-K, one block per line, whitespace normalized."""
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "head", "title", "meta"]):
        tag.decompose()
    for tag in soup.find_all(["ix:header"]):  # inline-XBRL machine data block
        tag.decompose()
    for tag in soup.find_all(style=re.compile(r"display\s*:\s*none", re.I)):
        tag.decompose()
    for tag in soup.find_all(_BLOCK_TAGS):
        tag.insert_before("\n")
        tag.insert_after("\n")
    text = soup.get_text()
    text = text.replace("\xa0", " ").replace("​", "")
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in text.splitlines()]
    return "\n".join(ln for ln in lines if ln)


@dataclass(frozen=True)
class Section:
    key: str  # "1A"
    title: str  # "Item 1A Risk Factors"
    text: str


def split_sections(text: str) -> tuple[list[Section], bool]:
    """Return (sections in document order, detected_ok)."""
    lines = text.split("\n")
    offsets, pos = [], 0
    for ln in lines:
        offsets.append(pos)
        pos += len(ln) + 1

    hits: list[tuple[int, str]] = []  # (char offset, key)
    for ln, off in zip(lines, offsets):
        m = _HEADING.match(ln)
        if m and len(ln) <= 160:  # headings are short lines
            hits.append((off, (m.group(1) + (m.group(2) or "")).upper()))
    if not hits:
        return [Section(FULL_DOCUMENT, section_label(FULL_DOCUMENT), text)], False

    # For each item, keep the occurrence followed by the longest stretch of text.
    ends = [h[0] for h in hits[1:]] + [len(text)]
    best: dict[str, tuple[int, int]] = {}  # key -> (length, start)
    for (start, key), end in zip(hits, ends):
        if key not in best or end - start > best[key][0]:
            best[key] = (end - start, start)

    starts = sorted((start, key) for key, (_, start) in best.items())
    sections = []
    for i, (start, key) in enumerate(starts):
        end = starts[i + 1][0] if i + 1 < len(starts) else len(text)
        body = text[start:end].split("\n", 1)
        sections.append(Section(key, section_label(key), body[1].strip() if len(body) > 1 else ""))

    ok = len(sections) >= 2 and bool({"1A", "7"} & set(best))
    if not ok:
        return [Section(FULL_DOCUMENT, section_label(FULL_DOCUMENT), text)], False
    return sections, True


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    text: str
    metadata: dict


def chunk_words(text: str, size: int = 220, overlap: int = 40) -> list[str]:
    """Sliding window over words: each chunk `size` words, consecutive chunks share `overlap`."""
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    words = text.split()
    if not words:
        return []
    step, out = size - overlap, []
    for i in range(0, len(words), step):
        out.append(" ".join(words[i : i + size]))
        if i + size >= len(words):
            break
    return out


def make_chunks(sections: list[Section], base_meta: dict, size: int = 220, overlap: int = 40,
                indexed: set[str] = INDEXED_SECTIONS) -> list[Chunk]:
    """Chunk the indexed sections. chunk_id is deterministic, so re-ingesting overwrites, never duplicates."""
    ticker, fy = base_meta["ticker"], base_meta["fiscal_year"]
    chunks = []
    for sec in sections:
        if sec.key not in indexed:
            continue
        for n, piece in enumerate(chunk_words(sec.text, size, overlap)):
            cid = f"{ticker}-10K-FY{fy}-{sec.key}-{n:03d}"
            chunks.append(Chunk(cid, piece, {**base_meta, "section_key": sec.key, "section": sec.title,
                                              "chunk_id": cid, "chunk_index": n}))
    return chunks
