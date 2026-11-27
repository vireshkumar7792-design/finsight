"""10-K HTML -> clean text -> sections (Item 1, 1A, 7, ...) -> small overlapping chunks."""

import re

from bs4 import BeautifulSoup

SECTION_TITLES = {
    "1": "Business", "1A": "Risk Factors", "1B": "Unresolved Staff Comments", "1C": "Cybersecurity",
    "2": "Properties", "3": "Legal Proceedings", "4": "Mine Safety Disclosures",
    "5": "Market for Common Equity", "6": "[Reserved]", "7": "Management's Discussion and Analysis",
    "7A": "Quantitative and Qualitative Disclosures About Market Risk",
    "8": "Financial Statements", "9": "Changes in and Disagreements with Accountants",
    "9A": "Controls and Procedures", "9B": "Other Information", "9C": "Foreign Jurisdictions",
    "10": "Directors and Corporate Governance", "11": "Executive Compensation",
    "12": "Security Ownership", "13": "Related Transactions", "14": "Accountant Fees",
    "15": "Exhibits", "16": "Form 10-K Summary",
}
FULL_DOCUMENT = "FULL"

# Only sections that answer research questions. Item 8 is tables (numbers are our tools' job).
INDEXED_SECTIONS = {"1", "1A", "1C", "2", "3", "7", "7A", FULL_DOCUMENT}

# A line that STARTS like "Item 1A. Risk Factors". Mid-sentence "see Item 7" doesn't match.
HEADING = re.compile(r"^\s*item\s*(1[0-6]|[1-9])([abc])?\s*[.:\-–—]?\s*(.{0,120})$", re.IGNORECASE)

CHUNK_WORDS = 220  # about 300 tokens: fits the embedding model's 512-token window
OVERLAP_WORDS = 40  # a sentence on a chunk border stays whole in at least one chunk


# =========================
# HTML -> text
# =========================


def html_to_text(html: str) -> str:
    """The visible text, one block per line."""
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "head", "ix:header"]):  # ix:header = hidden XBRL data
        tag.decompose()
    for tag in soup.find_all(style=re.compile(r"display\s*:\s*none", re.IGNORECASE)):
        tag.decompose()
    for tag in soup.find_all(["p", "div", "br", "tr", "li", "h1", "h2", "h3", "h4", "table"]):
        tag.insert_before("\n")
        tag.insert_after("\n")

    text = soup.get_text().replace("\xa0", " ")
    lines = []
    for line in text.splitlines():
        line = re.sub(r"\s+", " ", line).strip()
        if line:
            lines.append(line)
    return "\n".join(lines)


# =========================
# Text -> sections
# =========================


def split_sections(text: str) -> list[dict]:
    """Split into Items. Each heading appears twice (table of contents + real start); the real
    start is the one followed by the most text. If that fails, return one "Full document"."""
    lines = text.split("\n")
    headings = []  # (character position, item key)
    position = 0
    for line in lines:
        match = HEADING.match(line)
        if match and len(line) <= 160:
            key = (match.group(1) + (match.group(2) or "")).upper()
            headings.append((position, key))
        position += len(line) + 1

    # For each item key, keep the occurrence followed by the longest stretch of text.
    best_start = {}
    best_length = {}
    for i, (start, key) in enumerate(headings):
        end = headings[i + 1][0] if i + 1 < len(headings) else len(text)
        if end - start > best_length.get(key, -1):
            best_length[key] = end - start
            best_start[key] = start

    if len(best_start) < 2 or not ({"1A", "7"} & set(best_start)):
        return [{"key": FULL_DOCUMENT, "title": "Full document", "text": text}]

    starts = sorted((start, key) for key, start in best_start.items())
    sections = []
    for i, (start, key) in enumerate(starts):
        end = starts[i + 1][0] if i + 1 < len(starts) else len(text)
        body = text[start:end].split("\n", 1)  # drop the heading line itself
        sections.append({
            "key": key,
            "title": f"Item {key} {SECTION_TITLES.get(key, '')}".strip(),
            "text": body[1].strip() if len(body) > 1 else "",
        })
    return sections


# =========================
# Sections -> chunks
# =========================


def chunk_words(text: str, size: int = CHUNK_WORDS, overlap: int = OVERLAP_WORDS) -> list[str]:
    """Sliding window over words: each chunk has `size` words; neighbours share `overlap`."""
    words = text.split()
    chunks = []
    step = size - overlap
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start:start + size]))
        if start + size >= len(words):
            break
    return chunks


def make_chunks(sections: list[dict], filing: dict) -> list[dict]:
    """Chunks with metadata. Ids are fixed (e.g. NVDA-10K-FY2026-1A-007), so re-ingesting overwrites."""
    chunks = []
    for section in sections:
        if section["key"] not in INDEXED_SECTIONS:
            continue
        for number, text in enumerate(chunk_words(section["text"])):
            chunk_id = f"{filing['ticker']}-10K-FY{filing['fiscal_year']}-{section['key']}-{number:03d}"
            metadata = {**filing, "section_key": section["key"], "section": section["title"]}
            chunks.append({"id": chunk_id, "text": text, "metadata": metadata})
    return chunks
