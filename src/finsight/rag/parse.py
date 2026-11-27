"""10-K HTML -> clean text -> one Document per section (Item 1, 1A, 7, ...) -> small overlapping chunks."""

import re

from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

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

CHUNK_CHARS = 1200  # about 300 tokens: fits the embedding model's 512-token window
OVERLAP_CHARS = 200  # a sentence on a chunk border stays whole in at least one chunk


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
# Text -> section Documents
# =========================


def split_sections(text: str) -> list[Document]:
    """One Document per Item. Each heading appears twice (table of contents + real start); the
    real start is the one followed by the most text. If that fails, return one "Full document"."""
    best_start = find_section_starts(text)
    if len(best_start) < 2 or not ({"1A", "7"} & set(best_start)):
        return [section_document(FULL_DOCUMENT, "Full document", text)]

    starts = sorted((start, key) for key, start in best_start.items())
    sections = []
    for i, (start, key) in enumerate(starts):
        end = starts[i + 1][0] if i + 1 < len(starts) else len(text)
        body = text[start:end].split("\n", 1)  # drop the heading line itself
        title = f"Item {key} {SECTION_TITLES.get(key, '')}".strip()
        sections.append(section_document(key, title, body[1].strip() if len(body) > 1 else ""))
    return sections


def find_section_starts(text: str) -> dict[str, int]:
    """Item key -> character position of its heading, keeping the occurrence with the most text after it."""
    headings = []  # (character position, item key)
    position = 0
    for line in text.split("\n"):
        match = HEADING.match(line)
        if match and len(line) <= 160:
            headings.append((position, (match.group(1) + (match.group(2) or "")).upper()))
        position += len(line) + 1

    best_start = {}
    best_length = {}
    for i, (start, key) in enumerate(headings):
        end = headings[i + 1][0] if i + 1 < len(headings) else len(text)
        if end - start > best_length.get(key, -1):
            best_length[key] = end - start
            best_start[key] = start
    return best_start


def section_document(key: str, title: str, text: str) -> Document:
    return Document(page_content=text, metadata={"section_key": key, "section": title})


# =========================
# Sections -> chunks
# =========================


def make_chunks(sections: list[Document], filing: dict) -> list[Document]:
    """Chunks carry the filing facts as metadata. Ids are fixed (e.g. NVDA-10K-FY2026-1A-007),
    so re-ingesting the same filing overwrites instead of duplicating."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_CHARS, chunk_overlap=OVERLAP_CHARS)
    chunks = []
    for section in sections:
        key = section.metadata["section_key"]
        if key not in INDEXED_SECTIONS:
            continue
        section_with_filing = Document(page_content=section.page_content, metadata={**filing, **section.metadata})
        for number, chunk in enumerate(splitter.split_documents([section_with_filing])):
            chunk.id = f"{filing['ticker']}-10K-FY{filing['fiscal_year']}-{key}-{number:03d}"
            chunks.append(chunk)
    return chunks
