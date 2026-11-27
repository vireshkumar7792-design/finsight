"""10-K parsing, chunking, idempotent ingest and filtered search, all offline (fake embedder)."""

from finsight.rag.ingest import ingest_html
from finsight.rag.parse import chunk_words, html_to_text, make_chunks, split_sections
from finsight.rag.retriever import indexed_tickers, normalize_section, search_filings
from tests.conftest import MINI_10K, fake_filing

# =========================
# Parsing
# =========================


def test_html_to_text_drops_scripts_styles_and_hidden_data():
    text = html_to_text(MINI_10K)
    assert "rocket powered roller skates" in text
    assert "SCRIPT TEXT" not in text
    assert "HIDDEN XBRL" not in text
    assert "color:red" not in text


def test_sections_use_the_real_heading_not_the_table_of_contents():
    sections = split_sections(html_to_text(MINI_10K))
    assert [section["key"] for section in sections] == ["1", "1A", "7", "8"]
    risk_factors = sections[1]
    assert risk_factors["title"] == "Item 1A Risk Factors"
    assert "suppliers of rocket engines" in risk_factors["text"]


def test_text_without_items_becomes_one_full_document():
    sections = split_sections("Just some text.\nNo headings here.")
    assert [section["key"] for section in sections] == ["FULL"]


def test_chunks_overlap():
    words = " ".join(str(number) for number in range(10))
    # size 4, overlap 2 -> starts at 0, 2, 4, 6 (the last one reaches the end)
    assert chunk_words(words, size=4, overlap=2) == ["0 1 2 3", "2 3 4 5", "4 5 6 7", "6 7 8 9"]


def test_chunk_ids_and_skipped_sections():
    chunks = make_chunks(split_sections(html_to_text(MINI_10K)), fake_filing("ACME"))
    ids = [chunk["id"] for chunk in chunks]
    assert ids == ["ACME-10K-FY2025-1-000", "ACME-10K-FY2025-1A-000", "ACME-10K-FY2025-7-000"]  # no Item 8
    assert chunks[1]["metadata"]["section"] == "Item 1A Risk Factors"


# =========================
# Ingest + search
# =========================


def test_ingest_twice_does_not_duplicate(filings_index):
    collection, embedder = filings_index
    count = collection.count()
    result = ingest_html(MINI_10K, fake_filing("ACME"), collection, embedder)
    assert result["status"] == "already ingested"
    assert collection.count() == count == 6  # 3 chunks x 2 companies


def test_ticker_filter_never_returns_another_company(filings_index):
    collection, embedder = filings_index
    hits = search_filings(collection, embedder, "supply chain risks rocket engines", ticker="acme")
    assert hits
    assert {hit["ticker"] for hit in hits} == {"ACME"}


def test_section_filter(filings_index):
    collection, embedder = filings_index
    hits = search_filings(collection, embedder, "anything", ticker="ACME", section="Item 1A")
    assert [hit["id"] for hit in hits] == ["ACME-10K-FY2025-1A-000"]
    assert hits[0]["url"] == "https://example.com/ACME"


def test_indexed_tickers_and_section_names(filings_index):
    collection, _ = filings_index
    assert indexed_tickers(collection) == {"ACME", "OTHER"}
    assert normalize_section("1a") == "1A"
    assert normalize_section("Item 7") == "7"
    assert normalize_section("Risk Factors") == "1A"
    assert normalize_section("") is None
