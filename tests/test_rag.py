"""10-K parsing, chunking, idempotent ingest and filtered search, all offline (fake embedder)."""

import pytest

from finsight.rag.ingest import ingest_html, open_vector_store
from finsight.rag.parse import CHUNK_CHARS, html_to_text, make_chunks, section_document, split_sections
from finsight.rag.retriever import indexed_tickers, normalize_section, search_filings
from tests.conftest import MINI_10K, fake_filing
from tests.fakes import HashEmbedder

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
    assert [section.metadata["section_key"] for section in sections] == ["1", "1A", "7", "8"]
    risk_factors = sections[1]
    assert risk_factors.metadata["section"] == "Item 1A Risk Factors"
    assert "suppliers of rocket engines" in risk_factors.page_content


def test_text_without_items_becomes_one_full_document():
    sections = split_sections("Just some text.\nNo headings here.")
    assert [section.metadata["section_key"] for section in sections] == ["FULL"]


# =========================
# Chunking
# =========================


def test_chunk_ids_and_skipped_sections():
    chunks = make_chunks(split_sections(html_to_text(MINI_10K)), fake_filing("ACME"))
    ids = [chunk.id for chunk in chunks]
    assert ids == ["ACME-10K-FY2025-1-000", "ACME-10K-FY2025-1A-000", "ACME-10K-FY2025-7-000"]  # no Item 8
    assert chunks[1].metadata["section"] == "Item 1A Risk Factors"
    assert chunks[1].metadata["ticker"] == "ACME"


def test_long_sections_become_small_overlapping_chunks():
    # 600 numbered words of 5-6 chars each: about 3,500 characters -> several chunks.
    text = " ".join(f"word{number:03d}" for number in range(600))
    chunks = make_chunks([section_document("1A", "Item 1A Risk Factors", text)], fake_filing("ACME"))
    assert len(chunks) >= 3
    assert [chunk.id[-3:] for chunk in chunks] == [f"{number:03d}" for number in range(len(chunks))]
    assert all(len(chunk.page_content) <= CHUNK_CHARS for chunk in chunks)
    first_words = chunks[0].page_content.split()
    second_words = chunks[1].page_content.split()
    assert first_words[-1] in second_words  # neighbours share text


# =========================
# Ingest + search
# =========================


def test_ingest_twice_does_not_duplicate(filings_index):
    count = len(filings_index.get(include=[])["ids"])
    result = ingest_html(MINI_10K, fake_filing("ACME"), filings_index)
    assert result["status"] == "already ingested"
    assert len(filings_index.get(include=[])["ids"]) == count == 6  # 3 chunks x 2 companies


def test_ticker_filter_never_returns_another_company(filings_index):
    chunks = search_filings(filings_index, "supply chain risks rocket engines", ticker="acme")
    assert chunks
    assert {chunk.metadata["ticker"] for chunk in chunks} == {"ACME"}


def test_section_filter(filings_index):
    chunks = search_filings(filings_index, "anything", ticker="ACME", section="Item 1A")
    assert [chunk.id for chunk in chunks] == ["ACME-10K-FY2025-1A-000"]
    assert chunks[0].metadata["url"] == "https://example.com/ACME"


def test_indexed_tickers_and_section_names(filings_index):
    assert indexed_tickers(filings_index) == {"ACME", "OTHER"}
    assert normalize_section("1a") == "1A"
    assert normalize_section("Item 7") == "7"
    assert normalize_section("Risk Factors") == "1A"
    assert normalize_section("") is None


def test_index_built_with_another_embedding_model_is_refused(filings_index, tmp_path):
    other_embedder = HashEmbedder()
    other_embedder.model_name = "some-other-model"
    with pytest.raises(ValueError, match="built with hash-test-embedder"):
        open_vector_store(str(tmp_path / "chroma"), other_embedder)
