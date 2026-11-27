"""10-K parsing, chunking, Chroma ingest + retrieval (offline: fixture HTML + HashEmbedder)."""

from pathlib import Path

import pytest

from tests.fakes import HashEmbedder
from finsight.rag.ingest import ingest_html, sample_equity_tickers
from finsight.rag.parse import FULL_DOCUMENT, chunk_words, html_to_text, make_chunks, split_sections
from finsight.rag.retriever import FilingRetriever, normalize_section, trim
from finsight.rag.store import EmbeddingModelMismatchError, open_collection

FIXTURE = (Path(__file__).parent / "fixtures" / "mini_10k.html").read_text(encoding="utf-8")


def meta(ticker="ACME", fy=2026, accession="0000000000-26-000001"):
    return {"ticker": ticker, "company": f"{ticker} Corp", "form": "10-K", "fiscal_year": fy,
            "filing_date": "2026-02-20", "report_date": f"{fy}-01-31", "accession": accession,
            "url": f"https://www.sec.gov/Archives/edgar/data/1/{ticker}.htm"}


# ---- HTML -> text -> sections -------------------------------------------------------------------


def test_html_to_text_drops_hidden_script_and_xbrl():
    text = html_to_text(FIXTURE)
    assert "HIDDEN XBRL" not in text and "SCRIPT TEXT" not in text and ".x{color" not in text
    assert "Acme designs rocket powered roller skates" in text
    assert "\nItem 1A. Risk Factors\n" in text  # block tags become line breaks


def test_split_sections_skips_table_of_contents():
    sections, ok = split_sections(html_to_text(FIXTURE))
    assert ok
    assert [s.key for s in sections] == ["1", "1A", "7", "8"]
    by_key = {s.key: s for s in sections}
    assert by_key["1A"].title == "Item 1A Risk Factors"
    assert by_key["1A"].text.startswith("Our business depends on a small number of suppliers")  # body, not TOC
    assert "see Item 7 of this report" in by_key["1A"].text  # mid-sentence reference is not a heading
    assert "Revenue grew" in by_key["7"].text


def test_split_sections_falls_back_to_full_document():
    sections, ok = split_sections("Just some text.\nNo item headings here at all.")
    assert not ok
    assert [(s.key, s.title) for s in sections] == [(FULL_DOCUMENT, "Full document")]


def test_chunk_words_known_answer():
    words = " ".join(f"w{i}" for i in range(10))
    # size 4, overlap 1 -> step 3: w0-w3, w3-w6, w6-w9
    assert chunk_words(words, size=4, overlap=1) == ["w0 w1 w2 w3", "w3 w4 w5 w6", "w6 w7 w8 w9"]
    assert chunk_words("", 4, 1) == []
    with pytest.raises(ValueError):
        chunk_words(words, size=4, overlap=4)


def test_make_chunks_metadata_ids_and_section_filter():
    sections, _ = split_sections(html_to_text(FIXTURE))
    chunks = make_chunks(sections, meta(), size=30, overlap=5)
    keys = {c.metadata["section_key"] for c in chunks}
    assert keys == {"1", "1A", "7"}  # Item 8 (financial tables) not indexed
    first_1a = next(c for c in chunks if c.metadata["section_key"] == "1A")
    assert first_1a.chunk_id == "ACME-10K-FY2026-1A-000"
    assert first_1a.metadata | {} == {**meta(), "section_key": "1A", "section": "Item 1A Risk Factors",
                                      "chunk_id": "ACME-10K-FY2026-1A-000", "chunk_index": 0}
    assert all(None not in c.metadata.values() for c in chunks)  # Chroma rejects None
    assert len({c.chunk_id for c in chunks}) == len(chunks)


# ---- Chroma ingest + retrieval ----------------------------------------------------------------------


@pytest.fixture
def collection(tmp_path):
    return open_collection(tmp_path / "chroma", HashEmbedder.name)


def other_company_html():
    return FIXTURE.replace("Acme", "Globex").replace("rocket engines", "semiconductor wafers")


def test_ingest_is_idempotent(collection):
    emb = HashEmbedder()
    first = ingest_html(FIXTURE, meta(), collection, emb)
    n = collection.count()
    assert first.status == "ingested" and first.chunks == n > 0
    second = ingest_html(FIXTURE, meta(), collection, emb)
    assert second.status == "already_ingested"
    assert collection.count() == n  # no duplicates
    forced = ingest_html(FIXTURE, meta(), collection, emb, force=True)
    assert forced.status == "ingested" and collection.count() == n


def test_retriever_filters_by_ticker_and_section(collection):
    emb = HashEmbedder()
    ingest_html(FIXTURE, meta("ACME"), collection, emb)
    ingest_html(other_company_html(), meta("GLBX", accession="0000000000-26-000002"), collection, emb)
    r = FilingRetriever(collection, emb, max_chars=200)

    assert r.available_tickers() == {"ACME", "GLBX"}
    unfiltered = r.search("supply chain disruption suppliers", k=10)
    assert {h.ticker for h in unfiltered} == {"ACME", "GLBX"}  # similar text from both companies

    acme = r.search("supply chain disruption suppliers", ticker="acme", k=10)
    assert acme and {h.ticker for h in acme} == {"ACME"}
    risks = r.search("supply chain disruption suppliers", ticker="ACME", section="Risk Factors", k=10)
    assert risks and {h.section_key for h in risks} == {"1A"}
    top = risks[0]
    assert top.chunk_id.startswith("ACME-10K-FY2026-1A-") and 0 < top.score <= 1
    assert len(top.text) <= 204  # trimmed to max_chars (+ " ...")


def test_embedding_model_mismatch_is_refused(tmp_path):
    open_collection(tmp_path / "c", "model-a")
    with pytest.raises(EmbeddingModelMismatchError):
        open_collection(tmp_path / "c", "model-b")


@pytest.mark.parametrize("raw, key", [("1A", "1A"), ("Item 1A", "1A"), ("item 7a. market risk", "7A"),
                                      ("Risk Factors", "1A"), ("business", "1"), ("", None), (None, None),
                                      ("banana", None)])
def test_normalize_section(raw, key):
    assert normalize_section(raw) == key


def test_trim():
    assert trim("short", 10) == "short"
    assert trim("one two three four", 9) == "one two ..."


def test_sample_equity_tickers_skip_funds_and_cash():
    tickers = sample_equity_tickers()
    assert {"AAPL", "MSFT", "NVDA", "TSLA", "COIN", "JPM", "JNJ", "GOOGL", "AMZN"} == set(tickers)
    assert not {"VTI", "BND", "SPY", "QQQ", "CASH"} & set(tickers)
