"""Research schemas and agent pipeline, offline: fixture 10-K in Chroma + fake Tavily + FakeLLM."""

from datetime import date
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage
from pydantic import ValidationError

from finsight.agents.research import clean_args, default_plan, post_check, resolve_tickers, run_research
from finsight.agents.research_tools import ResearchDeps, ResearchRunContext, web_source_id
from tests.fakes import HashEmbedder
from finsight.rag.ingest import ingest_html
from finsight.rag.retriever import FilingRetriever
from finsight.rag.store import open_collection
from finsight.schemas.research import Insight, ResearchDraft, ResearchFindings, Source
from finsight.tools.web_search import WebSearch
from tests.fakes import FakeLLM, ai_tools, tool_call

FIXTURE = (Path(__file__).parent / "fixtures" / "mini_10k.html").read_text(encoding="utf-8")
RISK_CHUNK = "ACME-10K-FY2026-1A-000"
NEWS_URL = "https://news.example/acme-recall"


def src(id_, type_="filing"):
    return Source(id=id_, type=type_, title="t", url="https://x", retrieved_at="2026-11-26T00:00:00+00:00")


# ---- schemas: grounding is enforced by validation -----------------------------------------------


def test_valid_findings():
    f = ResearchFindings(tickers=["ACME"], question="q", as_of="2026-11-26", sources=[src("a"), src("b")],
                         insights=[Insight(claim="Acme depends on few suppliers.", source_ids=["a"],
                                           confidence="high", category="risk")])
    assert f.source("a").id == "a"


def test_uncited_insight_fails():
    with pytest.raises(ValidationError, match="source_ids"):
        Insight(claim="Acme is great.", source_ids=[], confidence="high", category="other")


def test_unknown_source_id_fails():
    with pytest.raises(ValidationError, match="unknown source id"):
        ResearchFindings(tickers=[], question="q", as_of="d", sources=[src("a")],
                         insights=[Insight(claim="Made-up claim.", source_ids=["zzz"], confidence="low", category="other")])


def test_duplicate_source_ids_fail():
    with pytest.raises(ValidationError, match="duplicate"):
        ResearchFindings(tickers=[], question="q", as_of="d", sources=[src("a"), src("a")], insights=[])


# ---- fixtures: tiny real pipeline --------------------------------------------------------------------


class FakeTavily:
    def __init__(self):
        self.calls = 0

    def search(self, query, **kw):
        self.calls += 1
        return {"results": [{"title": "Acme recalls skates", "url": NEWS_URL, "content": "Acme recalled rocket skates.",
                             "published_date": "2026-11-20", "score": 0.8}]}


@pytest.fixture
def deps(tmp_path):
    emb = HashEmbedder()
    col = open_collection(tmp_path / "chroma", emb.name)
    ingest_html(FIXTURE, {"ticker": "ACME", "company": "Acme Corp", "form": "10-K", "fiscal_year": 2026,
                          "filing_date": "2026-02-20", "report_date": "2026-01-31",
                          "accession": "0000000000-26-000001", "url": "https://sec.example/acme.htm"}, col, emb)
    fake = FakeTavily()
    web = WebSearch("k", tmp_path / "tavily", client=fake, today=lambda: date(2026, 11, 26))
    d = ResearchDeps(retriever=FilingRetriever(col, emb), web=web)
    d.fake_tavily = fake  # for assertions
    return d


def tools_msg(*calls):
    return ai_tools(*[tool_call(n, a, id=f"c{i}") for i, (n, a) in enumerate(calls)])


def draft(*insights, gaps=()):
    return ResearchDraft.model_validate({"insights": [
        {"claim": c, "source_ids": ids, "confidence": "high", "category": cat} for c, ids, cat in insights],
        "gaps": list(gaps)})


# ---- agent pipeline -----------------------------------------------------------------------------------


def test_filing_and_web_research_produces_cited_findings(deps):
    tool_llm = FakeLLM(script=[tools_msg(
        ("search_filings", {"query": "supply chain risks", "ticker": "ACME", "section": "1A"}),
        ("web_search", {"query": "Acme news", "recent_days": 30}),
    )])
    writer = FakeLLM(structured=[draft(
        ("Acme relies on a small number of rocket engine suppliers.", [RISK_CHUNK], "risk"),
        ("Acme recently recalled rocket skates.", [web_source_id(NEWS_URL)], "news"),
        gaps=["No market share data in the sources."],
    )])
    run = run_research("What are ACME's risks and recent news?", deps, tool_llm, writer)
    f = run.findings

    assert f.tickers == ["ACME"]
    assert [c["name"] for c in run.tool_calls] == ["search_filings", "web_search"]
    assert len(f.insights) == 2 and run.dropped == []
    assert {s.id for s in f.sources} == {RISK_CHUNK, web_source_id(NEWS_URL)}
    filing = f.source(RISK_CHUNK)
    assert (filing.type, filing.section, filing.form, filing.fiscal_year) == ("filing", "Item 1A Risk Factors", "10-K", 2026)
    web = f.source(web_source_id(NEWS_URL))
    assert (web.type, web.url, web.published_date) == ("web", NEWS_URL, "2026-11-20")
    assert f.gaps == ["No market share data in the sources."]
    assert run.llm_calls == 2 and run.synthesis_mode == "structured"

    # The writer saw the retrieved evidence, with source ids, and nothing else.
    evidence_prompt = writer._structured_calls[0][-1].content
    assert RISK_CHUNK in evidence_prompt and "small number of suppliers" in evidence_prompt


def test_fabricated_source_id_is_dropped(deps):
    tool_llm = FakeLLM(script=[tools_msg(("search_filings", {"query": "suppliers", "ticker": "ACME"}))])
    writer = FakeLLM(structured=[draft(
        ("Acme relies on few suppliers.", [RISK_CHUNK, "ACME-10K-FY2026-1A-999"], "risk"),  # one fake id
        ("Acme will double revenue next year.", ["web-deadbeef"], "growth"),  # only a fake id
    )])
    run = run_research("ACME risks?", deps, tool_llm, writer)
    f = run.findings
    assert [i.claim for i in f.insights] == ["Acme relies on few suppliers."]
    assert f.insights[0].source_ids == [RISK_CHUNK]  # fake id stripped, real one kept
    assert run.dropped == ["Acme will double revenue next year."]
    assert any("removed because they cited no retrieved source" in g for g in f.gaps)
    assert "web-deadbeef" not in {s.id for s in f.sources}


def test_structured_failure_falls_back_to_two_step(deps):
    tool_llm = FakeLLM(script=[tools_msg(("search_filings", {"query": "suppliers", "ticker": "ACME"}))])
    writer = FakeLLM(
        structured=[{"insights": "not a list"}, RuntimeError("400 schema"),
                    draft(("Acme relies on few suppliers.", [RISK_CHUNK], "risk"))],
        script=[AIMessage(content=f"- Acme relies on few suppliers [{RISK_CHUNK}]")],
    )
    run = run_research("ACME risks?", deps, tool_llm, writer)
    assert run.synthesis_mode == "two_step"
    assert len(run.findings.insights) == 1
    assert run.llm_calls == 1 + 2 + 2  # tools + 2 structured attempts + notes + structure
    assert len(run.errors) == 2 and all("structured synthesis attempt" in e for e in run.errors)


def test_total_synthesis_failure_returns_gap_not_crash(deps):
    tool_llm = FakeLLM(script=[tools_msg(("search_filings", {"query": "suppliers", "ticker": "ACME"}))])
    writer = FakeLLM(structured=[RuntimeError("x"), RuntimeError("y")], script=[RuntimeError("down")])
    run = run_research("ACME risks?", deps, tool_llm, writer)
    assert run.synthesis_mode == "failed" and run.findings.insights == []
    assert any("synthesis failed" in g for g in run.findings.gaps)


def test_no_tool_calls_uses_default_plan(deps):
    tool_llm = FakeLLM(script=[AIMessage(content="I think Acme is fine.")])
    writer = FakeLLM(structured=[draft(("Acme relies on few suppliers.", [RISK_CHUNK], "risk"))])
    run = run_research("What's the latest news on ACME?", deps, tool_llm, writer)
    assert [c["name"] for c in run.tool_calls] == ["search_filings", "web_search"]  # 'latest news' -> web too
    assert "model called no tools" in run.log[0]


def test_tool_caps_protect_free_quota(deps):
    calls = [("web_search", {"query": f"acme {i}"}) for i in range(4)]
    tool_llm = FakeLLM(script=[tools_msg(*calls)])
    writer = FakeLLM(structured=[draft(("Acme recently recalled rocket skates.", [web_source_id(NEWS_URL)], "news"))])
    run = run_research("ACME news", deps, tool_llm, writer)
    assert deps.fake_tavily.calls == 2  # 2 different queries allowed through, the rest refused
    assert [c["error"] for c in run.tool_calls][2:] == ["limit of 2 web searches per question reached"] * 2


def test_missing_filing_and_web_down_recorded_as_gaps(deps):
    deps.web = WebSearch(None, "unused")  # no key
    tool_llm = FakeLLM(script=[tools_msg(("search_filings", {"query": "risks", "ticker": "ZZZ"}),
                                         ("web_search", {"query": "ZZZ news"}))])
    run = run_research("ZZZ risks and news?", deps, tool_llm, FakeLLM())
    f = run.findings
    assert run.synthesis_mode == "no_evidence" and run.llm_calls == 1  # writer never called
    assert any("no 10-K ingested for ZZZ" in g for g in f.gaps)
    assert any("TAVILY_API_KEY is not set" in g for g in f.gaps)


def test_resolve_tickers():
    assert resolve_tickers("What are NVDA's main risks?", None, None) == ["NVDA"]
    assert resolve_tickers("How does Apple describe its competition?", None, None) == ["AAPL"]
    assert resolve_tickers("Any risks?", None, ["TSLA", "COIN", "AAPL", "SPY"]) == ["TSLA", "COIN", "AAPL"]
    assert resolve_tickers("Is my ETF exposure risky?", None, []) == []  # 'ETF' is not a ticker
    assert resolve_tickers("x", ["msft"], ["TSLA"]) == ["MSFT"]


def test_clean_args_and_default_plan():
    assert clean_args("search_filings", {"query": "q", "ticker": "", "section": None, "max_results": 9}) == {"query": "q"}
    assert clean_args("web_search", {"query": "q", "recent_days": "9999"}) == {"query": "q", "recent_days": 365}
    assert default_plan("NVDA risks", ["NVDA"]) == [("search_filings", {"query": "NVDA risks", "ticker": "NVDA"})]


def test_post_check_dedupes_ids():
    ctx = ResearchRunContext(ResearchDeps(None, None))
    ctx.sources["a"] = src("a")
    run = type("R", (), {"log": [], "dropped": []})()
    kept = post_check(draft(("Claim one here.", ["a", "a"], "risk")), ctx, run)
    assert kept[0].source_ids == ["a"]
