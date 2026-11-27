"""Research: grounding in code, Tavily cache rules, and the research agent with fake models."""

from langgraph.checkpoint.memory import InMemorySaver

from finsight.agents.research_agent import choose_searches, keep_grounded, pick_tickers, run_research
from finsight.graph import build_graph, run_question
from finsight.schemas import Insight, Source
from finsight.state import RunContext
from finsight.tools import research_tools
from finsight.tools.research_tools import FoundSources, ResearchSources, tavily_search
from tests.fakes import FakeLLM, tool_calls_message

# =========================
# Grounding
# =========================


def test_claims_citing_unknown_sources_are_dropped():
    found = FoundSources()
    found.add(Source(id="ACME-1", type="filing", title="t", url="u"), "text")
    insights = [
        Insight(claim="Real claim.", source_ids=["ACME-1", "made-up"], confidence="high"),
        Insight(claim="Invented claim.", source_ids=["made-up"], confidence="high"),
        Insight(claim="No citation.", source_ids=[], confidence="low"),
    ]
    kept, dropped = keep_grounded(insights, found)
    assert [insight.claim for insight in kept] == ["Real claim."]
    assert kept[0].source_ids == ["ACME-1"]  # the made-up id is removed too
    assert dropped == 2


# =========================
# Tavily
# =========================


def fake_tavily(monkeypatch, response):
    calls = []

    def fake_call(api_key, query, params):
        calls.append(params)
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr(research_tools, "call_tavily", fake_call)
    return calls


def test_tavily_results_are_filtered_and_cached(monkeypatch, tmp_path):
    response = {"results": [
        {"title": "Dated news", "url": "https://news.example/a", "content": "Big   news", "published_date": "2026-11-20"},
        {"title": "Undated page", "url": "https://example.com/b", "content": "x"},
        {"title": "Profile", "url": "https://tavily.com/c", "content": "x", "published_date": "2026-11-20"},
    ]}
    calls = fake_tavily(monkeypatch, response)
    results, error = tavily_search("Acme news", "key", str(tmp_path), recent_days=30)
    assert error is None
    assert [result["title"] for result in results] == ["Dated news"]
    assert results[0]["snippet"] == "Big news"
    assert calls[0]["topic"] == "news"

    again, _ = tavily_search("acme NEWS", "key", str(tmp_path), recent_days=30)
    assert again == results
    assert len(calls) == 1  # the second search came from the disk cache


def test_tavily_problems_become_gaps_not_crashes(monkeypatch, tmp_path):
    fake_tavily(monkeypatch, RuntimeError("432 quota exceeded"))
    results, error = tavily_search("Acme", "key", str(tmp_path))
    assert results == []
    assert "web search unavailable" in error
    _, error = tavily_search("Acme", None, str(tmp_path / "other"))
    assert "TAVILY_API_KEY is not set" in error


# =========================
# Research agent
# =========================


def research_sources(filings_index, tmp_path):
    return ResearchSources(filings_index, tavily_api_key=None, tavily_cache_dir=str(tmp_path))


def writer_draft():
    return {
        "insights": [
            {"claim": "Acme depends on a few rocket engine suppliers.",
             "source_ids": ["ACME-10K-FY2025-1A-000"], "confidence": "high"},
            {"claim": "Acme will double revenue.", "source_ids": ["ACME-10K-FY2099-9-999"], "confidence": "low"},
        ],
        "gaps": [],
    }


def test_run_research_cites_only_retrieved_sources(filings_index, tmp_path):
    search_llm = FakeLLM(script=[tool_calls_message(
        ("SearchFilings", {"query": "supply risks", "ticker": "ACME", "section": "1A"}),
        ("WebSearch", {"query": "Acme news", "recent_days": "30"}),
    )])
    writer_llm = FakeLLM(structured=[writer_draft()])
    result = run_research("What are ACME's risks?", ["ACME"], research_sources(filings_index, tmp_path),
                          search_llm, writer_llm)
    findings = result["findings"]

    assert [insight["claim"] for insight in findings["insights"]] == ["Acme depends on a few rocket engine suppliers."]
    assert [source["id"] for source in findings["sources"]] == ["ACME-10K-FY2025-1A-000"]
    assert "1 claim(s) removed because they cited no retrieved source" in findings["gaps"]
    assert any("TAVILY_API_KEY" in gap for gap in findings["gaps"])  # web search became a gap
    assert result["tool_calls"][1]["args"]["recent_days"] == 30  # "30" cleaned to a number
    assert search_llm.total_calls + writer_llm.total_calls == 2


def test_search_planning_falls_back_to_defaults():
    failing_llm = FakeLLM(script=[RuntimeError("down"), RuntimeError("down")])
    errors = []
    searches = choose_searches("Any latest news on ACME?", ["ACME"], failing_llm, errors)
    assert searches == [("SearchFilings", {"query": "Any latest news on ACME?", "ticker": "ACME"}),
                        ("WebSearch", {"query": "Any latest news on ACME?", "recent_days": 30})]
    assert len(errors) == 2


def test_pick_tickers():
    assert pick_tickers("What are NVDA's main risks?", [], ["AAPL"]) == ["NVDA"]
    assert pick_tickers("How is Tesla doing?", [], []) == ["TSLA"]
    assert pick_tickers("Any risks in my stocks?", [], ["AAPL", "MSFT", "JPM", "JNJ"]) == ["AAPL", "MSFT", "JPM"]
    assert pick_tickers("anything", ["nvda"], []) == ["NVDA"]


def test_graph_routes_research_question_end_to_end(portfolio_context, filings_index, tmp_path):
    report = {"title": "ACME risks", "summary": "Supplier risk [ACME-10K-FY2025-1A-000].",
              "sections": [], "key_metrics": [], "caveats": []}
    run_context = RunContext(
        portfolio=portfolio_context,
        router_llm=FakeLLM(structured=[{"agents": ["research"], "reason": "company risks"}]),
        agent_llm=FakeLLM(script=[tool_calls_message(("SearchFilings", {"query": "risks", "ticker": "ACME"}))]),
        report_llm=FakeLLM(structured=[writer_draft(), report]),  # research writer, then report
        settings=None,
        research=research_sources(filings_index, tmp_path),
    )
    state = run_question(build_graph(InMemorySaver()), run_context, "What are ACME's main risks?", "r1")

    assert [step["to"] for step in state["route_log"]] == ["research", "report"]
    research_data = state["findings"]["research"]["data"]["research"]
    assert research_data["tickers"] == ["ACME"]
    assert research_data["sources"][0]["id"] == "ACME-10K-FY2025-1A-000"
    assert state["tool_calls"][0]["name"] == "SearchFilings"
    assert state["report"]["title"] == "ACME risks"
