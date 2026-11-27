"""Multi-agent graph, fully offline with scripted fake models.

Fake portfolio (conftest): AAA 48% (Technology), BBB 36% (ETF), CASH 16%.
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from tests.fakes import FakeLLM, ScriptExhausted, ai_tools, tool_call
from finsight.agents.report import enforce_traceability
from finsight.agents.research_tools import ResearchDeps
from finsight.agents.supervisor import RouteDecision, make_supervisor_node, plan_from_decision
from finsight.graph import build_graph, is_stale, open_checkpointer, resume_thread, run_question, thread_config
from finsight.schemas.report import KeyMetric, Report
from finsight.state import MAX_ITERATIONS

SECTOR_SRC = "portfolio:get_allocation(by=sector)"


def a_report(metrics=(("Technology weight", 48.0, "%", SECTOR_SRC),), summary="Tech is 48%.") -> dict:
    return {
        "title": "Sector exposure",
        "summary": summary,
        "sections": [{"heading": "Allocation", "body": "Mostly technology."}],
        "key_metrics": [{"name": n, "value": v, "unit": u, "source_tool": s} for n, v, u, s in metrics],
        "caveats": [],
        "disclaimer": "whatever the model wrote",
    }


def route(next_, then=(), reason="test route"):
    return RouteDecision(next=next_, then=list(then), reason=reason)


def make(test_settings, router=(), agent=(), report=(), report_script=(), **kw):
    r, a, p = FakeLLM(structured=list(router)), FakeLLM(script=list(agent)), FakeLLM(
        structured=list(report), script=list(report_script))
    # Default: research has no sources configured, so it makes no LLM call (see test_research.py).
    deps = kw.pop("research_deps", ResearchDeps(retriever=None, web=None))
    g = build_graph(router=r, agent_model=a, report_model=p, settings=test_settings,
                    checkpointer=kw.pop("checkpointer", InMemorySaver()), research_deps=deps, **kw)
    return g, r, a, p


def path(values):
    return [e["to"] for e in values["route_log"]]


# ---- routing --------------------------------------------------------------------------------


def test_sector_question_routes_portfolio_then_report(fake_ctx, test_settings):
    g, r, a, p = make(
        test_settings,
        router=[route("portfolio", reason="allocation question")],
        agent=[ai_tools(tool_call("get_allocation", {"by": "sector"})), AIMessage(content="Technology is 48%.")],
        report=[a_report()],
    )
    v = run_question(g, fake_ctx, "What's my sector exposure?", "t1", "fake.csv").values

    assert path(v) == ["portfolio", "report"]
    assert v["route_log"][0] == {"to": "portfolio", "reason": "allocation question", "by": "llm"}
    assert v["completed_agents"] == ["portfolio"]
    f = v["portfolio_findings"]
    assert f["summary"] == "Technology is 48%."
    assert f["data"][SECTOR_SRC]["allocation"][0] == {"group": "Technology", "weight_pct": 48.0, "tickers": ["AAA"]}
    assert v["report_mode"] == "structured"
    assert v["report"]["key_metrics"][0]["value"] == 48.0
    assert v["llm_calls"] == 4  # router 1 + portfolio 2 + report 1
    assert r.n_calls + a.n_calls + p.n_calls == 4
    assert [type(m) for m in v["messages"]] == [HumanMessage, AIMessage]  # conversation memory


def test_two_specialists_run_in_planned_order(fake_ctx, test_settings):
    g, *_ = make(
        test_settings,
        router=[route("portfolio", ["risk"], "diversification needs both")],
        agent=[
            ai_tools(tool_call("get_allocation", {"by": "asset_type"})), AIMessage(content="Equity 48%."),
            ai_tools(tool_call("get_risk_metrics"), tool_call("get_correlations")), AIMessage(content="HHI 0.386."),
        ],
        report=[a_report(metrics=[("HHI", 0.386, "ratio", "risk:get_risk_metrics")])],
    )
    v = run_question(g, fake_ctx, "Is my portfolio diversified?", "t2", "fake.csv").values
    assert path(v) == ["portfolio", "risk", "report"]
    assert v["route_log"][1]["by"] == "plan"  # second hop is deterministic, no LLM
    assert set(v["risk_findings"]["data"]) == {"risk:get_risk_metrics", "risk:get_correlations"}
    assert v["llm_calls"] == 6  # 1 + 2 + 2 + 1
    assert v["report"]["key_metrics"][0]["name"] == "HHI"


def test_research_without_sources_records_gap_without_llm(fake_ctx, test_settings):
    g, _, agent, _ = make(test_settings, router=[route("research")], report=[a_report(metrics=())])
    v = run_question(g, fake_ctx, "Research NVDA", "t3", "fake.csv").values
    assert path(v) == ["research", "report"]
    rf = v["research_findings"]["data"]["research:findings"]
    assert rf["tickers"] == ["NVDA"] and rf["insights"] == []
    assert any("no research sources configured" in g for g in rf["gaps"])
    assert agent.n_calls == 0
    assert v["llm_calls"] == 2


# ---- fallbacks ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_router_output",
    [{"next": "banana", "then": [], "reason": "x"}, RuntimeError("429 Too Many Requests"), {"oops": 1}],
    ids=["invalid-choice", "exception", "wrong-shape"],
)
def test_router_failure_falls_back_to_report(fake_ctx, test_settings, bad_router_output):
    g, *_ = make(test_settings, router=[bad_router_output], report=[a_report(metrics=())])
    v = run_question(g, fake_ctx, "What's my sector exposure?", "t4", "fake.csv").values
    assert path(v) == ["report"]
    assert v["plan"]["source"] == "fallback"
    assert any("router failed" in e for e in v["errors"])
    assert v["report"] is not None


def test_finish_is_coerced_to_report(fake_ctx, test_settings):
    g, *_ = make(test_settings, router=[route("finish", ["risk"])], report=[a_report(metrics=())])
    v = run_question(g, fake_ctx, "hi", "t5", "fake.csv").values
    assert path(v) == ["report"]


def test_plan_skips_completed_and_duplicates():
    d = route("risk", ["risk", "portfolio", "research"])
    assert plan_from_decision(d, completed=["portfolio"])["agents"] == ["risk", "research"]
    assert plan_from_decision(route("report", ["risk"]), [])["agents"] == []


def test_max_iterations_forces_report():
    node = make_supervisor_node(FakeLLM())  # would raise if called
    out = node({"question": "q", "iteration": MAX_ITERATIONS, "plan": {"agents": ["risk"], "reason": "", "source": "llm"}})
    assert out["next_agent"] == "report" and out["route_log"][0]["by"] == "guard"
    assert "max iterations" in out["errors"][0]


def test_specialist_failure_is_recorded_and_report_still_runs(fake_ctx, test_settings):
    g, *_ = make(test_settings, router=[route("risk")], agent=[RuntimeError("provider down")],
                 report=[a_report(metrics=())])
    v = run_question(g, fake_ctx, "How risky?", "t6", "fake.csv").values
    assert path(v) == ["risk", "report"]
    assert v["risk_findings"] is None
    assert any("risk agent failed" in e for e in v["errors"])
    assert v["report_mode"] == "structured"


def test_specialist_without_tool_calls_gets_default_data(fake_ctx, test_settings):
    g, *_ = make(test_settings, router=[route("risk")], agent=[AIMessage(content="Looks risky.")],
                 report=[a_report(metrics=())])
    v = run_question(g, fake_ctx, "How risky?", "t7", "fake.csv").values
    f = v["risk_findings"]
    assert set(f["data"]) == {"risk:get_risk_metrics", "risk:get_correlations"}
    assert "default tool data attached" in f["summary"]


def test_report_falls_back_to_plain_text_then_deterministic(fake_ctx, test_settings):
    bad = {"title": "missing fields"}
    g, *_ = make(test_settings, router=[route("research")], report=[bad, RuntimeError("400")],
                 report_script=[AIMessage(content="Plain answer.")])
    v = run_question(g, fake_ctx, "Research NVDA", "t8", "fake.csv").values
    assert v["report_mode"] == "plain_text_fallback"
    assert v["report"]["summary"] == "Plain answer."
    assert v["llm_calls"] == 1 + 3  # router + (2 structured attempts + 1 plain)

    g, *_ = make(test_settings, router=[route("research")], report=[bad, bad], report_script=[RuntimeError("down")])
    v = run_question(g, fake_ctx, "Research NVDA", "t9", "fake.csv").values
    assert v["report_mode"] == "deterministic_fallback"
    assert v["report"]["sections"][0]["heading"] == "Research findings"


def test_traceability_drops_untraceable_metrics():
    findings = {"portfolio": {"data": {SECTOR_SRC: {"allocation": [{"weight_pct": 48.0}]}}}}
    r = Report.model_validate(a_report(metrics=[
        ("Tech", 48.0, "%", SECTOR_SRC),          # kept
        ("Made up", 99.9, "%", SECTOR_SRC),       # value not in source
        ("Wrong source", 48.0, "%", "risk:x"),    # source doesn't exist
    ]))
    out = enforce_traceability(r, findings)
    assert [m.name for m in out.key_metrics] == ["Tech"]
    assert "Removed 2 metric(s)" in out.caveats[-1]
    assert out.disclaimer.endswith("Not financial advice.")
    assert isinstance(out.key_metrics[0], KeyMetric)


def test_fake_llm_detects_extra_calls():
    with pytest.raises(ScriptExhausted):
        FakeLLM().invoke("hello")


# ---- checkpoints, resume, follow-ups ---------------------------------------------------------


def test_state_persists_in_sqlite_and_reloads(fake_ctx, test_settings, tmp_path):
    db = tmp_path / "ckpt.sqlite"
    with open_checkpointer(db) as saver:
        g, *_ = make(test_settings, router=[route("research")], report=[a_report(metrics=())], checkpointer=saver)
        run_question(g, fake_ctx, "Research NVDA", "persist", "fake.csv")

    # A brand-new connection + graph (like a new process) sees the saved thread.
    with open_checkpointer(db) as saver:
        g2, *_ = make(test_settings, checkpointer=saver)
        snap = g2.get_state(thread_config("persist"))
        assert snap.values["question"] == "Research NVDA"
        assert snap.values["report"]["title"] == "Sector exposure"
        assert snap.next == ()  # finished
        assert len(list(g2.get_state_history(thread_config("persist")))) >= 4
        assert g2.get_state(thread_config("unknown-thread")).values == {}


def test_interrupted_run_resumes_from_checkpoint(fake_ctx, test_settings, tmp_path):
    db = tmp_path / "ckpt.sqlite"
    with open_checkpointer(db) as saver:  # run 1: stops before the report node
        g, *_ = make(test_settings, router=[route("research")], checkpointer=saver, interrupt_before=["report"])
        snap = run_question(g, fake_ctx, "Research NVDA", "resume-me", "fake.csv")
        assert snap.next == ("report",)
        assert snap.values["research_findings"]["agent"] == "research"
        assert snap.values.get("report") is None

    with open_checkpointer(db) as saver:  # run 2 (new process): continue, no router call needed
        g2, router2, _, _ = make(test_settings, report=[a_report(metrics=())], checkpointer=saver)
        snap = resume_thread(g2, fake_ctx, "resume-me")
        assert snap.next == ()
        assert snap.values["report"] is not None
        assert router2.n_calls == 0  # the plan came from the checkpoint
        assert path(snap.values) == ["research", "report"]


def test_follow_up_question_sees_prior_conversation(fake_ctx, test_settings):
    g, router, *_ = make(
        test_settings,
        router=[route("research"), route("research")],
        report=[a_report(metrics=(), summary="NVDA research is a stub."), a_report(metrics=(), summary="Still a stub.")],
    )
    run_question(g, fake_ctx, "Research NVDA", "conv", "fake.csv")
    v = run_question(g, fake_ctx, "And AAPL?", "conv", "fake.csv").values

    assert [m.content for m in v["messages"] if isinstance(m, HumanMessage)] == ["Research NVDA", "And AAPL?"]
    assert len(v["messages"]) == 4
    assert v["question"] == "And AAPL?"
    assert v["completed_agents"] == ["research"]  # per-question fields were reset
    assert v["llm_calls"] == 2  # counter reset too
    second_router_prompt = router._structured_calls[1][-1].content
    assert "NVDA research is a stub." in second_router_prompt  # memory reached the router


# ---- real LLM (free Groq router), skipped by default ------------------------------------------


@pytest.mark.llm
@pytest.mark.parametrize(
    "question, expected",
    [("Research NVDA", {"research"}), ("What's my sector exposure?", {"portfolio"}),
     ("Is my portfolio diversified?", {"portfolio", "risk"})],
)
def test_real_router_plans_sensibly(question, expected):
    from finsight.agents.supervisor import route_with_llm
    from finsight.config import get_llm, get_settings

    if get_settings().groq_api_key is None:
        pytest.skip("GROQ_API_KEY not set")
    d = route_with_llm(get_llm("router"), {"question": question, "messages": []})
    assert {d.next, *d.then} == expected


def test_specialist_transcript_is_not_checkpointed(fake_ctx, test_settings, tmp_path):
    import sqlite3

    db = tmp_path / "ckpt.sqlite"
    with open_checkpointer(db) as saver:
        g, *_ = make(test_settings, router=[route("portfolio")], checkpointer=saver,
                     agent=[ai_tools(tool_call("get_allocation", {"by": "sector"})), AIMessage(content="ok")],
                     report=[a_report()])
        run_question(g, fake_ctx, "sector?", "ns", "fake.csv")
    namespaces = {r[0] for r in sqlite3.connect(db).execute("select distinct checkpoint_ns from checkpoints")}
    assert namespaces == {""}  # only the parent graph; no 'portfolio:...' subgraph checkpoints


# ---- stale data on resume ----------------------------------------------------------------------


def _ctx_on_later_day(fake_portfolio, test_settings):
    """Same portfolio, but market data ending a month later (like resuming on another day)."""
    import pandas as pd

    from finsight.context import PortfolioContext
    from tests.conftest import FAKE_CLOSES, FAKE_SECTORS

    def history(tickers, period):
        idx = pd.date_range("2024-02-01", periods=4, name="Date")
        return pd.DataFrame({t: FAKE_CLOSES[t] for t in tickers}, index=idx)

    return PortfolioContext.load(fake_portfolio, test_settings, fetch_history=history,
                                 fetch_sectors=lambda ts: {t: FAKE_SECTORS[t] for t in ts})


def test_stale_resume_restarts_question_on_new_data(fake_ctx, fake_portfolio, test_settings, tmp_path):
    db = tmp_path / "ckpt.sqlite"
    with open_checkpointer(db) as saver:
        g, *_ = make(test_settings, router=[route("research")], checkpointer=saver, interrupt_before=["report"])
        snap = run_question(g, fake_ctx, "Research NVDA", "stale", "fake.csv")
        assert snap.values["data_as_of"] == "2024-01-04"

    later = _ctx_on_later_day(fake_portfolio, test_settings)
    assert later.as_of == "2024-02-04"
    with open_checkpointer(db) as saver:
        g2, router2, _, _ = make(test_settings, router=[route("research")], report=[a_report(metrics=())],
                                 checkpointer=saver)
        assert is_stale(g2, later, "stale") and not is_stale(g2, fake_ctx, "stale")
        snap = resume_thread(g2, later, "stale")
    v = snap.values
    assert snap.next == () and v["report"] is not None
    assert v["data_as_of"] == "2024-02-04"
    assert router2.n_calls == 1  # question re-planned from scratch on the new data
    assert path(v) == ["research", "report"]  # fresh per-question log, not the old one + new


def test_same_day_resume_is_unaffected(fake_ctx, test_settings, tmp_path):
    db = tmp_path / "ckpt.sqlite"
    with open_checkpointer(db) as saver:
        g, *_ = make(test_settings, router=[route("research")], checkpointer=saver, interrupt_before=["report"])
        run_question(g, fake_ctx, "Research NVDA", "same", "fake.csv")
        g2, router2, _, _ = make(test_settings, report=[a_report(metrics=())], checkpointer=saver)
        snap = resume_thread(g2, fake_ctx, "same")  # same day: continues, no re-plan
    assert snap.next == () and router2.n_calls == 0
