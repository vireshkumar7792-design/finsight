"""The whole graph, offline: scripted fake models, fake market data, real checkpoints."""

from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from finsight.agents.supervisor import keyword_plan, supervisor_node
from finsight.graph import build_graph, open_checkpointer, resume_thread, run_question, thread_config
from finsight.state import RunContext
from finsight.tools.research_tools import ResearchSources
from tests.fakes import FakeLLM, tool_calls_message

SECTOR_SOURCE = "get_allocation(by=sector)"


# =========================
# Helpers
# =========================


def make_run_context(portfolio_context, router=(), agent=(), report=()) -> RunContext:
    return RunContext(
        portfolio=portfolio_context,
        router_llm=FakeLLM(structured=list(router)),
        agent_llm=FakeLLM(script=list(agent)),
        report_llm=FakeLLM(structured=list(report)),
        settings=None,  # only used to open research sources, which tests pass in directly
        research=ResearchSources(vector_store=None, tavily_api_key=None, tavily_cache_dir=""),
    )


def plan(*agents, reason="test plan"):
    return {"agents": list(agents), "reason": reason}


def report(metrics=(("Technology weight", 48.0, "%", SECTOR_SOURCE),)):
    return {
        "title": "Sector exposure",
        "summary": "Technology is 48% as of 2024-01-04.",
        "sections": [{"heading": "Allocation", "body": "Mostly technology."}],
        "key_metrics": [{"name": n, "value": v, "unit": u, "source": s} for n, v, u, s in metrics],
        "caveats": [],
    }


def sector_agent_script():
    return [tool_calls_message(("get_allocation", {"by": "sector"})), AIMessage(content="Technology is 48%.")]


def route(state):
    return [step["to"] for step in state["route_log"]]


def total_calls(run_context):
    return run_context.router_llm.total_calls + run_context.agent_llm.total_calls + run_context.report_llm.total_calls


# =========================
# Routing
# =========================


def test_sector_question_runs_portfolio_then_report(portfolio_context):
    run_context = make_run_context(portfolio_context, [plan("portfolio")], sector_agent_script(), [report()])
    state = run_question(build_graph(InMemorySaver()), run_context, "What's my sector exposure?", "t1")

    assert route(state) == ["portfolio", "report"]
    finding = state["findings"]["portfolio"]
    assert finding["summary"] == "Technology is 48%."
    assert finding["data"][SECTOR_SOURCE]["allocation"][0]["weight_pct"] == 48.0
    assert state["report"]["key_metrics"][0]["value"] == 48.0
    assert state["tool_calls"] == [{"agent": "portfolio", "name": "get_allocation", "args": {"by": "sector"}}]
    assert total_calls(run_context) == 4  # router 1 + portfolio agent 2 + report 1
    assert [message.type for message in state["messages"]] == ["human", "ai"]


def test_two_agents_run_in_planned_order(portfolio_context):
    agent_script = sector_agent_script() + [
        tool_calls_message(("get_risk_metrics", {}), ("get_correlations", {})),
        AIMessage(content="HHI is 0.386."),
    ]
    metrics = [("HHI", 0.386, "ratio", "get_risk_metrics")]
    run_context = make_run_context(portfolio_context, [plan("portfolio", "risk")], agent_script, [report(metrics)])
    state = run_question(build_graph(InMemorySaver()), run_context, "Is my portfolio diversified?", "t2")

    assert route(state) == ["portfolio", "risk", "report"]
    assert [step["by"] for step in state["route_log"]] == ["llm", "plan", "plan"]  # only one router call
    assert set(state["findings"]["risk"]["data"]) == {"get_risk_metrics", "get_correlations"}
    assert state["report"]["key_metrics"][0]["name"] == "HHI"
    assert total_calls(run_context) == 6


def test_router_retries_once_then_succeeds(portfolio_context):
    bad_plan = {"agents": ["banana"], "reason": "x"}
    run_context = make_run_context(portfolio_context, [bad_plan, plan("portfolio")], sector_agent_script(), [report()])
    state = run_question(build_graph(InMemorySaver()), run_context, "Sector exposure?", "t3")
    assert state["route_log"][0]["by"] == "llm"
    assert route(state) == ["portfolio", "report"]


def test_router_failing_twice_uses_keyword_fallback(portfolio_context):
    failures = [RuntimeError("429 Too Many Requests"), RuntimeError("429 again")]
    run_context = make_run_context(portfolio_context, failures, sector_agent_script(), [report()])
    state = run_question(build_graph(InMemorySaver()), run_context, "What's my sector exposure?", "t4")
    assert state["route_log"][0]["by"] == "fallback"
    assert route(state) == ["portfolio", "report"]
    assert any("keyword routing" in error for error in state["errors"])


def test_keyword_plan():
    assert keyword_plan("Is my portfolio diversified?") == ["portfolio", "risk"]
    assert keyword_plan("How risky is it?") == ["risk"]
    assert keyword_plan("Any news on NVDA?") == ["research"]
    assert keyword_plan("hello") == ["portfolio"]


def test_supervisor_stops_after_too_many_visits():
    state = {"supervisor_visits": 6, "plan": ["risk"], "plan_reason": "", "completed_agents": [],
             "route_log": [], "errors": []}
    update = supervisor_node(state, runtime=None)  # no LLM is needed after the first visit
    assert update["next_agent"] == "report"


# =========================
# Fallbacks and grounding
# =========================


def test_everything_fails_but_the_run_still_finishes(portfolio_context):
    agent_failures = [RuntimeError("down"), RuntimeError("still down")]
    report_failures = [RuntimeError("down"), RuntimeError("still down")]
    run_context = make_run_context(portfolio_context, [plan("portfolio")], agent_failures, report_failures)
    state = run_question(build_graph(InMemorySaver()), run_context, "Sector exposure?", "t5")

    # The agent's default tools ran without the LLM, so real numbers are still there.
    assert set(state["findings"]["portfolio"]["data"]) == {"get_portfolio_summary", SECTOR_SOURCE}
    assert state["report"]["title"] == "FinSight report (automatic summary)"
    assert len(state["errors"]) == 3  # two agent attempts + the report


def test_agent_that_calls_no_tools_gets_default_data(portfolio_context):
    lazy_answer = [AIMessage(content="I think tech is big.")]
    run_context = make_run_context(portfolio_context, [plan("risk")], lazy_answer, [report(metrics=())])
    state = run_question(build_graph(InMemorySaver()), run_context, "How risky?", "t6")
    assert set(state["findings"]["risk"]["data"]) == {"get_risk_metrics", "get_correlations"}


def test_untraceable_metrics_are_dropped(portfolio_context):
    metrics = [
        ("Technology weight", 48.0, "%", SECTOR_SOURCE),  # real
        ("Made-up weight", 99.9, "%", SECTOR_SOURCE),  # number not in the data
        ("Wrong source", 48.0, "%", "get_nothing"),  # source doesn't exist
    ]
    run_context = make_run_context(portfolio_context, [plan("portfolio")], sector_agent_script(), [report(metrics)])
    state = run_question(build_graph(InMemorySaver()), run_context, "Sector exposure?", "t7")
    assert [metric["name"] for metric in state["report"]["key_metrics"]] == ["Technology weight"]
    assert "Made-up weight, Wrong source" in state["report"]["caveats"][-1]


# =========================
# Memory and checkpoints
# =========================


def test_follow_up_question_keeps_the_conversation(portfolio_context):
    graph = build_graph(InMemorySaver())
    first = make_run_context(portfolio_context, [plan("portfolio")], sector_agent_script(), [report()])
    run_question(graph, first, "What's my sector exposure?", "chat")

    second = make_run_context(portfolio_context, [plan()], [], [report(metrics=())])
    state = run_question(graph, second, "Thanks, can you repeat that?", "chat")

    assert len(state["messages"]) == 4  # question, answer, question, answer
    assert state["findings"] == {}  # per-question data was reset
    assert route(state) == ["report"]


def test_resume_after_a_stop(portfolio_context, tmp_path):
    database = str(tmp_path / "checkpoints.sqlite")
    stopped_graph = build_graph(open_checkpointer(database), interrupt_before=["report"])
    run_context = make_run_context(portfolio_context, [plan("portfolio")], sector_agent_script(), [report()])
    run_question(stopped_graph, run_context, "Sector exposure?", "t8")
    assert stopped_graph.get_state(thread_config("t8")).next == ("report",)

    # A fresh graph on the same database, like restarting the program.
    new_graph = build_graph(open_checkpointer(database))
    state = resume_thread(new_graph, run_context, "t8")
    assert state["report"]["title"] == "Sector exposure"
    assert run_context.router_llm.total_calls == 1  # the plan was not made again


def test_resume_with_newer_market_data_starts_over(portfolio_context):
    checkpointer = InMemorySaver()
    stopped_graph = build_graph(checkpointer, interrupt_before=["report"])
    run_context = make_run_context(portfolio_context, [plan("portfolio")] * 2, sector_agent_script() * 2, [report()])
    run_question(stopped_graph, run_context, "Sector exposure?", "t9")

    portfolio_context.as_of = "2099-01-01"  # pretend it's a new day with new prices
    state = resume_thread(build_graph(checkpointer), run_context, "t9")
    assert state["data_as_of"] == "2099-01-01"
    assert run_context.router_llm.total_calls == 2  # the question was asked again from the start
