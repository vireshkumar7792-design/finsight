"""LangGraph StateGraph assembly: supervisor routing, specialists, report, SqliteSaver checkpoints.

    START -> supervisor -(conditional)-> portfolio | risk | research | report
             portfolio / risk / research -> supervisor
             report -> END

Persistence: compiled with a checkpointer, LangGraph saves the full state after every node
("super-step") under the run's thread_id. That gives us:
  - resume:     graph.invoke(None, config) continues an unfinished run from its last checkpoint
  - follow-ups: a new question on the same thread_id starts from the saved state, so
                `messages` (conversation memory) carries over
  - inspection: graph.get_state(config) / get_state_history(config)
PortfolioContext is passed per run via `context=` and is NOT checkpointed (see state.py).
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from finsight.agents.portfolio import make_portfolio_node
from finsight.agents.report import make_report_node
from finsight.agents.research import research_node
from finsight.agents.risk import make_risk_node
from finsight.agents.supervisor import make_supervisor_node, route_after_supervisor
from finsight.config import Settings, get_llm, get_settings
from finsight.context import PortfolioContext
from finsight.state import FinSightState, new_question_input

NODES = ("supervisor", "portfolio", "risk", "research", "report")


def build_graph(
    *,
    router: BaseChatModel | None = None,
    agent_model: BaseChatModel | None = None,
    report_model: BaseChatModel | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    settings: Settings | None = None,
    interrupt_before: list[str] | None = None,
):
    """Compile the FinSight graph. Pass fake models in tests; defaults come from get_llm(role)."""
    s = settings or get_settings()
    router = router or get_llm("router", s)
    agent_model = agent_model or get_llm("agent", s)
    report_model = report_model or get_llm("report", s)

    g = StateGraph(FinSightState, context_schema=PortfolioContext)
    g.add_node("supervisor", make_supervisor_node(router))
    g.add_node("portfolio", make_portfolio_node(agent_model))
    g.add_node("risk", make_risk_node(agent_model))
    g.add_node("research", research_node)
    g.add_node("report", make_report_node(report_model))

    g.add_edge(START, "supervisor")
    g.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {"portfolio": "portfolio", "risk": "risk", "research": "research", "report": "report"},
    )
    for specialist in ("portfolio", "risk", "research"):
        g.add_edge(specialist, "supervisor")
    g.add_edge("report", END)
    return g.compile(checkpointer=checkpointer, interrupt_before=interrupt_before)


@contextmanager
def open_checkpointer(path: str | Path) -> Iterator[SqliteSaver]:
    """SqliteSaver on a local file (created if missing). check_same_thread=False because
    LangGraph may run nodes on worker threads."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    try:
        yield SqliteSaver(conn)
    finally:
        conn.close()


def thread_config(thread_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": thread_id}}


UpdateCallback = Callable[[str, dict[str, Any]], None]


def _stream(graph, graph_input, thread_id: str, ctx: PortfolioContext, on_update: UpdateCallback | None):
    config = thread_config(thread_id)
    for chunk in graph.stream(graph_input, config, context=ctx, stream_mode="updates"):
        for node, update in chunk.items():
            if on_update and node in NODES:
                on_update(node, update or {})
    return graph.get_state(config)


def run_question(graph, ctx: PortfolioContext, question: str, thread_id: str, portfolio_source: str,
                 on_update: UpdateCallback | None = None):
    """Ask a (new or follow-up) question on a thread. Returns the final StateSnapshot."""
    return _stream(graph, new_question_input(question, portfolio_source), thread_id, ctx, on_update)


def resume_thread(graph, ctx: PortfolioContext, thread_id: str, on_update: UpdateCallback | None = None):
    """Continue an unfinished run from its last checkpoint (input=None means 'resume')."""
    return _stream(graph, None, thread_id, ctx, on_update)
