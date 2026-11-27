"""The FinSight graph: supervisor -> specialist agents -> report, with SQLite checkpoints.

    START -> supervisor -> portfolio | risk | research | report
             each specialist -> back to supervisor
             report -> END
"""

import sqlite3
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from finsight.agents.portfolio_agent import portfolio_node
from finsight.agents.report_agent import report_node
from finsight.agents.research_agent import research_node
from finsight.agents.risk_agent import risk_node
from finsight.agents.supervisor import supervisor_node
from finsight.config import Settings, get_llm, get_settings
from finsight.portfolio.context import load_context
from finsight.state import FinSightState, RunContext, new_question

# =========================
# 1. LLMs + data for one run
# =========================


def make_run_context(csv_path: str, settings: Settings | None = None) -> RunContext:
    """Load the portfolio and its market data, and pick the free model for each role."""
    settings = settings or get_settings()
    return RunContext(
        portfolio=load_context(csv_path, settings),
        router_llm=get_llm("router", settings),
        agent_llm=get_llm("agent", settings),
        report_llm=get_llm("report", settings),
        settings=settings,
    )


# =========================
# 2. State
# =========================
# FinSightState (state.py) is the shared notebook every node reads and writes.
# The agent nodes (supervisor, portfolio, risk, research, report) live in agents/.


# =========================
# 3. Routing
# =========================


def route_after_supervisor(state: FinSightState) -> str:
    """The supervisor already decided; this just tells LangGraph where to go."""
    return state["next_agent"]


# =========================
# 4. Build the graph
# =========================


def build_graph(checkpointer, interrupt_before=None):
    graph = StateGraph(FinSightState, context_schema=RunContext)

    graph.add_node("supervisor", supervisor_node)
    graph.add_node("portfolio", portfolio_node)
    graph.add_node("risk", risk_node)
    graph.add_node("research", research_node)
    graph.add_node("report", report_node)

    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges("supervisor", route_after_supervisor, ["portfolio", "risk", "research", "report"])
    graph.add_edge("portfolio", "supervisor")
    graph.add_edge("risk", "supervisor")
    graph.add_edge("research", "supervisor")
    graph.add_edge("report", END)

    # interrupt_before lets tests stop the run part-way, like a crash, to test resume.
    return graph.compile(checkpointer=checkpointer, interrupt_before=interrupt_before)


def open_checkpointer(path: str) -> SqliteSaver:
    """SqliteSaver saves the state after every node, so a run can be resumed by thread id."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    # check_same_thread=False: LangGraph may run nodes on another thread.
    connection = sqlite3.connect(path, check_same_thread=False)
    return SqliteSaver(connection)


# =========================
# 5. Run a question
# =========================


def thread_config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def run_question(graph, run_context: RunContext, question: str, thread_id: str,
                 on_update=None, callbacks=None) -> dict:
    """Ask a new question (or a follow-up on an existing thread). Returns the final state."""
    portfolio = run_context.portfolio
    graph_input = new_question(question, portfolio.csv_path, portfolio.as_of)
    return stream_graph(graph, graph_input, run_context, thread_id, on_update, callbacks)


def resume_thread(graph, run_context: RunContext, thread_id: str, on_update=None, callbacks=None) -> dict:
    """Continue an unfinished run from its last checkpoint."""
    saved = graph.get_state(thread_config(thread_id)).values
    if saved["data_as_of"] != run_context.portfolio.as_of:
        # Continuing would mix two days of prices in one report, so start the question again.
        return run_question(graph, run_context, saved["question"], thread_id, on_update, callbacks)
    # Input None means "carry on from the checkpoint".
    return stream_graph(graph, None, run_context, thread_id, on_update, callbacks)


def stream_graph(graph, graph_input, run_context, thread_id, on_update, callbacks) -> dict:
    """Run the graph, calling on_update(node_name, update) after each node finishes."""
    config = {"configurable": {"thread_id": thread_id}, "callbacks": callbacks or []}
    for chunk in graph.stream(graph_input, config, context=run_context, stream_mode="updates"):
        for node_name, update in chunk.items():
            if on_update and isinstance(update, dict):
                on_update(node_name, update)
    return graph.get_state(thread_config(thread_id)).values
