"""What the graph remembers (FinSightState, saved in checkpoints) and what each run is given (RunContext)."""

from dataclasses import dataclass
from typing import Annotated, Any, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.graph.message import add_messages

from finsight.config import Settings
from finsight.portfolio.context import PortfolioContext
from finsight.tools.research_tools import ResearchSources

# =========================
# Graph state (checkpointed)
# =========================


class FinSightState(TypedDict):
    # Conversation memory: questions and short answers. add_messages APPENDS instead of
    # replacing, so follow-up questions on the same thread see earlier turns.
    messages: Annotated[list[AnyMessage], add_messages]

    # The current question. Every field below is reset when a new question starts.
    question: str
    csv_path: str
    data_as_of: str  # date of the market data this question was answered with

    plan: list[str]  # agents the supervisor chose, in order
    plan_reason: str
    next_agent: str
    completed_agents: list[str]
    supervisor_visits: int

    findings: dict[str, dict]  # agent name -> {"summary": ..., "data": {source_id: tool output}}
    route_log: list[dict]  # [{"to": "risk", "reason": "..."}]
    tool_calls: list[dict]  # [{"agent": "risk", "name": "get_risk_metrics", "args": {}}]
    errors: list[str]
    report: dict | None


def new_question(question: str, csv_path: str, data_as_of: str) -> dict[str, Any]:
    """Graph input for a new question: add it to the conversation and reset everything else."""
    return {
        "messages": [HumanMessage(content=question)],
        "question": question,
        "csv_path": csv_path,
        "data_as_of": data_as_of,
        "plan": [],
        "plan_reason": "",
        "next_agent": "",
        "completed_agents": [],
        "supervisor_visits": 0,
        "findings": {},
        "route_log": [],
        "tool_calls": [],
        "errors": [],
        "report": None,
    }


def earlier_conversation(state: FinSightState, max_turns: int = 2) -> str:
    """Last few question/answer turns as text, without the current question."""
    messages = state["messages"][:-1]  # the last message is the current question
    lines = []
    for message in messages[-2 * max_turns:]:
        speaker = "User" if isinstance(message, HumanMessage) else "FinSight"
        lines.append(f"{speaker}: {message.content}")
    return "\n".join(lines) or "(none)"


# =========================
# Run context (NOT checkpointed)
# =========================


@dataclass
class RunContext:
    """Handed to every node and tool for one run.

    It holds DataFrames and model clients, which don't belong in checkpoints. On resume it
    is rebuilt from state["csv_path"].
    """

    portfolio: PortfolioContext
    router_llm: BaseChatModel
    agent_llm: BaseChatModel
    report_llm: BaseChatModel
    settings: Settings
    research: ResearchSources | None = None  # 10-K index + web search, opened on first research question
