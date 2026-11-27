"""Phase 2: a single ReAct-style portfolio analyst agent.

ReAct loop (run by LangChain's create_agent, which builds a small LangGraph graph):
  model node: LLM reads system prompt + messages + tool schemas -> answers OR emits tool calls
  tools node: our Python executes each tool call -> ToolMessage(s) appended
  ...repeat until the LLM replies without tool calls.

The PortfolioContext is passed at invoke time (`context=`) and reaches tools via ToolRuntime.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, ToolMessage

from finsight.agents.structured import message_text
from finsight.agents.tools import ALL_TOOLS
from finsight.config import Role, get_llm
from finsight.context import PortfolioContext

SYSTEM_PROMPT = """You are FinSight, a portfolio analyst. You explain a user's portfolio using \
data from your tools. You are an analyst, not a financial advisor: you describe and interpret, \
you do not tell the user what to buy or sell.

Rules:
1. Every number you state must come from a tool result in this conversation. Call a tool before \
answering any question that involves figures. Never estimate, recompute, or invent numbers, and \
never do arithmetic on tool outputs (no summing, subtracting or converting units yourself). If a \
figure you need is not in any tool output, say it isn't available.
2. Copy units exactly as the tools give them: fields ending in _pct are already percentages \
(22.4 means 22.4%), fields ending in _usd are US dollars.
3. For broad questions (risk, diversification), combine the relevant tools, e.g. allocation, \
risk metrics and correlations for diversification.
4. Compare with the benchmark figures the tools provide when judging whether something is high \
or low, and say what the comparison is.
5. Be concise: lead with the direct answer, then the key supporting figures, then a short \
interpretation. Mention the 'as_of' date once.
6. End with one line: "Not financial advice."
"""


def build_analyst(model: BaseChatModel | None = None, role: Role = "agent"):
    """Create the compiled agent graph. Pass `model` to inject a fake model in tests."""
    return create_agent(
        model=model or get_llm(role),
        tools=ALL_TOOLS,
        system_prompt=SYSTEM_PROMPT,
        context_schema=PortfolioContext,
        name="portfolio_analyst",
    )


@dataclass
class ToolCallRecord:
    name: str
    args: dict[str, Any]
    output: str


@dataclass
class AnalystResult:
    answer: str
    tool_calls: list[ToolCallRecord]
    messages: list[AnyMessage]


def ask(question: str, ctx: PortfolioContext, agent=None) -> AnalystResult:
    """Run one question through the agent and collect the answer plus a tool-call trace."""
    agent = agent or build_analyst()
    state = agent.invoke({"messages": [{"role": "user", "content": question}]}, context=ctx)
    messages: list[AnyMessage] = state["messages"]

    outputs = {m.tool_call_id: m.content for m in messages if isinstance(m, ToolMessage)}
    calls = [
        ToolCallRecord(tc["name"], tc["args"], str(outputs.get(tc["id"], "")))
        for m in messages
        if isinstance(m, AIMessage)
        for tc in m.tool_calls
    ]
    final = next((m for m in reversed(messages) if isinstance(m, AIMessage) and not m.tool_calls), None)
    return AnalystResult(answer=message_text(final) if final else "", tool_calls=calls, messages=messages)

