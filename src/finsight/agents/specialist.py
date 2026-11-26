"""Shared machinery for specialist agents (Portfolio, Risk).

Each specialist is a small ReAct agent (LangChain create_agent) restricted to its own tools.
After it runs, we DON'T keep its chat transcript. We turn it into a compact Finding:
  data     every tool payload, keyed by source id ("risk:get_risk_metrics", ...)
  summary  the specialist's final 2-4 sentence message
That's what the Report agent reads. Tool payloads are already compact JSON, so the report
prompt stays small (tokens = free-tier budget) and every number has a source id.

LLM calls per specialist: 1 (choose tools, ideally all in parallel) + 1 (summary) = 2.
If a free model answers without calling any tool, we run the specialist's default tools
deterministically (no LLM) so the report still gets real data.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.runtime import Runtime

from finsight.agents.analyst import message_text
from finsight.context import PortfolioContext
from finsight.schemas.findings import AgentName, Finding, ToolCallInfo, source_id
from finsight.state import FINDING_KEYS, FinSightState, prior_conversation

SPECIALIST_RULES = """
Rules:
- Call every tool you need IN ONE TURN (parallel tool calls), then stop calling tools.
- Then reply with 2-4 sentences answering your part of the question, quoting exact figures from \
the tool outputs with their units (_pct = percent, _usd = US dollars). Never compute or estimate numbers.
- You are an analyst, not an advisor: no buy/sell recommendations.
"""

# (tool name, args, payload function) run when the model called no tools at all
DefaultTool = tuple[str, dict[str, Any], Callable[[PortfolioContext], dict[str, Any]]]


def finding_from_messages(agent: AgentName, messages: list, defaults_used: list[str] | None = None) -> Finding:
    """Pair each tool call with its ToolMessage result and build a Finding."""
    results = {m.tool_call_id: m.content for m in messages if isinstance(m, ToolMessage)}
    data: dict[str, Any] = {}
    called: list[ToolCallInfo] = []
    for m in messages:
        if isinstance(m, AIMessage):
            for tc in m.tool_calls:
                called.append(ToolCallInfo(name=tc["name"], args=tc["args"]))
                raw = results.get(tc["id"], "")
                try:
                    data[source_id(agent, tc["name"], tc["args"])] = json.loads(raw)
                except (TypeError, json.JSONDecodeError):
                    data[source_id(agent, tc["name"], tc["args"])] = {"raw": str(raw)}
    final = next((m for m in reversed(messages) if isinstance(m, AIMessage) and not m.tool_calls), None)
    summary = message_text(final).strip() if final else ""
    return Finding(agent=agent, summary=summary or "(no summary)", data=data, tools_called=called)


def make_specialist_node(
    agent: AgentName,
    model: BaseChatModel,
    tools: list,
    system_prompt: str,
    default_tools: list[DefaultTool],
):
    """Build a graph node that runs one specialist and writes its Finding into state."""
    react = create_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt + SPECIALIST_RULES,
        context_schema=PortfolioContext,
        name=f"{agent}_agent",
        # Don't inherit the parent graph's checkpointer: the inner ReAct transcript is scratch
        # work (we keep only the Finding), and an interrupted node re-runs from scratch anyway.
        checkpointer=False,
    )

    def node(state: FinSightState, runtime: Runtime[PortfolioContext]) -> dict[str, Any]:
        task = f"Question: {state['question']}\n\nEarlier conversation on this thread:\n{prior_conversation(state)}"
        try:
            result = react.invoke({"messages": [HumanMessage(content=task)]}, context=runtime.context)
        except Exception as e:  # rate limits after retries, provider outages, ...
            return {
                FINDING_KEYS[agent]: None,
                "completed_agents": [agent],  # don't retry forever; report works with what exists
                "errors": [f"{agent} agent failed: {type(e).__name__}: {str(e)[:200]}"],
            }
        messages = result["messages"]
        n_llm = sum(isinstance(m, AIMessage) for m in messages)
        finding = finding_from_messages(agent, messages)

        if not finding.data:  # model skipped tools: fetch default data deterministically
            for name, args, payload_fn in default_tools:
                finding.data[source_id(agent, name, args)] = payload_fn(runtime.context)
                finding.tools_called.append(ToolCallInfo(name=name, args=args))
            finding.summary += " [Note: model called no tools; default tool data attached automatically.]"

        return {
            FINDING_KEYS[agent]: finding.model_dump(),
            "completed_agents": [agent],
            "tool_calls": [{"agent": agent, "name": t.name, "args": t.args} for t in finding.tools_called],
            "llm_calls": n_llm,
        }

    return node
