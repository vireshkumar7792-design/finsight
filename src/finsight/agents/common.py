"""Helpers shared by the agents: structured output with one retry, and running a tool-calling agent."""

import json

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from finsight.state import FinSightState, RunContext, earlier_conversation

# =========================
# Structured output (try, then retry once)
# =========================


def ask_structured(llm, schema, messages):
    """Ask the LLM to fill a Pydantic schema. Tries twice; if both fail, raises the last error."""
    structured_llm = llm.with_structured_output(schema, method="json_schema", strict=True)
    last_error = None
    for attempt in range(2):
        try:
            result = structured_llm.invoke(messages)
            return schema.model_validate(result)
        except Exception as error:
            last_error = error
    raise last_error


def short_error(error: Exception) -> str:
    return f"{type(error).__name__}: {str(error)[:150]}"


# =========================
# Tool-calling agent (used by Portfolio and Risk)
# =========================

AGENT_RULES = """
Rules:
- Call all the tools you need in ONE turn, then stop calling tools.
- Then answer your part of the question in 2-4 sentences, quoting exact figures from the
  tool outputs with their units (_pct = percent, _usd = US dollars). Never calculate numbers.
- You are an analyst, not an advisor: no buy or sell recommendations."""


def run_tool_agent(name, prompt, tools, default_tools, state: FinSightState, run_context: RunContext) -> dict:
    """Run a small agent with its own tools and turn its work into a "finding" for the report.

    default_tools: [(tool_name, args, payload_function)] used when the LLM fails or calls no tools.
    """
    agent = create_agent(
        model=run_context.agent_llm,
        tools=tools,
        system_prompt=prompt + AGENT_RULES,
        context_schema=RunContext,
        # The inner agent's chat is scratch work, so it is not saved in our checkpoints.
        checkpointer=False,
    )
    task = f"Question: {state['question']}\n\nEarlier conversation:\n{earlier_conversation(state)}"
    errors = list(state["errors"])

    messages = []
    for attempt in range(2):
        try:
            result = agent.invoke({"messages": [HumanMessage(content=task)]}, context=run_context)
            messages = result["messages"]
            break
        except Exception as error:
            errors.append(f"{name} agent attempt {attempt + 1} failed: {short_error(error)}")

    summary, data, calls = read_agent_messages(messages)

    if not data:
        # Safe fallback without the LLM: run the default tools so the report still gets real numbers.
        for tool_name, args, payload_function in default_tools:
            data[source_id(tool_name, args)] = payload_function(run_context.portfolio, **args)
            calls.append({"name": tool_name, "args": args})
        summary = summary or "(The model gave no summary; default tool data was attached.)"

    return {
        "findings": {**state["findings"], name: {"summary": summary, "data": data}},
        "completed_agents": state["completed_agents"] + [name],
        "tool_calls": state["tool_calls"] + [{"agent": name, **call} for call in calls],
        "errors": errors,
    }


def read_agent_messages(messages: list) -> tuple[str, dict, list]:
    """Pair each tool call with its result. Returns (final summary, data by source id, calls)."""
    results = {}
    for message in messages:
        if isinstance(message, ToolMessage):
            results[message.tool_call_id] = message.content

    data = {}
    calls = []
    for message in messages:
        if not isinstance(message, AIMessage):
            continue
        for call in message.tool_calls:
            calls.append({"name": call["name"], "args": call["args"]})
            try:
                data[source_id(call["name"], call["args"])] = json.loads(results[call["id"]])
            except (KeyError, ValueError):
                pass  # the tool failed or returned an error message: no data to keep

    summary = ""
    if messages and isinstance(messages[-1], AIMessage):
        summary = messages[-1].text.strip()
    return summary, data, calls


def source_id(tool_name: str, args: dict) -> str:
    """A readable id for one tool result, e.g. "get_allocation(by=sector)". Reports cite these."""
    if not args:
        return tool_name
    arg_text = ",".join(f"{key}={value}" for key, value in sorted(args.items()))
    return f"{tool_name}({arg_text})"
