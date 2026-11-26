"""The agent loop, offline: a scripted fake chat model stands in for the LLM.

Script: 1st model call -> emit a tool call; 2nd model call -> final answer.
This proves the loop runs our tool with the injected context and feeds the result back.
"""

import json

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import PrivateAttr

from finsight.agents.analyst import SYSTEM_PROMPT, ask, build_analyst
from finsight.config import get_settings


class ScriptedChatModel(BaseChatModel):
    """Returns pre-written AIMessages in order and records what it was shown."""

    script: list[AIMessage]
    _calls: list = PrivateAttr(default_factory=list)
    _bound_tools: list = PrivateAttr(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        self._bound_tools = list(tools)
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self._calls.append(list(messages))
        return ChatResult(generations=[ChatGeneration(message=self.script[len(self._calls) - 1])])


def test_loop_calls_tool_with_context_and_returns_answer(fake_ctx):
    model = ScriptedChatModel(
        script=[
            AIMessage(content="", tool_calls=[{"name": "get_allocation", "args": {"by": "sector"}, "id": "call_1"}]),
            AIMessage(content="Technology is 48% of your portfolio. Not financial advice."),
        ]
    )
    result = ask("What's my sector exposure?", fake_ctx, build_analyst(model=model))

    # The model was offered all five tools.
    assert {t.name for t in model._bound_tools} == {
        "get_portfolio_summary", "get_allocation", "get_risk_metrics", "get_correlations", "get_position_detail"
    }
    # First call saw the system prompt + the question.
    first = model._calls[0]
    assert isinstance(first[0], SystemMessage) and first[0].content == SYSTEM_PROMPT
    # Second call saw the tool result, computed from OUR context (not from LLM-supplied data).
    tool_msg = model._calls[1][-1]
    assert isinstance(tool_msg, ToolMessage) and tool_msg.tool_call_id == "call_1"
    payload = json.loads(tool_msg.content)
    assert payload["allocation"][0] == {"group": "Technology", "weight_pct": 48.0, "tickers": ["AAA"]}

    assert result.answer.startswith("Technology is 48%")
    assert [(c.name, c.args) for c in result.tool_calls] == [("get_allocation", {"by": "sector"})]


def test_loop_handles_multiple_tool_calls(fake_ctx):
    model = ScriptedChatModel(
        script=[
            AIMessage(content="", tool_calls=[
                {"name": "get_risk_metrics", "args": {}, "id": "a"},
                {"name": "get_correlations", "args": {"top_n": 1}, "id": "b"},
            ]),
            AIMessage(content="", tool_calls=[{"name": "get_position_detail", "args": {"ticker": "ZZZ"}, "id": "c"}]),
            AIMessage(content="Done."),
        ]
    )
    result = ask("Is my portfolio diversified?", fake_ctx, build_analyst(model=model))
    assert [c.name for c in result.tool_calls] == ["get_risk_metrics", "get_correlations", "get_position_detail"]
    assert "not in the portfolio" in result.tool_calls[2].output  # error goes back to the LLM, no crash
    assert result.answer == "Done."


@pytest.mark.llm
def test_real_llm_uses_tool_for_numbers(fake_ctx):
    """Real Groq call (router role, cheapest) over the FAKE offline portfolio: isolates LLM behavior."""
    s = get_settings()
    if s.groq_api_key is None:
        pytest.skip("GROQ_API_KEY not set")
    result = ask("What is the total market value of my portfolio?", fake_ctx, build_analyst(role="router"))
    assert "get_portfolio_summary" in [c.name for c in result.tool_calls]
    assert "2,500" in result.answer or "2500" in result.answer
