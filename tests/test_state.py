"""Reducers and per-question reset."""

from langchain_core.messages import AIMessage, HumanMessage

from finsight.state import accumulate, add_or_reset, new_question_input, prior_conversation


def test_accumulate_appends_and_none_resets():
    assert accumulate(None, ["a"]) == ["a"]
    assert accumulate(["a"], ["b"]) == ["a", "b"]
    assert accumulate(["a", "b"], None) == []


def test_add_or_reset():
    assert add_or_reset(None, 2) == 2
    assert add_or_reset(2, 3) == 5
    assert add_or_reset(5, None) == 0


def test_new_question_input_resets_per_question_fields():
    i = new_question_input("How risky?", "p.csv")
    assert i["question"] == "How risky?" and i["portfolio_source"] == "p.csv"
    assert isinstance(i["messages"][0], HumanMessage)
    for k in ("completed_agents", "errors", "route_log", "tool_calls", "llm_calls", "plan", "report"):
        assert i[k] is None
    assert i["iteration"] == 0


def test_prior_conversation_excludes_current_question():
    state = {"messages": [HumanMessage("Q1"), AIMessage("A1"), HumanMessage("Q2")]}
    assert prior_conversation(state) == "User: Q1\nFinSight: A1"
    assert prior_conversation({"messages": [HumanMessage("Q1")]}) == "(none)"
