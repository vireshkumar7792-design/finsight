"""Report agent: turns the agents' findings into a structured Report with traceable key metrics."""

import json

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.runtime import Runtime

from finsight.agents.common import ask_structured, short_error
from finsight.schemas import Report
from finsight.state import FinSightState, RunContext, earlier_conversation

DISCLAIMER = "FinSight is an analysis tool, not a financial advisor. This is not financial advice."

# =========================
# Prompt
# =========================

PROMPT = """You write the final report for FinSight, a portfolio analysis tool. Answer the
user's question using ONLY the findings provided.

Rules:
- Copy every number exactly from a finding's "data". Never calculate, round or convert.
  _pct fields are already percentages; _usd fields are US dollars.
- Don't create new numbers like differences, sums or "three times higher". To compare,
  state both figures (e.g. "44.43% vs 14.95% for SPY").
- key_metrics: 3-8 headline numbers. "source" must be the exact data key the number came
  from (e.g. "get_risk_metrics"). unit: "%" for _pct, "USD" for _usd, "ratio" otherwise.
- sections: 1-4 short sections explaining what the numbers mean in plain language.
- If research insights are given, keep their source ids in square brackets after the
  sentence, e.g. "... export controls [NVDA-10K-1A-004]". Never invent ids.
- If an agent failed or data is missing, say so in caveats. Never fill gaps yourself.
- Describe and explain; do not recommend buying or selling.
- Mention the data's as_of date in the summary."""

# =========================
# Node
# =========================


def report_node(state: FinSightState, runtime: Runtime[RunContext]) -> dict:
    errors = list(state["errors"])
    try:
        report = ask_structured(runtime.context.report_llm, Report, build_messages(state))
    except Exception as error:
        errors.append(f"report LLM failed ({short_error(error)}); used the automatic summary")
        report = fallback_report(state)

    report, dropped = keep_traceable_metrics(report, state["findings"])
    if dropped:
        report.caveats.append(f"Removed metrics not found in the tool data: {', '.join(dropped)}.")

    return {
        "report": report.model_dump(),
        "errors": errors,
        # Conversation memory: the next question on this thread can see this answer.
        "messages": [AIMessage(content=f"{report.title}: {report.summary}")],
    }


# =========================
# Helpers
# =========================


def build_messages(state: FinSightState) -> list:
    question_text = (
        f"Question: {state['question']}\n\n"
        f"Earlier conversation:\n{earlier_conversation(state)}\n\n"
        f"Agent errors: {state['errors'] or 'none'}\n\n"
        f"Findings (JSON):\n{json.dumps(state['findings'])}"
    )
    return [SystemMessage(content=PROMPT), HumanMessage(content=question_text)]


def fallback_report(state: FinSightState) -> Report:
    """A plain report built without the LLM: just the agents' own summaries."""
    sections = []
    for agent, finding in state["findings"].items():
        sections.append({"heading": f"{agent.title()} findings", "body": finding["summary"]})
    return Report(
        title="FinSight report (automatic summary)",
        summary=f"The report writer was unavailable, so these are the agents' own notes for: {state['question']}",
        sections=sections,
        key_metrics=[],
        caveats=["The report LLM failed twice; this summary was built without it."],
    )


def keep_traceable_metrics(report: Report, findings: dict) -> tuple[Report, list[str]]:
    """Keep a key metric only if its value really appears in the tool output it cites."""
    all_data = {}
    for finding in findings.values():
        all_data.update(finding["data"])

    kept = []
    dropped = []
    for metric in report.key_metrics:
        source_data = all_data.get(metric.source)
        if source_data is not None and metric.value in numbers_in(source_data):
            kept.append(metric)
        else:
            dropped.append(metric.name)
    report.key_metrics = kept
    return report, dropped


def numbers_in(value) -> set[float]:
    """Every number inside a nested dict/list of tool output."""
    if isinstance(value, bool):
        return set()
    if isinstance(value, (int, float)):
        return {float(value)}
    if isinstance(value, dict):
        value = list(value.values())
    found = set()
    if isinstance(value, list):
        for item in value:
            found.update(numbers_in(item))
    return found
