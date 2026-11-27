"""Report agent: turns the specialists' Findings into a structured Report.

Input is ONLY the findings (tool payloads + short summaries) and the conversation so far, never
specialist transcripts, so the prompt stays small and every number has a source id.

Fallback chain (a run must never crash on a flaky free model):
  1. structured Report (strict JSON schema), retried once
  2. plain-text answer from the same model, wrapped in a Report
  3. deterministic Report assembled from finding summaries (no LLM)
Then a light traceability check: each key metric's source_tool must exist and its value must
appear in that source's payload; anything else is dropped with a caveat. (Full number-consistency
guardrails come in Phase 6.)
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from finsight.agents.structured import message_text, structured_invoke
from finsight.schemas.report import Report
from finsight.state import FINDING_KEYS, FinSightState, prior_conversation

DISCLAIMER = "FinSight is an analysis tool, not a financial advisor. Not financial advice."
STRUCTURED_ATTEMPTS = 2  # 1 try + 1 retry before the plain-text fallback

REPORT_PROMPT = """You are the Report writer for FinSight, a portfolio analysis tool. Write a clear, \
concise report answering the user's question using ONLY the findings provided.

Rules:
- Every number must be copied exactly from a finding's `data` (no arithmetic, rounding or unit \
conversion). _pct fields are already percentages; _usd fields are US dollars.
- Do not derive new numbers, including multiples ("three times higher"), differences or sums. \
Compare by stating both figures (e.g. "44.43% vs 14.95% for SPY").
- key_metrics: 3-8 headline numbers. `source_tool` must be the exact data key the value came from \
(e.g. "risk:get_risk_metrics"). unit: "%" for _pct, "USD" for _usd, "ratio" otherwise.
- sections: 1-4 short sections with plain-language interpretation; compare with the benchmark when \
the data includes one.
- Research findings contain cited insights: when you use one, keep its source ids in square brackets after the sentence, e.g. "... export controls [NVDA-10K-FY2026-1A-004]". Never cite ids that aren't there.
- If an agent failed or research lists gaps, say so in caveats; never fill the gap yourself.
- Describe and interpret; do not recommend buying or selling.
- Mention the data's as_of date in the summary."""


def collect_findings(state: FinSightState) -> dict[str, dict[str, Any]]:
    return {agent: state[key] for agent, key in FINDING_KEYS.items() if state.get(key)}


def _numbers(obj: Any) -> set[float]:
    """All numeric leaves in a payload (for the traceability check)."""
    if isinstance(obj, bool):
        return set()
    if isinstance(obj, int | float):
        return {float(obj)}
    if isinstance(obj, dict):
        return set().union(*(_numbers(v) for v in obj.values())) if obj else set()
    if isinstance(obj, list):
        return set().union(*(_numbers(v) for v in obj)) if obj else set()
    return set()


def enforce_traceability(report: Report, findings: dict[str, dict[str, Any]]) -> Report:
    """Keep only key metrics whose source exists and whose value appears in that source."""
    sources = {sid: payload for f in findings.values() for sid, payload in f.get("data", {}).items()}
    kept, dropped = [], []
    for m in report.key_metrics:
        payload = sources.get(m.source_tool)
        if payload is not None and float(m.value) in _numbers(payload):
            kept.append(m)
        else:
            dropped.append(m.name)
    caveats = list(report.caveats)
    if dropped:
        caveats.append(f"Removed {len(dropped)} metric(s) not traceable to tool data: {', '.join(dropped)}.")
    return report.model_copy(update={"key_metrics": kept, "caveats": caveats, "disclaimer": DISCLAIMER})


def _build_messages(state: FinSightState, findings: dict[str, dict[str, Any]]) -> list:
    compact = {
        agent: {"summary": f["summary"], "data": f["data"]}
        for agent, f in findings.items()
    }
    body = (
        f"Question: {state['question']}\n\n"
        f"Earlier conversation:\n{prior_conversation(state)}\n\n"
        f"Routing plan: {(state.get('plan') or {}).get('reason', 'n/a')}\n"
        f"Agent errors: {state.get('errors') or 'none'}\n\n"
        f"Findings (JSON):\n{json.dumps(compact, separators=(',', ':'))}"
    )
    return [SystemMessage(content=REPORT_PROMPT), HumanMessage(content=body)]


def deterministic_report(state: FinSightState, findings: dict[str, dict[str, Any]], why: str) -> Report:
    sections = [{"heading": f"{a.title()} findings", "body": f["summary"]} for a, f in findings.items()]
    return Report.model_validate({
        "title": "FinSight report (automatic fallback)",
        "summary": f"The report writer was unavailable, so this report lists the specialists' findings "
                   f"for: {state['question']}",
        "sections": sections or [{"heading": "No findings", "body": "No specialist produced data for this question."}],
        "key_metrics": [],
        "caveats": [f"Report LLM unavailable: {why}"],
        "disclaimer": DISCLAIMER,
    })


def make_report_node(llm: BaseChatModel):
    def report(state: FinSightState) -> dict[str, Any]:
        findings = collect_findings(state)
        messages = _build_messages(state, findings)
        errors: list[str] = []
        calls = 0
        result: Report | None = None
        mode = "structured"

        for _ in range(STRUCTURED_ATTEMPTS):
            calls += 1
            try:
                result = structured_invoke(llm, Report, messages)
                break
            except Exception as e:
                errors.append(f"report structured output failed: {str(e)[:200]}")

        if result is None:  # fallback 2: plain text from the same model
            mode = "plain_text_fallback"
            calls += 1
            try:
                text = message_text(llm.invoke(messages + [HumanMessage(content="Answer in plain text (no JSON).")])).strip()
                result = Report(title="FinSight report", summary=text or "(empty response)", sections=[],
                                key_metrics=[], caveats=["Structured output failed; plain-text report."],
                                disclaimer=DISCLAIMER)
            except Exception as e:  # fallback 3: no LLM at all
                mode = "deterministic_fallback"
                errors.append(f"report plain-text fallback failed: {str(e)[:200]}")
                result = deterministic_report(state, findings, str(e)[:150])

        result = enforce_traceability(result, findings)
        return {
            "report": result.model_dump(),
            "report_mode": mode,
            "llm_calls": calls,
            "errors": errors,
            # Conversation memory: the next question on this thread sees this answer.
            "messages": [AIMessage(content=f"{result.title}: {result.summary}")],
        }

    return report
