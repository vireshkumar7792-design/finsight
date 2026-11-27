"""What a specialist agent hands to the rest of the graph.

A Finding is deliberately NOT a chat transcript. It holds:
  data     the exact tool payloads, keyed by a source id like "risk:get_risk_metrics" or
           "portfolio:get_allocation(by=sector)". These are the numbers of record.
  summary  the specialist's short interpretation (its final LLM message).
The report agent reads Findings only, so its prompt stays small and every number it can
quote has a traceable source id.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

AgentName = Literal["portfolio", "risk", "research"]


class Finding(BaseModel):
    agent: AgentName
    summary: str
    data: dict[str, Any] = Field(default_factory=dict)  # source_id -> parsed tool payload
    tools_called: list[dict[str, Any]] = Field(default_factory=list)  # [{"name": ..., "args": ...}]


def source_id(agent: str, tool_name: str, args: dict[str, Any]) -> str:
    """Stable id for one tool result, used as Report.key_metrics[].source_tool."""
    arg_str = ",".join(f"{k}={v}" for k, v in sorted(args.items()))
    return f"{agent}:{tool_name}({arg_str})" if arg_str else f"{agent}:{tool_name}"
