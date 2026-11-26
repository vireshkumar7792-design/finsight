"""The final report schema the Report agent must produce (structured output).

Groq strict json_schema mode requires every field to be required (no defaults/optionals),
so this model has none. `source_tool` must be one of the Finding.data keys, which keeps every
headline number traceable to the tool call that produced it (full checks come in Phase 6).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class Section(BaseModel):
    model_config = ConfigDict(extra="forbid")
    heading: str
    body: str


class KeyMetric(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(description="e.g. 'Annualized volatility'")
    value: float = Field(description="Exact number copied from a finding's data, no rounding or arithmetic")
    unit: str = Field(description="'%' for *_pct fields, 'USD' for *_usd fields, 'ratio' for unitless")
    source_tool: str = Field(description="The finding data key this value was copied from")


class Report(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str
    summary: str = Field(description="2-4 sentence direct answer to the question")
    sections: list[Section]
    key_metrics: list[KeyMetric]
    caveats: list[str]
    disclaimer: str


def render_report(r: Report | dict) -> str:
    """Readable plain-text rendering for the CLI."""
    r = Report.model_validate(r)
    lines = [r.title, "=" * len(r.title), "", r.summary, ""]
    for s in r.sections:
        lines += [f"## {s.heading}", s.body, ""]
    if r.key_metrics:
        lines.append("Key metrics:")
        lines += [f"  - {m.name}: {m.value:g} {m.unit}   [source: {m.source_tool}]" for m in r.key_metrics]
        lines.append("")
    if r.caveats:
        lines.append("Caveats:")
        lines += [f"  - {c}" for c in r.caveats]
        lines.append("")
    lines.append(r.disclaimer)
    return "\n".join(lines)
