"""Research output schemas: grounded, source-cited findings.

Two layers, on purpose:
  ResearchDraft     what the LLM writes: claims + the source ids they rely on + gaps.
                    It contains NO Source objects, so the model cannot invent a URL, title or section.
  ResearchFindings  what code assembles: the draft's insights plus Source objects taken from the
                    registry of things our tools ACTUALLY returned this run. Its validator rejects
                    any insight citing an id that isn't in `sources`, so an unsupported claim cannot
                    exist inside a valid ResearchFindings object.
That is what "grounded" means here: every claim points at retrieved text, enforced in code.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SourceType = Literal["filing", "web"]
Confidence = Literal["high", "medium", "low"]
Category = Literal["risk", "growth", "competition", "financials", "strategy", "news", "other"]


class Source(BaseModel):
    """Something a tool retrieved: a 10-K chunk or a web page."""

    model_config = ConfigDict(extra="forbid")

    id: str  # short and copyable by an LLM: "NVDA-10K-FY2026-1A-007" or "web-3f9a1c2b"
    type: SourceType
    title: str
    url: str
    ticker: str | None = None
    form: str | None = None  # "10-K" for filings
    fiscal_year: int | None = None
    section: str | None = None  # "Item 1A Risk Factors"
    published_date: str | None = None  # web only, when the search API provides it
    retrieved_at: str  # ISO timestamp of retrieval


class Insight(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim: str = Field(min_length=5, description="One sentence, supported by the cited sources")
    source_ids: list[str] = Field(min_length=1, description="Ids of the sources supporting the claim")
    confidence: Confidence
    category: Category


class ResearchFindings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tickers: list[str]
    question: str
    insights: list[Insight]
    sources: list[Source]
    gaps: list[str] = Field(default_factory=list, description="What could not be answered, and why")
    as_of: str  # ISO date the research was performed

    @model_validator(mode="after")
    def _every_citation_resolves(self) -> ResearchFindings:
        ids = [s.id for s in self.sources]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate source ids")
        known = set(ids)
        for i, ins in enumerate(self.insights):
            unknown = [sid for sid in ins.source_ids if sid not in known]
            if unknown:
                raise ValueError(f"insight {i} cites unknown source id(s): {', '.join(unknown)}")
        return self

    def source(self, source_id: str) -> Source:
        return next(s for s in self.sources if s.id == source_id)


# ---- what the LLM is asked to produce (strict structured output: all fields required) ----------


class DraftInsight(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim: str = Field(description="One sentence stating a fact found in the cited sources")
    source_ids: list[str] = Field(description="Exact ids of the sources that support this claim (at least one)")
    confidence: Confidence = Field(description="high = stated directly in the source; medium = clearly implied; low = partial")
    category: Category


class ResearchDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    insights: list[DraftInsight]
    gaps: list[str] = Field(description="Parts of the question the sources could not answer")
