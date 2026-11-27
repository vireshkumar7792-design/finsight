"""All Pydantic models in one place: portfolio input, supervisor plan, final report and research."""

import re
from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

# =========================
# Portfolio input
# =========================

# US tickers: letters/digits, plus "." or "-" for share classes (BRK.B, BF-B).
TICKER_PATTERN = re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}$")

# A cash row's quantity is the dollar amount, so its price is always $1.
CASH_PRICE = 1.0

AssetType = Literal["equity", "etf", "bond", "crypto", "cash"]


class Holding(BaseModel):
    """One position = one row of the portfolio CSV."""

    ticker: str
    quantity: float = Field(gt=0)
    avg_buy_price: float = Field(gt=0)
    buy_date: date | None = None
    asset_type: AssetType = "equity"
    currency: Literal["USD"] = "USD"

    @field_validator("ticker", mode="before")
    @classmethod
    def clean_ticker(cls, value):
        ticker = str(value).strip().upper()
        if not TICKER_PATTERN.match(ticker):
            raise ValueError(f"invalid ticker '{ticker}' (expected something like AAPL or BRK.B)")
        return ticker

    @field_validator("asset_type", mode="before")
    @classmethod
    def clean_asset_type(cls, value):
        return str(value).strip().lower()

    @field_validator("currency", mode="before")
    @classmethod
    def clean_currency(cls, value):
        return str(value).strip().upper()

    @field_validator("buy_date")
    @classmethod
    def date_not_in_future(cls, value):
        if value is not None and value > date.today():
            raise ValueError(f"buy_date {value} is in the future")
        return value

    @model_validator(mode="after")
    def cash_costs_one_dollar(self):
        if self.asset_type == "cash" and self.avg_buy_price != CASH_PRICE:
            raise ValueError("cash must have avg_buy_price 1.0 (quantity is the dollar amount)")
        return self

    @property
    def is_cash(self) -> bool:
        return self.asset_type == "cash"

    @property
    def cost_basis(self) -> float:
        return self.quantity * self.avg_buy_price


class Portfolio(BaseModel):
    """All holdings. At least one, and each ticker only once."""

    holdings: list[Holding] = Field(min_length=1)

    @model_validator(mode="after")
    def tickers_are_unique(self):
        seen = set()
        duplicates = set()
        for holding in self.holdings:
            if holding.ticker in seen:
                duplicates.add(holding.ticker)
            seen.add(holding.ticker)
        if duplicates:
            names = ", ".join(sorted(duplicates))
            raise ValueError(f"duplicate tickers: {names} (merge them into one row)")
        return self

    @property
    def tickers(self) -> list[str]:
        return [holding.ticker for holding in self.holdings]

    @property
    def market_tickers(self) -> list[str]:
        """Tickers that need market data: everything except cash."""
        return [holding.ticker for holding in self.holdings if not holding.is_cash]


# =========================
# Supervisor plan (structured output)
# =========================

AgentName = Literal["portfolio", "risk", "research"]


class RoutePlan(BaseModel):
    """The supervisor's one decision per question: which agents to run, in order."""

    model_config = {"extra": "forbid"}  # strict JSON mode needs "no extra fields"

    agents: list[AgentName] = Field(description="Agents to run in order. Empty list if none are needed.")
    reason: str = Field(description="One short sentence explaining the choice")


# =========================
# Final report (structured output)
# =========================

# Strict JSON mode needs every field to be required, so these models have no defaults.


class Section(BaseModel):
    model_config = {"extra": "forbid"}

    heading: str
    body: str


class KeyMetric(BaseModel):
    model_config = {"extra": "forbid"}

    name: str = Field(description="e.g. 'Annualized volatility'")
    value: float = Field(description="Number copied exactly from the findings data")
    unit: str = Field(description="'%' for _pct fields, 'USD' for _usd fields, 'ratio' otherwise")
    source: str = Field(description="The findings data key the value was copied from")


class Report(BaseModel):
    model_config = {"extra": "forbid"}

    title: str
    summary: str = Field(description="2-4 sentences that directly answer the question")
    sections: list[Section]
    key_metrics: list[KeyMetric]
    caveats: list[str]


# =========================
# Research (grounded, cited)
# =========================


class Source(BaseModel):
    """Something a tool really returned: a 10-K chunk or a web page. Only code creates these."""

    id: str  # "NVDA-10K-FY2026-1A-007" or "web-3f9a1c2b"
    type: Literal["filing", "web"]
    title: str
    url: str
    section: str | None = None
    published_date: str | None = None


class Insight(BaseModel):
    """One claim from the research writer. It may only point at sources by id."""

    model_config = {"extra": "forbid"}

    claim: str = Field(description="One sentence stating something the cited evidence says")
    source_ids: list[str] = Field(description="Exact source_id(s) of the evidence behind the claim")
    confidence: Literal["high", "medium", "low"] = Field(
        description="high = stated directly; medium = clearly implied; low = partly supported"
    )


class ResearchDraft(BaseModel):
    """What the research writer LLM returns (structured output)."""

    model_config = {"extra": "forbid"}

    insights: list[Insight]
    gaps: list[str] = Field(description="Parts of the question the evidence could not answer")
