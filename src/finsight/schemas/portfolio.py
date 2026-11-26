"""Portfolio input schemas: a validated Holding (one CSV row) and a Portfolio (all rows).

Once data passes these models, downstream code can trust it: tickers are normalized,
quantities and prices are positive, dates are not in the future, currency is USD.
"""

from __future__ import annotations

import re
from datetime import date
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# US tickers: 1-10 chars, letters/digits plus "." or "-" for share classes (BRK.B, BF-B).
_TICKER_RE = re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}$")

CASH_PRICE = 1.0  # cash rows: quantity = USD amount, unit price fixed at $1


class AssetType(StrEnum):
    """Economic exposure of a holding (used for allocation), not its legal wrapper."""

    EQUITY = "equity"
    ETF = "etf"
    BOND = "bond"
    CRYPTO = "crypto"
    CASH = "cash"


class Holding(BaseModel):
    """One position in the portfolio, i.e. one validated CSV row."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    ticker: str
    quantity: float = Field(gt=0, description="Shares/units held; fractional allowed.")
    avg_buy_price: float = Field(gt=0, description="Average cost per unit in USD.")
    buy_date: date | None = None
    asset_type: AssetType = AssetType.EQUITY
    currency: Literal["USD"] = "USD"

    @field_validator("ticker", mode="before")
    @classmethod
    def _normalize_ticker(cls, v: object) -> object:
        if isinstance(v, str):
            v = v.strip().upper()
            if not _TICKER_RE.match(v):
                raise ValueError(f"invalid ticker {v!r} (expected e.g. AAPL, BRK.B)")
        return v

    @field_validator("asset_type", mode="before")
    @classmethod
    def _lower_asset_type(cls, v: object) -> object:
        return v.strip().lower() if isinstance(v, str) else v

    @field_validator("currency", mode="before")
    @classmethod
    def _upper_currency(cls, v: object) -> object:
        return v.strip().upper() if isinstance(v, str) else v

    @field_validator("buy_date")
    @classmethod
    def _not_in_future(cls, v: date | None) -> date | None:
        if v is not None and v > date.today():
            raise ValueError(f"buy_date {v} is in the future")
        return v

    @model_validator(mode="after")
    def _cash_is_priced_at_one(self) -> Holding:
        # For cash, quantity is the dollar amount and the unit price is fixed at $1.
        if self.asset_type is AssetType.CASH and self.avg_buy_price != CASH_PRICE:
            raise ValueError(f"cash must have avg_buy_price {CASH_PRICE} (quantity is the USD amount)")
        return self

    @property
    def is_cash(self) -> bool:
        return self.asset_type is AssetType.CASH

    @property
    def cost_basis(self) -> float:
        """Total amount paid for this position."""
        return self.quantity * self.avg_buy_price


class Portfolio(BaseModel):
    """A non-empty set of holdings with unique tickers."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    holdings: list[Holding] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_tickers(self) -> Portfolio:
        seen: set[str] = set()
        dupes: set[str] = set()
        for h in self.holdings:
            if h.ticker in seen:
                dupes.add(h.ticker)
            seen.add(h.ticker)
        if dupes:
            raise ValueError(f"duplicate tickers: {', '.join(sorted(dupes))} (merge them into one row)")
        return self

    @property
    def tickers(self) -> list[str]:
        return [h.ticker for h in self.holdings]

    @property
    def market_tickers(self) -> list[str]:
        """Tickers that need market data (everything except cash)."""
        return [h.ticker for h in self.holdings if not h.is_cash]

    def get(self, ticker: str) -> Holding | None:
        t = ticker.strip().upper()
        return next((h for h in self.holdings if h.ticker == t), None)
