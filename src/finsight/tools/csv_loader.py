"""Load and validate a portfolio CSV into a Portfolio model.

Design: read every cell as a string, then let Pydantic do all type conversion and checks.
Errors from *all* rows are collected and raised together with CSV line numbers, so a user
fixing an upload sees every problem at once.
"""

from __future__ import annotations

import csv
import io
from os import PathLike
from typing import IO

from pydantic import ValidationError

from finsight.schemas.portfolio import Holding, Portfolio

REQUIRED_COLUMNS = ("ticker", "quantity", "avg_buy_price")
OPTIONAL_COLUMNS = ("buy_date", "asset_type", "currency")
ALLOWED_COLUMNS = REQUIRED_COLUMNS + OPTIONAL_COLUMNS


class PortfolioValidationError(ValueError):
    """Raised when a CSV cannot be turned into a valid Portfolio. `.errors` lists each problem."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("Invalid portfolio CSV:\n  - " + "\n  - ".join(errors))


def _format_pydantic_error(line: int, err: ValidationError) -> list[str]:
    out = []
    for e in err.errors():
        field = ".".join(str(p) for p in e["loc"]) or "row"
        msg = e["msg"].removeprefix("Value error, ")
        out.append(f"line {line}, {field}: {msg} (got {e.get('input')!r})")
    return out


def load_portfolio_csv(source: str | PathLike[str] | IO[str] | IO[bytes]) -> Portfolio:
    """Parse a CSV file path or file-like object (e.g. a Streamlit upload) into a Portfolio.

    Raises PortfolioValidationError listing every problem found.
    """
    if hasattr(source, "read"):
        raw = source.read()
        text = raw.decode("utf-8-sig") if isinstance(raw, bytes) else raw
    else:
        with open(source, encoding="utf-8-sig", newline="") as f:  # utf-8-sig strips Excel BOM
            text = f.read()

    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        raise PortfolioValidationError(["file is empty or has no header row"])

    columns = [c.strip().lower() for c in reader.fieldnames]
    header_errors = []
    missing = [c for c in REQUIRED_COLUMNS if c not in columns]
    unknown = [c for c in columns if c not in ALLOWED_COLUMNS]
    if missing:
        header_errors.append(f"missing required column(s): {', '.join(missing)}")
    if unknown:
        header_errors.append(
            f"unknown column(s): {', '.join(unknown)} (allowed: {', '.join(ALLOWED_COLUMNS)})"
        )
    if header_errors:
        raise PortfolioValidationError(header_errors)

    holdings: list[Holding] = []
    errors: list[str] = []
    # Header is line 1, so the first data row is line 2.
    for line, row in enumerate(csv.reader(io.StringIO(text)), start=1):
        if line == 1 or not any(cell.strip() for cell in row):
            continue  # skip header and blank lines
        if len(row) != len(columns):
            errors.append(f"line {line}: expected {len(columns)} values, got {len(row)}")
            continue
        # Blank optional cells are dropped so the model's defaults apply.
        data = {col: cell.strip() for col, cell in zip(columns, row) if cell.strip()}
        try:
            holdings.append(Holding(**data))
        except ValidationError as e:
            errors.extend(_format_pydantic_error(line, e))

    if not holdings and not errors:
        errors.append("file has a header but no holdings")
    if errors:
        raise PortfolioValidationError(errors)

    try:
        return Portfolio(holdings=holdings)
    except ValidationError as e:
        raise PortfolioValidationError([err["msg"].removeprefix("Value error, ") for err in e.errors()]) from e
