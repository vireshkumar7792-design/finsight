"""Read a portfolio CSV into a validated Portfolio, reporting every error with its line number."""

import csv
import io
from pathlib import Path

from pydantic import ValidationError

from finsight.schemas import Holding, Portfolio

REQUIRED_COLUMNS = ["ticker", "quantity", "avg_buy_price"]
OPTIONAL_COLUMNS = ["buy_date", "asset_type", "currency"]
ALLOWED_COLUMNS = REQUIRED_COLUMNS + OPTIONAL_COLUMNS


# =========================
# Public function
# =========================


def load_portfolio_csv(source) -> Portfolio:
    """Load a CSV from a file path or an open file (e.g. a web upload).

    Raises ValueError listing all problems at once.
    """
    text = read_text(source)
    reader = csv.reader(io.StringIO(text))

    header = next(reader, None)
    if not header:
        raise_errors(["the file is empty"])
    columns = [name.strip().lower() for name in header]
    raise_errors(check_columns(columns))

    holdings = []
    errors = []
    for row in reader:
        line_number = reader.line_num
        if not any(cell.strip() for cell in row):
            continue  # skip blank lines
        if len(row) != len(columns):
            errors.append(f"line {line_number}: expected {len(columns)} values, got {len(row)}")
            continue
        # Leave out empty cells so the model's defaults are used.
        values = {}
        for column, cell in zip(columns, row):
            if cell.strip():
                values[column] = cell.strip()
        try:
            holdings.append(Holding(**values))
        except ValidationError as error:
            errors.extend(format_errors(error, f"line {line_number}"))

    if not holdings and not errors:
        errors.append("the file has a header but no holdings")
    raise_errors(errors)

    try:
        return Portfolio(holdings=holdings)
    except ValidationError as error:
        raise_errors(format_errors(error, "portfolio"))


# =========================
# Helpers
# =========================


def read_text(source) -> str:
    if hasattr(source, "read"):
        content = source.read()
    else:
        content = Path(source).read_bytes()
    if isinstance(content, bytes):
        # "utf-8-sig" removes the invisible marker Excel puts at the start of a file.
        content = content.decode("utf-8-sig")
    return content


def check_columns(columns: list[str]) -> list[str]:
    errors = []
    missing = [name for name in REQUIRED_COLUMNS if name not in columns]
    unknown = [name for name in columns if name not in ALLOWED_COLUMNS]
    if missing:
        errors.append(f"missing required column(s): {', '.join(missing)}")
    if unknown:
        errors.append(f"unknown column(s): {', '.join(unknown)} (allowed: {', '.join(ALLOWED_COLUMNS)})")
    return errors


def format_errors(error: ValidationError, where: str) -> list[str]:
    """Turn Pydantic's error list into short readable lines."""
    messages = []
    for item in error.errors():
        message = item["msg"].removeprefix("Value error, ")
        if item["loc"]:
            field = ".".join(str(part) for part in item["loc"])
            messages.append(f"{where}, {field}: {message} (got {item['input']!r})")
        else:
            # A whole-row check (e.g. cash price) has no single field.
            messages.append(f"{where}: {message}")
    return messages


def raise_errors(errors: list[str]) -> None:
    if errors:
        raise ValueError("Invalid portfolio CSV:\n  - " + "\n  - ".join(errors))
