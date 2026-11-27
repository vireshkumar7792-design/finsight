"""CSV loading: the sample files load, and bad files list every error with its line number."""

import io

import pytest

from finsight.portfolio.loader import load_portfolio_csv
from tests.conftest import SAMPLES_DIR


def load_text(text: str):
    return load_portfolio_csv(io.StringIO(text))


def test_all_sample_files_load():
    for name in ["balanced.csv", "tech_heavy.csv", "concentrated_risky.csv"]:
        portfolio = load_portfolio_csv(SAMPLES_DIR / name)
        assert len(portfolio.holdings) >= 4


def test_values_are_cleaned_and_defaults_applied():
    portfolio = load_text("ticker,quantity,avg_buy_price\n brk.b ,2.5,300\n")
    holding = portfolio.holdings[0]
    assert holding.ticker == "BRK.B"
    assert holding.quantity == 2.5
    assert holding.asset_type == "equity"
    assert holding.currency == "USD"


def test_every_bad_row_is_reported_with_its_line_number():
    text = (
        "ticker,quantity,avg_buy_price,buy_date,asset_type,currency\n"
        "AAPL,10,150,2024-01-01,equity,USD\n"  # line 2: fine
        "MSFT,-5,300,,equity,USD\n"  # line 3: negative quantity
        "GOOG,1,100,2999-01-01,equity,USD\n"  # line 4: future date
        "TSLA,1,100,,stock,USD\n"  # line 5: unknown asset type
        "CASH,500,2,,cash,USD\n"  # line 6: cash must cost $1
        "AMZN,1\n"  # line 7: too few values
    )
    with pytest.raises(ValueError) as error:
        load_text(text)
    message = str(error.value)
    assert "line 3, quantity" in message
    assert "line 4, buy_date" in message
    assert "line 5, asset_type" in message
    assert "line 6: cash must have avg_buy_price 1.0" in message
    assert "line 7: expected 6 values, got 2" in message
    assert "line 2" not in message


def test_blank_lines_are_skipped():
    portfolio = load_text("ticker,quantity,avg_buy_price\nAAPL,1,100\n\nMSFT,1,200\n")
    assert portfolio.tickers == ["AAPL", "MSFT"]


def test_missing_and_unknown_columns():
    with pytest.raises(ValueError) as error:
        load_text("ticker,qty,avg_buy_price\nAAPL,1,100\n")
    message = str(error.value)
    assert "missing required column(s): quantity" in message
    assert "unknown column(s): qty" in message


def test_duplicate_tickers_are_rejected():
    with pytest.raises(ValueError, match="duplicate tickers: AAPL"):
        load_text("ticker,quantity,avg_buy_price\nAAPL,1,100\naapl,2,100\n")


def test_empty_file_and_header_only():
    with pytest.raises(ValueError, match="the file is empty"):
        load_text("")
    with pytest.raises(ValueError, match="no holdings"):
        load_text("ticker,quantity,avg_buy_price\n")


def test_excel_bom_is_removed():
    portfolio = load_portfolio_csv(io.BytesIO("﻿ticker,quantity,avg_buy_price\nAAPL,1,100\n".encode()))
    assert portfolio.tickers == ["AAPL"]
