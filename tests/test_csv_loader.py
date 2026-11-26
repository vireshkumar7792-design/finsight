"""CSV -> Portfolio loading, including clear errors for bad files."""

import io

import pytest

from finsight.schemas.portfolio import AssetType
from finsight.tools.csv_loader import PortfolioValidationError, load_portfolio_csv

HEADER = "ticker,quantity,avg_buy_price,buy_date,asset_type,currency\n"


@pytest.mark.parametrize(
    "name, n_holdings", [("balanced.csv", 9), ("tech_heavy.csv", 6), ("concentrated_risky.csv", 4)]
)
def test_sample_csvs_validate(samples_dir, name, n_holdings):
    p = load_portfolio_csv(samples_dir / name)
    assert len(p.holdings) == n_holdings
    assert all(h.currency == "USD" for h in p.holdings)


def test_blank_optional_cell_uses_default(samples_dir):
    p = load_portfolio_csv(samples_dir / "balanced.csv")
    agg = next(h for h in p.holdings if h.ticker == "AGG")
    assert agg.buy_date is None
    assert agg.asset_type is AssetType.BOND


def test_minimal_columns_and_file_like_bytes():
    # Streamlit uploads are bytes file-likes, possibly with an Excel BOM.
    data = "﻿ticker,quantity,avg_buy_price\nmsft,3,300\n".encode()
    p = load_portfolio_csv(io.BytesIO(data))
    assert p.holdings[0].ticker == "MSFT"
    assert p.holdings[0].asset_type is AssetType.EQUITY


def test_blank_lines_are_skipped():
    p = load_portfolio_csv(io.StringIO(HEADER + "AAPL,1,100,,,\n\n,,,,,\nMSFT,1,100,,,\n"))
    assert p.tickers == ["AAPL", "MSFT"]


def test_collects_all_row_errors_with_line_numbers():
    csv_text = HEADER + "AAPL,-5,100,,,\nMSFT,1,100,2024-01-01,stock,USD\nGOOG,1,100,,,EUR\n"
    with pytest.raises(PortfolioValidationError) as exc:
        load_portfolio_csv(io.StringIO(csv_text))
    errors = exc.value.errors
    assert len(errors) == 3
    assert errors[0].startswith("line 2, quantity")
    assert errors[1].startswith("line 3, asset_type")
    assert errors[2].startswith("line 4, currency")


def test_missing_required_column():
    with pytest.raises(PortfolioValidationError, match="missing required column.*avg_buy_price"):
        load_portfolio_csv(io.StringIO("ticker,quantity\nAAPL,1\n"))


def test_unknown_column():
    with pytest.raises(PortfolioValidationError, match="unknown column.*sector"):
        load_portfolio_csv(io.StringIO("ticker,quantity,avg_buy_price,sector\nAAPL,1,1,Tech\n"))


def test_wrong_number_of_values():
    with pytest.raises(PortfolioValidationError, match="line 2: expected 6 values, got 3"):
        load_portfolio_csv(io.StringIO(HEADER + "AAPL,1,100\n"))


def test_empty_file_and_header_only():
    with pytest.raises(PortfolioValidationError, match="empty"):
        load_portfolio_csv(io.StringIO(""))
    with pytest.raises(PortfolioValidationError, match="no holdings"):
        load_portfolio_csv(io.StringIO(HEADER))


def test_duplicate_tickers():
    with pytest.raises(PortfolioValidationError, match="duplicate tickers: AAPL"):
        load_portfolio_csv(io.StringIO(HEADER + "AAPL,1,100,,,\naapl,2,110,,,\n"))
