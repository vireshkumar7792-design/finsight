"""Every formula checked against numbers worked out by hand (shown in the comments)."""

import math

import numpy as np
import pandas as pd
import pytest

from finsight.portfolio import calculations
from finsight.schemas import Holding, Portfolio

# =========================
# Portfolio math
# =========================


def two_stock_portfolio() -> Portfolio:
    # AAPL 10 @ $100 and MSFT 5 @ $200: $1,000 cost each.
    return Portfolio(holdings=[
        Holding(ticker="AAPL", quantity=10, avg_buy_price=100),
        Holding(ticker="MSFT", quantity=5, avg_buy_price=200),
    ])


def test_position_details_and_totals():
    # Now AAPL $150 -> worth 1500 (+500, +50%); MSFT $180 -> worth 900 (-100, -10%).
    positions = calculations.position_details(two_stock_portfolio(), {"AAPL": 150.0, "MSFT": 180.0})
    aapl, msft = positions
    assert aapl["market_value"] == 1500
    assert aapl["unrealized_pnl"] == 500
    assert aapl["unrealized_pnl_pct"] == pytest.approx(0.50)
    assert aapl["weight"] == pytest.approx(1500 / 2400)
    assert msft["unrealized_pnl_pct"] == pytest.approx(-0.10)
    assert msft["weight"] == pytest.approx(900 / 2400)

    totals = calculations.portfolio_totals(positions)
    assert totals["total_cost_basis"] == 2000
    assert totals["total_market_value"] == 2400
    assert totals["total_unrealized_pnl"] == 400
    assert totals["total_unrealized_pnl_pct"] == pytest.approx(0.20)


def test_allocation_groups_and_sorts():
    weights = {"AAPL": 0.5, "MSFT": 0.2, "XOM": 0.3}
    labels = {"AAPL": "Technology", "MSFT": "Technology"}  # XOM has no label
    assert calculations.allocation(weights, labels) == {"Technology": 0.7, "Unknown": 0.3}


# =========================
# Returns
# =========================


def test_daily_returns_and_portfolio_returns():
    prices = pd.Series([100.0, 110.0, 99.0])
    # 110/100 - 1 = +10%, 99/110 - 1 = -10%
    assert calculations.daily_returns(prices).tolist() == pytest.approx([0.10, -0.10])

    returns = pd.DataFrame({"A": [0.10, 0.0], "B": [0.0, 0.20]})
    # 50/50 weights: day 1 = 0.5*0.10 = 0.05, day 2 = 0.5*0.20 = 0.10
    result = calculations.portfolio_returns(returns, {"A": 0.5, "B": 0.5})
    assert result.tolist() == pytest.approx([0.05, 0.10])


# =========================
# Risk math
# =========================

# Mean = 0.005. Squared deviations: 0.000025 + 0.000625 + 0.000625 + 0.000025 = 0.0013.
# Sample variance = 0.0013 / 3, so daily std = sqrt(0.0013 / 3) = 0.020817.
RETURNS = pd.Series([0.01, -0.02, 0.03, 0.00])
DAILY_STD = math.sqrt(0.0013 / 3)


def test_annualized_volatility():
    expected = DAILY_STD * math.sqrt(252)  # = 0.3305
    assert calculations.annualized_volatility(RETURNS) == pytest.approx(expected)
    assert expected == pytest.approx(0.3305, abs=1e-4)


def test_sharpe_ratio():
    # Without risk-free rate: 0.005 / 0.020817 * sqrt(252) = 3.813
    assert calculations.sharpe_ratio(RETURNS, risk_free_rate=0.0) == pytest.approx(3.813, abs=1e-3)
    # 2.52% a year = 0.0001 a day, so the excess mean is 0.0049.
    expected = 0.0049 / DAILY_STD * math.sqrt(252)
    assert calculations.sharpe_ratio(RETURNS, risk_free_rate=0.0252) == pytest.approx(expected)


def test_sharpe_needs_some_volatility():
    with pytest.raises(ValueError, match="zero volatility"):
        calculations.sharpe_ratio(pd.Series([0.01, 0.01, 0.01]))


def test_beta():
    market = pd.Series([0.01, -0.01, 0.02, 0.0])
    # An asset that always moves twice as much as the market has beta 2.
    assert calculations.beta(market * 2, market) == pytest.approx(2.0)
    # Adding a constant doesn't change covariance, so beta stays 2.
    assert calculations.beta(market * 2 + 0.001, market) == pytest.approx(2.0)


def test_historical_var():
    returns = pd.Series([0.02, -0.05, 0.01, -0.01, -0.03])
    # Sorted: -0.05, -0.03, -0.01, 0.01, 0.02. The 5% point sits 4 * 0.05 = 0.2 of the way
    # from -0.05 to -0.03: -0.05 + 0.2 * 0.02 = -0.046, so VaR = 0.046 (a 4.6% loss).
    assert calculations.historical_var(returns, 0.95) == pytest.approx(0.046)


def test_max_drawdown():
    prices = pd.Series([100.0, 120.0, 90.0, 110.0, 60.0, 80.0])
    # Peak 120, lowest after it 60: 60 / 120 - 1 = -50%.
    assert calculations.max_drawdown(prices) == pytest.approx(0.5)
    assert calculations.max_drawdown(pd.Series([1.0, 2.0, 3.0])) == 0.0


def test_herfindahl_index():
    # 0.5^2 + 0.3^2 + 0.2^2 = 0.25 + 0.09 + 0.04 = 0.38, so it acts like 1 / 0.38 = 2.63 positions.
    assert calculations.herfindahl_index({"A": 0.5, "B": 0.3, "C": 0.2}) == pytest.approx(0.38)
    assert calculations.herfindahl_index({"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}) == pytest.approx(0.25)


# =========================
# Correlation
# =========================


def test_correlations_skip_cash_and_sort():
    returns = pd.DataFrame({
        "A": [1.0, 2.0, 3.0, 4.0],
        "B": [2.0, 4.0, 6.0, 8.0],  # moves exactly with A: +1
        "C": [4.0, 3.0, 2.0, 1.0],  # moves exactly against A and B: -1
        "CASH": [0.0, 0.0, 0.0, 0.0],  # never moves: correlation undefined
    })
    matrix = calculations.correlation_matrix(returns)
    assert np.isnan(matrix.loc["A", "CASH"])

    pairs = calculations.correlation_pairs(matrix)
    assert pairs[0] == ("A", "B", pytest.approx(1.0))
    assert len(pairs) == 3  # no CASH pairs

    # A: (1 + -1) / 2 = 0, B: 0, C: (-1 + -1) / 2 = -1. Lowest first; CASH left out.
    averages = calculations.average_correlation_by_ticker(matrix)
    assert list(averages) == ["C", "A", "B"]
    assert averages["C"] == pytest.approx(-1.0)
    assert averages["A"] == pytest.approx(0.0)
