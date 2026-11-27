"""Known-answer tests for portfolio_math.

Fixture (see conftest): AAPL 10 @ $100 -> $150, MSFT 5 @ $200 -> $180.
  market values: AAPL 10*150 = 1500, MSFT 5*180 = 900, total 2400
  weights:       1500/2400 = 0.625, 900/2400 = 0.375
  cost basis:    1000 + 1000 = 2000
  P&L:           AAPL +500 (+50%), MSFT -100 (-10%), total +400 (+20%)
"""

import pytest

from finsight.tools import portfolio_math as pm


def test_market_values(two_stock_portfolio, two_stock_prices):
    assert pm.market_values(two_stock_portfolio, two_stock_prices) == {"AAPL": 1500.0, "MSFT": 900.0}


def test_weights(two_stock_portfolio, two_stock_prices):
    w = pm.portfolio_weights(two_stock_portfolio, two_stock_prices)
    assert w == pytest.approx({"AAPL": 0.625, "MSFT": 0.375})


def test_summary(two_stock_portfolio, two_stock_prices):
    s = pm.portfolio_summary(two_stock_portfolio, two_stock_prices)
    assert s.total_cost_basis == pytest.approx(2000)
    assert s.total_market_value == pytest.approx(2400)
    assert s.total_unrealized_pnl == pytest.approx(400)
    assert s.total_unrealized_pnl_pct == pytest.approx(0.20)
    aapl, msft = s.positions
    assert (aapl.unrealized_pnl, aapl.unrealized_pnl_pct, aapl.weight) == pytest.approx((500, 0.5, 0.625))
    assert (msft.unrealized_pnl, msft.unrealized_pnl_pct, msft.weight) == pytest.approx((-100, -0.1, 0.375))


def test_missing_or_bad_price_raises(two_stock_portfolio):
    with pytest.raises(ValueError, match="missing current price for: MSFT"):
        pm.market_values(two_stock_portfolio, {"AAPL": 150})
    with pytest.raises(ValueError, match="non-positive price for: MSFT"):
        pm.market_values(two_stock_portfolio, {"AAPL": 150, "MSFT": 0})


def test_allocation_by_group_sums_sorts_and_defaults_unknown():
    weights = {"A": 0.2, "B": 0.3, "C": 0.5}
    alloc = pm.allocation_by_group(weights, {"A": "Tech", "B": "Tech"})  # C unlabeled
    assert alloc == pytest.approx({"Tech": 0.5, "Unknown": 0.5})
    alloc = pm.allocation_by_group(weights, {"A": "Tech", "B": "Health", "C": "Energy"})
    assert list(alloc) == ["Energy", "Health", "Tech"]  # 0.5, 0.3, 0.2

