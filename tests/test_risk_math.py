"""Known-answer tests for risk_math. Each expected value is derived by hand in the comment."""

import math

import numpy as np
import pandas as pd
import pytest

from finsight.tools import risk_math as rm


def test_simple_returns():
    # 100 -> 110 is +10%, 110 -> 99 is -10%
    r = rm.simple_returns(pd.Series([100.0, 110.0, 99.0]))
    assert r.tolist() == pytest.approx([0.10, -0.10])


def test_annualized_volatility():
    # returns [0.1, -0.1]: mean 0, sample var = (0.01 + 0.01) / (2-1) = 0.02, std = 0.1*sqrt(2)
    r = pd.Series([0.1, -0.1])
    assert rm.annualized_volatility(r, periods_per_year=1) == pytest.approx(0.1 * math.sqrt(2))
    assert rm.annualized_volatility(r) == pytest.approx(0.1 * math.sqrt(2) * math.sqrt(252))


def test_sharpe_ratio_zero_rf():
    # returns [0.01, 0.03]: mean 0.02, std = sqrt((0.0001+0.0001)/1) = 0.01*sqrt(2)
    # Sharpe (1 period/yr) = 0.02 / (0.01*sqrt 2) = sqrt(2)
    assert rm.sharpe_ratio(pd.Series([0.01, 0.03]), periods_per_year=1) == pytest.approx(math.sqrt(2))


def test_sharpe_ratio_with_risk_free():
    # rf 4%/yr over 4 periods -> 0.01/period; excess mean 0.02 - 0.01 = 0.01
    # 0.01 / (0.01*sqrt 2) * sqrt(4) = 2/sqrt(2) = sqrt(2)
    r = pd.Series([0.01, 0.03])
    assert rm.sharpe_ratio(r, risk_free_rate=0.04, periods_per_year=4) == pytest.approx(math.sqrt(2))


def test_sharpe_zero_volatility_raises():
    with pytest.raises(ValueError, match="zero volatility"):
        rm.sharpe_ratio(pd.Series([0.01, 0.01, 0.01]))


def test_beta_exact_multiple():
    # asset = 2*market + 0.001 -> cov = 2*var(m) -> beta exactly 2 (constant shift doesn't matter)
    m = pd.Series([0.01, 0.02, -0.01, 0.0, 0.015])
    assert rm.beta(2 * m + 0.001, m) == pytest.approx(2.0)
    assert rm.beta(-0.5 * m, m) == pytest.approx(-0.5)


def test_beta_aligns_on_dates():
    idx = pd.date_range("2024-01-01", periods=4)
    m = pd.Series([0.01, 0.02, -0.01, 0.03], index=idx)
    a = (2 * m).iloc[1:]  # asset missing first day
    assert rm.beta(a, m) == pytest.approx(2.0)


def test_beta_zero_market_variance_raises():
    with pytest.raises(ValueError, match="zero variance"):
        rm.beta(pd.Series([0.01, 0.02]), pd.Series([0.01, 0.01]))


def test_historical_var():
    # 20 returns: -0.10, -0.09, ..., 0.09 (sorted). 5% quantile, linear interpolation:
    # position = (n-1)*q = 19*0.05 = 0.95 -> -0.10 + 0.95*(0.01) = -0.0905 -> VaR 0.0905
    r = pd.Series(np.arange(-10, 10) / 100)
    assert rm.historical_var(r, 0.95) == pytest.approx(0.0905)


def test_historical_var_bad_confidence():
    with pytest.raises(ValueError):
        rm.historical_var(pd.Series([0.1, 0.2]), confidence=95)


def test_max_drawdown():
    # peaks: 100,120,120,120,130,130; worst drop 120 -> 90 = -25%
    prices = pd.Series([100.0, 120, 90, 95, 130, 104])
    assert rm.max_drawdown(prices) == pytest.approx(0.25)
    assert rm.max_drawdown(pd.Series([1.0, 2, 3])) == 0.0  # only goes up


def test_concentration():
    # HHI = 0.5^2 + 0.3^2 + 0.2^2 = 0.25 + 0.09 + 0.04 = 0.38; effective N = 1/0.38
    w = {"A": 0.5, "B": 0.3, "C": 0.2}
    assert rm.herfindahl_index(w) == pytest.approx(0.38)
    assert rm.effective_number_of_positions(w) == pytest.approx(1 / 0.38)
    assert rm.max_weight(w) == ("A", 0.5)
    # Equal weights: HHI = 1/N
    assert rm.herfindahl_index({t: 0.25 for t in "ABCD"}) == pytest.approx(0.25)


def test_herfindahl_requires_weights_summing_to_one():
    with pytest.raises(ValueError, match="sum to 1"):
        rm.herfindahl_index({"A": 0.5, "B": 0.2})


def test_correlation_matrix():
    df = pd.DataFrame({"A": [0.01, 0.02, 0.03], "B": [0.02, 0.04, 0.06], "C": [0.03, 0.02, 0.01]})
    c = rm.correlation_matrix(df)
    assert c.loc["A", "B"] == pytest.approx(1.0)  # B = 2*A
    assert c.loc["A", "C"] == pytest.approx(-1.0)  # C moves opposite
    assert c.loc["A", "A"] == pytest.approx(1.0)


def test_portfolio_returns():
    # 50/50: day1 0.5*0.10 + 0.5*0.00 = 0.05; day2 0.5*-0.10 + 0.5*0.20 = 0.05
    df = pd.DataFrame({"A": [0.10, -0.10], "B": [0.00, 0.20]})
    assert rm.portfolio_returns(df, {"A": 0.5, "B": 0.5}).tolist() == pytest.approx([0.05, 0.05])


def test_portfolio_returns_missing_history_raises():
    with pytest.raises(ValueError, match="no return history for: C"):
        rm.portfolio_returns(pd.DataFrame({"A": [0.1]}), {"A": 0.5, "C": 0.5})


def test_too_few_observations_raises():
    with pytest.raises(ValueError, match="at least 2"):
        rm.annualized_volatility(pd.Series([0.01]))


# ---- cash: a constant price series (zero returns) is valid data, not missing data ----------


def test_constant_series_risk_is_zero():
    prices = pd.Series([1.0, 1.0, 1.0, 1.0])
    r = rm.simple_returns(prices)
    assert rm.annualized_volatility(r) == 0.0
    assert rm.max_drawdown(prices) == 0.0
    var = rm.historical_var(r)
    assert var == 0.0 and math.copysign(1, var) == 1  # +0.0, not -0.0
    assert rm.beta(r, pd.Series([0.01, -0.02, 0.03])) == 0.0


def test_cash_halves_volatility():
    # 50% cash: portfolio return = 0.5 * asset return, so vol is exactly half.
    df = pd.DataFrame({"A": [0.02, -0.01, 0.03, -0.02], "CASH": [0.0] * 4})
    full = rm.annualized_volatility(df["A"])
    half = rm.annualized_volatility(rm.portfolio_returns(df, {"A": 0.5, "CASH": 0.5}))
    assert half == pytest.approx(full / 2)


def test_correlation_with_cash_is_undefined_and_skipped():
    df = pd.DataFrame({"A": [0.01, 0.02, 0.03], "B": [0.02, 0.04, 0.06], "CASH": [0.0, 0.0, 0.0]})
    corr = rm.correlation_matrix(df)
    assert np.isnan(corr.loc["A", "CASH"])
    assert rm.correlation_pairs(corr) == [("A", "B", pytest.approx(1.0))]
    assert rm.average_pairwise_correlation(corr) == pytest.approx(1.0)


def test_correlation_pairs_sorted_high_to_low():
    df = pd.DataFrame({"A": [0.01, 0.02, 0.03, 0.01], "B": [0.02, 0.04, 0.06, 0.02], "C": [0.03, 0.02, 0.01, 0.02]})
    pairs = rm.correlation_pairs(rm.correlation_matrix(df))
    assert [p[:2] for p in pairs][0] == ("A", "B")
    assert [p[2] for p in pairs] == sorted([p[2] for p in pairs], reverse=True)
    assert len(pairs) == 3  # 3 choose 2


def test_average_pairwise_correlation_none_when_no_pairs():
    corr = rm.correlation_matrix(pd.DataFrame({"A": [0.0, 0.0], "B": [0.0, 0.0]}))
    assert rm.average_pairwise_correlation(corr) is None


def test_average_correlation_by_asset():
    # A-B 0.8, A-C 0.2, B-C -0.4; D is cash (all NaN).
    # A: (0.8+0.2)/2 = 0.5   B: (0.8-0.4)/2 = 0.2   C: (0.2-0.4)/2 = -0.1   -> sorted low to high
    nan = float("nan")
    corr = pd.DataFrame(
        [[1, 0.8, 0.2, nan], [0.8, 1, -0.4, nan], [0.2, -0.4, 1, nan], [nan, nan, nan, nan]],
        index=list("ABCD"), columns=list("ABCD"),
    )
    avg = rm.average_correlation_by_asset(corr)
    assert list(avg) == ["C", "B", "A"]
    assert avg == pytest.approx({"C": -0.1, "B": 0.2, "A": 0.5})
