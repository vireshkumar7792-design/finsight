"""Tool payloads (what the LLM receives) and tool schemas (what the LLM sees).

Fake portfolio (conftest): AAA 48%, BBB 36%, CASH 16%; MV $2,500; cost $2,400; P&L +$100.
"""

import json

import pytest

from finsight.agents import tools as T
from finsight.tools import risk_math


def test_summary_payload(fake_ctx):
    s = T.summary_payload(fake_ctx)
    assert s["total_market_value_usd"] == 2500.0
    assert s["total_cost_basis_usd"] == 2400.0
    assert s["total_unrealized_pnl_usd"] == 100.0
    assert s["total_unrealized_pnl_pct"] == 4.17  # 100 / 2400 = 4.1666...%
    assert [p["ticker"] for p in s["positions_by_weight"]] == ["AAA", "BBB", "CASH"]
    assert [p["weight_pct"] for p in s["positions_by_weight"]] == [48.0, 36.0, 16.0]
    assert s["worst_position_by_pnl"] == {"ticker": "BBB", "unrealized_pnl_usd": -100.0, "unrealized_pnl_pct": -10.0}
    assert s["best_position_by_pnl"]["ticker"] == "AAA"
    assert s["n_positions_with_loss"] == 1
    assert s["as_of"] == "2024-01-04"


@pytest.mark.parametrize(
    "by, expected",
    [
        ("asset_type", [("equity", 48.0, ["AAA"]), ("etf", 36.0, ["BBB"]), ("cash", 16.0, ["CASH"])]),
        ("sector", [("Technology", 48.0, ["AAA"]), ("ETF / Fund", 36.0, ["BBB"]), ("Cash", 16.0, ["CASH"])]),
    ],
)
def test_allocation_payload(fake_ctx, by, expected):
    a = T.allocation_payload(fake_ctx, by)
    assert [(g["group"], g["weight_pct"], g["tickers"]) for g in a["allocation"]] == expected


def test_allocation_bad_grouping(fake_ctx):
    assert "error" in T.allocation_payload(fake_ctx, "country")


def test_risk_payload(fake_ctx):
    r = T.risk_payload(fake_ctx)
    c = r["concentration"]
    # HHI = 0.48^2 + 0.36^2 + 0.16^2 = 0.2304 + 0.1296 + 0.0256 = 0.3856
    assert c["herfindahl_index"] == 0.386
    assert c["effective_number_of_positions"] == round(1 / 0.3856, 2)
    assert c["largest_position"] == {"ticker": "AAA", "weight_pct": 48.0}
    # Wrapper must report risk_math's numbers, converted to percent and rounded.
    expected_vol = risk_math.annualized_volatility(fake_ctx.portfolio_returns) * 100
    assert r["portfolio"]["annualized_volatility_pct"] == round(expected_vol, 2)
    var_frac = risk_math.historical_var(fake_ctx.portfolio_returns, 0.95)
    assert r["portfolio"]["var_1day_usd"] == round(var_frac * 2500, 2)
    assert r["benchmark"]["beta_vs_SPY"] == 1.0  # SPY vs itself
    assert r["method"]["risk_free_rate_pct"] == 4.0
    assert r["method"]["daily_observations"] == 3


def test_correlations_payload_excludes_cash(fake_ctx):
    c = T.correlations_payload(fake_ctx)
    assert c["excluded_cash"] == ["CASH"]
    assert c["n_pairs"] == 1
    assert c["most_correlated_pairs"][0]["pair"] == ["AAA", "BBB"]
    assert -1 <= c["average_pairwise_correlation"] <= 1
    assert set(c["avg_correlation_with_other_holdings"]) == {"AAA", "BBB"}  # cash omitted


def test_position_payload(fake_ctx):
    p = T.position_payload(fake_ctx, "aaa")  # case-insensitive
    assert (p["ticker"], p["sector"], p["market_value_usd"], p["unrealized_pnl_pct"]) == ("AAA", "Technology", 1200.0, 20.0)
    assert "annualized_volatility_pct" in p["risk"]
    assert isinstance(T.position_payload(fake_ctx, "CASH")["risk"], str)


def test_position_payload_unknown_ticker(fake_ctx):
    p = T.position_payload(fake_ctx, "TSLA")
    assert "not in the portfolio" in p["error"] and p["holdings"] == ["AAA", "BBB", "CASH"]


# ---- what the LLM sees --------------------------------------------------------------------


@pytest.mark.parametrize("t", T.ALL_TOOLS, ids=lambda t: t.name)
def test_runtime_is_hidden_from_llm_schema(t):
    schema = t.tool_call_schema.model_json_schema()
    assert "runtime" not in schema.get("properties", {})
    assert len(t.description) > 80  # a real description, not a one-liner


def test_tool_argument_schemas():
    props = T.get_allocation.tool_call_schema.model_json_schema()["properties"]
    assert set(props) == {"by"} and props["by"]["enum"] == ["sector", "asset_type"]
    assert T.get_portfolio_summary.tool_call_schema.model_json_schema().get("properties", {}) == {}


def test_payloads_are_compact_json(fake_ctx):
    for payload in (T.summary_payload(fake_ctx), T.risk_payload(fake_ctx), T.correlations_payload(fake_ctx)):
        text = T._dump(payload)
        assert json.loads(text) == payload
        assert len(text) < 2500  # keeps us well inside Groq's tokens-per-minute budget
