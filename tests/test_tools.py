"""Tool outputs: rounded numbers with the right units, on the fake portfolio from conftest.py.

AAA 10 @ $100, now $120 -> $1,200 (+$200, +20%)     weight 48%
BBB 20 @ $50,  now $45  -> $900   (-$100, -10%)     weight 36%
CASH $400                                           weight 16%
"""

import json

from finsight.tools import portfolio_tools


def test_summary_units_and_best_worst(portfolio_context):
    summary = portfolio_tools.summary_payload(portfolio_context)
    assert summary["total_market_value_usd"] == 2500.0
    assert summary["total_cost_basis_usd"] == 2400.0
    assert summary["total_unrealized_pnl_usd"] == 100.0
    assert summary["total_unrealized_pnl_pct"] == 4.17  # 100 / 2400 = 4.1667%
    assert summary["best_position"]["ticker"] == "AAA"
    assert summary["best_position"]["unrealized_pnl_pct"] == 20.0
    assert summary["worst_position"]["ticker"] == "BBB"
    assert [row["weight_pct"] for row in summary["positions_by_weight"]] == [48.0, 36.0, 16.0]


def test_allocation_by_sector_and_asset_type(portfolio_context):
    by_sector = portfolio_tools.allocation_payload(portfolio_context, "sector")["allocation"]
    assert by_sector[0] == {"group": "Technology", "weight_pct": 48.0, "tickers": ["AAA"]}
    assert by_sector[2] == {"group": "Cash", "weight_pct": 16.0, "tickers": ["CASH"]}

    by_type = portfolio_tools.allocation_payload(portfolio_context, "asset_type")["allocation"]
    assert [row["group"] for row in by_type] == ["equity", "etf", "cash"]


def test_risk_concentration(portfolio_context):
    risk = portfolio_tools.risk_payload(portfolio_context)
    # HHI = 0.48^2 + 0.36^2 + 0.16^2 = 0.2304 + 0.1296 + 0.0256 = 0.3856 -> 1 / 0.3856 = 2.59
    assert risk["concentration"]["herfindahl_index"] == 0.386
    assert risk["concentration"]["effective_number_of_positions"] == 2.59
    assert risk["concentration"]["largest_position"] == {"ticker": "AAA", "weight_pct": 48.0}
    for key in ["annualized_volatility_pct", "sharpe_ratio", "beta_vs_SPY", "var_1day_95_pct", "max_drawdown_pct"]:
        assert key in risk["portfolio"]
        assert key in risk["benchmark"]
    assert "var_1day_95_usd" in risk["portfolio"]


def test_correlations_leave_out_cash(portfolio_context):
    result = portfolio_tools.correlations_payload(portfolio_context)
    assert result["most_correlated_pairs"][0]["pair"] == ["AAA", "BBB"]
    assert "CASH" not in result["average_correlation_with_others"]


def test_position_detail(portfolio_context):
    aaa = portfolio_tools.position_payload(portfolio_context, "aaa")
    assert aaa["current_price_usd"] == 120.0
    assert aaa["unrealized_pnl_pct"] == 20.0
    assert "annualized_volatility_pct" in aaa["risk"]
    assert "no market risk" in portfolio_tools.position_payload(portfolio_context, "CASH")["risk"]
    assert "error" in portfolio_tools.position_payload(portfolio_context, "ZZZ")


def test_llm_tools_hide_the_portfolio_from_the_model():
    # The model only sees real arguments, never the `runtime` that carries the data.
    schema = portfolio_tools.get_allocation.tool_call_schema.model_json_schema()
    assert list(schema["properties"]) == ["by"]
    assert portfolio_tools.get_risk_metrics.tool_call_schema.model_json_schema()["properties"] == {}


def test_payloads_are_json_ready(portfolio_context):
    for payload in [
        portfolio_tools.summary_payload(portfolio_context),
        portfolio_tools.risk_payload(portfolio_context),
        portfolio_tools.correlations_payload(portfolio_context),
    ]:
        json.dumps(payload)
