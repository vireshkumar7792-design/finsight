# FinSight — Agentic Portfolio Analyst

A multi-agent financial research and portfolio analysis system built with **LangGraph**.
Upload a portfolio CSV and a team of agents (Supervisor, Portfolio, Research, Risk, Report)
produce a structured, source-cited analysis, with guardrails and human approval for high-risk recommendations.

> 🚧 Work in progress — see [PLAN.md](PLAN.md) for status.

## Quickstart
```bash
uv sync                 # create .venv and install dependencies
cp .env.example .env    # then fill in your own API keys
uv run pytest           # run the test suite
```

## Portfolio CSV format
```
ticker,quantity,avg_buy_price,buy_date,asset_type,currency
AAPL,25,184.90,2023-06-15,equity,USD
```
Required: `ticker`, `quantity`, `avg_buy_price`. `asset_type` ∈ equity|etf|bond|crypto|cash. USD only.
Samples live in `data/samples/`.

## Docs
- [PLAN.md](PLAN.md): build phases
- [DECISIONS.md](DECISIONS.md): architecture decision log
