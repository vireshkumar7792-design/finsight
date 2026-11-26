# CLAUDE.md — FinSight project memory

Read this at the start of every session. Also check `PLAN.md` (current phase) and `DECISIONS.md` (why things are the way they are).

## Project summary
FinSight is an **agentic portfolio analyst**: a LangGraph multi-agent system that takes a user's
portfolio CSV (US stocks/ETFs only for now) and produces a structured, source-cited analysis report.

Agents:
- **Supervisor**: routes work between agents dynamically using structured output (Groq, fast model).
- **Portfolio**: holdings, weights, P&L, sector allocation.
- **Research**: company/market research via Chroma RAG over SEC 10-K filings + Tavily web search.
- **Risk**: volatility, beta, Sharpe, VaR, max drawdown, concentration, correlation.
- **Report**: final structured report; every claim cites a Source.

Cross-cutting: tools exposed as MCP servers (FastMCP) and consumed via langchain-mcp-adapters;
SqliteSaver checkpoints; guardrails (schema validation, grounding, number consistency);
human-in-the-loop approval (LangGraph `interrupt`) for high-risk recommendations; Streamlit UI (last).

This is a learning/resume project: the user must understand everything. Explain the what and the why.

## Stack
Python 3.11+ (pinned 3.12 locally) · uv · LangGraph (+ langgraph-checkpoint-sqlite) · LangChain ·
langchain-groq · langchain-openai (pointed at OpenRouter) · chromadb · sentence-transformers ·
tavily-python · yfinance · pandas · numpy · pydantic v2 · mcp (FastMCP) · langchain-mcp-adapters ·
streamlit · python-dotenv · pytest · LangSmith tracing.

## Folder structure
```
data/samples/          sample portfolio CSVs (balanced, tech_heavy, concentrated_risky)
data/filings/          10-K files for RAG (contents git-ignored)
src/finsight/
  config.py            settings + get_llm(role)
  schemas/             Pydantic models (holdings, metrics, Insight/Source, report)
  tools/               deterministic math + market data (portfolio_math, risk_math, market_data)
  mcp_servers/         FastMCP servers wrapping tools/ (portfolio_server, market_server)
  rag/                 ingest.py (10-K -> Chroma), retriever.py (query with citations)
  agents/              supervisor, portfolio, research, risk, report
  guardrails/          validators.py, risk_classifier.py
  graph.py             StateGraph wiring, checkpointer, interrupt
app/streamlit_app.py   UI
tests/                 pytest suite
```

CSV schema: `ticker,quantity,avg_buy_price,buy_date,asset_type,currency`.
Required: ticker, quantity, avg_buy_price. `asset_type` ∈ equity|etf|bond|crypto|cash. `currency` = USD.

## Commands
- Install / sync env: `uv sync`
- Run tests: `uv run pytest`
- Add a dependency: `uv add <pkg>` (dev-only: `uv add --dev <pkg>`)

## RULES
1. **The LLM never does math.** All calculations live in deterministic Python in `tools/`. Agents call tools and interpret results.
2. **Every claim in a report must carry a Source** (filing chunk or URL).
3. **Never read or print `.env` values.** Don't ask the user for keys; they fill `.env` themselves. `.env.example` is the only env file to edit.
4. **Every new function gets a pytest test**; tools are tested with known-answer data (hand-computed expected values).
5. **Build bottom-up**; don't start a phase until the previous phase's tests pass.
6. **Don't commit unless the user asks.**
7. **LangGraph / MCP / langchain-mcp-adapters change fast** — check current docs rather than guessing APIs.
8. **Explain design decisions** to the user and suggest a one-line entry for `DECISIONS.md`.
