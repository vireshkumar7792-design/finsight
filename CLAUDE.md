# CLAUDE.md — FinSight project memory

Read this at the start of every session. Also check `PLAN.md` (current phase) and `DECISIONS.md` (why things are the way they are).

## Project summary
FinSight is an **agentic portfolio analyst**: a LangGraph multi-agent system that takes a user's
portfolio CSV (US stocks/ETFs only for now) and produces a structured, source-cited analysis report.

Agents:
- **Supervisor**: plans which specialists a question needs with ONE structured-output LLM call, then walks the plan deterministically.
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
  config.py            Settings + get_llm(role) + cost guard; all roles default to FREE Groq models
  llm_limits.py        FreeTierRateLimiter (requests+tokens/min), TokenLedger callback, SQLiteLLMCache
  context.py           PortfolioContext: CSV + market data loaded once per run, injected via runtime context
  state.py             FinSightState (TypedDict) + reducers; new_question_input() resets per-question fields
  graph.py             StateGraph wiring, SqliteSaver checkpoints, run_question / resume_thread
  cli.py               Phase 2 single agent: `python -m finsight.cli <csv> "<question>" [--free] [--messages]`
  cli_graph.py         Phase 3 graph: `python -m finsight.cli_graph <csv> "<q>" --thread ID [--resume] [--show-state]`
  schemas/             Pydantic models (holdings, metrics, Insight/Source, report)
  tools/               deterministic math + market data (portfolio_math, risk_math, market_data)
  mcp_servers/         FastMCP servers wrapping tools/ (portfolio_server, market_server)
  rag/                 ingest.py (10-K -> Chroma), retriever.py (query with citations)
  agents/              tools.py (payload fns + @tool wrappers), analyst.py (Phase 2 single agent),
                       supervisor.py, specialist.py (+ portfolio.py, risk.py), research.py (stub until P4),
                       report.py, structured.py (structured-output helper with failure handling)
  guardrails/          validators.py, risk_classifier.py
app/streamlit_app.py   UI
tests/                 pytest suite
```

CSV schema: `ticker,quantity,avg_buy_price,buy_date,asset_type,currency`.
Required: ticker, quantity, avg_buy_price. `asset_type` ∈ equity|etf|bond|crypto|cash. `currency` = USD.

## Commands
- Install / sync env: `uv sync`
- Run tests: `uv run pytest` (offline). Live tests: `uv run pytest -m network` (yfinance), `uv run pytest -m llm` (real LLM, costs tokens)
- Try the single agent: `uv run python -m finsight.cli data/samples/balanced.csv "What's my sector exposure?"`
- Try the graph: `uv run python -m finsight.cli_graph data/samples/concentrated_risky.csv "How risky is my portfolio?" --thread demo1`
  (then `--thread demo1 --show-state`, or `--resume` after an interruption)
- Offline tests use `tests/fakes.py::FakeLLM` (scripted responses; raises if the graph makes unscripted calls)
- Add a dependency: `uv add --no-sync <pkg>` then `uv sync` (dev-only: `--dev`)

**OneDrive gotcha:** the project lives in OneDrive, which locks files in `.venv` and makes uv's reinstall of
the editable `finsight` package fail with "Access is denied (os error 5)". Workarounds: `uv run --no-sync ...`
(editable install means code changes are live); if `import finsight` breaks, delete the empty
`.venv/Lib/site-packages/finsight-0.1.0.dist-info` folder and `uv sync`. Permanent fix: move the project out of OneDrive.

## RULES
1. **The LLM never does math.** All calculations live in deterministic Python in `tools/`. Agents call tools and interpret results.
2. **Every claim in a report must carry a Source** (filing chunk or URL).
3. **Never read or print `.env` values.** Don't ask the user for keys; they fill `.env` themselves. `.env.example` is the only env file to edit.
4. **Every new function gets a pytest test**; tools are tested with known-answer data (hand-computed expected values).
5. **Build bottom-up**; don't start a phase until the previous phase's tests pass.
6. **Don't commit unless the user asks.**
7. **LangGraph / MCP / langchain-mcp-adapters change fast** — check current docs rather than guessing APIs.
8. **Explain design decisions** to the user and suggest a one-line entry for `DECISIONS.md`.
9. **ZERO COST. The user has no budget: every LLM call must use a free model.**
   - Defaults are Groq free tier (router `openai/gpt-oss-20b`, agent `qwen/qwen3.8-27b`, report `openai/gpt-oss-120b`).
     Measured free limits per model: 30 RPM, 1000 RPD, **8000 TPM** (tokens/min is the binding one).
   - OpenRouter only with `:free` model ids (≈50 requests/day without purchased credits). `get_llm()` refuses
     anything else unless `FINSIGHT_ALLOW_PAID=true`; never set that flag or suggest paid models.
   - Never add paid APIs/services. Keep LLM calls per question minimal and prompts compact; report the expected
     number of LLM calls when changing the graph. Tests stay offline (FakeLLM); real calls only under `-m llm`.
   - Before changing models, re-check current free-tier docs/limits (they change); don't guess.
