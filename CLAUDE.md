# CLAUDE.md — FinSight project memory

Read this at the start of every session. Also check `PLAN.md` (current phase), `DECISIONS.md`
(short decision log) and `docs/ARCHITECTURE.md` (why each module looks the way it does).

## Project summary
FinSight is an **agentic portfolio analyst**: a LangGraph multi-agent system that takes a portfolio
CSV (US stocks/ETFs, USD only) and produces a structured, source-cited report.

Agents (all in `src/finsight/agents/`):
- **Supervisor**: ONE structured-output LLM call per question makes a plan; code then follows it.
- **Portfolio**: holdings, weights, P&L, allocation by sector / asset type.
- **Risk**: volatility, beta, Sharpe, VaR, max drawdown, concentration (HHI), correlations.
- **Research**: 10-K RAG (Chroma, local embeddings) + Tavily web news; every claim cites a source id.
- **Report**: final structured report; key metrics must be traceable to tool data.

Planned: tools as MCP servers (Phase 5), guardrails + human approval via `interrupt` (Phase 6),
Streamlit UI (Phase 7), evals (Phase 8).

This is a learning/resume project: the user must understand and explain every file. Explain the
what and the why, in simple English.

## Stack
Python 3.11+ (3.12 locally) · uv · LangGraph (+ langgraph-checkpoint-sqlite) · LangChain ·
langchain-groq · langchain-openai (pointed at OpenRouter) · langchain-chroma (chromadb) ·
langchain-huggingface (local sentence-transformers) · langchain-text-splitters ·
tavily-python · yfinance · pandas · numpy · pydantic v2 · httpx · beautifulsoup4 + lxml ·
python-dotenv · pytest · LangSmith tracing. (mcp, langchain-mcp-adapters, streamlit: for later phases.)

## Folder structure
```
data/samples/            sample CSVs (balanced, tech_heavy, concentrated_risky)
data/cache/, data/chroma/, data/filings/, data/checkpoints/   local data (git-ignored)
docs/ARCHITECTURE.md     the "why" for each module, in plain language
src/finsight/
  config.py              Settings, get_llm(role), cost guard, rate limiters
  schemas.py             all Pydantic models (Holding, Portfolio, RoutePlan, Report, Source, Insight...)
  portfolio/
    loader.py            CSV -> Portfolio (all errors with line numbers)
    market.py            yfinance downloads + daily disk cache
    calculations.py      portfolio math + risk math (pure functions)
    context.py           PortfolioContext: data loaded once per run
  tools/
    portfolio_tools.py   LLM tools for portfolio/risk (payload functions + @tool wrappers)
    research_tools.py    10-K search, Tavily search + cache, source registry (FoundSources)
  rag/
    edgar.py             download the latest 10-K from SEC EDGAR
    parse.py             HTML -> text -> one Document per 10-K Item -> chunks
    ingest.py            HuggingFaceEmbeddings, Chroma vector store, idempotent ingest
    retriever.py         similarity_search with ticker/section filters
  agents/
    common.py            ask_structured (try + retry once), run_tool_agent
    supervisor.py  portfolio_agent.py  risk_agent.py  research_agent.py  report_agent.py
  state.py               FinSightState (checkpointed) + RunContext (not checkpointed)
  graph.py               make_run_context, build_graph, run_question, resume_thread
  cli.py                 finsight ask | ingest | research
tests/                   offline tests; fakes.py has FakeLLM and HashEmbedder
```

CSV schema: `ticker,quantity,avg_buy_price,buy_date,asset_type,currency`.
Required: ticker, quantity, avg_buy_price. `asset_type` ∈ equity|etf|bond|crypto|cash. `currency` = USD.
Cash rows: quantity = dollar amount, avg_buy_price = 1.

## Commands
- Install / sync env: `uv sync`
- Tests (offline, free): `uv run pytest`
- Ask: `uv run finsight ask data/samples/balanced.csv "What's my sector exposure?"`
- Follow-up / inspect / resume: `--thread ID`, `--thread ID --show-state`, `--thread ID --resume`
- Ingest 10-Ks: `uv run finsight ingest --from-samples` (or tickers). Needs SEC_USER_AGENT in .env.
- Research: `uv run finsight research "What are NVDA's main risks?" --ticker NVDA`
- Add a dependency: `uv add --no-sync <pkg>` then `uv sync` (dev-only: `--dev`)

**OneDrive gotcha:** the project lives in OneDrive, which can lock files in `.venv` so uv's reinstall
of the editable `finsight` package fails with "Access is denied (os error 5)". Workarounds:
`uv run --no-sync ...` (the editable install makes code changes live); if `import finsight` breaks,
delete the empty `.venv/Lib/site-packages/finsight-0.1.0.dist-info` folder and `uv sync`.
Permanent fix: move the project out of OneDrive.

## RULES
1. **The LLM never does math.** All calculations live in Python (`portfolio/calculations.py`, tools).
   Tools return finished, rounded numbers with units in the names (`_pct`, `_usd`).
2. **Every claim must be traceable.** Key metrics must appear in the tool output they cite; research
   claims cite source ids, and code drops claims citing ids the tools never returned.
3. **Never read or print `.env` values.** Don't ask the user for keys; they fill `.env` themselves.
   `.env.example` is the only env file to edit.
4. **Every new function gets a test** where it makes sense; math is tested with hand-computed answers.
   All tests run offline (FakeLLM, fake market data, HashEmbedder).
5. **Build in stages**; don't start a phase until the previous phase's tests pass.
6. **Don't commit or merge unless the user asks.**
7. **LangGraph / LangChain / MCP APIs change fast** — check installed versions or current docs, don't guess.
8. **Explain design decisions** to the user and suggest a one-line entry for `DECISIONS.md`.
9. **ZERO COST. The user has no budget: every LLM call must use a free model.**
   - Defaults are Groq free tier (router `openai/gpt-oss-20b`, agent `qwen/qwen3.8-27b`, report
     `openai/gpt-oss-120b`). Measured free limits per model: 30 RPM, 1000 RPD, **8000 TPM**.
   - OpenRouter only with `:free` model ids. `get_llm()` refuses anything else unless
     `FINSIGHT_ALLOW_PAID=true`; never set that flag or suggest paid models.
   - Never add paid APIs/services. Keep LLM calls per question minimal and prompts compact; report
     the expected number of LLM calls when changing the graph (now: 4 for one agent, 6 for two).
   - Before changing models, re-check current free-tier docs/limits; don't guess.
   - Embeddings are LOCAL (sentence-transformers); never add an embedding API. Tavily: free tier,
     basic depth, max 2 searches/question, disk-cached. SEC EDGAR: free, ≤4 requests/second.
10. **SEC_USER_AGENT is personal contact info**: read it only via Settings (SecretStr); never print,
    log, hardcode or commit it, and never put it in error messages.
11. **Each LLM step: try → retry once → simple safe fallback.** No deeper fallback chains.

## Style rules (follow these in every file)
1. Each file reads top to bottom with section headings:
   ```
   # =========================
   # Risk Agent
   # =========================
   ```
2. File docstring: 1-3 plain-English lines. No essays in code.
3. Explanations of WHY go in `docs/ARCHITECTURE.md`, in simple language, one short section per module.
4. Comments only where something is surprising.
5. Functions do one job, aim under ~30 lines. Files aim under ~200 lines.
6. One idea per line. No dense one-liners or clever tricks. Plain for-loops are fine.
7. Full, clear names (`portfolio_returns`, not `pr`).
8. Every agent file has the same shape: PROMPT -> TOOLS -> node function -> helpers.
9. The LLM never does math. Tools calculate; the LLM explains.
10. Simple first. Don't add a layer, class or abstraction unless it removes real repetition.

## Coding guidelines (summary)
- Priorities: **Correctness > Simplicity > Readability > Maintainability > Optimization.**
- KISS, YAGNI, DRY: implement only what is needed now; accept small duplication over a needless
  abstraction; no custom exceptions, caching, retries or config unless they solve a real problem.
- Prefer functions over classes unless state or encapsulation is really needed. Prefer native
  LangChain/LangGraph features over custom infrastructure.
- Handle realistic errors; never silently swallow them; never expose secrets.
- When modifying code: understand it first, make focused changes, keep public interfaces and
  existing features, remove unused code, don't touch unrelated files.
- Before a significant feature, briefly explain: the problem, the simplest approach, why, and the
  files touched. After it: what was built, the execution flow, key libraries, trade-offs.
- Final self-review: can it be simpler? Is every function/parameter needed? Can a beginner follow
  the flow? Do the tests pass?
