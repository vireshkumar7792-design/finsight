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
  research_cli.py      Phase 4 research: `python -m finsight.research_cli "<q>" [--ticker T] [--csv C] [--no-web]`
  cli_graph.py         Phase 3 graph: `python -m finsight.cli_graph <csv> "<q>" --thread ID [--resume] [--show-state]`
  schemas/             Pydantic models (holdings, metrics, Insight/Source, report)
  tools/               deterministic math + market data (portfolio_math, risk_math, market_data, market_cache)
  mcp_servers/         FastMCP servers wrapping tools/ (portfolio_server, market_server)
  rag/                 edgar.py (SEC client), parse.py (HTML -> sections -> chunks), embeddings.py
                       (local bge-small), store.py (Chroma), ingest.py, retriever.py
  agents/              research.py + research_tools.py (RAG + web, cited ResearchFindings),
                       tools.py (payload fns + @tool wrappers), analyst.py (Phase 2 single agent),
                       supervisor.py, specialist.py (Portfolio + Risk agents),
                       report.py, structured.py (structured output with failure handling, message_text)
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
- Ingest 10-Ks: `uv run python -m finsight.rag.ingest --from-samples` (or tickers). Needs SEC_USER_AGENT in .env.
- Research: `uv run python -m finsight.research_cli "What are NVDA's main risks?" --ticker NVDA`
- Offline tests use `tests/fakes.py`: FakeLLM (scripted responses; raises if the graph makes unscripted calls) and HashEmbedder
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
   - Embeddings are LOCAL (sentence-transformers); never add an embedding API. Tavily: free tier (1,000
     credits/month), basic depth, max 2 searches/question, disk-cached. SEC EDGAR: free, ≤4 req/s.
10. **SEC_USER_AGENT is personal contact info**: read it only via Settings (SecretStr); never print, log,
    hardcode or commit it, and never put it in error messages.


# Coding Guidelines — Simplicity, Readability and Maintainability

## 1. Core Philosophy

Write clean, simple, readable, and maintainable code that a beginner-to-intermediate developer can understand independently.

I am building this project not only to make it functional but also to learn its architecture, implementation, and underlying concepts. I should be able to explain every part of the code in an interview.

Prioritize:

**Correctness > Simplicity > Readability > Maintainability > Optimization**

Do not sacrifice readability for cleverness or unnecessary optimization.

## 2. Code Simplicity

- Use the simplest approach that correctly solves the problem.
- Prefer straightforward Python syntax over advanced language features.
- Avoid unnecessary abstractions, wrappers, decorators, factories, and design patterns.
- Avoid creating helper functions that merely wrap one or two straightforward statements without adding meaningful value.
- Avoid excessive inheritance, nested functions, and complicated functional programming.
- Prefer explicit logic over clever one-liners.
- Use meaningful and descriptive variable and function names.
- Use type hints where they improve clarity, but avoid unnecessarily complex typing.
- Do not introduce a class when a simple function is sufficient.

## 3. Avoid Overengineering

Follow the KISS (Keep It Simple), YAGNI (You Aren't Gonna Need It), and DRY (Don't Repeat Yourself) principles.

- Implement only what the current requirements demand.
- Do not add features based on hypothetical future requirements.
- Do not introduce custom exception classes unless distinct exception handling is necessary.
- Do not add caching, concurrency, retries, rate limiting, or complex configuration systems unless they solve an actual problem.
- Do not introduce unnecessary dependencies.
- Prefer existing library functionality over custom implementations when appropriate.
- Accept small amounts of duplication if removing them would create unnecessary abstraction.

Every new function, class, dependency, and configuration parameter must have a justified purpose.

## 4. Project Architecture

- Maintain a simple and intuitive folder structure.
- Separate responsibilities logically.
- Keep related functionality together.
- Do not split small modules into multiple files unnecessarily.
- Avoid excessive layers of abstraction.
- Prefer direct execution flow that is easy to trace from input to output.

Architecture should match the actual complexity of the project, not the complexity of an enterprise-scale application.

## 5. Comments and Documentation

- Write short, meaningful comments explaining WHY something is done.
- Explain non-obvious logic and important design decisions.
- Avoid comments that merely repeat what the code already says.
- Use concise docstrings for important functions and classes.
- Avoid excessively long module-level documentation.

## 6. Error Handling and Reliability

- Handle realistic and relevant error scenarios.
- Prefer standard Python exceptions unless custom exceptions are justified.
- Do not silently suppress errors.
- Preserve essential security, input validation, and reliability measures.
- Never expose API keys, credentials, or other sensitive information.

Simplicity must not come at the expense of correctness, security, or existing requirements.

## 7. Modifying Existing Code

- Understand the existing implementation before modifying it.
- Prefer minimal, focused changes instead of rewriting entire modules.
- Preserve existing functionality and public interfaces unless explicitly asked to change them.
- Reuse existing functions and components where appropriate.
- Remove redundant code and unused imports.
- Do not modify unrelated files.
- If simplification requires removing functionality, explain the trade-off before removing it.

## 8. Learning-Oriented Development

Before implementing a significant feature, briefly explain:

1. What problem is being solved?
2. What is the simplest approach?
3. Why is this approach suitable?
4. Which files need to be created or modified?

After implementation, explain:

1. What was implemented?
2. How does the execution flow work?
3. Why were the important libraries, functions, or classes used?
4. Were any important design trade-offs made?

Keep explanations concise, beginner-friendly, and focused on the implementation.

Do not overwhelm me with unnecessary technical details.

## 9. Final Self-Review

Before completing any coding task, review the implementation:

- Can this code be made easier to understand?
- Are all functions and classes necessary?
- Are all configuration parameters actually needed?
- Is there any unnecessary abstraction?
- Can the same functionality be achieved using a simpler approach?
- Can a beginner-to-intermediate developer explain the execution flow?
- Have existing functionality and requirements been preserved?
- Do the relevant tests pass?

If a simpler implementation provides the same functionality, choose it.

**Write code that is easy to understand first, easy to maintain second, and optimized only when necessary. Do not write complicated code merely to demonstrate advanced software engineering knowledge.**


## FinSight-Specific Guidelines

- Use native LangChain and LangGraph functionality instead of building unnecessary custom abstractions.
- Keep agent definitions, tools, graph nodes, state schemas, and prompts straightforward.
- Use Pydantic for meaningful data validation and structured outputs.
- Keep configuration simple and expose only genuinely tunable parameters.
- Maintain the zero-cost API policy.
- Retain rate limiting and token accounting where necessary to respect free-tier quotas.
- Do not introduce custom infrastructure when the existing libraries already solve the problem adequately.
- Prefer functional implementations over class-based implementations unless persistent state or encapsulation is required.
- Implement features incrementally, ensuring each stage works before introducing additional complexity.
- Ensure financial calculations remain correct, transparent, and independently testable.
- Never remove existing functionality just to reduce the number of lines of code.