# PLAN.md — FinSight build plan

Bottom-up. A phase starts only when the previous phase's tests pass.

- [x] **Phase 0: Project setup**
  - Done when: `uv sync` succeeds, `uv run pytest` runs, skeleton + docs + sample CSVs exist.

- [x] **Phase 1: CSV validation + financial tools (no LLM)**
  - Pydantic holding schema + CSV loader; `portfolio_math`, `risk_math`, `market_data` (yfinance).
  - Done when: all three sample CSVs validate (and bad CSVs fail with clear errors), and every tool function has a known-answer pytest that passes offline (market data mocked).

- [x] **Phase 2: Single agent with tools + `get_llm(role)` config**
  - `config.py` (Settings + `get_llm(role)`), cash support, `PortfolioContext`, 5 LangChain tools, `create_agent` analyst, CLI.
  - Done when: offline `uv run pytest` passes, and on the real LLM these call the right tools: sector exposure (balanced), how risky (concentrated_risky), biggest loser (tech_heavy), diversified? (allocation + HHI + correlations).
  - LangSmith tracing confirmed (runs visible in project `finsight`).

- [x] **Phase 3: Multi-agent LangGraph + SqliteSaver (zero-cost)**
  - Supervisor (one structured-output planning call), Portfolio + Risk specialists, Research stub, structured Report with fallbacks; FinSightState with reducers; SqliteSaver checkpoints; follow-ups on a thread; free-tier models only, cost guard, request+token rate limiter, optional SQLite LLM cache; `cli_graph` with `--resume` / `--show-state`.
  - Done when: offline `uv run pytest` passes; on FREE models, sector (balanced), risky (concentrated_risky), diversified+performance (tech_heavy, portfolio+risk) and "Research NVDA" (stub) run end-to-end with sensible routes; a killed run resumes by thread_id.
  - Measured: 2 LLM calls (research stub), 4 (one specialist), 6 (two specialists); ~1.3K / 4-5K / 8-10K tokens.

- [ ] **Phase 4: Research agent — 10-K RAG + Tavily, cited schemas** (built, offline tests pass; real 10-K runs pending SEC_USER_AGENT)
  - Ingest 10-Ks into Chroma with local embeddings; retriever with citation metadata; Tavily search; `Insight`/`Source` schemas.
  - Wired into the Phase 3 graph (`research` node replaces the stub; deps load lazily). Standalone: `research_cli`.
  - Remaining: set SEC_USER_AGENT, `python -m finsight.rag.ingest --from-samples`, then real Q1 (NVDA risks -> Item 1A) and Q3 (Apple competition + news).
  - Done when: Research returns Insights that each carry ≥1 Source (chunk id or URL), retrieval is tested on a small fixture corpus, and the stub is replaced in the graph.

- [ ] **Phase 5: Move tools to MCP servers**
  - FastMCP `portfolio_server` and `market_server`; agents consume them via langchain-mcp-adapters.
  - Done when: agents get all tools via MCP, the graph output matches Phase 4 behavior, and MCP servers have tests calling their tools.

- [ ] **Phase 6: Guardrails + HITL interrupt**
  - Schema validation, grounding check, number consistency; risk classifier; LangGraph `interrupt` for high-risk recommendations.
  - Done when: `concentrated_risky.csv` pauses for human approval and resumes correctly on approve/reject, and an ungrounded claim or mismatched number is caught by a test.

- [ ] **Phase 7: Streamlit UI**
  - Upload CSV, run graph, show progress, approve/reject high-risk items, render cited report.
  - Done when: the full flow (including HITL approval) works in the browser for all three sample CSVs.

- [ ] **Phase 8: Evals, README, demo**
  - Eval set (routing accuracy, grounding rate, number consistency), README with architecture diagram, demo recording.
  - Done when: evals run with one command and report scores, README lets a stranger run the project, and a demo is recorded.
