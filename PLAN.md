# PLAN.md — FinSight build plan

Bottom-up. A phase starts only when the previous phase's tests pass.
The code was rewritten from scratch for readability (branch `rewrite`; the old code is tag `before-rewrite`).

- [x] **Phase 0: Project setup**: uv project, src layout, sample CSVs, docs.

- [x] **Phase 1: CSV validation + math (no LLM)**
  - `portfolio/loader.py`, `market.py` (yfinance + daily cache), `calculations.py`, `context.py`.
  - Done: sample CSVs load; bad CSVs list every error with line numbers; every formula has a
    hand-checked test.

- [x] **Phase 2: Config + tools**
  - `config.py` (free models, cost guard, rate limiter), `tools/portfolio_tools.py` (rounded,
    unit-labeled outputs; portfolio data hidden from the LLM).

- [x] **Phase 3: Multi-agent graph + checkpoints**
  - Supervisor (1 planning call), Portfolio + Risk agents, Report agent with traceable metrics,
    SqliteSaver checkpoints, follow-ups, resume (refuses to mix market-data days), `finsight ask`.
  - Measured on free Groq models: 4 LLM calls for one agent (~20-30 s), 6 for two (~35 s).

- [x] **Phase 4: Research agent (10-K RAG + Tavily)**
  - SEC download, parsing into Items, chunking, local bge-small embeddings, Chroma with
    ticker/section filters, idempotent ingest, Tavily with disk cache, grounding enforced in code.
  - `finsight ingest` and `finsight research`; the research node is wired into the graph.
  - Verified live: `finsight research` and a research question through `finsight ask` (4 calls).
  - Still to do by the user: set `SEC_USER_AGENT` in `.env`, run `finsight ingest --from-samples`,
    then re-run `finsight research "What are NVDA's main risks?" --ticker NVDA` to see 10-K citations.

- [ ] **Phase 5: Move tools to MCP servers**
  - FastMCP servers for portfolio and research tools; agents consume them via langchain-mcp-adapters.
  - Done when: agents get all tools via MCP, the graph behaves as in Phase 4, and the MCP servers
    have tests that call their tools.

- [ ] **Phase 6: Guardrails + human-in-the-loop**
  - Number-consistency check on report text (not just key metrics), risk classifier, LangGraph
    `interrupt` for high-risk recommendations.
  - Done when: `concentrated_risky.csv` pauses for approval and resumes correctly on approve/reject,
    and an ungrounded claim or mismatched number is caught by a test.

- [ ] **Phase 7: Streamlit UI**
  - Upload CSV, run the graph with progress, approve/reject high-risk items, show the cited report.
  - Done when: the full flow (including approval) works in the browser for all three sample CSVs.

- [ ] **Phase 8: Evals, demo**
  - Eval set (routing accuracy, grounding rate, number consistency), demo recording.
  - Done when: evals run with one command and report scores, and a demo is recorded.
