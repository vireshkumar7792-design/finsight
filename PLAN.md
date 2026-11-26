# PLAN.md — FinSight build plan

Bottom-up. A phase starts only when the previous phase's tests pass.

- [x] **Phase 0: Project setup**
  - Done when: `uv sync` succeeds, `uv run pytest` runs, skeleton + docs + sample CSVs exist.

- [x] **Phase 1: CSV validation + financial tools (no LLM)**
  - Pydantic holding schema + CSV loader; `portfolio_math`, `risk_math`, `market_data` (yfinance).
  - Done when: all three sample CSVs validate (and bad CSVs fail with clear errors), and every tool function has a known-answer pytest that passes offline (market data mocked).

- [ ] **Phase 2: Single agent with tools + `get_llm(role)` config**
  - `config.py` with `get_llm(role)` (Groq vs OpenRouter); one tool-calling agent over the Phase 1 tools.
  - Done when: one agent answers "summarize my portfolio" for a sample CSV using only tool outputs for numbers, `get_llm` is unit-tested, and the run is visible in LangSmith.

- [ ] **Phase 3: Multi-agent LangGraph + SqliteSaver**
  - Supervisor (structured-output routing), Portfolio, Risk, Report agents; Research stubbed; shared state; SqliteSaver checkpoints.
  - Done when: the graph runs end-to-end on all sample CSVs producing a report, routing logic is tested, and a run can be resumed from its checkpoint by thread_id.

- [ ] **Phase 4: Research agent — 10-K RAG + Tavily, cited schemas**
  - Ingest 10-Ks into Chroma with local embeddings; retriever with citation metadata; Tavily search; `Insight`/`Source` schemas.
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
