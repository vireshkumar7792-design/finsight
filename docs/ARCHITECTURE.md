# FinSight architecture

This file explains **why** the code is shaped the way it is, one short section per module.
The code itself only has short comments; the reasons live here.

## The big picture

```
CSV ──► portfolio/ (load + market data + math) ──► PortfolioContext
                                                        │
question ──► graph.py ──► supervisor ──► portfolio / risk / research agents ──► report agent ──► Report
                              ▲                    │
                              └────── back ────────┘
```

Three ideas run through everything:

1. **The LLM never does math.** Python calculates every number. The LLM picks tools and explains.
2. **Every claim is traceable.** Numbers point to a tool output; research claims point to a source id.
3. **Free models fail sometimes, so nothing crashes.** Each LLM step: try → retry once → simple fallback.

---

## config.py — settings and models

- `Settings` reads `.env`. API keys are `SecretStr`, so printing settings shows `**********`.
- `get_llm(role)` gives each role its model. Roles: `router` (small, fast), `agent` (tool calling),
  `report` (strongest writer). All defaults are Groq's free tier.
- **Cost guard:** an OpenRouter model must end in `:free`, unless `FINSIGHT_ALLOW_PAID=true`.
- **Rate limiting:** LangChain's built-in `InMemoryRateLimiter` keeps us under 30 requests/minute.
  Groq's real limit is 8,000 *tokens*/minute; when we hit it, Groq answers "429, retry after N
  seconds" and the SDK (`max_retries=5`) waits and retries. Tested on a 6-call question: no errors.
- One rate limiter per model, because Groq's limits are per model.

## schemas.py — all Pydantic models

- `Holding` / `Portfolio`: one CSV row / all rows. Validation happens here (positive numbers,
  valid ticker, no future dates, cash at $1, no duplicate tickers).
- `RoutePlan`, `Report`, `ResearchDraft`: what the LLM must return (structured output).
  Groq's strict JSON mode needs every field to be required, so these have no defaults.
- `Source`: something a tool really returned. Only code creates these, never the LLM.

## portfolio/ — data and math (no LLM)

- **loader.py** reads the CSV as text and lets Pydantic do all checks. It collects *every* error
  with its line number, so you fix the file once instead of ten times.
- **market.py** is the only code that talks to Yahoo Finance. Prices are cached per day (every
  run on the same day sees the same numbers); sectors are cached for 30 days, but "Unknown" is
  never cached because it usually means a network hiccup. Cache files are JSON, not pickle,
  because loading a pickle can run code.
- **calculations.py** holds every formula as a plain function. Conventions: fractions
  (0.25 = 25%), daily returns, sample statistics (ddof=1), 252 trading days a year, losses as
  positive numbers. Portfolio returns assume today's weights were held all year (the standard
  simplification for risk estimates).
- **context.py** builds `PortfolioContext` once per run: one download for all holdings plus SPY,
  cash as a flat $1 line (it is never sent to Yahoo), and pre-computed prices, weights and returns.

## tools/ — what the agents can call

- **portfolio_tools.py**: each tool is a plain payload function (easy to test) plus a small
  `@tool` wrapper (what the LLM sees). Outputs are already rounded and carry the unit in the
  name (`weight_pct`, `market_value_usd`), because converting units is math too.
  The portfolio data reaches the tool through `ToolRuntime`, which LangChain hides from the
  model, so the model can't pass in (or invent) data.
- **research_tools.py**: 10-K search and Tavily web search. Every result becomes a `Source` in
  a per-question registry (`FoundSources`). The LLM chooses searches, but code runs them, so
  code can enforce the limits (3 filing searches, 2 web searches per question).
  Tavily never raises: no key, used-up quota or a network error becomes a "gap" in the
  research, and the run continues. Results are cached on disk to save the 1,000 free credits.

## rag/ — 10-K filings

- **edgar.py** downloads the latest 10-K from the SEC (free). The SEC asks for a User-Agent with
  contact details (`SEC_USER_AGENT`); it goes only into the request header and is never printed.
  We wait 0.25 s between requests (the SEC allows 10 per second).
- **parse.py** turns HTML into text and splits it into Items (1 Business, 1A Risk Factors, 7 MD&A…).
  Every heading appears twice (table of contents + real start); the real start is the one followed
  by the most text. If that fails, the whole filing becomes one "Full document" section.
  Chunks are 220 words with 40 words of overlap (about 300 tokens, which fits the embedding
  model's 512-token window). Item 8 (financial tables) is skipped: numbers are our tools' job.
- **ingest.py** embeds chunks **locally** with `BAAI/bge-small-en-v1.5` (free, no API) and stores
  them in Chroma. Chunk ids are fixed (`NVDA-10K-FY2026-1A-007`) and we *upsert*, so running
  ingest twice never duplicates; if nothing changed, embedding is skipped. We pass our own
  vectors to Chroma so it never downloads its own model, and Chroma telemetry is off.
- **retriever.py** searches by meaning, with ticker and section filters applied **before** ranking.
  Without the ticker filter, Apple's supply-chain paragraph could be cited for a NVIDIA question.

## state.py — what the graph remembers

- `FinSightState` is the shared notebook, saved after every node. `messages` is the conversation
  memory (it grows); every other field is reset by `new_question()` when a new question starts.
- Nodes run one at a time, so each node simply reads a list, adds to it and writes it back.
  No custom reducers are needed.
- `RunContext` carries the portfolio data and the LLMs into every node. It is **not** saved
  (DataFrames and API clients don't belong in a database); on resume it is rebuilt from the
  saved `csv_path`.

## agents/ — the team

Every agent file has the same shape: PROMPT → TOOLS → node function → helpers.

- **supervisor.py** makes **one** LLM call per question: "which agents, in what order?"
  After that it just walks the plan. This saves LLM calls on a free tier. If the LLM fails
  twice, `keyword_plan` routes by words like "risk" or "sector". A 6-visit limit stops loops.
- **portfolio_agent.py / risk_agent.py** are small tool-calling agents (LangChain `create_agent`),
  each with its own tools. They usually need 2 LLM calls: pick tools, then summarize.
- **common.py** holds what they share: `ask_structured` (try + retry once) and `run_tool_agent`.
  If an agent fails twice or calls no tools, its default tools run without the LLM, so the
  report still gets real numbers. Agents hand over *findings* (tool outputs + a short summary),
  not chat transcripts, which keeps the report prompt small.
- **research_agent.py** is a fixed 2-call pipeline instead of an agent loop:
  (1) the LLM chooses searches, (2) code runs them, (3) the LLM writes insights citing source ids,
  (4) **code drops any claim that cites an id the tools never returned**. That is grounding
  enforced in code, not just asked for in the prompt.
- **report_agent.py** writes the final `Report`. If it fails twice, a plain report is built from
  the agents' own summaries. Then `keep_traceable_metrics` removes any key metric whose number
  doesn't actually appear in the tool output it cites. The disclaimer is added by code.

## graph.py — the wiring

Reads top to bottom: build the run context (data + LLMs) → routing → build the graph → run.
`SqliteSaver` saves the state after every node under a `thread_id`, which gives three things:
follow-up questions (same thread = same conversation), resume after a crash (`input=None`), and
inspection (`--show-state`). Resume refuses to mix two days of market data: if the prices have
changed since the run started, the question is asked again from the start.

## cli.py — the commands

`ask`, `ingest` and `research`. A tiny LangChain callback (`LLMCallCounter`) counts every model
request, including those inside agents, so nodes don't have to count anything.

## LLM calls per question

| Question type | Calls |
|---|---|
| one specialist (portfolio or risk) | 4 (router, agent ×2, report) |
| two specialists | 6 |
| research | 4 (router, search planning, writer, report) |
| a model keeps failing | each step tries twice, then falls back; you still get a report |
