# DECISIONS.md — Architecture decision log

One line per decision: **what** we chose and **why**. Newest at the bottom. Phase in brackets.

- [P0] **uv** for env + deps: one fast tool for Python install, venv, lockfile (`uv.lock`) and running commands; reproducible installs.
- [P0] **src/ layout + installed package** (`uv init --package`): tests import the installed `finsight` package, catching packaging/import bugs early.
- [P0] **Local embeddings (sentence-transformers)** for RAG: no API key or per-call cost, deterministic, works offline; quality is sufficient for 10-K retrieval.
- [P0] **Groq vs OpenRouter split**: Groq (fast, cheap) for Supervisor routing + guardrail checks, which are frequent and simple; OpenRouter (stronger model) for Research/Risk/Report, where reasoning and writing quality matter.
- [P0] **US-only, USD-only** portfolios for now: avoids FX conversion, multiple exchanges and ticker-suffix handling; keeps Phase 1 math simple.
- [P0] **langgraph-checkpoint-sqlite** as an explicit dependency: SqliteSaver ships separately from core `langgraph`.
- [P0] **Bond ETFs tagged `asset_type=bond`**: asset_type describes economic exposure (used for allocation), not the wrapper.
- [P1] **Math takes data, never fetches it**: prices/history are passed in, so tools are deterministic, testable offline, and MCP-ready.
- [P1] **CSV parsed with stdlib csv as strings; Pydantic is the only type converter**; all row errors reported together with line numbers.
- [P1] **Losses (VaR, max drawdown) returned as positive fractions**; portfolio returns assume constant current weights.
- [P2] **Cash = fixed $1 price, constant history**: never sent to yfinance; zero returns are valid data; correlations with cash are undefined (NaN) and skipped explicitly.
- [P2] **Current price = last close of the same batched 1y download** used for risk, so prices and history can never disagree.
- [P2] **Settings via pydantic-settings + `load_dotenv()`**: keys are `SecretStr` (masked in logs); load_dotenv is still needed to export `LANGSMITH_*` to os.environ for tracing.
- [P2] **Groq router default `openai/gpt-oss-20b`; OpenRouter agent `claude-haiku-5.5`, report `claude-sonnet-5.5`**: verified against live docs (Groq retired Llama 3.x in Aug 2026); all overridable via `FINSIGHT_*_MODEL`.
- [P2] **Retries at the SDK level (`max_retries`)**, not agent middleware: the provider SDKs back off exponentially and honor 429 `Retry-After` headers.
- [P2] **`max_output_tokens=2000` cap**: without it OpenRouter pre-authorizes credits for the model's full 65k output and returns 402 on low balances.
- [P2] **`create_agent` (langchain v1), not `create_react_agent`** (deprecated in LangGraph v1).
- [P2] **Context injected via `ToolRuntime`**: hidden from the LLM's tool schema, so the model can't pass (or fabricate) portfolio data as arguments.
- [P2] **Payload functions separate from `@tool` wrappers**: payloads are unit-tested directly and will be reused by MCP servers in Phase 5.
- [P2] **Tools return percentages already converted (`_pct`/`_usd` suffixes) and pre-ranked fields** (e.g. worst position): unit conversion and ranking are math, so the LLM doesn't do them.
- [P2] **When the LLM overgeneralizes, give it the missing data**: added per-holding average correlation after the model invented an MSFT claim from a single pair.
- [P3] **Zero-cost policy**: all roles default to Groq free tier; OpenRouter only with `:free` ids; `get_llm()` refuses paid ids unless `FINSIGHT_ALLOW_PAID=true`.
- [P3] **Models per role from measured capabilities**: router `gpt-oss-20b` (strict structured output, fast), agent `qwen3.8-27b` (only one with parallel tool calls -> 2 calls per specialist), report `gpt-oss-120b` (strongest, strict structured output). Qwen is preview; fallback is a one-line env change.
- [P3] **Free limits measured from Groq response headers** (30 RPM, 1000 RPD, 8000 TPM per model) because the docs page didn't render free-plan numbers.
- [P3] **Custom request+token limiter** instead of InMemoryRateLimiter: tokens/min is the binding free-tier limit and InMemoryRateLimiter only counts requests; SDK retry on 429 stays as a backstop.
- [P3] **One limiter per model id, shared across roles**: Groq limits are per model, so roles on the same model share one budget.
- [P3] **Own 40-line SQLiteLLMCache** (core BaseCache) instead of adding langchain-community; deserialization restricted to Generation/ChatGeneration/AIMessage.
- [P3] **Supervisor plans once, then walks the plan deterministically**: 1 router call per question instead of one per hop; routing is still LLM-decided.
- [P3] **Every question ends in the report node** ("finish" coerced to report): the user always gets an answer, and fallbacks have one place to land.
- [P3] **Specialists hand over Findings (tool payloads + short summary), not transcripts**: the report prompt stays small, and every number has a source id like `risk:get_risk_metrics`.
- [P3] **Deterministic fallbacks at every LLM step**: router failure -> report; specialist without tool calls -> default tool data; report: structured x2 -> plain text -> no-LLM summary.
- [P3] **Light traceability check now** (key metric value must appear in its source payload); full number-consistency guardrails are Phase 6.
- [P3] **PortfolioContext stays out of state** (DataFrames would bloat checkpoints); state stores `portfolio_source` so resume can rebuild it.
- [P3] **State holds plain dicts (model_dump), not Pydantic objects**: checkpoints serialize cleanly and safely; Pydantic validates at the node boundaries.
- [P3] **Custom resettable reducers** (`accumulate`, `add_or_reset`; `None` resets): per-question logs restart on a follow-up while `messages` keeps the conversation.
