# DECISIONS.md — Architecture decision log

One line per decision: **what** we chose and **why**. Newest at the bottom. Phase in brackets.

- [P0] **uv** for env + deps: one fast tool for Python install, venv, lockfile (`uv.lock`) and running commands; reproducible installs.
- [P0] **src/ layout + installed package** (`uv init --package`): tests import the installed `finsight` package, catching packaging/import bugs early.
- [P0] **Local embeddings (sentence-transformers)** for RAG: no API key or per-call cost, deterministic, works offline; quality is sufficient for 10-K retrieval.
- [P0] **Groq vs OpenRouter split**: Groq (fast, cheap) for Supervisor routing + guardrail checks, which are frequent and simple; OpenRouter (stronger model) for Research/Risk/Report, where reasoning and writing quality matter.
- [P0] **US-only, USD-only** portfolios for now: avoids FX conversion, multiple exchanges and ticker-suffix handling; keeps Phase 1 math simple.
- [P0] **langgraph-checkpoint-sqlite** as an explicit dependency: SqliteSaver ships separately from core `langgraph`.
- [P0] **Bond ETFs tagged `asset_type=bond`**: asset_type describes economic exposure (used for allocation), not the wrapper.
