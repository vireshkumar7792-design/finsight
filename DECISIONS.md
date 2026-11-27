# DECISIONS.md — short decision log

One line each: **what** we chose and **why**. Longer explanations are in `docs/ARCHITECTURE.md`.

- **Rewrite from scratch** (old code kept at tag `before-rewrite`): the old code worked but was too dense to read and explain.
- **uv + src layout**: one tool for Python, venv and lockfile; tests import the installed package.
- **US-only, USD-only**: no currency conversion or foreign exchanges to handle.
- **Math in plain Python, data passed in**: deterministic, testable offline with hand-checked answers.
- **Losses as positive numbers; today's weights held all year**: standard simplifications for risk estimates.
- **Cash = flat $1 price, never sent to Yahoo**: its returns are zero and its correlations are skipped.
- **Daily price cache in JSON** (not pickle): same numbers all day, fewer Yahoo calls, no code-execution risk.
- **Free models only, with a cost guard**: the user has no budget; paid OpenRouter ids are refused.
- **Built-in InMemoryRateLimiter + SDK retries** instead of a custom token limiter: Groq's 429 "retry after" handles the token limit; tested on 6-call questions with no errors.
- **No LLM response cache**: the built-in one is in-memory only, and a disk one needs another dependency.
- **Supervisor plans once, code follows the plan**: one routing call per question instead of one per step.
- **Keyword routing as the supervisor fallback**: still sends the question somewhere useful when the router fails.
- **Try → retry once → simple fallback at every LLM step**: free models are flaky; a run must never crash.
- **RunContext (data + LLMs) outside the checkpointed state**: DataFrames and clients don't belong in a database.
- **Plain state fields, no custom reducers**: nodes run one at a time, so read-append-write is enough.
- **Agents hand over findings, not transcripts**: small report prompt; each number has a source id.
- **Key metrics checked in code** against the tool output they cite; untraceable ones are dropped.
- **Research = fixed 2-call pipeline**: choose searches, then write; an agent loop would add a useless 3rd call.
- **Grounding in code**: the LLM only cites ids; Sources come from tool results; unknown-id claims are dropped.
- **Local bge-small-en-v1.5 embeddings**: free, retrieval-tuned, 512-token window for ~300-token chunks.
- **Ticker/section filters before ranking**: a similar paragraph from another company must never be cited.
- **Fixed chunk ids + upsert**: ingest is idempotent; unchanged filings skip embedding.
- **Tavily never raises; failures become gaps**: research continues on filings; undated news is dropped.
- **Chroma opened only when research runs**: importing it takes ~3 s, which other questions shouldn't pay.
