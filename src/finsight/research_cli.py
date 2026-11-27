"""Standalone research CLI (no graph, no portfolio required).

    uv run python -m finsight.research_cli "What are NVDA's main risks?" --ticker NVDA
    uv run python -m finsight.research_cli "What's the latest news on TSLA?"
    uv run python -m finsight.research_cli "Any risks in my holdings?" --csv data/samples/tech_heavy.csv

Prints tool calls, each insight with its citations, the sources, gaps, and LLM/Tavily call counts.
Requires ingested 10-Ks (python -m finsight.rag.ingest ...) for filing results.
"""

from __future__ import annotations

import argparse
import sys

from finsight.agents.research import research_deps_from_settings, run_research
from finsight.config import MissingAPIKeyError, PaidModelRefusedError, get_llm, get_settings


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(prog="finsight-research", description="Grounded research from 10-Ks + web.")
    p.add_argument("question")
    p.add_argument("--ticker", action="append", help="focus ticker (repeatable); default: detected from question")
    p.add_argument("--csv", help="portfolio CSV: focus on its stock holdings when no ticker is given")
    p.add_argument("--no-web", action="store_true", help="filings only (saves Tavily credits)")
    args = p.parse_args(argv)

    s = get_settings()
    try:
        tool_llm, writer_llm = get_llm("agent", s), get_llm("report", s)
    except (MissingAPIKeyError, PaidModelRefusedError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    holdings = None
    if args.csv:
        from finsight.tools.csv_loader import load_portfolio_csv

        holdings = [h.ticker for h in load_portfolio_csv(args.csv).holdings if h.asset_type.value == "equity"]

    print(f"Loading local embedder ({s.embedding_model}) and the 10-K index...")
    deps = research_deps_from_settings(s)
    if args.no_web:
        deps.web = None
    indexed = sorted(deps.retriever.available_tickers()) if deps.retriever else []
    print(f"10-Ks indexed: {', '.join(indexed) or 'none (run python -m finsight.rag.ingest ...)'}")
    print(f"Models: tools={s.agent_model} | writer={s.report_model}\n")

    run = run_research(args.question, deps, tool_llm, writer_llm, tickers=args.ticker, holdings=holdings)
    f = run.findings

    print(f"Focus: {', '.join(f.tickers) or '-'}")
    print("TOOL CALLS")
    for c in run.tool_calls:
        status = f"ERROR: {c['error']}" if c["error"] else f"{c['results']} result(s)"
        print(f"  - {c['name']}({', '.join(f'{k}={v!r}' for k, v in c['args'].items())}) -> {status}")

    print(f"\nINSIGHTS ({len(f.insights)})")
    for i, ins in enumerate(f.insights, 1):
        print(f"  {i}. [{ins.category}/{ins.confidence}] {ins.claim}")
        print(f"     cites: {', '.join(ins.source_ids)}")

    print(f"\nSOURCES ({len(f.sources)})")
    for src in f.sources:
        where = src.section if src.type == "filing" else (src.published_date or "date n/a")
        print(f"  [{src.id}] {src.title} | {where}\n      {src.url}")

    if f.gaps:
        print("\nGAPS")
        for g in f.gaps:
            print(f"  - {g}")
    if run.dropped:
        print(f"\nDROPPED by post-check (cited no retrieved source): {len(run.dropped)}")
        for d in run.dropped:
            print(f"  - {d}")

    web = deps.web
    tavily = f"{web.api_calls} API call(s), {web.cache_hits} cache hit(s)" if web else "disabled"
    print(f"\nLLM calls: {run.llm_calls} (synthesis: {run.synthesis_mode}) | Tavily: {tavily}")
    for line in run.errors:
        print(f"  error (fallback used): {line}")
    for line in run.log:
        print(f"  log: {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
