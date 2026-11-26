"""Command-line entry point for trying the Phase 2 analyst.

    uv run python -m finsight.cli data/samples/balanced.csv "What's my sector exposure?"
    uv run python -m finsight.cli data/samples/balanced.csv "..." --free       # Groq (router role)
    uv run python -m finsight.cli data/samples/balanced.csv "..." --messages   # dump full transcript

LangSmith tracing turns on automatically when LANGSMITH_TRACING/LANGSMITH_API_KEY are in .env.
"""

from __future__ import annotations

import argparse
import json
import sys

from finsight.agents.analyst import ask, build_analyst
from finsight.config import MissingAPIKeyError, get_settings
from finsight.context import PortfolioContext
from finsight.tools.csv_loader import PortfolioValidationError


def _short(text: str, limit: int = 160) -> str:
    return text if len(text) <= limit else text[:limit] + "..."


def main(argv: list[str] | None = None) -> int:
    # Windows consoles default to cp1252; LLM output often contains characters it can't encode.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    p = argparse.ArgumentParser(prog="finsight", description="Ask the FinSight analyst about a portfolio CSV.")
    p.add_argument("csv", help="path to portfolio CSV")
    p.add_argument("question", help="question in quotes")
    p.add_argument("--free", action="store_true", help="use the Groq 'router' model instead of OpenRouter")
    p.add_argument("--messages", action="store_true", help="print the full message transcript")
    args = p.parse_args(argv)

    settings = get_settings()  # also loads .env -> enables LangSmith tracing
    role = "router" if args.free else "agent"
    model_name = settings.router_model if args.free else settings.agent_model

    try:
        print(f"Loading portfolio and market data ({settings.history_period} history)...")
        ctx = PortfolioContext.from_csv(args.csv, settings)
        agent = build_analyst(role=role)
    except (PortfolioValidationError, MissingAPIKeyError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print(f"Asking {model_name} ...\n")
    try:
        result = ask(args.question, ctx, agent)
    except Exception as e:  # provider errors: credits (402), rate limits (429) after retries, timeouts
        print(f"LLM call failed ({type(e).__name__}): {e}", file=sys.stderr)
        return 2

    print("=" * 72)
    print(f"TOOL CALLS ({len(result.tool_calls)})")
    for i, tc in enumerate(result.tool_calls, 1):
        print(f"  {i}. {tc.name}({json.dumps(tc.args)})")
        print(f"     -> {_short(tc.output)}")
    if args.messages:
        print("=" * 72 + "\nTRANSCRIPT")
        for m in result.messages:
            m.pretty_print()
    print("=" * 72 + "\nANSWER\n")
    print(result.answer)

    usage = [m.usage_metadata for m in result.messages if getattr(m, "usage_metadata", None)]
    if usage:
        tin, tout = sum(u["input_tokens"] for u in usage), sum(u["output_tokens"] for u in usage)
        print(f"\n[{len(usage)} LLM calls, {tin} input + {tout} output tokens]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
