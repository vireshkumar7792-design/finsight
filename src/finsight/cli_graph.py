"""CLI for the Phase 3 multi-agent graph.

    uv run python -m finsight.cli_graph data/samples/concentrated_risky.csv "How risky is my portfolio?" --thread demo1
    uv run python -m finsight.cli_graph data/samples/concentrated_risky.csv "Which position drives that?" --thread demo1   # follow-up
    uv run python -m finsight.cli_graph --thread demo1 --show-state     # inspect the checkpoint
    uv run python -m finsight.cli_graph --thread demo1 --resume         # continue an unfinished run (e.g. after Ctrl-C)

All models are free-tier by default (see config.py). Checkpoints: data/checkpoints/finsight.sqlite.
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from typing import Any

from finsight.config import _LIMITERS, MissingAPIKeyError, PaidModelRefusedError, get_settings
from finsight.context import PortfolioContext
from finsight.graph import build_graph, open_checkpointer, resume_thread, run_question, thread_config
from finsight.schemas.report import render_report
from finsight.tools.csv_loader import PortfolioValidationError


def _short(text: str, n: int = 140) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 3] + "..."


def print_update(node: str, update: dict[str, Any]) -> None:
    """Live progress line(s) for one node's state update."""
    if node == "supervisor":
        for entry in update.get("route_log") or []:
            print(f"[supervisor] -> {entry['to']:<9} ({entry['by']}) {_short(entry['reason'], 110)}")
    elif node in ("portfolio", "risk", "research"):
        finding = update.get(f"{node}_findings")
        calls = ", ".join(f"{t['name']}({json.dumps(t['args'])})" for t in update.get("tool_calls") or []) or "none"
        print(f"[{node}] tools: {calls} | LLM calls: {update.get('llm_calls', 0)}")
        if finding:
            print(f"   finding: {_short(finding['summary'])}")
        for e in update.get("errors") or []:
            print(f"   ERROR: {e}")
    elif node == "report":
        print(f"[report] mode: {update.get('report_mode')} | LLM calls: {update.get('llm_calls', 0)}")


def print_summary(snap) -> None:
    v = snap.values
    route = " -> ".join(["supervisor"] + [x for e in v.get("route_log", []) for x in (e["to"], "supervisor")][:-1])
    print("\n" + "=" * 76)
    print(f"Path: {route}")
    waited = sum(lim.total_wait_seconds for lim in _LIMITERS.values())
    print(f"LLM calls this question: {v.get('llm_calls', 0)} | rate-limit waits: {waited:.1f}s")
    if v.get("errors"):
        print("Errors:\n  - " + "\n  - ".join(v["errors"]))
    print("=" * 76 + "\n")
    if v.get("report"):
        print(render_report(v["report"]))
    elif snap.next:
        print(f"Run is unfinished (next: {', '.join(snap.next)}). Continue with --resume.")


def show_state(graph, thread_id: str) -> int:
    config = thread_config(thread_id)
    snap = graph.get_state(config)
    if not snap.values:
        print(f"No checkpoint for thread {thread_id!r}.")
        return 1
    v = snap.values
    history = list(graph.get_state_history(config))
    findings = {
        k: {"summary": _short(f["summary"], 100), "data_keys": list(f["data"])}
        for k in ("portfolio_findings", "risk_findings", "research_findings")
        if (f := v.get(k))
    }
    view = {
        "thread_id": thread_id,
        "checkpoints_saved": len(history),
        "next_nodes": list(snap.next),
        "question": v.get("question"),
        "portfolio_source": v.get("portfolio_source"),
        "plan": v.get("plan"),
        "route_log": v.get("route_log"),
        "completed_agents": v.get("completed_agents"),
        "iteration": v.get("iteration"),
        "llm_calls": v.get("llm_calls"),
        "tool_calls": v.get("tool_calls"),
        "findings": findings,
        "report_mode": v.get("report_mode"),
        "report_title": (v.get("report") or {}).get("title"),
        "errors": v.get("errors"),
        "conversation": [f"{m.type}: {_short(m.content, 90)}" for m in v.get("messages", [])],
    }
    print(json.dumps(view, indent=2, default=str))
    return 0


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles default to cp1252
    p = argparse.ArgumentParser(prog="finsight-graph", description="Ask the FinSight multi-agent graph.")
    p.add_argument("csv", nargs="?", help="portfolio CSV (optional with --resume/--show-state)")
    p.add_argument("question", nargs="?", help="question in quotes")
    p.add_argument("--thread", help="thread id (conversation + checkpoints); default: new random id")
    p.add_argument("--resume", action="store_true", help="continue an unfinished run on --thread")
    p.add_argument("--show-state", action="store_true", help="print the checkpointed state for --thread")
    p.add_argument("--db", help="checkpoint SQLite path (default from settings)")
    args = p.parse_args(argv)

    settings = get_settings()  # loads .env (also enables LangSmith tracing if configured)
    thread_id = args.thread or f"run-{uuid.uuid4().hex[:8]}"
    if (args.resume or args.show_state) and not args.thread:
        p.error("--resume/--show-state need --thread")
    if not (args.resume or args.show_state) and not (args.csv and args.question):
        p.error("give a CSV and a question (or use --resume / --show-state with --thread)")

    with open_checkpointer(args.db or settings.checkpoint_path) as saver:
        try:
            graph = build_graph(checkpointer=saver, settings=settings)
        except (MissingAPIKeyError, PaidModelRefusedError) as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        if args.show_state:
            return show_state(graph, thread_id)

        csv_path = args.csv
        if args.resume:
            snap = graph.get_state(thread_config(thread_id))
            if not snap.next:
                print(f"Thread {thread_id!r} has nothing to resume (finished or unknown).")
                return 1
            csv_path = csv_path or snap.values.get("portfolio_source")

        try:
            print(f"Thread: {thread_id} | loading {csv_path} + market data ({settings.history_period})...")
            ctx = PortfolioContext.from_csv(csv_path, settings)
        except (PortfolioValidationError, ValueError, OSError) as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        print(f"Models: router={settings.router_model} | agent={settings.agent_model} | report={settings.report_model}\n")

        if args.resume:
            snap = resume_thread(graph, ctx, thread_id, on_update=print_update)
        else:
            pending = graph.get_state(thread_config(thread_id)).next
            if pending:
                print(f"Note: thread has an unfinished run (next: {pending}); starting the new question anyway.")
            snap = run_question(graph, ctx, args.question, thread_id, str(csv_path), on_update=print_update)
        print_summary(snap)
        print(f"\n(thread: {thread_id} - ask a follow-up with --thread {thread_id}, inspect with --show-state)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
