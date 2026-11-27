"""Command line with three commands:

    finsight ask <csv> "<question>" [--thread ID] [--resume] [--show-state]
    finsight ingest NVDA AAPL ... | --from-samples
    finsight research "<question>" [--ticker NVDA]
"""

import argparse
import json
import sys
import uuid

from langchain_core.callbacks import BaseCallbackHandler

from finsight.agents.report_agent import DISCLAIMER
from finsight.agents.research_agent import pick_tickers, run_research
from finsight.config import get_llm, get_settings
from finsight.graph import build_graph, make_run_context, open_checkpointer, resume_thread, run_question, thread_config
from finsight.rag.ingest import LocalEmbedder, ingest_ticker, open_collection, sample_equity_tickers
from finsight.rag.retriever import indexed_tickers
from finsight.tools.research_tools import open_research_sources

# =========================
# Entry point
# =========================


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # Windows terminals default to an older encoding
    parser = argparse.ArgumentParser(prog="finsight", description="FinSight: an agentic portfolio analyst.")
    commands = parser.add_subparsers(dest="command", required=True)

    ask = commands.add_parser("ask", help="ask a question about a portfolio CSV")
    ask.add_argument("csv", nargs="?", help="portfolio CSV (not needed with --resume or --show-state)")
    ask.add_argument("question", nargs="?")
    ask.add_argument("--thread", help="conversation id; reuse it for follow-up questions")
    ask.add_argument("--resume", action="store_true", help="continue an unfinished run on --thread")
    ask.add_argument("--show-state", action="store_true", help="print the saved state of --thread")

    ingest = commands.add_parser("ingest", help="download and index the latest 10-K filings")
    ingest.add_argument("tickers", nargs="*")
    ingest.add_argument("--from-samples", action="store_true", help="all stocks in data/samples/*.csv")
    ingest.add_argument("--force", action="store_true", help="re-embed even if already ingested")

    research = commands.add_parser("research", help="cited research from 10-Ks and web news")
    research.add_argument("question")
    research.add_argument("--ticker", action="append", default=[], help="focus ticker (can repeat)")

    args = parser.parse_args(argv)
    try:
        if args.command == "ask":
            return ask_command(args)
        if args.command == "ingest":
            return ingest_command(args)
        return research_command(args)
    except ValueError as error:  # bad CSV, missing API key, refused paid model, ...
        print(f"Error: {error}", file=sys.stderr)
        return 1


# =========================
# ask
# =========================


def ask_command(args) -> int:
    if (args.resume or args.show_state) and not args.thread:
        raise ValueError("--resume and --show-state need --thread")
    if not (args.resume or args.show_state) and not (args.csv and args.question):
        raise ValueError('give a CSV and a question, e.g. finsight ask data/samples/balanced.csv "How risky is it?"')

    settings = get_settings()
    graph = build_graph(open_checkpointer(settings.checkpoint_path))
    if args.show_state:
        return show_state(graph, args.thread)

    thread_id = args.thread or f"run-{uuid.uuid4().hex[:8]}"
    if args.resume:
        saved = graph.get_state(thread_config(thread_id))
        if not saved.next:
            raise ValueError(f"thread '{thread_id}' has nothing to resume")
        csv_path = saved.values["csv_path"]
    else:
        csv_path = args.csv

    print(f"Thread: {thread_id} | loading {csv_path} and market data...")
    run_context = make_run_context(csv_path, settings)
    print(f"Models: router={settings.router_model} | agent={settings.agent_model} | report={settings.report_model}\n")

    counter = LLMCallCounter()
    if args.resume:
        final_state = resume_thread(graph, run_context, thread_id, print_update, [counter])
    else:
        final_state = run_question(graph, run_context, args.question, thread_id, print_update, [counter])

    print_summary(final_state, counter.count)
    print(f"\nFollow up with: --thread {thread_id}")
    return 0


class LLMCallCounter(BaseCallbackHandler):
    """LangChain calls this before every chat model request, including inside agents."""

    def __init__(self):
        self.count = 0

    def on_chat_model_start(self, serialized, messages, **kwargs):
        self.count += 1


# =========================
# ingest
# =========================


def ingest_command(args) -> int:
    tickers = [ticker.upper() for ticker in args.tickers]
    if args.from_samples:
        tickers += sample_equity_tickers()
    tickers = list(dict.fromkeys(tickers))
    if not tickers:
        raise ValueError("give tickers (e.g. finsight ingest NVDA AAPL) or --from-samples")

    settings = get_settings()
    if not settings.sec_user_agent or not settings.sec_user_agent.get_secret_value().strip():
        raise ValueError('SEC_USER_AGENT is not set in .env (the SEC asks for "Name email@example.com")')
    user_agent = settings.sec_user_agent.get_secret_value()

    embedder = LocalEmbedder(settings.embedding_model)
    collection = open_collection(settings.chroma_path, embedder.name)
    print(f"Ingesting {', '.join(tickers)} (embeddings run locally with {embedder.name})")
    failures = 0
    for ticker in tickers:
        result = ingest_ticker(ticker, user_agent, settings, collection, embedder, args.force)
        if result["status"].startswith("error"):
            failures += 1
        sections = ",".join(result["sections"]) or "-"
        print(f"  {result['ticker']:<6} {result['status']:<17} chunks={result['chunks']:<4} sections={sections}")
    print(f"Done. The index now holds {collection.count()} chunks.")
    return 1 if failures == len(tickers) else 0


# =========================
# research
# =========================


def research_command(args) -> int:
    settings = get_settings()
    counter = LLMCallCounter()
    search_llm = get_llm("agent", settings)
    writer_llm = get_llm("report", settings)
    search_llm.callbacks = [counter]
    writer_llm.callbacks = [counter]

    print("Loading the local embedder and the 10-K index...")
    research = open_research_sources(settings)
    print(f"10-Ks indexed: {', '.join(sorted(indexed_tickers(research.collection))) or 'none (run finsight ingest)'}")

    tickers = pick_tickers(args.question, args.ticker, [])
    result = run_research(args.question, tickers, research, search_llm, writer_llm)

    print(f"\nFocus: {', '.join(tickers) or '-'}")
    for call in result["tool_calls"]:
        print(f"  searched {call['name']}({json.dumps(call['args'])}) -> {call['results']} result(s)")
    findings = result["findings"]
    print(f"\nINSIGHTS ({len(findings['insights'])})")
    for number, insight in enumerate(findings["insights"], start=1):
        print(f"  {number}. [{insight['confidence']}] {insight['claim']}")
        print(f"     cites: {', '.join(insight['source_ids'])}")
    print_sources(findings)
    for gap in findings["gaps"]:
        print(f"Gap: {gap}")
    for error in result["errors"]:
        print(f"Note: {error}")
    print(f"\nLLM calls: {counter.count}")
    return 0


# =========================
# Printing
# =========================


def print_update(node_name: str, update: dict) -> None:
    """One progress line after each node."""
    if node_name == "supervisor":
        step = update["route_log"][-1]
        print(f"[supervisor] -> {step['to']} ({step['by']}): {step['reason']}")
    elif node_name in ("portfolio", "risk", "research"):
        calls = [call for call in update.get("tool_calls", []) if call["agent"] == node_name]
        names = [f"{call['name']}({json.dumps(call['args'])})" for call in calls]
        print(f"[{node_name}] tools: {', '.join(names) or 'none'}")
    elif node_name == "report":
        print("[report] done")


def print_summary(state: dict, llm_calls: int) -> None:
    route = " -> ".join(["supervisor"] + [step["to"] for step in state["route_log"]])
    print("\n" + "=" * 70)
    print(f"Route: {route}")
    print(f"LLM calls: {llm_calls}")
    for error in state["errors"]:
        print(f"Note: {error}")
    print("=" * 70 + "\n")
    if state["report"]:
        print(render_report(state["report"]))
        research = state["findings"].get("research")
        if research and research["data"]:
            print_sources(research["data"]["research"])
    else:
        print("The run is unfinished. Continue it with --resume.")


def render_report(report: dict) -> str:
    lines = [report["title"], "=" * len(report["title"]), "", report["summary"], ""]
    for section in report["sections"]:
        lines += [f"## {section['heading']}", section["body"], ""]
    if report["key_metrics"]:
        lines.append("Key metrics:")
        for metric in report["key_metrics"]:
            lines.append(f"  - {metric['name']}: {metric['value']:g} {metric['unit']}   [source: {metric['source']}]")
        lines.append("")
    if report["caveats"]:
        lines.append("Caveats:")
        for caveat in report["caveats"]:
            lines.append(f"  - {caveat}")
        lines.append("")
    lines.append(DISCLAIMER)
    return "\n".join(lines)


def print_sources(findings: dict) -> None:
    if not findings["sources"]:
        return
    print(f"\nSOURCES ({len(findings['sources'])})")
    for source in findings["sources"]:
        published = f" ({source['published_date']})" if source["published_date"] else ""
        print(f"  [{source['id']}] {source['title']}{published}\n      {source['url']}")


def show_state(graph, thread_id: str) -> int:
    snapshot = graph.get_state(thread_config(thread_id))
    if not snapshot.values:
        print(f"No saved state for thread '{thread_id}'.")
        return 1
    values = snapshot.values
    view = {
        "thread_id": thread_id,
        "next_nodes": list(snapshot.next),
        "question": values["question"],
        "csv_path": values["csv_path"],
        "data_as_of": values["data_as_of"],
        "plan": values["plan"],
        "route_log": values["route_log"],
        "tool_calls": values["tool_calls"],
        "findings": {agent: finding["summary"] for agent, finding in values["findings"].items()},
        "report_title": (values["report"] or {}).get("title"),
        "errors": values["errors"],
        "conversation": [f"{message.type}: {message.content[:100]}" for message in values["messages"]],
    }
    print(json.dumps(view, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
