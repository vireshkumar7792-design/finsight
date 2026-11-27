"""CLI modules import cleanly and the graph CLI prints research sources (offline)."""

import importlib
from types import SimpleNamespace

import pytest

from finsight.cli_graph import print_summary


@pytest.mark.parametrize("module", ["finsight.cli", "finsight.cli_graph", "finsight.research_cli",
                                    "finsight.rag.ingest"])
def test_cli_modules_import(module):
    importlib.import_module(module)  # catches syntax errors that no other test would reach


def test_print_summary_lists_research_sources(capsys):
    report = {"title": "T", "summary": "S", "sections": [], "key_metrics": [], "caveats": [], "disclaimer": "D"}
    source = {"id": "NVDA-10K-FY2026-1A-000", "title": "NVIDIA 10-K", "section": "Item 1A Risk Factors",
              "url": "https://sec.example/nvda"}
    values = {"report": report, "route_log": [], "llm_calls": 3,
              "research_findings": {"data": {"research:findings": {"sources": [source]}}}}
    print_summary(SimpleNamespace(values=values, next=()))
    out = capsys.readouterr().out
    assert "\nResearch sources:\n" in out
    assert "  [NVDA-10K-FY2026-1A-000] NVIDIA 10-K | Item 1A Risk Factors\n      https://sec.example/nvda" in out
