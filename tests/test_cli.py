"""CLI: argument checks and report printing (no network, no LLM)."""

from finsight.cli import main, render_report


def test_ask_needs_a_csv_and_question(capsys):
    assert main(["ask"]) == 1
    assert "give a CSV and a question" in capsys.readouterr().err


def test_resume_needs_a_thread(capsys):
    assert main(["ask", "--resume"]) == 1
    assert "need --thread" in capsys.readouterr().err


def test_render_report_shows_metrics_sources_and_disclaimer():
    report = {
        "title": "Risk",
        "summary": "Volatility is 44.43%.",
        "sections": [{"heading": "Volatility", "body": "High."}],
        "key_metrics": [{"name": "Volatility", "value": 44.43, "unit": "%", "source": "get_risk_metrics"}],
        "caveats": ["Prices only."],
    }
    text = render_report(report)
    assert "Volatility: 44.43 %   [source: get_risk_metrics]" in text
    assert "## Volatility" in text
    assert "not financial advice" in text
