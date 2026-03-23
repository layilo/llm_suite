import json
from pathlib import Path

from llm_benchmark_suite.reports.generator import render_markdown, summary_to_rows
from llm_benchmark_suite.schemas.models import BenchmarkSummary


def test_markdown_report_contains_table() -> None:
    summary = BenchmarkSummary.model_validate(
        json.loads(Path("artifacts/sample_run/summary.json").read_text(encoding="utf-8"))
    )
    output = render_markdown(summary)
    assert "| Backend | Dataset |" in output


def test_report_rows_tolerate_missing_quality_and_cost() -> None:
    summary = BenchmarkSummary.model_validate(
        json.loads(Path("artifacts/sample_run/summary.json").read_text(encoding="utf-8"))
    )
    summary.accuracy_metrics.clear()
    summary.cost_metrics.clear()

    rows = summary_to_rows(summary)

    assert rows[0]["status"] == "partial"
    assert rows[0]["quality"] is None
    assert rows[0]["cost_per_million_tokens_usd"] is None
    assert "benchmark_wall_time_s" in rows[0]
    assert "concurrency" in rows[0]

    output = render_markdown(summary)
    assert "n/a" in output
    assert "Wall Time (s)" in output
