import json
from pathlib import Path

from llm_benchmark_suite.reports.generator import render_markdown
from llm_benchmark_suite.schemas.models import BenchmarkSummary


def test_markdown_report_contains_table() -> None:
    summary = BenchmarkSummary.model_validate(
        json.loads(Path("artifacts/sample_run/summary.json").read_text(encoding="utf-8"))
    )
    output = render_markdown(summary)
    assert "| Backend | Dataset |" in output
