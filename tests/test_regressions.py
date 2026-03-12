import json
from pathlib import Path

from llm_benchmark_suite.regressions.checks import compare_summaries
from llm_benchmark_suite.schemas.models import BenchmarkSummary


def test_regression_compare_detects_failure() -> None:
    sample = BenchmarkSummary.model_validate(
        json.loads(Path("artifacts/sample_run/summary.json").read_text(encoding="utf-8"))
    )
    current = sample.model_copy(deep=True)
    current.backend_metrics[0].latency_ms_p95 = sample.backend_metrics[0].latency_ms_p95 * 1.5
    results = compare_summaries(current, sample, "configs/thresholds/default.yaml")
    assert any(item.check_name == "p95_latency" and not item.passed for item in results)
