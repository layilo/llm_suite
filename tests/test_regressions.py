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


def test_regression_compare_covers_each_backend_dataset_pair() -> None:
    sample = BenchmarkSummary.model_validate(
        json.loads(Path("artifacts/sample_run/summary.json").read_text(encoding="utf-8"))
    )
    current = sample.model_copy(deep=True)
    baseline = sample.model_copy(deep=True)

    current.backend_metrics.append(current.backend_metrics[0].model_copy(deep=True))
    current.backend_metrics[1].backend_name = "onnx_runtime"
    current.backend_metrics[1].latency_ms_p95 = 120.0
    current.backend_metrics[1].ttft_ms_avg = 50.0
    current.backend_metrics[1].tokens_per_second = 250.0

    current.accuracy_metrics.append(current.accuracy_metrics[0].model_copy(deep=True))
    current.accuracy_metrics[1].backend_name = "onnx_runtime"
    current.accuracy_metrics[1].exact_match = 0.4
    current.accuracy_metrics[1].token_f1 = 0.75
    current.accuracy_metrics[1].bleu = 0.7
    current.accuracy_metrics[1].rouge_l = 0.74
    current.accuracy_metrics[1].pass_rate = 0.8

    current.cost_metrics.append(current.cost_metrics[0].model_copy(deep=True))
    current.cost_metrics[1].backend_name = "onnx_runtime"
    current.cost_metrics[1].cost_per_million_tokens_usd = 5.5

    baseline.backend_metrics.append(current.backend_metrics[1].model_copy(deep=True))
    baseline.backend_metrics[1].latency_ms_p95 = 100.0
    baseline.backend_metrics[1].ttft_ms_avg = 42.0
    baseline.backend_metrics[1].tokens_per_second = 300.0

    baseline.accuracy_metrics.append(current.accuracy_metrics[1].model_copy(deep=True))
    baseline.cost_metrics.append(current.cost_metrics[1].model_copy(deep=True))
    baseline.cost_metrics[1].cost_per_million_tokens_usd = 4.0

    results = compare_summaries(current, baseline, "configs/thresholds/default.yaml")

    onnx_results = [
        item
        for item in results
        if item.backend_name == "onnx_runtime" and item.dataset_name == "summarization"
    ]
    assert len(onnx_results) == 6
    assert any(item.check_name == "p95_latency" and not item.passed for item in onnx_results)
    assert any(item.check_name == "throughput" and not item.passed for item in onnx_results)
    assert any(
        item.check_name == "cost_per_million_tokens" and not item.passed
        for item in onnx_results
    )


def test_regression_compare_flags_missing_pair() -> None:
    sample = BenchmarkSummary.model_validate(
        json.loads(Path("artifacts/sample_run/summary.json").read_text(encoding="utf-8"))
    )
    current = sample.model_copy(deep=True)
    current.backend_metrics.append(current.backend_metrics[0].model_copy(deep=True))
    current.backend_metrics[1].backend_name = "onnx_runtime"
    current.accuracy_metrics.append(current.accuracy_metrics[0].model_copy(deep=True))
    current.accuracy_metrics[1].backend_name = "onnx_runtime"
    current.cost_metrics.append(current.cost_metrics[0].model_copy(deep=True))
    current.cost_metrics[1].backend_name = "onnx_runtime"

    results = compare_summaries(current, sample, "configs/thresholds/default.yaml")

    assert any(
        item.check_name == "missing_pair"
        and item.backend_name == "onnx_runtime"
        and item.dataset_name == "summarization"
        and not item.passed
        for item in results
    )
