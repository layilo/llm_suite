"""Regression comparison logic for current versus baseline runs."""

from __future__ import annotations

from llm_benchmark_suite.config import load_yaml_file
from llm_benchmark_suite.schemas.models import BenchmarkSummary, RegressionCheckResult


def _delta_pct(current: float, baseline: float) -> float:
    if baseline == 0:
        return 0.0
    return ((current - baseline) / baseline) * 100.0


def compare_summaries(
    current: BenchmarkSummary,
    baseline: BenchmarkSummary,
    thresholds_path: str,
) -> list[RegressionCheckResult]:
    thresholds = load_yaml_file(thresholds_path)
    current_bm = current.backend_metrics[0]
    baseline_bm = baseline.backend_metrics[0]
    current_acc = current.accuracy_metrics[0]
    baseline_cost = baseline.cost_metrics[0]
    current_cost = current.cost_metrics[0]

    checks: list[RegressionCheckResult] = []
    scenarios = [
        (
            "p95_latency",
            current_bm.latency_ms_p95,
            baseline_bm.latency_ms_p95,
            thresholds["p95_latency_regression_pct"],
            False,
        ),
        (
            "ttft",
            current_bm.ttft_ms_avg,
            baseline_bm.ttft_ms_avg,
            thresholds["ttft_regression_pct"],
            False,
        ),
        (
            "throughput",
            current_bm.tokens_per_second,
            baseline_bm.tokens_per_second,
            thresholds["throughput_regression_pct"],
            True,
        ),
        (
            "cost_per_million_tokens",
            current_cost.cost_per_million_tokens_usd,
            baseline_cost.cost_per_million_tokens_usd,
            thresholds["cost_per_million_tokens_regression_pct"],
            False,
        ),
    ]

    for name, current_value, baseline_value, threshold, improve_when_higher in scenarios:
        delta_pct = _delta_pct(current_value, baseline_value)
        passed = delta_pct >= -threshold if improve_when_higher else delta_pct <= threshold
        checks.append(
            RegressionCheckResult(
                check_name=name,
                passed=passed,
                threshold=threshold,
                current_value=current_value,
                baseline_value=baseline_value,
                delta_pct=delta_pct,
                message=f"{name} delta={delta_pct:.2f}% threshold={threshold:.2f}%",
            )
        )

    checks.append(
        RegressionCheckResult(
            check_name="error_rate",
            passed=current_bm.error_rate <= thresholds["error_rate_max"],
            threshold=thresholds["error_rate_max"],
            current_value=current_bm.error_rate,
            baseline_value=baseline_bm.error_rate,
            delta_pct=_delta_pct(current_bm.error_rate, baseline_bm.error_rate),
            message=(
                f"error_rate={current_bm.error_rate:.4f} max={thresholds['error_rate_max']:.4f}"
            ),
        )
    )
    checks.append(
        RegressionCheckResult(
            check_name="accuracy_min",
            passed=current_acc.aggregate_quality >= thresholds["accuracy_min"],
            threshold=thresholds["accuracy_min"],
            current_value=current_acc.aggregate_quality,
            baseline_value=current_acc.aggregate_quality,
            delta_pct=0.0,
            message=(
                f"aggregate_quality={current_acc.aggregate_quality:.4f} "
                f"min={thresholds['accuracy_min']:.4f}"
            ),
        )
    )
    return checks
