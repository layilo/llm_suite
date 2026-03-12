"""Regression comparison logic for current versus baseline runs."""

from __future__ import annotations

from typing import Optional

from llm_benchmark_suite.config import load_yaml_file
from llm_benchmark_suite.schemas.models import (
    AccuracyMetrics,
    BackendMetrics,
    BenchmarkSummary,
    CostMetrics,
    RegressionCheckResult,
)


def _delta_pct(current: float, baseline: float) -> float:
    if baseline == 0:
        return 0.0
    return ((current - baseline) / baseline) * 100.0


def _pair_key(item: object) -> tuple[str, str]:
    return (str(getattr(item, "backend_name")), str(getattr(item, "dataset_name")))


def _build_lookup(items: list[object]) -> dict[tuple[str, str], object]:
    return {_pair_key(item): item for item in items}


def _missing_pair_result(
    pair: tuple[str, str],
    side: str,
) -> RegressionCheckResult:
    backend_name, dataset_name = pair
    return RegressionCheckResult(
        check_name="missing_pair",
        backend_name=backend_name,
        dataset_name=dataset_name,
        passed=False,
        threshold=0.0,
        current_value=0.0,
        baseline_value=0.0,
        delta_pct=0.0,
        message=f"{backend_name}/{dataset_name} missing from {side} summary",
    )


def _paired_metrics(
    current: BenchmarkSummary,
    baseline: BenchmarkSummary,
) -> list[
    tuple[
        tuple[str, str],
        Optional[BackendMetrics],
        Optional[BackendMetrics],
        Optional[AccuracyMetrics],
        Optional[AccuracyMetrics],
        Optional[CostMetrics],
        Optional[CostMetrics],
    ]
]:
    current_bm = _build_lookup(current.backend_metrics)
    baseline_bm = _build_lookup(baseline.backend_metrics)
    current_acc = _build_lookup(current.accuracy_metrics)
    baseline_acc = _build_lookup(baseline.accuracy_metrics)
    current_cost = _build_lookup(current.cost_metrics)
    baseline_cost = _build_lookup(baseline.cost_metrics)
    pairs = sorted(set(current_bm) | set(baseline_bm) | set(current_acc) | set(baseline_acc) | set(current_cost) | set(baseline_cost))
    return [
        (
            pair,
            current_bm.get(pair),
            baseline_bm.get(pair),
            current_acc.get(pair),
            baseline_acc.get(pair),
            current_cost.get(pair),
            baseline_cost.get(pair),
        )
        for pair in pairs
    ]


def compare_summaries(
    current: BenchmarkSummary,
    baseline: BenchmarkSummary,
    thresholds_path: str,
) -> list[RegressionCheckResult]:
    thresholds = load_yaml_file(thresholds_path)
    checks: list[RegressionCheckResult] = []
    for pair, current_bm, baseline_bm, current_acc, baseline_acc, current_cost, baseline_cost in _paired_metrics(
        current, baseline
    ):
        backend_name, dataset_name = pair
        if current_bm is None or baseline_bm is None:
            checks.append(_missing_pair_result(pair, "current" if current_bm is None else "baseline"))
            continue
        if current_acc is None or baseline_acc is None:
            checks.append(_missing_pair_result(pair, "current" if current_acc is None else "baseline"))
            continue
        if current_cost is None or baseline_cost is None:
            checks.append(_missing_pair_result(pair, "current" if current_cost is None else "baseline"))
            continue

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
                    backend_name=backend_name,
                    dataset_name=dataset_name,
                    passed=passed,
                    threshold=threshold,
                    current_value=current_value,
                    baseline_value=baseline_value,
                    delta_pct=delta_pct,
                    message=(
                        f"{backend_name}/{dataset_name} {name} "
                        f"delta={delta_pct:.2f}% threshold={threshold:.2f}%"
                    ),
                )
            )

        checks.append(
            RegressionCheckResult(
                check_name="error_rate",
                backend_name=backend_name,
                dataset_name=dataset_name,
                passed=current_bm.error_rate <= thresholds["error_rate_max"],
                threshold=thresholds["error_rate_max"],
                current_value=current_bm.error_rate,
                baseline_value=baseline_bm.error_rate,
                delta_pct=_delta_pct(current_bm.error_rate, baseline_bm.error_rate),
                message=(
                    f"{backend_name}/{dataset_name} error_rate={current_bm.error_rate:.4f} "
                    f"max={thresholds['error_rate_max']:.4f}"
                ),
            )
        )
        checks.append(
            RegressionCheckResult(
                check_name="accuracy_min",
                backend_name=backend_name,
                dataset_name=dataset_name,
                passed=current_acc.aggregate_quality >= thresholds["accuracy_min"],
                threshold=thresholds["accuracy_min"],
                current_value=current_acc.aggregate_quality,
                baseline_value=baseline_acc.aggregate_quality,
                delta_pct=_delta_pct(current_acc.aggregate_quality, baseline_acc.aggregate_quality),
                message=(
                    f"{backend_name}/{dataset_name} aggregate_quality="
                    f"{current_acc.aggregate_quality:.4f} min={thresholds['accuracy_min']:.4f}"
                ),
            )
        )
    return checks
