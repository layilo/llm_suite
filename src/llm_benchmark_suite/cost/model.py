"""Cost estimation model for benchmark runs."""

from __future__ import annotations

from llm_benchmark_suite.config import load_yaml_file
from llm_benchmark_suite.schemas.models import AccuracyMetrics, BackendMetrics, CostMetrics


def compute_cost_metrics(
    backend_metrics: BackendMetrics,
    accuracy_metrics: AccuracyMetrics,
    cost_profile_path: str,
) -> CostMetrics:
    profile = load_yaml_file(cost_profile_path)
    total_run_hours = max(
        (backend_metrics.latency_ms_avg * backend_metrics.request_count) / 3_600_000,
        1e-6,
    )
    infra_cost = (
        (
            profile["gpu_hourly_cost_usd"]
            + profile["cpu_hourly_cost_usd"]
            + (backend_metrics.peak_host_memory_gb * profile["memory_gb_hourly_cost_usd"])
        )
        * total_run_hours
        * profile.get("amortization_factor", 1.0)
    )
    cost_per_request = infra_cost / max(backend_metrics.request_count, 1)
    cost_per_successful_response = infra_cost / max(
        int(backend_metrics.request_count * backend_metrics.success_rate), 1
    )
    cost_per_million_tokens = infra_cost / max(backend_metrics.total_tokens, 1) * 1_000_000
    cost_per_1k_prompts = cost_per_request * 1_000
    cost_per_throughput_unit = infra_cost / max(backend_metrics.tokens_per_second, 1.0)
    cost_adjusted_quality_score = accuracy_metrics.aggregate_quality / max(cost_per_request, 1e-9)
    return CostMetrics(
        backend_name=backend_metrics.backend_name,
        dataset_name=backend_metrics.dataset_name,
        cost_per_1k_prompts_usd=cost_per_1k_prompts,
        cost_per_million_tokens_usd=cost_per_million_tokens,
        cost_per_request_usd=cost_per_request,
        cost_per_successful_response_usd=cost_per_successful_response,
        cost_per_throughput_unit_usd=cost_per_throughput_unit,
        cost_adjusted_quality_score=cost_adjusted_quality_score,
        estimated_total_run_cost_usd=infra_cost,
    )
