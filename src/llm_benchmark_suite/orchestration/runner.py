"""End-to-end benchmark orchestration."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from uuid import uuid4

from llm_benchmark_suite.adapters.factory import create_adapter
from llm_benchmark_suite.config import RunConfig
from llm_benchmark_suite.cost.model import compute_cost_metrics
from llm_benchmark_suite.evaluators.quality import evaluate_responses
from llm_benchmark_suite.regressions.checks import compare_summaries
from llm_benchmark_suite.reports.generator import write_reports
from llm_benchmark_suite.schemas.models import BenchmarkSummary
from llm_benchmark_suite.utils.io import ensure_dir, load_jsonl_dataset, write_json
from llm_benchmark_suite.utils.system import collect_environment_info, get_git_commit

LOGGER = logging.getLogger(__name__)


def run_benchmark(
    config: RunConfig, baseline_summary: Optional[BenchmarkSummary] = None
) -> BenchmarkSummary:
    output_dir = ensure_dir(config.output_dir)
    summary = BenchmarkSummary(
        run_id=f"{config.profile_name}-{uuid4().hex[:8]}",
        profile_name=config.profile_name,
        timestamp=datetime.now(timezone.utc),
        mode="mock" if config.mock_mode else "real",
        git_commit=get_git_commit(),
        environment=collect_environment_info(),
        backends=config.selected_backends,
        datasets=[dataset.name for dataset in config.datasets],
    )

    defaults = config.backend_defaults.model_dump()
    for dataset_cfg in config.datasets:
        requests = load_jsonl_dataset(dataset_cfg.path)
        for backend_name in config.selected_backends:
            backend_config = dict(config.backend_overrides.get(backend_name, {}))
            if config.mock_mode:
                backend_config["mode"] = "mock"
            adapter = create_adapter(backend_name, backend_config, defaults)
            try:
                adapter.start_server()
                if not adapter.health_check():
                    raise RuntimeError(f"health check failed for {backend_name}")
                responses, backend_metrics = adapter.benchmark(dataset_cfg.name, requests)
                accuracy_metrics = evaluate_responses(
                    backend_name, dataset_cfg.name, requests, responses
                )
                cost_metrics = compute_cost_metrics(
                    backend_metrics, accuracy_metrics, config.cost_profile
                )
                summary.backend_metrics.append(backend_metrics)
                summary.accuracy_metrics.append(accuracy_metrics)
                summary.cost_metrics.append(cost_metrics)
                summary.raw_responses.extend(responses)
            except Exception as exc:
                LOGGER.exception("Backend failed", extra={"context": {"backend": backend_name}})
                if not config.continue_on_error:
                    raise
                summary.metadata.setdefault("errors", []).append(
                    {"backend": backend_name, "dataset": dataset_cfg.name, "error": str(exc)}
                )
            finally:
                adapter.stop_server()

    summary.rankings = build_rankings(summary, config.composite_weights)
    if (
        baseline_summary is not None
        and summary.backend_metrics
        and baseline_summary.backend_metrics
    ):
        summary.regression_results = compare_summaries(
            summary,
            baseline_summary,
            config.thresholds_profile,
        )
    write_reports(summary, str(output_dir), config.report_formats)
    write_json(
        Path(output_dir) / "raw_responses.json",
        [item.model_dump(mode="json") for item in summary.raw_responses],
    )
    return summary


def build_rankings(summary: BenchmarkSummary, weights: dict[str, float]) -> list[dict[str, object]]:
    rankings: list[dict[str, object]] = []
    quality_lookup = {
        (item.backend_name, item.dataset_name): item for item in summary.accuracy_metrics
    }
    cost_lookup = {(item.backend_name, item.dataset_name): item for item in summary.cost_metrics}
    max_latency = max((item.latency_ms_p95 for item in summary.backend_metrics), default=1.0)
    max_cost = max((item.cost_per_million_tokens_usd for item in summary.cost_metrics), default=1.0)
    max_throughput = max((item.tokens_per_second for item in summary.backend_metrics), default=1.0)

    for metrics in summary.backend_metrics:
        quality = quality_lookup[(metrics.backend_name, metrics.dataset_name)].aggregate_quality
        cost = cost_lookup[(metrics.backend_name, metrics.dataset_name)].cost_per_million_tokens_usd
        latency_score = max(0.0, 1 - (metrics.latency_ms_p95 / max_latency))
        throughput_score = metrics.tokens_per_second / max_throughput
        cost_score = max(0.0, 1 - (cost / max_cost))
        reliability_score = metrics.success_rate
        composite = (
            weights.get("quality", 0.35) * quality
            + weights.get("latency", 0.20) * latency_score
            + weights.get("throughput", 0.20) * throughput_score
            + weights.get("cost", 0.15) * cost_score
            + weights.get("reliability", 0.10) * reliability_score
        )
        rankings.append(
            {
                "backend": metrics.backend_name,
                "dataset": metrics.dataset_name,
                "composite_score": round(composite, 4),
                "quality": round(quality, 4),
                "latency_score": round(latency_score, 4),
                "throughput_score": round(throughput_score, 4),
                "cost_score": round(cost_score, 4),
                "reliability_score": round(reliability_score, 4),
            }
        )
    return sorted(rankings, key=lambda item: item["composite_score"], reverse=True)
