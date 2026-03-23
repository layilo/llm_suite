"""Core Pydantic schemas used across the benchmarking suite."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class BenchmarkRequest(BaseModel):
    """Single dataset prompt and metadata passed to a backend."""

    request_id: str
    prompt: str
    reference: Optional[str] = None
    task_type: str
    expected_contains: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


class BenchmarkResponse(BaseModel):
    """Normalized backend response for a single request."""

    request_id: str
    backend_name: str
    model_name: str
    output_text: str
    success: bool
    error_message: Optional[str] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    ttft_ms: float = 0.0
    tpot_ms: float = 0.0
    latency_ms: float = 0.0
    started_at_offset_ms: float = 0.0
    finished_at_offset_ms: float = 0.0


class BackendMetrics(BaseModel):
    """Aggregate serving metrics normalized across backends."""

    backend_name: str
    model_name: str
    dataset_name: str
    run_timestamp: datetime
    request_count: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    ttft_ms_avg: float
    tpot_ms_avg: float
    latency_ms_avg: float
    latency_ms_p50: float
    latency_ms_p95: float
    latency_ms_p99: float
    benchmark_wall_time_s: float = 0.0
    tokens_per_second: float
    requests_per_second: float
    success_rate: float
    error_rate: float
    gpu_memory_gb: float
    peak_host_memory_gb: float
    gpu_utilization_pct: Optional[float] = None
    warmup_time_s: float = 0.0
    model_load_time_s: float = 0.0
    concurrency: int = 1
    batch_size: int = 1
    measured_request_count: int = 0
    warmup_request_count: int = 0
    hardware_metadata: dict[str, Any] = Field(default_factory=dict)
    precision: str = "unknown"
    backend_version: str = "unknown"
    quantization: str = "none"
    diagnostics: dict[str, Any] = Field(default_factory=dict)


class AccuracyMetrics(BaseModel):
    """Evaluation outputs from text quality metrics."""

    backend_name: str
    dataset_name: str
    exact_match: float
    token_f1: float
    bleu: float
    rouge_l: float
    semantic_similarity: Optional[float] = None
    rubric_score: Optional[float] = None
    pass_rate: float = 0.0
    golden_pass_rate: float = 0.0

    @property
    def aggregate_quality(self) -> float:
        values = [self.exact_match, self.token_f1, self.bleu, self.rouge_l, self.pass_rate]
        if self.semantic_similarity is not None:
            values.append(self.semantic_similarity)
        if self.rubric_score is not None:
            values.append(self.rubric_score)
        return sum(values) / len(values)


class CostMetrics(BaseModel):
    """Estimated serving cost outputs."""

    backend_name: str
    dataset_name: str
    cost_per_1k_prompts_usd: float
    cost_per_million_tokens_usd: float
    cost_per_request_usd: float
    cost_per_successful_response_usd: float
    cost_per_throughput_unit_usd: float
    cost_adjusted_quality_score: float
    estimated_total_run_cost_usd: float


class RegressionCheckResult(BaseModel):
    """Single regression rule evaluation result."""

    check_name: str
    backend_name: Optional[str] = None
    dataset_name: Optional[str] = None
    passed: bool
    threshold: float
    current_value: float
    baseline_value: float
    delta_pct: float
    message: str


class BenchmarkSummary(BaseModel):
    """Top-level benchmark artifact written to disk and consumed by reports."""

    run_id: str
    profile_name: str
    timestamp: datetime
    mode: str
    git_commit: Optional[str] = None
    environment: dict[str, Any] = Field(default_factory=dict)
    backends: list[str] = Field(default_factory=list)
    datasets: list[str] = Field(default_factory=list)
    backend_metrics: list[BackendMetrics] = Field(default_factory=list)
    accuracy_metrics: list[AccuracyMetrics] = Field(default_factory=list)
    cost_metrics: list[CostMetrics] = Field(default_factory=list)
    regression_results: list[RegressionCheckResult] = Field(default_factory=list)
    rankings: list[dict[str, Any]] = Field(default_factory=list)
    raw_responses: list[BenchmarkResponse] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
