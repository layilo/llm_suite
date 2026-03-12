from datetime import datetime, timezone

from llm_benchmark_suite.cost.model import compute_cost_metrics
from llm_benchmark_suite.schemas.models import AccuracyMetrics, BackendMetrics


def test_cost_metrics_are_positive() -> None:
    backend = BackendMetrics(
        backend_name="vllm",
        model_name="demo",
        dataset_name="qa",
        run_timestamp=datetime.now(timezone.utc),
        request_count=2,
        prompt_tokens=10,
        completion_tokens=6,
        total_tokens=16,
        ttft_ms_avg=10,
        tpot_ms_avg=1,
        latency_ms_avg=20,
        latency_ms_p50=18,
        latency_ms_p95=25,
        latency_ms_p99=26,
        tokens_per_second=100,
        requests_per_second=10,
        success_rate=1,
        error_rate=0,
        gpu_memory_gb=8,
        peak_host_memory_gb=2,
        precision="fp16",
        backend_version="test",
        quantization="none",
    )
    accuracy = AccuracyMetrics(
        backend_name="vllm",
        dataset_name="qa",
        exact_match=1,
        token_f1=1,
        bleu=1,
        rouge_l=1,
        pass_rate=1,
        golden_pass_rate=1,
    )
    result = compute_cost_metrics(backend, accuracy, "configs/costs/default.yaml")
    assert result.cost_per_request_usd > 0
    assert result.cost_adjusted_quality_score > 0
