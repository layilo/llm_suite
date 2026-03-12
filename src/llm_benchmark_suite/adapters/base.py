"""Backend adapter abstraction and deterministic mock helpers."""

from __future__ import annotations

import abc
import hashlib
import random
from datetime import datetime, timezone
from statistics import mean

from llm_benchmark_suite.schemas.models import BackendMetrics, BenchmarkRequest, BenchmarkResponse


class BaseBackendAdapter(abc.ABC):
    """Abstract interface that all backend integrations implement."""

    def __init__(
        self, backend_name: str, config: dict[str, object], defaults: dict[str, object]
    ) -> None:
        self.backend_name = backend_name
        self.config = config
        self.defaults = defaults

    @property
    def mode(self) -> str:
        return str(self.config.get("mode", "mock"))

    def start_server(self) -> None:
        return None

    def stop_server(self) -> None:
        return None

    def health_check(self) -> bool:
        return True

    @abc.abstractmethod
    def infer(self, request: BenchmarkRequest) -> BenchmarkResponse:
        raise NotImplementedError

    def benchmark(
        self, dataset_name: str, requests: list[BenchmarkRequest]
    ) -> tuple[list[BenchmarkResponse], BackendMetrics]:
        responses = [self.infer(request) for request in requests]
        latencies = [item.latency_ms for item in responses]
        ttft = [item.ttft_ms for item in responses]
        tpot = [item.tpot_ms for item in responses]
        total_tokens = sum(item.total_tokens for item in responses)
        total_latency_s = max(sum(latencies) / 1000.0, 1e-6)
        return responses, BackendMetrics(
            backend_name=self.backend_name,
            model_name=str(self.defaults["model_name"]),
            dataset_name=dataset_name,
            run_timestamp=datetime.now(timezone.utc),
            request_count=len(responses),
            prompt_tokens=sum(item.prompt_tokens for item in responses),
            completion_tokens=sum(item.completion_tokens for item in responses),
            total_tokens=total_tokens,
            ttft_ms_avg=mean(ttft) if ttft else 0.0,
            tpot_ms_avg=mean(tpot) if tpot else 0.0,
            latency_ms_avg=mean(latencies) if latencies else 0.0,
            latency_ms_p50=_percentile(latencies, 50),
            latency_ms_p95=_percentile(latencies, 95),
            latency_ms_p99=_percentile(latencies, 99),
            tokens_per_second=total_tokens / total_latency_s,
            requests_per_second=len(responses) / total_latency_s,
            success_rate=sum(1 for item in responses if item.success) / max(len(responses), 1),
            error_rate=sum(1 for item in responses if not item.success) / max(len(responses), 1),
            gpu_memory_gb=float(self.config.get("mock_gpu_memory_gb", 8.0)),
            peak_host_memory_gb=float(self.config.get("mock_host_memory_gb", 2.5)),
            gpu_utilization_pct=float(self.config.get("mock_gpu_utilization_pct", 72.0)),
            warmup_time_s=float(self.config.get("mock_warmup_time_s", 1.5)),
            model_load_time_s=float(self.config.get("mock_model_load_time_s", 4.2)),
            concurrency=int(self.defaults.get("concurrency", 1)),
            batch_size=int(self.defaults.get("batch_size", 1)),
            hardware_metadata={"device": self.config.get("device", "mock-gpu"), "mode": self.mode},
            precision=str(self.defaults.get("precision", "fp16")),
            backend_version=str(self.config.get("backend_version", "mock-0.1")),
            quantization=str(self.config.get("quantization", "none")),
            diagnostics={"config": self.config},
        )

    def collect_metrics(self) -> dict[str, object]:
        return {"backend_name": self.backend_name, "mode": self.mode}

    def _mock_response(self, request: BenchmarkRequest, flavor: str) -> BenchmarkResponse:
        seed_key = f"{self.backend_name}:{request.request_id}:{flavor}"
        seed = int(hashlib.sha256(seed_key.encode("utf-8")).hexdigest(), 16) % (2**32)
        rng = random.Random(seed)
        prompt_tokens = max(4, len(request.prompt.split()))
        completion_tokens = max(
            1, len((request.reference or request.prompt).split()) + rng.randint(0, 3)
        )
        latency_base = float(self.config.get("latency_base_ms", 75.0))
        latency_ms = latency_base + rng.uniform(0, 25)
        ttft_ms = latency_ms * 0.4
        tpot_ms = max((latency_ms - ttft_ms) / max(completion_tokens, 1), 1.0)
        output_text = self._generate_mock_text(request, rng)
        return BenchmarkResponse(
            request_id=request.request_id,
            backend_name=self.backend_name,
            model_name=str(self.defaults["model_name"]),
            output_text=output_text,
            success=True,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            ttft_ms=ttft_ms,
            tpot_ms=tpot_ms,
            latency_ms=latency_ms,
        )

    def _generate_mock_text(self, request: BenchmarkRequest, rng: random.Random) -> str:
        if request.reference:
            variants = [
                request.reference,
                f"{request.reference}.",
                f"{request.reference} ({self.backend_name})",
            ]
            return variants[rng.randint(0, len(variants) - 1)]
        return request.prompt[:80]


def _percentile(values: list[float], percentile: int) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((percentile / 100) * (len(ordered) - 1)))
    return ordered[index]
