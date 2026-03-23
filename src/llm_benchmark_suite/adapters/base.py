"""Backend adapter abstraction and deterministic mock helpers."""

from __future__ import annotations

import abc
import hashlib
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from statistics import mean
from typing import Any

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
        warmup_requests = int(self.defaults.get("warmup_requests", 0))
        configured_concurrency = max(int(self.defaults.get("concurrency", 1)), 1)

        if warmup_requests > 0 and requests:
            warmup_slice = requests[: min(warmup_requests, len(requests))]
            self._run_requests(warmup_slice, concurrency=configured_concurrency, measure_offsets=False)

        measured_started_at = time.perf_counter()
        responses = self._run_requests(
            requests,
            concurrency=configured_concurrency,
            measure_offsets=True,
            run_start=measured_started_at,
        )
        benchmark_wall_time_s = max(time.perf_counter() - measured_started_at, 1e-6)
        latencies = [item.latency_ms for item in responses]
        ttft = [item.ttft_ms for item in responses]
        tpot = [item.tpot_ms for item in responses]
        total_tokens = sum(item.total_tokens for item in responses)
        effective_concurrency = min(configured_concurrency, max(len(requests), 1))
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
            benchmark_wall_time_s=benchmark_wall_time_s,
            tokens_per_second=total_tokens / benchmark_wall_time_s,
            requests_per_second=len(responses) / benchmark_wall_time_s,
            success_rate=sum(1 for item in responses if item.success) / max(len(responses), 1),
            error_rate=sum(1 for item in responses if not item.success) / max(len(responses), 1),
            gpu_memory_gb=float(self.config.get("mock_gpu_memory_gb", 8.0)),
            peak_host_memory_gb=float(self.config.get("mock_host_memory_gb", 2.5)),
            gpu_utilization_pct=float(self.config.get("mock_gpu_utilization_pct", 72.0)),
            warmup_time_s=float(self.config.get("mock_warmup_time_s", 1.5)),
            model_load_time_s=float(self.config.get("mock_model_load_time_s", 4.2)),
            concurrency=configured_concurrency,
            batch_size=int(self.defaults.get("batch_size", 1)),
            measured_request_count=len(responses),
            warmup_request_count=min(warmup_requests, len(requests)),
            hardware_metadata={"device": self.config.get("device", "mock-gpu"), "mode": self.mode},
            precision=str(self.defaults.get("precision", "fp16")),
            backend_version=str(self.config.get("backend_version", "mock-0.1")),
            quantization=str(self.config.get("quantization", "none")),
            diagnostics={
                "config": self.config,
                "configured_concurrency": configured_concurrency,
                "effective_concurrency": effective_concurrency,
                "execution_mode": "concurrent" if effective_concurrency > 1 else "sequential",
            },
        )

    def _run_requests(
        self,
        requests: list[BenchmarkRequest],
        *,
        concurrency: int,
        measure_offsets: bool,
        run_start: float | None = None,
    ) -> list[BenchmarkResponse]:
        if not requests:
            return []

        def invoke(request: BenchmarkRequest) -> BenchmarkResponse:
            started_at = time.perf_counter()
            try:
                response = self.infer(request)
            except Exception as exc:
                response = self._failure_response(request, str(exc))
            finished_at = time.perf_counter()
            if measure_offsets and run_start is not None:
                response.started_at_offset_ms = max((started_at - run_start) * 1000.0, 0.0)
                response.finished_at_offset_ms = max((finished_at - run_start) * 1000.0, 0.0)
            return response

        if concurrency <= 1 or len(requests) <= 1:
            return [invoke(request) for request in requests]

        ordered: list[BenchmarkResponse | None] = [None] * len(requests)
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [
                executor.submit(invoke, request)
                for request in requests
            ]
            for index, future in enumerate(futures):
                ordered[index] = future.result()
        return [response for response in ordered if response is not None]

    def collect_metrics(self) -> dict[str, object]:
        return {"backend_name": self.backend_name, "mode": self.mode}

    def _response_from_payload(
        self,
        request: BenchmarkRequest,
        payload: dict[str, Any],
        *,
        fallback_text: str = "",
    ) -> BenchmarkResponse:
        output_text = self._extract_output_text(payload, fallback_text=fallback_text)
        prompt_tokens = self._coerce_int(
            payload.get("prompt_tokens"),
            default=self._coerce_int(self._get_usage_value(payload, "prompt_tokens"), default=0),
        )
        completion_tokens = self._coerce_int(
            payload.get("completion_tokens"),
            default=self._coerce_int(
                self._get_usage_value(payload, "completion_tokens"),
                default=max(1, len(output_text.split())) if output_text else 0,
            ),
        )
        total_tokens = self._coerce_int(
            payload.get("total_tokens"),
            default=self._coerce_int(
                self._get_usage_value(payload, "total_tokens"),
                default=prompt_tokens + completion_tokens,
            ),
        )
        latency_ms = self._extract_latency_ms(payload)
        ttft_ms = self._extract_metric(payload, "ttft_ms")
        tpot_ms = self._extract_metric(payload, "tpot_ms")
        success = bool(payload.get("success", True))
        error_message = payload.get("error_message")
        return BenchmarkResponse(
            request_id=request.request_id,
            backend_name=self.backend_name,
            model_name=str(payload.get("model_name", self.defaults["model_name"])),
            output_text=output_text,
            success=success,
            error_message=str(error_message) if error_message is not None else None,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=max(total_tokens, prompt_tokens + completion_tokens),
            ttft_ms=ttft_ms,
            tpot_ms=tpot_ms,
            latency_ms=latency_ms,
        )

    def _parse_json_output(self, stdout: str) -> dict[str, Any]:
        payload = json.loads(stdout.strip() or "{}")
        if not isinstance(payload, dict):
            raise ValueError("adapter command output must be a JSON object")
        return payload

    def _extract_output_text(self, payload: dict[str, Any], *, fallback_text: str = "") -> str:
        output_text = payload.get("output_text")
        if isinstance(output_text, str):
            return output_text
        choices = payload.get("choices")
        if isinstance(choices, list) and choices:
            first_choice = choices[0]
            if isinstance(first_choice, dict):
                message = first_choice.get("message")
                if isinstance(message, dict) and isinstance(message.get("content"), str):
                    return str(message["content"])
                if isinstance(first_choice.get("text"), str):
                    return str(first_choice["text"])
        return fallback_text

    def _get_usage_value(self, payload: dict[str, Any], key: str) -> Any:
        usage = payload.get("usage", {})
        if isinstance(usage, dict):
            return usage.get(key)
        return None

    def _extract_latency_ms(self, payload: dict[str, Any]) -> float:
        latency_payload = payload.get("latency_ms")
        if isinstance(latency_payload, dict):
            for key in ("avg", "mean", "p50", "total"):
                if key in latency_payload:
                    return self._coerce_float(latency_payload[key], default=0.0)
            return 0.0
        return self._coerce_float(latency_payload, default=0.0)

    def _extract_metric(self, payload: dict[str, Any], key: str) -> float:
        value = payload.get(key)
        if value is not None:
            return self._coerce_float(value, default=0.0)
        metrics = payload.get("metrics", {})
        if isinstance(metrics, dict):
            return self._coerce_float(metrics.get(key), default=0.0)
        return 0.0

    def _coerce_int(self, value: object, *, default: int = 0) -> int:
        try:
            if value is None:
                return default
            return int(value)
        except (TypeError, ValueError):
            return default

    def _coerce_float(self, value: object, *, default: float = 0.0) -> float:
        try:
            if value is None:
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

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

    def _failure_response(self, request: BenchmarkRequest, error_message: str) -> BenchmarkResponse:
        return BenchmarkResponse(
            request_id=request.request_id,
            backend_name=self.backend_name,
            model_name=str(self.defaults["model_name"]),
            output_text="",
            success=False,
            error_message=error_message,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            ttft_ms=0.0,
            tpot_ms=0.0,
            latency_ms=0.0,
        )


def _percentile(values: list[float], percentile: int) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((percentile / 100) * (len(ordered) - 1)))
    return ordered[index]
