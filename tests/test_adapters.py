import json
import subprocess
import time

from llm_benchmark_suite.adapters.factory import create_adapter
from llm_benchmark_suite.adapters.base import BaseBackendAdapter
from llm_benchmark_suite.schemas.models import BenchmarkRequest, BenchmarkResponse


def test_adapter_contract_mock_mode() -> None:
    adapter = create_adapter(
        "vllm",
        {"mode": "mock"},
        {"model_name": "demo", "precision": "fp16", "concurrency": 1, "batch_size": 1},
    )
    response = adapter.infer(
        BenchmarkRequest(request_id="1", prompt="What is Mars?", reference="Mars", task_type="qa")
    )
    assert response.success is True
    assert response.total_tokens > 0


def test_vllm_real_mode_normalizes_openai_style_payload(monkeypatch) -> None:
    adapter = create_adapter(
        "vllm",
        {"mode": "real", "endpoint": "http://localhost:8000/v1/chat/completions"},
        {"model_name": "demo", "precision": "fp16", "concurrency": 1, "batch_size": 1},
    )

    class DummyResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {
                "choices": [{"message": {"content": "Mars"}}],
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 2,
                    "total_tokens": 7,
                },
                "ttft_ms": 12.5,
                "tpot_ms": 3.1,
                "latency_ms": 20.0,
            }

    monkeypatch.setattr(
        "llm_benchmark_suite.adapters.vllm.httpx.post",
        lambda *args, **kwargs: DummyResponse(),
    )

    response = adapter.infer(
        BenchmarkRequest(request_id="1", prompt="What is Mars?", reference="Mars", task_type="qa")
    )

    assert response.output_text == "Mars"
    assert response.prompt_tokens == 5
    assert response.completion_tokens == 2
    assert response.total_tokens == 7
    assert response.latency_ms == 20.0


def test_tensorrt_real_mode_normalizes_command_payload(monkeypatch) -> None:
    adapter = create_adapter(
        "tensorrt_llm",
        {"mode": "real", "command": "python scripts/mock_tensorrt_runner.py"},
        {"model_name": "demo", "precision": "fp16", "concurrency": 1, "batch_size": 1},
    )

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=0,
            stdout=json.dumps(
                {
                    "output_text": "Mars",
                    "latency_ms": {"p50": 18.0, "p95": 22.0},
                    "ttft_ms": 8.0,
                    "prompt_tokens": 4,
                    "completion_tokens": 2,
                    "total_tokens": 6,
                }
            ),
            stderr="",
        )

    monkeypatch.setattr("llm_benchmark_suite.adapters.tensorrt_llm.subprocess.run", fake_run)

    response = adapter.infer(
        BenchmarkRequest(request_id="2", prompt="What is Mars?", reference="Mars", task_type="qa")
    )

    assert response.output_text == "Mars"
    assert response.latency_ms == 18.0
    assert response.ttft_ms == 8.0
    assert response.total_tokens == 6


def test_onnx_real_mode_normalizes_script_payload(monkeypatch) -> None:
    adapter = create_adapter(
        "onnx_runtime",
        {"mode": "real", "benchmark_script": "scripts/run_onnx_benchmark.py"},
        {"model_name": "demo", "precision": "fp16", "concurrency": 1, "batch_size": 1},
    )

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=0,
            stdout=json.dumps(
                {
                    "output_text": "Mars",
                    "metrics": {"ttft_ms": 7.5, "tpot_ms": 1.9},
                    "latency_ms": 16.0,
                }
            ),
            stderr="",
        )

    monkeypatch.setattr("llm_benchmark_suite.adapters.onnx_runtime.subprocess.run", fake_run)

    response = adapter.infer(
        BenchmarkRequest(request_id="3", prompt="What is Mars?", reference="Mars", task_type="qa")
    )

    assert response.output_text == "Mars"
    assert response.ttft_ms == 7.5
    assert response.tpot_ms == 1.9
    assert response.latency_ms == 16.0
    assert response.prompt_tokens > 0
    assert response.total_tokens >= response.prompt_tokens


class SleepyAdapter(BaseBackendAdapter):
    def infer(self, request: BenchmarkRequest) -> BenchmarkResponse:
        time.sleep(0.05)
        return BenchmarkResponse(
            request_id=request.request_id,
            backend_name=self.backend_name,
            model_name=str(self.defaults["model_name"]),
            output_text=request.reference or request.prompt,
            success=True,
            prompt_tokens=5,
            completion_tokens=5,
            total_tokens=10,
            ttft_ms=10.0,
            tpot_ms=2.0,
            latency_ms=50.0,
        )


class FlakyAdapter(BaseBackendAdapter):
    def infer(self, request: BenchmarkRequest) -> BenchmarkResponse:
        if request.request_id == "1":
            raise RuntimeError("boom")
        return BenchmarkResponse(
            request_id=request.request_id,
            backend_name=self.backend_name,
            model_name=str(self.defaults["model_name"]),
            output_text=request.reference or request.prompt,
            success=True,
            prompt_tokens=2,
            completion_tokens=3,
            total_tokens=5,
            ttft_ms=4.0,
            tpot_ms=1.0,
            latency_ms=20.0,
        )


def test_benchmark_uses_concurrency_and_wall_clock_metrics() -> None:
    adapter = SleepyAdapter(
        "sleepy",
        {"mode": "mock"},
        {
            "model_name": "demo",
            "precision": "fp16",
            "concurrency": 4,
            "batch_size": 1,
            "warmup_requests": 0,
        },
    )
    requests = [
        BenchmarkRequest(
            request_id=str(index),
            prompt=f"Prompt {index}",
            reference=f"Answer {index}",
            task_type="qa",
        )
        for index in range(4)
    ]

    responses, metrics = adapter.benchmark("qa", requests)

    assert [response.request_id for response in responses] == ["0", "1", "2", "3"]
    assert metrics.concurrency == 4
    assert metrics.diagnostics["execution_mode"] == "concurrent"
    assert metrics.benchmark_wall_time_s < 0.15
    assert metrics.tokens_per_second > 200
    assert all(response.finished_at_offset_ms >= response.started_at_offset_ms for response in responses)
    assert max(response.started_at_offset_ms for response in responses) < 50.0


def test_benchmark_records_failures_without_aborting() -> None:
    adapter = FlakyAdapter(
        "flaky",
        {"mode": "mock"},
        {
            "model_name": "demo",
            "precision": "fp16",
            "concurrency": 2,
            "batch_size": 1,
            "warmup_requests": 0,
        },
    )
    requests = [
        BenchmarkRequest(request_id=str(index), prompt="Prompt", reference="Answer", task_type="qa")
        for index in range(3)
    ]

    responses, metrics = adapter.benchmark("qa", requests)

    assert len(responses) == 3
    assert responses[1].success is False
    assert responses[1].error_message == "boom"
    assert metrics.success_rate == 2 / 3
    assert metrics.error_rate == 1 / 3


def test_warmup_requests_are_excluded_from_measured_metrics() -> None:
    adapter = SleepyAdapter(
        "sleepy",
        {"mode": "mock"},
        {
            "model_name": "demo",
            "precision": "fp16",
            "concurrency": 2,
            "batch_size": 1,
            "warmup_requests": 2,
        },
    )
    requests = [
        BenchmarkRequest(
            request_id=str(index),
            prompt=f"Prompt {index}",
            reference=f"Answer {index}",
            task_type="qa",
        )
        for index in range(3)
    ]

    responses, metrics = adapter.benchmark("qa", requests)

    assert len(responses) == 3
    assert metrics.measured_request_count == 3
    assert metrics.warmup_request_count == 2
