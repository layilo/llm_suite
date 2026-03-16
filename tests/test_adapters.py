import json
import subprocess

from llm_benchmark_suite.adapters.factory import create_adapter
from llm_benchmark_suite.schemas.models import BenchmarkRequest


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
