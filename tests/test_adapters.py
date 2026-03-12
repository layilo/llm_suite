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
