"""Adapter registry and factory."""

from __future__ import annotations

from llm_benchmark_suite.adapters.base import BaseBackendAdapter
from llm_benchmark_suite.adapters.onnx_runtime import ONNXRuntimeAdapter
from llm_benchmark_suite.adapters.tensorrt_llm import TensorRTLLMAdapter
from llm_benchmark_suite.adapters.vllm import VLLMAdapter

ADAPTERS: dict[str, type[BaseBackendAdapter]] = {
    "vllm": VLLMAdapter,
    "tensorrt_llm": TensorRTLLMAdapter,
    "onnx_runtime": ONNXRuntimeAdapter,
}


def create_adapter(
    backend_name: str,
    backend_config: dict[str, object],
    defaults: dict[str, object],
) -> BaseBackendAdapter:
    adapter_cls = ADAPTERS[backend_name]
    return adapter_cls(backend_name=backend_name, config=backend_config, defaults=defaults)
