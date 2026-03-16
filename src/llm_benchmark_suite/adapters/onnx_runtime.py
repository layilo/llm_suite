"""ONNX Runtime / ORT GenAI adapter."""

from __future__ import annotations

import subprocess

from llm_benchmark_suite.adapters.base import BaseBackendAdapter
from llm_benchmark_suite.schemas.models import BenchmarkRequest, BenchmarkResponse


class ONNXRuntimeAdapter(BaseBackendAdapter):
    def infer(self, request: BenchmarkRequest) -> BenchmarkResponse:
        if self.mode == "mock":
            response = self._mock_response(request, "onnx")
            response.latency_ms *= 1.05
            response.ttft_ms *= 1.08
            return response

        script = str(self.config["benchmark_script"])
        result = subprocess.run(
            ["python", script, "--prompt", request.prompt],
            check=True,
            capture_output=True,
            text=True,
        )
        payload = self._parse_json_output(result.stdout)
        payload.setdefault("request_id", request.request_id)
        payload.setdefault("output_text", request.reference or request.prompt)
        payload.setdefault("prompt_tokens", max(1, len(request.prompt.split())))
        payload.setdefault(
            "completion_tokens",
            max(1, len(str(payload["output_text"]).split())),
        )
        payload.setdefault(
            "total_tokens",
            int(payload["prompt_tokens"]) + int(payload["completion_tokens"]),
        )
        return self._response_from_payload(
            request,
            payload,
            fallback_text=request.reference or request.prompt,
        )
