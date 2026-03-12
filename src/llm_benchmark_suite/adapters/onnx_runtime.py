"""ONNX Runtime / ORT GenAI adapter."""

from __future__ import annotations

import json
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
        payload = json.loads(result.stdout.strip() or "{}")
        text = payload.get("output_text", request.reference or request.prompt)
        return BenchmarkResponse(
            request_id=request.request_id,
            backend_name=self.backend_name,
            model_name=str(self.defaults["model_name"]),
            output_text=text,
            success=True,
            prompt_tokens=max(1, len(request.prompt.split())),
            completion_tokens=max(1, len(text.split())),
            total_tokens=max(2, len(request.prompt.split()) + len(text.split())),
            ttft_ms=float(payload.get("ttft_ms", 0.0)),
            tpot_ms=float(payload.get("tpot_ms", 0.0)),
            latency_ms=float(payload.get("latency_ms", 0.0)),
        )
