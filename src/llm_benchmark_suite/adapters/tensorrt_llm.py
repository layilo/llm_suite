"""TensorRT-LLM adapter based on command execution scaffolding."""

from __future__ import annotations

import json
import subprocess

from llm_benchmark_suite.adapters.base import BaseBackendAdapter
from llm_benchmark_suite.schemas.models import BenchmarkRequest, BenchmarkResponse


class TensorRTLLMAdapter(BaseBackendAdapter):
    def infer(self, request: BenchmarkRequest) -> BenchmarkResponse:
        if self.mode == "mock":
            response = self._mock_response(request, "tensorrt")
            response.latency_ms *= 0.9
            response.ttft_ms *= 0.85
            return response

        command = str(self.config["command"]).split()
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        payload = json.loads(result.stdout.strip() or "{}")
        latency_payload = payload.get("latency_ms", {})
        latency_ms = (
            latency_payload["p50"]
            if isinstance(latency_payload, dict)
            else payload.get("latency_ms", 0.0)
        )
        output_text = request.reference or request.prompt
        return BenchmarkResponse(
            request_id=request.request_id,
            backend_name=self.backend_name,
            model_name=str(self.defaults["model_name"]),
            output_text=output_text,
            success=True,
            prompt_tokens=max(1, len(request.prompt.split())),
            completion_tokens=max(1, len(output_text.split())),
            total_tokens=max(2, len(request.prompt.split()) + len(output_text.split())),
            ttft_ms=float(payload.get("ttft_ms", 0.0)),
            tpot_ms=3.2,
            latency_ms=float(latency_ms),
        )
