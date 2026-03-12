"""vLLM backend adapter."""

from __future__ import annotations

import httpx

from llm_benchmark_suite.adapters.base import BaseBackendAdapter
from llm_benchmark_suite.schemas.models import BenchmarkRequest, BenchmarkResponse


class VLLMAdapter(BaseBackendAdapter):
    def health_check(self) -> bool:
        if self.mode == "mock":
            return True
        health_url = str(self.config.get("health_endpoint", self.config.get("endpoint", "")))
        if not health_url:
            return False
        try:
            response = httpx.get(health_url, timeout=5.0)
            return response.status_code < 500
        except Exception:
            return False

    def infer(self, request: BenchmarkRequest) -> BenchmarkResponse:
        if self.mode == "mock":
            return self._mock_response(request, "vllm")
        endpoint = str(self.config["endpoint"])
        payload = {
            "model": self.defaults["model_name"],
            "messages": [{"role": "user", "content": request.prompt}],
            "temperature": 0,
            "max_tokens": self.defaults.get("max_output_tokens", 128),
        }
        response = httpx.post(endpoint, json=payload, timeout=30.0)
        response.raise_for_status()
        body = response.json()
        text = body["choices"][0]["message"]["content"]
        usage = body.get("usage", {})
        return BenchmarkResponse(
            request_id=request.request_id,
            backend_name=self.backend_name,
            model_name=str(self.defaults["model_name"]),
            output_text=text,
            success=True,
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            total_tokens=int(usage.get("total_tokens", 0)),
            ttft_ms=float(body.get("ttft_ms", 0.0)),
            tpot_ms=float(body.get("tpot_ms", 0.0)),
            latency_ms=float(body.get("latency_ms", 0.0)),
        )
