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
        return self._response_from_payload(
            request,
            body,
            fallback_text=request.reference or request.prompt,
        )
