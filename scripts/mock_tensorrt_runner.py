"""Mock TensorRT-LLM benchmark helper for demonstration and CI smoke tests."""

from __future__ import annotations

import json
from datetime import datetime, timezone


def main() -> None:
    payload = {
        "backend": "tensorrt_llm",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "latency_ms": {"p50": 82.1, "p95": 113.4, "p99": 131.2},
        "throughput_tokens_per_s": 188.2,
        "throughput_requests_per_s": 8.9,
        "ttft_ms": 37.0,
        "success_rate": 1.0,
    }
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
