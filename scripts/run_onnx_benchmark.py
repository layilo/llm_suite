"""Scaffold entry point for real ONNX Runtime benchmarking integration."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone


def main() -> None:
    payload = {
        "backend": "onnx_runtime",
        "mode": "scaffold",
        "message": "Replace this script with environment-specific ONNX Runtime benchmark logic.",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "argv": sys.argv[1:],
    }
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
