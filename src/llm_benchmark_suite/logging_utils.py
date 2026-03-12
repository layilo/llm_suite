"""Structured logging configuration."""

from __future__ import annotations

import json
import logging
import os
from typing import Any


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
        }
        if hasattr(record, "context"):
            payload["context"] = record.context
        return json.dumps(payload)


def configure_logging() -> None:
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(os.getenv("LLM_BENCHMARK_LOG_LEVEL", "INFO"))
    handler = logging.StreamHandler()
    if os.getenv("LLM_BENCHMARK_JSON_LOGS", "false").lower() == "true":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s %(message)s"))
    root.addHandler(handler)
