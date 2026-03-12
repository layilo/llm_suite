"""I/O helpers for dataset loading and artifact writing."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, Union

from llm_benchmark_suite.schemas.models import BenchmarkRequest


def ensure_dir(path: Union[str, Path]) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def load_jsonl_dataset(path: Union[str, Path]) -> list[BenchmarkRequest]:
    items: list[BenchmarkRequest] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            items.append(
                BenchmarkRequest(
                    request_id=row["id"],
                    prompt=row["prompt"],
                    reference=row.get("reference"),
                    task_type=row["task_type"],
                    expected_contains=row.get("expected_contains", []),
                    tags=row.get("tags", []),
                )
            )
    return items


def write_json(path: Union[str, Path], payload: object) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)


def write_csv(path: Union[str, Path], rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        return
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
