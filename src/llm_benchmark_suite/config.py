"""Configuration loading and normalization helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Union

import yaml
from pydantic import BaseModel, Field


class DatasetConfig(BaseModel):
    name: str
    path: str
    task_type: str


class BackendDefaults(BaseModel):
    model_name: str
    precision: str
    concurrency: int = 1
    batch_size: int = 1
    max_output_tokens: int = 128


class RunConfig(BaseModel):
    profile_name: str
    mock_mode: bool = True
    continue_on_error: bool = True
    random_seed: int = 7
    output_dir: str = "artifacts/generated"
    report_formats: list[str] = Field(default_factory=lambda: ["json"])
    selected_backends: list[str] = Field(default_factory=list)
    datasets: list[DatasetConfig] = Field(default_factory=list)
    backend_defaults: BackendDefaults
    cost_profile: str
    thresholds_profile: str
    composite_weights: dict[str, float] = Field(default_factory=dict)
    backend_overrides: dict[str, dict[str, Any]] = Field(default_factory=dict)


def load_yaml_file(path: Union[str, Path]) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def apply_env_overrides(data: dict[str, Any]) -> dict[str, Any]:
    output_dir = os.getenv("LLM_BENCHMARK_OUTPUT_DIR")
    if output_dir:
        data["output_dir"] = output_dir
    profile_name = os.getenv("LLM_BENCHMARK_PROFILE_NAME")
    if profile_name:
        data["profile_name"] = profile_name
    return data


def load_run_config(path: Union[str, Path]) -> RunConfig:
    payload = apply_env_overrides(load_yaml_file(path))
    return RunConfig.model_validate(payload)
