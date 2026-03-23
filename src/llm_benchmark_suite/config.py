"""Configuration loading and validation helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Union

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator

from llm_benchmark_suite.adapters.factory import ADAPTERS

SUPPORTED_REPORT_FORMATS = {"json", "csv", "markdown", "html"}
SUPPORTED_COMPOSITE_WEIGHT_KEYS = {"quality", "latency", "throughput", "cost", "reliability"}


class ConfigValidationError(Exception):
    """Raised when one or more configuration validation checks fail."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        message = "Configuration validation failed:\n" + "\n".join(
            f"- {error}" for error in errors
        )
        super().__init__(message)


class DatasetConfig(BaseModel):
    name: str
    path: str
    task_type: str

    @field_validator("name", "task_type")
    @classmethod
    def _require_non_empty_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be empty")
        return value


class BackendDefaults(BaseModel):
    model_name: str
    precision: str
    concurrency: int = 1
    batch_size: int = 1
    max_output_tokens: int = 128
    warmup_requests: int = 0

    @field_validator("model_name", "precision")
    @classmethod
    def _require_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be empty")
        return value

    @field_validator("concurrency", "batch_size", "max_output_tokens")
    @classmethod
    def _require_positive_int(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("must be greater than 0")
        return value

    @field_validator("warmup_requests")
    @classmethod
    def _require_non_negative_int(cls, value: int) -> int:
        if value < 0:
            raise ValueError("must be greater than or equal to 0")
        return value


class CostProfile(BaseModel):
    gpu_hourly_cost_usd: float
    cpu_hourly_cost_usd: float
    memory_gb_hourly_cost_usd: float
    amortization_factor: float = 1.0

    @field_validator(
        "gpu_hourly_cost_usd",
        "cpu_hourly_cost_usd",
        "memory_gb_hourly_cost_usd",
        "amortization_factor",
    )
    @classmethod
    def _require_non_negative_float(cls, value: float) -> float:
        if value < 0:
            raise ValueError("must be greater than or equal to 0")
        return value


class ThresholdsProfile(BaseModel):
    p95_latency_regression_pct: float
    ttft_regression_pct: float
    throughput_regression_pct: float
    accuracy_min: float
    cost_per_million_tokens_regression_pct: float
    error_rate_max: float

    @field_validator(
        "p95_latency_regression_pct",
        "ttft_regression_pct",
        "throughput_regression_pct",
        "accuracy_min",
        "cost_per_million_tokens_regression_pct",
        "error_rate_max",
    )
    @classmethod
    def _require_non_negative_float(cls, value: float) -> float:
        if value < 0:
            raise ValueError("must be greater than or equal to 0")
        return value


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

    @field_validator("profile_name")
    @classmethod
    def _require_profile_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be empty")
        return value


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


def _resolve_path(base_dir: Path, value: str) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    profile_relative = (base_dir / path).resolve()
    if profile_relative.exists():
        return str(profile_relative)
    return str(path.resolve())


def _resolve_profile_paths(payload: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    resolved = dict(payload)
    if "output_dir" in resolved and isinstance(resolved["output_dir"], str):
        resolved["output_dir"] = _resolve_path(base_dir, resolved["output_dir"])
    if "cost_profile" in resolved and isinstance(resolved["cost_profile"], str):
        resolved["cost_profile"] = _resolve_path(base_dir, resolved["cost_profile"])
    if "thresholds_profile" in resolved and isinstance(resolved["thresholds_profile"], str):
        resolved["thresholds_profile"] = _resolve_path(base_dir, resolved["thresholds_profile"])

    datasets = []
    for dataset in resolved.get("datasets", []):
        dataset_payload = dict(dataset)
        if isinstance(dataset_payload.get("path"), str):
            dataset_payload["path"] = _resolve_path(base_dir, dataset_payload["path"])
        datasets.append(dataset_payload)
    if datasets:
        resolved["datasets"] = datasets

    backend_overrides: dict[str, dict[str, Any]] = {}
    for backend_name, override in resolved.get("backend_overrides", {}).items():
        override_payload = dict(override)
        if backend_name == "onnx_runtime" and isinstance(
            override_payload.get("benchmark_script"), str
        ):
            override_payload["benchmark_script"] = _resolve_path(
                base_dir, str(override_payload["benchmark_script"])
            )
        backend_overrides[backend_name] = override_payload
    if backend_overrides:
        resolved["backend_overrides"] = backend_overrides

    return resolved


def _format_pydantic_errors(exc: ValidationError, prefix: str) -> list[str]:
    formatted: list[str] = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"])
        if location:
            formatted.append(f"{prefix}.{location}: {error['msg']}")
        else:
            formatted.append(f"{prefix}: {error['msg']}")
    return formatted


def _validate_unique_strings(values: list[str], field_name: str, errors: list[str]) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    for duplicate in sorted(duplicates):
        errors.append(f"{field_name}: duplicate value '{duplicate}'")


def _validate_run_config_payload(config: RunConfig) -> None:
    errors: list[str] = []

    if not config.selected_backends:
        errors.append("selected_backends: must contain at least one backend")
    else:
        _validate_unique_strings(config.selected_backends, "selected_backends", errors)
        for backend_name in config.selected_backends:
            if backend_name not in ADAPTERS:
                errors.append(f"selected_backends: unknown backend '{backend_name}'")

    if not config.datasets:
        errors.append("datasets: must contain at least one dataset")
    else:
        dataset_names = [dataset.name for dataset in config.datasets]
        _validate_unique_strings(dataset_names, "datasets", errors)
        for dataset in config.datasets:
            if not Path(dataset.path).exists():
                errors.append(
                    f"datasets.{dataset.name}.path: file does not exist at '{dataset.path}'"
                )

    if config.report_formats:
        _validate_unique_strings(config.report_formats, "report_formats", errors)
        for report_format in config.report_formats:
            if report_format not in SUPPORTED_REPORT_FORMATS:
                errors.append(f"report_formats: unsupported format '{report_format}'")

    invalid_weight_keys = sorted(set(config.composite_weights) - SUPPORTED_COMPOSITE_WEIGHT_KEYS)
    for key in invalid_weight_keys:
        errors.append(f"composite_weights: unsupported key '{key}'")
    for key, value in config.composite_weights.items():
        if value < 0:
            errors.append(f"composite_weights.{key}: must be greater than or equal to 0")

    if not Path(config.cost_profile).exists():
        errors.append(f"cost_profile: file does not exist at '{config.cost_profile}'")
    if not Path(config.thresholds_profile).exists():
        errors.append(
            f"thresholds_profile: file does not exist at '{config.thresholds_profile}'"
        )

    unknown_override_backends = sorted(set(config.backend_overrides) - set(ADAPTERS))
    for backend_name in unknown_override_backends:
        errors.append(f"backend_overrides: unknown backend '{backend_name}'")

    for backend_name, override in config.backend_overrides.items():
        if backend_name not in ADAPTERS:
            continue
        mode = str(override.get("mode", "mock")).strip() or "mock"
        if mode not in {"mock", "real"}:
            errors.append(
                f"backend_overrides.{backend_name}.mode: unsupported mode '{mode}'"
            )
            continue
        if mode != "real":
            continue
        if backend_name == "vllm" and not str(override.get("endpoint", "")).strip():
            errors.append("backend_overrides.vllm.endpoint: required when mode=real")
        if backend_name == "tensorrt_llm" and not str(override.get("command", "")).strip():
            errors.append("backend_overrides.tensorrt_llm.command: required when mode=real")
        if backend_name == "onnx_runtime":
            script = str(override.get("benchmark_script", "")).strip()
            if not script:
                errors.append(
                    "backend_overrides.onnx_runtime.benchmark_script: required when mode=real"
                )
            elif not Path(script).exists():
                errors.append(
                    "backend_overrides.onnx_runtime.benchmark_script: "
                    f"file does not exist at '{script}'"
                )

    if errors:
        raise ConfigValidationError(errors)


def _load_and_validate_profile(
    path: str,
    model_cls: type[BaseModel],
    label: str,
) -> None:
    errors: list[str] = []
    try:
        payload = load_yaml_file(path)
    except FileNotFoundError:
        raise
    except Exception as exc:
        errors.append(f"{label}: failed to load '{path}': {exc}")
    else:
        try:
            model_cls.model_validate(payload)
        except ValidationError as exc:
            errors.extend(_format_pydantic_errors(exc, label))
    if errors:
        raise ConfigValidationError(errors)


def load_run_config(path: Union[str, Path]) -> RunConfig:
    config_path = Path(path).resolve()
    errors: list[str] = []

    try:
        payload = apply_env_overrides(load_yaml_file(config_path))
    except Exception as exc:
        raise ConfigValidationError([f"config: failed to load '{config_path}': {exc}"]) from exc

    payload = _resolve_profile_paths(payload, config_path.parent)

    try:
        config = RunConfig.model_validate(payload)
    except ValidationError as exc:
        errors.extend(_format_pydantic_errors(exc, "config"))
        raise ConfigValidationError(errors) from exc

    try:
        _validate_run_config_payload(config)
    except ConfigValidationError as exc:
        errors.extend(exc.errors)

    if Path(config.cost_profile).exists():
        try:
            _load_and_validate_profile(config.cost_profile, CostProfile, "cost_profile")
        except ConfigValidationError as exc:
            errors.extend(exc.errors)

    if Path(config.thresholds_profile).exists():
        try:
            _load_and_validate_profile(
                config.thresholds_profile, ThresholdsProfile, "thresholds_profile"
            )
        except ConfigValidationError as exc:
            errors.extend(exc.errors)

    if errors:
        raise ConfigValidationError(errors)

    return config
