from __future__ import annotations

from pathlib import Path

import pytest

from llm_benchmark_suite.config import ConfigValidationError, load_run_config


def _write_base_files(tmp_path: Path) -> dict[str, Path]:
    profile_dir = tmp_path / "profiles"
    data_dir = tmp_path / "data"
    costs_dir = tmp_path / "costs"
    thresholds_dir = tmp_path / "thresholds"
    scripts_dir = tmp_path / "scripts"
    profile_dir.mkdir()
    data_dir.mkdir()
    costs_dir.mkdir()
    thresholds_dir.mkdir()
    scripts_dir.mkdir()

    dataset_path = data_dir / "sample.jsonl"
    dataset_path.write_text(
        '{"id":"1","prompt":"Hello","reference":"Hello","task_type":"qa"}\n',
        encoding="utf-8",
    )
    cost_path = costs_dir / "default.yaml"
    cost_path.write_text(
        "\n".join(
            [
                "gpu_hourly_cost_usd: 1",
                "cpu_hourly_cost_usd: 1",
                "memory_gb_hourly_cost_usd: 1",
                "amortization_factor: 1",
                "",
            ]
        ),
        encoding="utf-8",
    )
    thresholds_path = thresholds_dir / "default.yaml"
    thresholds_path.write_text(
        "\n".join(
            [
                "p95_latency_regression_pct: 10",
                "ttft_regression_pct: 10",
                "throughput_regression_pct: 10",
                "accuracy_min: 0.5",
                "cost_per_million_tokens_regression_pct: 10",
                "error_rate_max: 0.1",
                "",
            ]
        ),
        encoding="utf-8",
    )
    benchmark_script = scripts_dir / "run_onnx_benchmark.py"
    benchmark_script.write_text("print('{}')\n", encoding="utf-8")

    return {
        "profile_dir": profile_dir,
        "dataset_path": dataset_path,
        "cost_path": cost_path,
        "thresholds_path": thresholds_path,
        "benchmark_script": benchmark_script,
    }


def _write_profile(
    profile_path: Path,
    *,
    body: list[str] | None = None,
) -> Path:
    lines = body or [
        "profile_name: custom",
        "mock_mode: true",
        "output_dir: ../artifacts/custom",
        "report_formats: [json]",
        "selected_backends: [vllm]",
        "datasets:",
        "  - name: sample",
        "    path: ../data/sample.jsonl",
        "    task_type: qa",
        "backend_defaults:",
        "  model_name: demo",
        "  precision: fp16",
        "  concurrency: 1",
        "  batch_size: 1",
        "  max_output_tokens: 64",
        "cost_profile: ../costs/default.yaml",
        "thresholds_profile: ../thresholds/default.yaml",
    ]
    profile_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return profile_path


def _assert_config_error(path: Path, expected_messages: list[str]) -> None:
    with pytest.raises(ConfigValidationError) as excinfo:
        load_run_config(path)
    message = str(excinfo.value)
    for expected in expected_messages:
        assert expected in message


def test_load_shipped_configs() -> None:
    for profile in [
        "configs/profiles/local-demo.yaml",
        "configs/profiles/ci-smoke.yaml",
        "configs/profiles/gpu-dev.yaml",
        "configs/profiles/perf-lab.yaml",
    ]:
        config = load_run_config(profile)
        assert config.profile_name
        assert config.selected_backends


def test_load_run_config_resolves_paths_relative_to_profile(tmp_path: Path) -> None:
    paths = _write_base_files(tmp_path)
    profile_path = _write_profile(paths["profile_dir"] / "custom.yaml")

    config = load_run_config(profile_path)

    assert Path(config.output_dir).is_absolute()
    assert Path(config.datasets[0].path) == paths["dataset_path"].resolve()
    assert Path(config.cost_profile) == paths["cost_path"].resolve()
    assert Path(config.thresholds_profile) == paths["thresholds_path"].resolve()


def test_dataset_path_missing_is_reported(tmp_path: Path) -> None:
    paths = _write_base_files(tmp_path)
    profile_path = _write_profile(
        paths["profile_dir"] / "custom.yaml",
        body=[
            "profile_name: custom",
            "mock_mode: true",
            "report_formats: [json]",
            "selected_backends: [vllm]",
            "datasets:",
            "  - name: missing",
            "    path: ../data/missing.jsonl",
            "    task_type: qa",
            "backend_defaults:",
            "  model_name: demo",
            "  precision: fp16",
            "cost_profile: ../costs/default.yaml",
            "thresholds_profile: ../thresholds/default.yaml",
        ],
    )

    _assert_config_error(profile_path, ["datasets.missing.path: file does not exist"])


def test_cost_profile_path_missing_is_reported(tmp_path: Path) -> None:
    paths = _write_base_files(tmp_path)
    profile_path = _write_profile(
        paths["profile_dir"] / "custom.yaml",
        body=[
            "profile_name: custom",
            "mock_mode: true",
            "report_formats: [json]",
            "selected_backends: [vllm]",
            "datasets:",
            "  - name: sample",
            "    path: ../data/sample.jsonl",
            "    task_type: qa",
            "backend_defaults:",
            "  model_name: demo",
            "  precision: fp16",
            "cost_profile: ../costs/missing.yaml",
            "thresholds_profile: ../thresholds/default.yaml",
        ],
    )

    _assert_config_error(profile_path, ["cost_profile: file does not exist"])


def test_thresholds_profile_path_missing_is_reported(tmp_path: Path) -> None:
    paths = _write_base_files(tmp_path)
    profile_path = _write_profile(
        paths["profile_dir"] / "custom.yaml",
        body=[
            "profile_name: custom",
            "mock_mode: true",
            "report_formats: [json]",
            "selected_backends: [vllm]",
            "datasets:",
            "  - name: sample",
            "    path: ../data/sample.jsonl",
            "    task_type: qa",
            "backend_defaults:",
            "  model_name: demo",
            "  precision: fp16",
            "cost_profile: ../costs/default.yaml",
            "thresholds_profile: ../thresholds/missing.yaml",
        ],
    )

    _assert_config_error(profile_path, ["thresholds_profile: file does not exist"])


def test_unknown_backend_in_selected_backends_is_reported(tmp_path: Path) -> None:
    paths = _write_base_files(tmp_path)
    profile_path = _write_profile(
        paths["profile_dir"] / "custom.yaml",
        body=[
            "profile_name: custom",
            "mock_mode: true",
            "report_formats: [json]",
            "selected_backends: [unknown_backend]",
            "datasets:",
            "  - name: sample",
            "    path: ../data/sample.jsonl",
            "    task_type: qa",
            "backend_defaults:",
            "  model_name: demo",
            "  precision: fp16",
            "cost_profile: ../costs/default.yaml",
            "thresholds_profile: ../thresholds/default.yaml",
        ],
    )

    _assert_config_error(profile_path, ["selected_backends: unknown backend 'unknown_backend'"])


def test_duplicate_backend_names_are_reported(tmp_path: Path) -> None:
    paths = _write_base_files(tmp_path)
    profile_path = _write_profile(
        paths["profile_dir"] / "custom.yaml",
        body=[
            "profile_name: custom",
            "mock_mode: true",
            "report_formats: [json]",
            "selected_backends: [vllm, vllm]",
            "datasets:",
            "  - name: sample",
            "    path: ../data/sample.jsonl",
            "    task_type: qa",
            "backend_defaults:",
            "  model_name: demo",
            "  precision: fp16",
            "cost_profile: ../costs/default.yaml",
            "thresholds_profile: ../thresholds/default.yaml",
        ],
    )

    _assert_config_error(profile_path, ["selected_backends: duplicate value 'vllm'"])


def test_duplicate_dataset_names_are_reported(tmp_path: Path) -> None:
    paths = _write_base_files(tmp_path)
    profile_path = _write_profile(
        paths["profile_dir"] / "custom.yaml",
        body=[
            "profile_name: custom",
            "mock_mode: true",
            "report_formats: [json]",
            "selected_backends: [vllm]",
            "datasets:",
            "  - name: sample",
            "    path: ../data/sample.jsonl",
            "    task_type: qa",
            "  - name: sample",
            "    path: ../data/sample.jsonl",
            "    task_type: qa",
            "backend_defaults:",
            "  model_name: demo",
            "  precision: fp16",
            "cost_profile: ../costs/default.yaml",
            "thresholds_profile: ../thresholds/default.yaml",
        ],
    )

    _assert_config_error(profile_path, ["datasets: duplicate value 'sample'"])


def test_unsupported_report_format_is_reported(tmp_path: Path) -> None:
    paths = _write_base_files(tmp_path)
    profile_path = _write_profile(
        paths["profile_dir"] / "custom.yaml",
        body=[
            "profile_name: custom",
            "mock_mode: true",
            "report_formats: [json, xml]",
            "selected_backends: [vllm]",
            "datasets:",
            "  - name: sample",
            "    path: ../data/sample.jsonl",
            "    task_type: qa",
            "backend_defaults:",
            "  model_name: demo",
            "  precision: fp16",
            "cost_profile: ../costs/default.yaml",
            "thresholds_profile: ../thresholds/default.yaml",
        ],
    )

    _assert_config_error(profile_path, ["report_formats: unsupported format 'xml'"])


def test_invalid_composite_weight_key_and_value_are_reported(tmp_path: Path) -> None:
    paths = _write_base_files(tmp_path)
    profile_path = _write_profile(
        paths["profile_dir"] / "custom.yaml",
        body=[
            "profile_name: custom",
            "mock_mode: true",
            "report_formats: [json]",
            "selected_backends: [vllm]",
            "datasets:",
            "  - name: sample",
            "    path: ../data/sample.jsonl",
            "    task_type: qa",
            "backend_defaults:",
            "  model_name: demo",
            "  precision: fp16",
            "cost_profile: ../costs/default.yaml",
            "thresholds_profile: ../thresholds/default.yaml",
            "composite_weights:",
            "  unsupported: 0.1",
            "  latency: -0.5",
        ],
    )

    _assert_config_error(
        profile_path,
        [
            "composite_weights: unsupported key 'unsupported'",
            "composite_weights.latency: must be greater than or equal to 0",
        ],
    )


def test_real_mode_vllm_requires_endpoint(tmp_path: Path) -> None:
    paths = _write_base_files(tmp_path)
    profile_path = _write_profile(
        paths["profile_dir"] / "custom.yaml",
        body=[
            "profile_name: custom",
            "mock_mode: true",
            "report_formats: [json]",
            "selected_backends: [vllm]",
            "datasets:",
            "  - name: sample",
            "    path: ../data/sample.jsonl",
            "    task_type: qa",
            "backend_defaults:",
            "  model_name: demo",
            "  precision: fp16",
            "cost_profile: ../costs/default.yaml",
            "thresholds_profile: ../thresholds/default.yaml",
            "backend_overrides:",
            "  vllm:",
            "    mode: real",
        ],
    )

    _assert_config_error(profile_path, ["backend_overrides.vllm.endpoint: required when mode=real"])


def test_real_mode_tensorrt_requires_command(tmp_path: Path) -> None:
    paths = _write_base_files(tmp_path)
    profile_path = _write_profile(
        paths["profile_dir"] / "custom.yaml",
        body=[
            "profile_name: custom",
            "mock_mode: true",
            "report_formats: [json]",
            "selected_backends: [tensorrt_llm]",
            "datasets:",
            "  - name: sample",
            "    path: ../data/sample.jsonl",
            "    task_type: qa",
            "backend_defaults:",
            "  model_name: demo",
            "  precision: fp16",
            "cost_profile: ../costs/default.yaml",
            "thresholds_profile: ../thresholds/default.yaml",
            "backend_overrides:",
            "  tensorrt_llm:",
            "    mode: real",
        ],
    )

    _assert_config_error(
        profile_path, ["backend_overrides.tensorrt_llm.command: required when mode=real"]
    )


def test_real_mode_onnx_requires_benchmark_script(tmp_path: Path) -> None:
    paths = _write_base_files(tmp_path)
    profile_path = _write_profile(
        paths["profile_dir"] / "custom.yaml",
        body=[
            "profile_name: custom",
            "mock_mode: true",
            "report_formats: [json]",
            "selected_backends: [onnx_runtime]",
            "datasets:",
            "  - name: sample",
            "    path: ../data/sample.jsonl",
            "    task_type: qa",
            "backend_defaults:",
            "  model_name: demo",
            "  precision: fp16",
            "cost_profile: ../costs/default.yaml",
            "thresholds_profile: ../thresholds/default.yaml",
            "backend_overrides:",
            "  onnx_runtime:",
            "    mode: real",
        ],
    )

    _assert_config_error(
        profile_path,
        ["backend_overrides.onnx_runtime.benchmark_script: required when mode=real"],
    )


def test_unknown_backend_override_is_reported(tmp_path: Path) -> None:
    paths = _write_base_files(tmp_path)
    profile_path = _write_profile(
        paths["profile_dir"] / "custom.yaml",
        body=[
            "profile_name: custom",
            "mock_mode: true",
            "report_formats: [json]",
            "selected_backends: [vllm]",
            "datasets:",
            "  - name: sample",
            "    path: ../data/sample.jsonl",
            "    task_type: qa",
            "backend_defaults:",
            "  model_name: demo",
            "  precision: fp16",
            "cost_profile: ../costs/default.yaml",
            "thresholds_profile: ../thresholds/default.yaml",
            "backend_overrides:",
            "  unknown_backend:",
            "    mode: mock",
        ],
    )

    _assert_config_error(profile_path, ["backend_overrides: unknown backend 'unknown_backend'"])


def test_invalid_cost_profile_contents_are_reported(tmp_path: Path) -> None:
    paths = _write_base_files(tmp_path)
    paths["cost_path"].write_text("gpu_hourly_cost_usd: 1\n", encoding="utf-8")
    profile_path = _write_profile(paths["profile_dir"] / "custom.yaml")

    _assert_config_error(
        profile_path,
        [
            "cost_profile.cpu_hourly_cost_usd: Field required",
            "cost_profile.memory_gb_hourly_cost_usd: Field required",
        ],
    )


def test_invalid_thresholds_profile_contents_are_reported(tmp_path: Path) -> None:
    paths = _write_base_files(tmp_path)
    paths["thresholds_path"].write_text("p95_latency_regression_pct: 10\n", encoding="utf-8")
    profile_path = _write_profile(paths["profile_dir"] / "custom.yaml")

    _assert_config_error(
        profile_path,
        [
            "thresholds_profile.ttft_regression_pct: Field required",
            "thresholds_profile.error_rate_max: Field required",
        ],
    )


def test_aggregated_errors_include_multiple_issues(tmp_path: Path) -> None:
    paths = _write_base_files(tmp_path)
    profile_path = _write_profile(
        paths["profile_dir"] / "custom.yaml",
        body=[
            "profile_name: custom",
            "mock_mode: true",
            "report_formats: [json, xml]",
            "selected_backends: [vllm, vllm]",
            "datasets:",
            "  - name: sample",
            "    path: ../data/missing.jsonl",
            "    task_type: qa",
            "backend_defaults:",
            "  model_name: demo",
            "  precision: fp16",
            "cost_profile: ../costs/default.yaml",
            "thresholds_profile: ../thresholds/default.yaml",
            "backend_overrides:",
            "  vllm:",
            "    mode: real",
        ],
    )

    _assert_config_error(
        profile_path,
        [
            "selected_backends: duplicate value 'vllm'",
            "report_formats: unsupported format 'xml'",
            "datasets.sample.path: file does not exist",
            "backend_overrides.vllm.endpoint: required when mode=real",
        ],
    )
