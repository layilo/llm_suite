from llm_benchmark_suite.config import load_run_config
from llm_benchmark_suite.orchestration.runner import run_benchmark


def test_mock_end_to_end_run(tmp_path) -> None:
    config = load_run_config("configs/profiles/ci-smoke.yaml")
    config.output_dir = str(tmp_path / "run")
    config.backend_defaults.concurrency = 2
    summary = run_benchmark(config)
    assert summary.backend_metrics
    assert summary.accuracy_metrics
    assert summary.cost_metrics
    assert all(item.benchmark_wall_time_s > 0 for item in summary.backend_metrics)
    assert all(item.measured_request_count > 0 for item in summary.backend_metrics)
    assert all(item.concurrency == 2 for item in summary.backend_metrics)
    assert all(
        response.finished_at_offset_ms >= response.started_at_offset_ms
        for response in summary.raw_responses
    )


def test_run_preserves_backend_metrics_when_cost_step_fails(tmp_path, monkeypatch) -> None:
    config = load_run_config("configs/profiles/ci-smoke.yaml")
    config.output_dir = str(tmp_path / "run")
    config.continue_on_error = True

    def fail_cost(*args, **kwargs):
        raise RuntimeError("cost model failed")

    monkeypatch.setattr("llm_benchmark_suite.orchestration.runner.compute_cost_metrics", fail_cost)

    summary = run_benchmark(config)

    assert summary.backend_metrics
    assert not summary.cost_metrics
    assert summary.raw_responses
    assert summary.metadata["errors"]
    assert all(item["status"] == "partial" for item in summary.rankings)


def test_warmup_requests_do_not_affect_measured_counts(tmp_path) -> None:
    config = load_run_config("configs/profiles/ci-smoke.yaml")
    config.output_dir = str(tmp_path / "run")
    config.backend_defaults.concurrency = 2
    config.backend_defaults.warmup_requests = 1

    summary = run_benchmark(config)

    assert summary.backend_metrics
    assert all(item.warmup_request_count == 1 for item in summary.backend_metrics)
