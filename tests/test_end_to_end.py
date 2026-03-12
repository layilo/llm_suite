from llm_benchmark_suite.config import load_run_config
from llm_benchmark_suite.orchestration.runner import run_benchmark


def test_mock_end_to_end_run(tmp_path) -> None:
    config = load_run_config("configs/profiles/ci-smoke.yaml")
    config.output_dir = str(tmp_path / "run")
    summary = run_benchmark(config)
    assert summary.backend_metrics
    assert summary.accuracy_metrics
    assert summary.cost_metrics
