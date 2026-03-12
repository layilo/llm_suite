from click.testing import CliRunner

from llm_benchmark_suite.cli import main


def test_demo_command_runs() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["demo", "--output-dir", "artifacts/generated/test-demo"])
    assert result.exit_code == 0, result.output
    assert "Benchmark Summary" in result.output
