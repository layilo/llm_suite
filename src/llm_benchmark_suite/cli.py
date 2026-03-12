"""CLI for the LLM Benchmark Suite."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.table import Table

from llm_benchmark_suite.config import load_run_config
from llm_benchmark_suite.logging_utils import configure_logging
from llm_benchmark_suite.orchestration.runner import run_benchmark
from llm_benchmark_suite.regressions.checks import compare_summaries
from llm_benchmark_suite.reports.generator import write_reports
from llm_benchmark_suite.schemas.models import BenchmarkSummary
from llm_benchmark_suite.utils.io import write_json

console = Console()


def _load_summary(path: str) -> BenchmarkSummary:
    return BenchmarkSummary.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))


def _print_summary(summary: BenchmarkSummary) -> None:
    table = Table(title=f"Benchmark Summary: {summary.run_id}")
    table.add_column("Backend")
    table.add_column("Dataset")
    table.add_column("p95 Latency (ms)")
    table.add_column("Tokens/s")
    table.add_column("Quality")
    for ranking in summary.rankings:
        metrics = next(
            item
            for item in summary.backend_metrics
            if item.backend_name == ranking["backend"] and item.dataset_name == ranking["dataset"]
        )
        table.add_row(
            ranking["backend"],
            ranking["dataset"],
            f"{metrics.latency_ms_p95:.2f}",
            f"{metrics.tokens_per_second:.2f}",
            f"{ranking['quality']:.4f}",
        )
    console.print(table)


@click.group()
def main() -> None:
    """Run LLM benchmarking, reporting, and regression workflows."""
    configure_logging()


@main.command()
@click.option("--config", "config_path", required=True, type=click.Path(exists=True))
@click.option("--baseline", type=click.Path(exists=True), default=None)
def run(config_path: str, baseline: Optional[str]) -> None:
    """Run a benchmark using a YAML profile."""
    config = load_run_config(config_path)
    baseline_summary = _load_summary(baseline) if baseline else None
    summary = run_benchmark(config, baseline_summary=baseline_summary)
    _print_summary(summary)


@main.command()
@click.option(
    "--config",
    "config_path",
    default="configs/profiles/local-demo.yaml",
    type=click.Path(exists=True),
)
@click.option("--output-dir", required=True, type=click.Path())
def demo(config_path: str, output_dir: str) -> None:
    """Run the built-in mock demo benchmark."""
    config = load_run_config(config_path)
    config.output_dir = output_dir
    config.mock_mode = True
    summary = run_benchmark(config)
    _print_summary(summary)


@main.command()
@click.option("--input", "input_path", required=True, type=click.Path(exists=True))
@click.option(
    "--format",
    "report_format",
    required=True,
    type=click.Choice(["json", "csv", "markdown", "html"]),
)
@click.option("--output-dir", required=True, type=click.Path())
def report(input_path: str, report_format: str, output_dir: str) -> None:
    """Generate a report from an existing summary artifact."""
    summary = _load_summary(input_path)
    outputs = write_reports(summary, output_dir, [report_format])
    console.print(outputs)


@main.command()
@click.option("--current", required=True, type=click.Path(exists=True))
@click.option("--baseline", required=True, type=click.Path(exists=True))
@click.option("--thresholds", required=True, type=click.Path(exists=True))
def compare(current: str, baseline: str, thresholds: str) -> None:
    """Compare two summary files and print regression checks."""
    current_summary = _load_summary(current)
    baseline_summary = _load_summary(baseline)
    results = compare_summaries(current_summary, baseline_summary, thresholds)
    for result in results:
        message = (
            f"{result.check_name}: passed={result.passed} "
            f"delta={result.delta_pct:.2f}% {result.message}"
        )
        console.print(message)


@main.command()
@click.option("--current", required=True, type=click.Path(exists=True))
@click.option("--baseline", required=True, type=click.Path(exists=True))
@click.option("--thresholds", required=True, type=click.Path(exists=True))
@click.option("--output", required=False, type=click.Path(), default=None)
def regress(current: str, baseline: str, thresholds: str, output: Optional[str]) -> None:
    """Run regression gates and optionally persist machine-readable results."""
    current_summary = _load_summary(current)
    baseline_summary = _load_summary(baseline)
    results = compare_summaries(current_summary, baseline_summary, thresholds)
    payload = [item.model_dump(mode="json") for item in results]
    if output:
        write_json(output, payload)
    failing = [item for item in results if not item.passed]
    for result in results:
        console.print(f"{result.check_name}: {'PASS' if result.passed else 'FAIL'}")
    if failing:
        raise SystemExit(1)


@main.command(name="export-baseline")
@click.option("--input", "input_path", required=True, type=click.Path(exists=True))
@click.option("--output", required=True, type=click.Path())
def export_baseline(input_path: str, output: str) -> None:
    """Export a summary artifact as a reusable baseline."""
    summary = _load_summary(input_path)
    write_json(output, summary.model_dump(mode="json"))
    console.print(f"Baseline written to {output}")


@main.command()
@click.option("--input", "input_path", required=True, type=click.Path(exists=True))
def evaluate(input_path: str) -> None:
    """Inspect evaluation metrics from an existing summary artifact."""
    summary = _load_summary(input_path)
    for metric in summary.accuracy_metrics:
        console.print(
            f"{metric.backend_name}/{metric.dataset_name}: quality={metric.aggregate_quality:.4f} "
            f"EM={metric.exact_match:.4f} F1={metric.token_f1:.4f} ROUGE-L={metric.rouge_l:.4f}"
        )


if __name__ == "__main__":
    main()
