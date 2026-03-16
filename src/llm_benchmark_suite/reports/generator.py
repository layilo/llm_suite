"""Artifact and report generation utilities."""

from __future__ import annotations

import html
from pathlib import Path

from llm_benchmark_suite.schemas.models import BenchmarkSummary
from llm_benchmark_suite.utils.io import ensure_dir, write_csv, write_json


def _format_value(value: object, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def summary_to_rows(summary: BenchmarkSummary) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    quality_lookup = {
        (item.backend_name, item.dataset_name): item for item in summary.accuracy_metrics
    }
    cost_lookup = {(item.backend_name, item.dataset_name): item for item in summary.cost_metrics}
    for metrics in summary.backend_metrics:
        pair = (metrics.backend_name, metrics.dataset_name)
        quality = quality_lookup.get(pair)
        cost = cost_lookup.get(pair)
        missing = []
        if quality is None:
            missing.append("quality")
        if cost is None:
            missing.append("cost")
        rows.append(
            {
                "backend": metrics.backend_name,
                "dataset": metrics.dataset_name,
                "latency_p95_ms": round(metrics.latency_ms_p95, 2),
                "ttft_ms": round(metrics.ttft_ms_avg, 2),
                "tokens_per_second": round(metrics.tokens_per_second, 2),
                "success_rate": round(metrics.success_rate, 4),
                "quality": round(quality.aggregate_quality, 4) if quality is not None else None,
                "cost_per_million_tokens_usd": (
                    round(cost.cost_per_million_tokens_usd, 2) if cost is not None else None
                ),
                "status": "complete" if not missing else "partial",
                "missing": ",".join(missing),
            }
        )
    return rows


def render_markdown(summary: BenchmarkSummary) -> str:
    rows = summary_to_rows(summary)
    lines = [
        f"# Benchmark Summary: {summary.run_id}",
        "",
        (
            "| Backend | Dataset | p95 Latency (ms) | TTFT (ms) | Tokens/s | "
            "Success Rate | Quality | Cost / 1M Tokens ($) |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        row_text = (
            f"| {row['backend']} | {row['dataset']} | {row['latency_p95_ms']} | "
            f"{row['ttft_ms']} | {row['tokens_per_second']} | {row['success_rate']} | "
            f"{_format_value(row['quality'])} | "
            f"{_format_value(row['cost_per_million_tokens_usd'], 2)} |"
        )
        lines.append(row_text)
    errors = summary.metadata.get("errors", [])
    if errors:
        lines.extend(
            [
                "",
                "## Partial Run Errors",
                "",
                "| Backend | Dataset | Error |",
                "|---|---|---|",
            ]
        )
        for error in errors:
            lines.append(
                f"| {error.get('backend', 'unknown')} | {error.get('dataset', 'unknown')} | "
                f"{error.get('error', 'unknown error')} |"
            )
    if summary.regression_results:
        lines.extend(
            [
                "",
                "## Regression Checks",
                "",
                "| Check | Passed | Delta % | Message |",
                "|---|---|---:|---|",
            ]
        )
        for result in summary.regression_results:
            regression_row = (
                f"| {result.check_name} | {result.passed} | "
                f"{result.delta_pct:.2f} | {result.message} |"
            )
            lines.append(regression_row)
    return "\n".join(lines)


def render_html(summary: BenchmarkSummary) -> str:
    rows = summary_to_rows(summary)
    bars = "".join(
        (
            f"<tr><td>{html.escape(str(row['backend']))}</td>"
            f"<td>{row['dataset']}</td>"
            f"<td><div style='background:#14342b;height:18px;"
            f"width:{max(1, int(float(row['tokens_per_second'])))}px'></div></td>"
            f"<td>{row['tokens_per_second']}</td>"
            f"<td>{row['latency_p95_ms']}</td><td>{html.escape(_format_value(row['quality']))}</td></tr>"
        )
        for row in rows
    )
    errors = summary.metadata.get("errors", [])
    error_rows = "".join(
        (
            f"<tr><td>{html.escape(str(error.get('backend', 'unknown')))}</td>"
            f"<td>{html.escape(str(error.get('dataset', 'unknown')))}</td>"
            f"<td>{html.escape(str(error.get('error', 'unknown error')))}</td></tr>"
        )
        for error in errors
    )
    error_card = ""
    if error_rows:
        error_card = f"""
  <div class="card">
    <h2>Partial Run Errors</h2>
    <table>
      <thead>
        <tr>
          <th>Backend</th>
          <th>Dataset</th>
          <th>Error</th>
        </tr>
      </thead>
      <tbody>{error_rows}</tbody>
    </table>
  </div>"""
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>LLM Benchmark Report</title>
  <style>
    body {{
      font-family: Helvetica, Arial, sans-serif;
      margin: 2rem;
      background: linear-gradient(180deg, #f6f8f7, #eef2ef);
      color: #17201d;
    }}
    table {{ border-collapse: collapse; width: 100%; background: white; }}
    th, td {{ border: 1px solid #cdd6d1; padding: 0.6rem; text-align: left; }}
    th {{ background: #dce7e1; }}
    .card {{
      background: white;
      padding: 1rem 1.25rem;
      border: 1px solid #d8e0db;
      margin-bottom: 1rem;
    }}
  </style>
</head>
<body>
  <div class="card">
    <h1>LLM Benchmark Report</h1>
    <p>
      Run: {html.escape(summary.run_id)} |
      Profile: {html.escape(summary.profile_name)} |
      Mode: {html.escape(summary.mode)}
    </p>
  </div>
  <div class="card">
    <h2>Throughput vs Latency</h2>
    <table>
      <thead>
        <tr>
          <th>Backend</th>
          <th>Dataset</th>
          <th>Tokens/s Bar</th>
          <th>Tokens/s</th>
          <th>p95 Latency (ms)</th>
          <th>Quality</th>
        </tr>
      </thead>
      <tbody>{bars}</tbody>
    </table>
  </div>
  {error_card}
</body>
</html>"""


def write_reports(summary: BenchmarkSummary, output_dir: str, formats: list[str]) -> dict[str, str]:
    target = ensure_dir(output_dir)
    outputs: dict[str, str] = {}
    if "json" in formats:
        path = target / "summary.json"
        write_json(path, summary.model_dump(mode="json"))
        outputs["json"] = str(path)
    if "csv" in formats:
        path = target / "summary.csv"
        write_csv(path, summary_to_rows(summary))
        outputs["csv"] = str(path)
    if "markdown" in formats:
        path = target / "summary.md"
        Path(path).write_text(render_markdown(summary), encoding="utf-8")
        outputs["markdown"] = str(path)
    if "html" in formats:
        path = target / "summary.html"
        Path(path).write_text(render_html(summary), encoding="utf-8")
        outputs["html"] = str(path)
    return outputs
