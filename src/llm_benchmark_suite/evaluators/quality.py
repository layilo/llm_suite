"""Quality evaluation orchestration."""

from __future__ import annotations

from collections import defaultdict

from llm_benchmark_suite.metrics.text import (
    bleu_score,
    exact_match_score,
    rouge_l_score,
    token_f1_score,
)
from llm_benchmark_suite.schemas.models import AccuracyMetrics, BenchmarkRequest, BenchmarkResponse


def evaluate_responses(
    backend_name: str,
    dataset_name: str,
    requests: list[BenchmarkRequest],
    responses: list[BenchmarkResponse],
) -> AccuracyMetrics:
    by_id = {item.request_id: item for item in requests}
    totals = defaultdict(float)
    golden_checks = 0
    golden_passes = 0

    for response in responses:
        request = by_id[response.request_id]
        reference = request.reference or ""
        prediction = response.output_text
        totals["exact_match"] += exact_match_score(prediction, reference)
        totals["token_f1"] += token_f1_score(prediction, reference)
        totals["bleu"] += bleu_score(prediction, reference)
        totals["rouge_l"] += rouge_l_score(prediction, reference)
        if request.expected_contains:
            golden_checks += 1
            if all(token.lower() in prediction.lower() for token in request.expected_contains):
                golden_passes += 1
        if response.success and reference:
            totals["pass_rate"] += float(token_f1_score(prediction, reference) >= 0.5)

    count = max(len(responses), 1)
    return AccuracyMetrics(
        backend_name=backend_name,
        dataset_name=dataset_name,
        exact_match=totals["exact_match"] / count,
        token_f1=totals["token_f1"] / count,
        bleu=totals["bleu"] / count,
        rouge_l=totals["rouge_l"] / count,
        pass_rate=totals["pass_rate"] / count,
        golden_pass_rate=golden_passes / max(golden_checks, 1),
    )
