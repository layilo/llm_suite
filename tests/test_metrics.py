from llm_benchmark_suite.metrics.text import (
    bleu_score,
    exact_match_score,
    rouge_l_score,
    token_f1_score,
)


def test_text_metrics_basic() -> None:
    assert exact_match_score("Mars", "Mars") == 1.0
    assert token_f1_score("Jane Austen", "Jane Austen") == 1.0
    assert bleu_score("electric buses", "electric buses pilot") > 0.0
    assert rouge_l_score("hello world", "hello brave world") > 0.0
