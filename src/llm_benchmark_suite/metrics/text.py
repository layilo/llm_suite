"""Lightweight evaluation metrics for deterministic local benchmarking."""

from __future__ import annotations

import math
import re
from collections import Counter

TOKEN_PATTERN = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def exact_match_score(prediction: str, reference: str) -> float:
    return float(prediction.strip().lower() == reference.strip().lower())


def token_f1_score(prediction: str, reference: str) -> float:
    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    overlap = Counter(pred_tokens) & Counter(ref_tokens)
    common = sum(overlap.values())
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def bleu_score(prediction: str, reference: str) -> float:
    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0
    matches = sum(1 for token in pred_tokens if token in ref_tokens)
    precision = matches / len(pred_tokens)
    brevity_penalty = min(1.0, math.exp(1 - (len(ref_tokens) / max(len(pred_tokens), 1))))
    return precision * brevity_penalty


def rouge_l_score(prediction: str, reference: str) -> float:
    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0
    dp = [[0] * (len(ref_tokens) + 1) for _ in range(len(pred_tokens) + 1)]
    for i, pred in enumerate(pred_tokens, start=1):
        for j, ref in enumerate(ref_tokens, start=1):
            if pred == ref:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[-1][-1]
    return (2 * lcs) / (len(pred_tokens) + len(ref_tokens))
