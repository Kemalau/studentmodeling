"""Dependency-light metrics shared by formal non-LLM baselines."""

from __future__ import annotations

import html
import math
import re
from fractions import Fraction
from typing import Any


TAG_BY_TASK = {
    "predict_response": "predicted_response",
    "diagnose_misconception": "misconception",
}


def extract_tag(text: str, tag: str) -> str | None:
    match = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", text, flags=re.I | re.S)
    return html.unescape(match.group(1).strip()) if match else None


def normalize_text(value: str | None) -> str:
    text = html.unescape(str(value or "")).casefold().strip()
    text = text.replace("−", "-").replace("×", "*").replace("÷", "/")
    return re.sub(r"\s+", "", text)


def numeric_value(value: str | None):
    text = normalize_text(value).replace(",", "")
    if not text or len(text) > 80 or not re.fullmatch(r"[+\-]?(?:\d+(?:\.\d+)?|\.\d+)(?:/[+\-]?(?:\d+(?:\.\d+)?|\.\d+))?", text):
        return None
    try:
        if "/" in text:
            numerator, denominator = text.split("/", 1)
            return Fraction(numerator) / Fraction(denominator)
        return Fraction(text)
    except (ValueError, ZeroDivisionError):
        return None


def mathematically_equivalent(prediction: str | None, gold: str | None) -> bool:
    if normalize_text(prediction) == normalize_text(gold):
        return True
    prediction_value = numeric_value(prediction)
    gold_value = numeric_value(gold)
    return prediction_value is not None and gold_value is not None and prediction_value == gold_value


def binary_auroc(labels: list[int], probabilities: list[float]) -> float | None:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    pairs = sorted(zip(probabilities, labels), key=lambda item: item[0])
    rank_sum = 0.0
    index = 0
    while index < len(pairs):
        end = index
        while end + 1 < len(pairs) and pairs[end + 1][0] == pairs[index][0]:
            end += 1
        average_rank = (index + 1 + end + 1) / 2
        rank_sum += average_rank * sum(label for _, label in pairs[index : end + 1])
        index = end + 1
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def binary_auprc(labels: list[int], probabilities: list[float]) -> float | None:
    positives = sum(labels)
    if positives == 0:
        return None
    true_positives = 0
    precision_sum = 0.0
    for rank, (_, label) in enumerate(sorted(zip(probabilities, labels), reverse=True), 1):
        if label:
            true_positives += 1
            precision_sum += true_positives / rank
    return precision_sum / positives


def class_f1(true_positive: int, false_positive: int, false_negative: int) -> float:
    denominator = 2 * true_positive + false_positive + false_negative
    return 0.0 if denominator == 0 else 2 * true_positive / denominator


def binary_metrics(records: list[dict[str, Any]], positive: str) -> dict[str, Any]:
    gold = [int(record["gold"] == positive) for record in records]
    probability = [float(record["probabilities"].get(positive, 0.0)) for record in records]
    prediction = [int(record["prediction"] == positive) for record in records]
    true_positive = sum(label == 1 and predicted == 1 for label, predicted in zip(gold, prediction))
    false_positive = sum(label == 0 and predicted == 1 for label, predicted in zip(gold, prediction))
    false_negative = sum(label == 1 and predicted == 0 for label, predicted in zip(gold, prediction))
    true_negative = sum(label == 0 and predicted == 0 for label, predicted in zip(gold, prediction))
    result = {
        "n": len(records),
        "positive": positive,
        "positive_rate": sum(gold) / len(gold) if gold else None,
        "accuracy": sum(label == predicted for label, predicted in zip(gold, prediction)) / max(1, len(gold)),
        "macro_f1": (
            class_f1(true_positive, false_positive, false_negative)
            + class_f1(true_negative, false_negative, false_positive)
        )
        / 2,
        "nll": sum(
            -math.log(max(min(probability_value if label else 1 - probability_value, 1 - 1e-12), 1e-12))
            for label, probability_value in zip(gold, probability)
        )
        / max(1, len(gold)),
        "brier": sum((probability_value - label) ** 2 for label, probability_value in zip(gold, probability))
        / max(1, len(gold)),
    }
    result["auroc"] = binary_auroc(gold, probability)
    result["auprc"] = binary_auprc(gold, probability)
    return result


def generation_metrics(records: list[dict[str, Any]], task: str) -> dict[str, Any]:
    if task in TAG_BY_TASK:
        tag = TAG_BY_TASK[task]
        parsed = [extract_tag(record["raw_output"], tag) for record in records]
        gold = [extract_tag(record["target"], tag) for record in records]
        result = {
            "n": len(records),
            "format_valid_rate": sum(value is not None for value in parsed) / max(1, len(records)),
            "normalized_exact_match": sum(
                normalize_text(prediction) == normalize_text(target)
                for prediction, target in zip(parsed, gold)
            )
            / max(1, len(records)),
        }
        if task == "predict_response":
            result["math_equivalence"] = sum(
                mathematically_equivalent(prediction, target) for prediction, target in zip(parsed, gold)
            ) / max(1, len(records))
        return result
    correctness = [extract_tag(record["raw_output"], "correctness") for record in records]
    response = [extract_tag(record["raw_output"], "predicted_response") for record in records]
    gold_correctness = [extract_tag(record["target"], "correctness") for record in records]
    gold_response = [extract_tag(record["target"], "predicted_response") for record in records]
    correctness_match = [
        normalize_text(prediction) == normalize_text(target)
        for prediction, target in zip(correctness, gold_correctness)
    ]
    response_match = [
        normalize_text(prediction) == normalize_text(target)
        for prediction, target in zip(response, gold_response)
    ]
    response_equivalence = [
        mathematically_equivalent(prediction, target) for prediction, target in zip(response, gold_response)
    ]
    return {
        "n": len(records),
        "format_valid_rate": sum(
            prediction_correctness is not None and prediction_response is not None
            for prediction_correctness, prediction_response in zip(correctness, response)
        )
        / max(1, len(records)),
        "correctness_accuracy": sum(correctness_match) / max(1, len(records)),
        "response_normalized_exact_match": sum(response_match) / max(1, len(records)),
        "response_math_equivalence": sum(response_equivalence) / max(1, len(records)),
        "joint_exact_match": sum(a and b for a, b in zip(correctness_match, response_match))
        / max(1, len(records)),
        "joint_math_equivalence": sum(a and b for a, b in zip(correctness_match, response_equivalence))
        / max(1, len(records)),
    }
