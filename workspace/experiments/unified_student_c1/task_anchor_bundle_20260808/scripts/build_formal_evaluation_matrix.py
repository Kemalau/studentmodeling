#!/usr/bin/env python3
"""Build the formal Base-vs-Unified-vs-specialist evaluation matrix."""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
from typing import Any

from formal_metric_utils import binary_metrics


TASK_NAMES = ["T1", "T2", "T3", "T5", "T6"]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def find_metric(root: Path, directory_name: str) -> Path:
    matches = [path for path in root.rglob("metrics.json") if path.parent.name == directory_name]
    if len(matches) != 1:
        raise ValueError(f"Expected one {directory_name}/metrics.json under {root}, found {matches}")
    return matches[0]


def weighted_metric(result: dict[str, Any], key: str) -> tuple[float, int]:
    groups = result["metrics_by_group"]
    numerator = 0.0
    denominator = 0
    for metrics in groups.values():
        if key not in metrics:
            continue
        count = int(metrics["n"])
        numerator += count * float(metrics[key])
        denominator += count
    if denominator == 0:
        raise ValueError(f"No groups contain {key}")
    return numerator / denominator, denominator


def classification_summary(path: Path, task: str, positive: str) -> dict[str, Any]:
    records = [row for row in read_jsonl(path) if row["task"] == task]
    return binary_metrics(records, positive)


def metric(value: float, n: int, secondary: float | None = None, secondary_name: str | None = None) -> dict[str, Any]:
    output = {"value": float(value), "n": int(n)}
    if secondary is not None:
        output["secondary"] = float(secondary)
        output["secondary_name"] = secondary_name
    return output


def wilson(value: float, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n <= 0:
        return value, value
    denominator = 1 + z * z / n
    center = (value + z * z / (2 * n)) / denominator
    margin = z * math.sqrt(value * (1 - value) / n + z * z / (4 * n * n)) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def cell_text(cell: dict[str, Any] | None) -> str:
    if cell is None:
        return "—"
    low, high = wilson(cell["value"], cell["n"])
    primary = f"{cell['value']:.3f} [{low:.3f}, {high:.3f}]"
    if "secondary" in cell:
        return f"{primary}; {cell['secondary_name']} {cell['secondary']:.3f}"
    return primary


def build_rows(
    classification_root: Path,
    generation_root: Path,
    specialist_root: Path,
    t3_root: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    base_predictions = next(classification_root.rglob("base_classification/predictions_redacted.jsonl"))
    adapter_predictions = next(classification_root.rglob("adapter_classification/predictions_redacted.jsonl"))
    base_t1 = classification_summary(base_predictions, "predict_correctness", "correct")
    adapter_t1 = classification_summary(adapter_predictions, "predict_correctness", "correct")
    base_t6 = classification_summary(base_predictions, "predict_error_persistence", "persistent")
    adapter_t6 = classification_summary(adapter_predictions, "predict_error_persistence", "persistent")

    base_t2_json = load_json(find_metric(generation_root, "base_predict_response"))
    adapter_t2_json = load_json(find_metric(generation_root, "adapter_predict_response"))
    t3_metric_root = t3_root or generation_root
    base_t3_json = load_json(find_metric(t3_metric_root, "base_diagnose_misconception"))
    adapter_t3_json = load_json(find_metric(t3_metric_root, "adapter_diagnose_misconception"))
    base_t5_json = load_json(find_metric(generation_root, "base_predict_prospective_error"))
    adapter_t5_json = load_json(find_metric(generation_root, "adapter_predict_prospective_error"))

    base_t2 = weighted_metric(base_t2_json, "math_equivalence")
    adapter_t2 = weighted_metric(adapter_t2_json, "math_equivalence")
    t3_metric_name = "llm_judge_accuracy" if t3_root else "normalized_exact_match"
    base_t3 = weighted_metric(base_t3_json, t3_metric_name)
    adapter_t3 = weighted_metric(adapter_t3_json, t3_metric_name)
    base_t5 = weighted_metric(base_t5_json, "joint_math_equivalence")
    adapter_t5 = weighted_metric(adapter_t5_json, "joint_math_equivalence")
    base_t5_correct = weighted_metric(base_t5_json, "correctness_accuracy")[0]
    adapter_t5_correct = weighted_metric(adapter_t5_json, "correctness_accuracy")[0]

    irt_pfa = load_json(specialist_root / "irt_pfa.json")
    dkt = load_json(specialist_root / "dkt.json")
    akt = load_json(specialist_root / "akt.json")
    bkt = load_json(specialist_root / "bkt.json")
    task_specialists = load_json(specialist_root / "task_specialists.json")
    t3_specialists = load_json(specialist_root / "t3_specialists.json")

    t2_specialist = task_specialists["T2"]["metrics"]["overall"]
    t5_specialist = task_specialists["T5"]["metrics"]["overall"]
    t6_specialist = task_specialists["T6"]["metrics"]["overall"]
    bge = t3_specialists["bge_full_taxonomy_eedi"]
    malrule_tfidf = t3_specialists["tfidf_linearsvc"]["by_source"]["malrule"]
    t3_suite_n = int(bge["n"]) + int(malrule_tfidf["n"])
    t3_suite_value = (
        float(bge["accuracy"]) * int(bge["n"])
        + float(malrule_tfidf["accuracy"]) * int(malrule_tfidf["n"])
    ) / t3_suite_n

    rows = [
        {
            "model": "Qwen3-32B Base",
            "scope": "Formal full test",
            "T1": metric(base_t1["accuracy"], base_t1["n"], base_t1["auroc"], "AUROC"),
            "T2": metric(base_t2[0], base_t2[1]),
            "T3": metric(base_t3[0], base_t3[1]),
            "T5": metric(base_t5[0], base_t5[1], base_t5_correct, "CorrAcc"),
            "T6": metric(base_t6["accuracy"], base_t6["n"], base_t6["auroc"], "AUROC"),
        },
        {
            "model": "Unified Qwen3-32B QLoRA",
            "scope": "Formal full test",
            "T1": metric(adapter_t1["accuracy"], adapter_t1["n"], adapter_t1["auroc"], "AUROC"),
            "T2": metric(adapter_t2[0], adapter_t2[1]),
            "T3": metric(adapter_t3[0], adapter_t3[1]),
            "T5": metric(adapter_t5[0], adapter_t5[1], adapter_t5_correct, "CorrAcc"),
            "T6": metric(adapter_t6["accuracy"], adapter_t6["n"], adapter_t6["auroc"], "AUROC"),
        },
        {
            "model": "Best traditional specialist suite",
            "scope": "Source/task specialist",
            "T1": metric(akt["test"]["accuracy"], akt["test"]["n"], akt["test"]["auc"], "AUROC"),
            "T2": metric(t2_specialist["math_equivalence"], t2_specialist["n"]),
            "T3": metric(t3_suite_value, t3_suite_n),
            "T5": metric(t5_specialist["joint_math_equivalence"], t5_specialist["n"], t5_specialist["correctness_accuracy"], "CorrAcc"),
            "T6": metric(t6_specialist["accuracy"], t6_specialist["n"], t6_specialist["auroc"], "AUROC"),
        },
        {
            "model": "Rasch IRT",
            "scope": "T1 full",
            "T1": metric(irt_pfa["rasch_irt"]["test"]["accuracy"], irt_pfa["rasch_irt"]["test"]["n"], irt_pfa["rasch_irt"]["test"]["auc"], "AUROC"),
        },
        {
            "model": "PFA",
            "scope": "T1 full",
            "T1": metric(irt_pfa["pfa"]["test"]["accuracy"], irt_pfa["pfa"]["test"]["n"], irt_pfa["pfa"]["test"]["auc"], "AUROC"),
        },
        {
            "model": "BKT",
            "scope": "T1 temporal/synthetic, n=9,929",
            "T1": metric(bkt["metrics"]["accuracy"], bkt["metrics"]["n"], bkt["metrics"]["auc"], "AUROC"),
        },
        {
            "model": "DKT item-LSTM",
            "scope": "T1 full",
            "T1": metric(dkt["baselines"]["dkt_item_lstm_mlp"]["test"]["accuracy"], dkt["baselines"]["dkt_item_lstm_mlp"]["test"]["n"], dkt["baselines"]["dkt_item_lstm_mlp"]["test"]["auc_positive_correct"], "AUROC"),
        },
        {
            "model": "AKT (official pyKT)",
            "scope": "T1 full",
            "T1": metric(akt["test"]["accuracy"], akt["test"]["n"], akt["test"]["auc"], "AUROC"),
        },
        {
            "model": "Conditional response specialist",
            "scope": "T2 full",
            "T2": metric(t2_specialist["math_equivalence"], t2_specialist["n"]),
        },
        {
            "model": "BGE-small full-taxonomy",
            "scope": "T3 Eedi, n=894",
            "T3": metric(bge["accuracy"], bge["n"], bge["mrr"], "MRR"),
        },
        {
            "model": "TF-IDF + LinearSVC",
            "scope": "T3 Malrule, n=600",
            "T3": metric(malrule_tfidf["accuracy"], malrule_tfidf["n"], malrule_tfidf["macro_f1"], "MacroF1"),
        },
        {
            "model": "PFA + conditioned response mode",
            "scope": "T5 full",
            "T5": metric(t5_specialist["joint_math_equivalence"], t5_specialist["n"], t5_specialist["correctness_accuracy"], "CorrAcc"),
        },
        {
            "model": "Persistence logistic",
            "scope": "T6 strict full",
            "T6": metric(t6_specialist["accuracy"], t6_specialist["n"], t6_specialist["auroc"], "AUROC"),
        },
    ]
    summary = {
        "unified_minus_best_specialist": {
            "T1_accuracy": adapter_t1["accuracy"] - akt["test"]["accuracy"],
            "T2_math_equivalence": adapter_t2[0] - t2_specialist["math_equivalence"],
            "T3_semantic_accuracy": adapter_t3[0] - t3_suite_value,
            "T5_joint_math_equivalence": adapter_t5[0] - t5_specialist["joint_math_equivalence"],
            "T6_accuracy": adapter_t6["accuracy"] - t6_specialist["accuracy"],
        }
    }
    return rows, summary


def markdown(rows: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    lines = [
        "# Formal Evaluation Matrix",
        "",
        "Primary metrics: T1/T6 accuracy, T2 mathematical equivalence, T3 semantic diagnosis accuracy, T5 joint correctness + mathematical equivalence. Brackets are Wilson 95% intervals for the primary metric.",
        "",
        "| Model | Scope | T1 Acc | T2 MathEq | T3 Semantic | T5 JointMath | T6 Acc |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| " + " | ".join(
                [row["model"], row["scope"]] + [cell_text(row.get(task)) for task in TASK_NAMES]
            ) + " |"
        )
    deltas = summary["unified_minus_best_specialist"]
    lines.extend(
        [
            "",
            "## Unified minus best traditional specialist",
            "",
            *[f"- {name}: {value:+.3f}" for name, value in deltas.items()],
            "",
            "## Protocol notes",
            "",
            "- T3 open generations use blinded GPT-5.5 semantic-equivalence judgments; normalized exact match remains a diagnostic secondary metric.",
            "- T3 specialists emit canonical taxonomy labels and use exact class accuracy, while LLM generations use semantic judgments; cross-paradigm differences are indicative rather than identical-score comparisons.",
            "- The T3 specialist suite uses BGE-small over the full 2,586-label Eedi taxonomy and TF-IDF + LinearSVC for Malrule; it never receives per-example oracle candidates.",
            "- BKT excludes the non-temporal Eedi Response profile protocol; its T1 cell therefore uses 9,929 examples rather than the full 10,913.",
            "- T5's specialist is a PFA-style correctness model plus item/student conditioned response modes; it is an explicit task adaptation, not a canonical universal KT architecture.",
        ]
    )
    return "\n".join(lines) + "\n"


def svg(rows: list[dict[str, Any]], output: Path) -> None:
    widths = [330, 300, 250, 220, 220, 250, 250]
    headers = ["Model", "Scope", "T1 Acc", "T2 MathEq", "T3 Semantic", "T5 JointMath", "T6 Acc"]
    row_height = 42
    width = sum(widths)
    height = row_height * (len(rows) + 2)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#17202a}.h{font-weight:700;font-size:15px}.c{font-size:13px}.key{font-weight:700;fill:#0b5345}.line{stroke:#d5d8dc;stroke-width:1}</style>',
    ]
    x = 0
    for header, cell_width in zip(headers, widths):
        parts.append(f'<rect x="{x}" y="0" width="{cell_width}" height="{row_height}" fill="#e8f6f3"/>')
        parts.append(f'<text class="h" x="{x + 8}" y="26">{html.escape(header)}</text>')
        x += cell_width
    for row_index, row in enumerate(rows, 1):
        y = row_index * row_height
        fill = "#f8f9f9" if row_index % 2 == 0 else "#ffffff"
        if row_index <= 3:
            fill = ["#fef9e7", "#eafaf1", "#ebf5fb"][row_index - 1]
        parts.append(f'<rect x="0" y="{y}" width="{width}" height="{row_height}" fill="{fill}"/>')
        values = [row["model"], row["scope"]] + [cell_text(row.get(task)) for task in TASK_NAMES]
        x = 0
        for column_index, (value, cell_width) in enumerate(zip(values, widths)):
            css_class = "key" if row_index <= 3 and column_index == 0 else "c"
            parts.append(f'<text class="{css_class}" x="{x + 8}" y="{y + 26}">{html.escape(value)}</text>')
            x += cell_width
        parts.append(f'<line class="line" x1="0" y1="{y + row_height}" x2="{width}" y2="{y + row_height}"/>')
    parts.append("</svg>")
    output.write_text("\n".join(parts) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--classification-root", type=Path, required=True)
    parser.add_argument("--generation-root", type=Path, required=True)
    parser.add_argument("--t3-root", type=Path)
    parser.add_argument("--specialist-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rows, summary = build_rows(args.classification_root, args.generation_root, args.specialist_root, args.t3_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {"rows": rows, "summary": summary}
    (args.output_dir / "formal_evaluation_matrix.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "formal_evaluation_matrix.md").write_text(markdown(rows, summary), encoding="utf-8")
    svg(rows, args.output_dir / "formal_evaluation_matrix.svg")
    print(markdown(rows, summary))


if __name__ == "__main__":
    main()
