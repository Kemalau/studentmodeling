#!/usr/bin/env python3
"""Build a matrix containing only published or clearly named strong baselines."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


TASKS = ["T1", "T2", "T3", "T4", "T5", "T6"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--t1-dir", type=Path, required=True)
    parser.add_argument("--t3-dir", type=Path, required=True)
    parser.add_argument("--simple-json", type=Path, required=True)
    parser.add_argument("--pending-json", type=Path, required=True)
    parser.add_argument("--qwen-json", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    dkt = json.loads((args.t1_dir / "dkt_results.json").read_text())
    classical = json.loads((args.t1_dir / "irt_pfa_results.json").read_text())
    bkt = json.loads((args.t1_dir / "bkt_results.json").read_text())
    akt = json.loads((args.t1_dir / "akt_results.json").read_text())
    bge = json.loads((args.t3_dir / "bge_small_results.json").read_text())
    simple = json.loads(args.simple_json.read_text())
    pending = json.loads(args.pending_json.read_text())
    qwen = json.loads(args.qwen_json.read_text())["model"]["metrics"]
    malrule = simple["models"]["task_heuristic"]["by_task_source"]["diagnose_misconception/malrule"]

    def qwen_cell(task: str) -> str:
        metrics = qwen[task]
        return f"{metrics['accuracy']:.3f} / F1 {metrics['macro_f1']:.3f}"

    rows = [
        ("Rasch IRT", f"{classical['rasch_irt']['test']['accuracy']:.3f} / {classical['rasch_irt']['test']['auc']:.3f}", "N/A", "N/A", "N/A", "N/A", "N/A"),
        ("PFA", f"{classical['pfa']['test']['accuracy']:.3f} / {classical['pfa']['test']['auc']:.3f}", "N/A", "N/A", f"{pending['T4']['accuracy']:.3f} / F1 {pending['T4']['macro_f1']:.3f}", "N/A", "N/A"),
        ("BKT (temporal sources)", f"{bkt['metrics']['accuracy']:.3f} / {bkt['metrics']['auc']:.3f}", "N/A", "N/A", "N/A", "N/A", "N/A"),
        ("DKT skill-LSTM", f"{dkt['baselines']['dkt_skill_lstm']['test']['accuracy']:.3f} / {dkt['baselines']['dkt_skill_lstm']['test']['auc_positive_correct']:.3f}", "N/A", "N/A", "N/A", "N/A", "N/A"),
        ("DKT item-LSTM", f"{dkt['baselines']['dkt_item_lstm_mlp']['test']['accuracy']:.3f} / {dkt['baselines']['dkt_item_lstm_mlp']['test']['auc_positive_correct']:.3f}", "N/A", "N/A", "N/A", "N/A", "N/A"),
        ("AKT (official pyKT)", f"{akt['test']['accuracy']:.3f} / {akt['test']['auc']:.3f}", "N/A", "N/A", "N/A", "N/A", "N/A"),
        ("BGE-small oracle-candidate retrieval†", "N/A", "N/A", f"{bge['accuracy']:.3f} / MRR {bge['mrr']:.3f}", "N/A", "N/A", "N/A"),
        ("TF-IDF + LinearSVC (Malrule)", "N/A", "N/A", f"{malrule['accuracy']:.3f} / F1 {malrule['macro_f1']:.3f}", "N/A", "N/A", "N/A"),
        ("Conditional response logit", "N/A", f"{pending['T2']['accuracy']:.3f} / F1 {pending['T2']['macro_f1']:.3f}", "N/A", "N/A", "N/A", "N/A"),
        ("PFA + item distractor", "N/A", "N/A", "N/A", "N/A", f"{pending['T5']['exact_response']['accuracy']:.3f} / AUC {pending['T5']['correctness']['auc']:.3f}", "N/A"),
        ("Persistence logistic", "N/A", "N/A", "N/A", "N/A", "N/A", f"{pending['T6']['accuracy']:.3f} / AUC {pending['T6']['auc']:.3f}"),
        (
            "Qwen3-32B mixed QLoRA (full fast eval)",
            qwen_cell("predict_correctness"),
            qwen_cell("predict_response"),
            qwen_cell("diagnose_misconception"),
            qwen_cell("estimate_mastery"),
            qwen_cell("predict_prospective_error"),
            qwen_cell("predict_error_persistence"),
        ),
    ]

    lines = [
        "# Published Strong Baseline Matrix",
        "",
        "T1 cells are `Accuracy / AUC`. T3 cells are source-specific as indicated. `N/A` means the model is not defined for that task; task adaptations are listed as separate model rows.",
        "",
        "| Model | T1 | T2 | T3 | T4 | T5 | T6 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    lines.extend(
        [
            "",
            "## Scope Notes",
            "",
            "- BKT excludes Eedi Response because that mirror has no temporal order.",
            "- † BGE-small uses each row's oracle set containing the gold misconception plus 31 negatives. It is a closed-set retrieval diagnostic and is not a real-world open-generation baseline.",
            "- BGE-small is evaluated on the 894 Eedi-Mining T3 examples; LinearSVC is evaluated on the 300 Malrule T3 examples.",
            "- T2 uses a candidate-level conditional response logit; T5 uses a PFA-style correctness model plus item distractor choice; T6 uses a persistence logistic survival proxy.",
            "- These explicit adaptations are not presented as canonical SOTA because T2/T5 exact-response and T6 persistence do not have universally accepted traditional KT architectures.",
            "- Qwen3-32B is evaluated on all 10,544 test rows with cached-prefix candidate-label log-probability scoring and maximum context length 1,536. Its T3 cell is also closed-set candidate ranking, not open generation.",
        ]
    )

    width, left, top, cell_w, cell_h = 1120, 300, 75, 130, 48
    height = top + len(rows) * cell_h + 50
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#172033}.h{font-size:16px;font-weight:700}.l{font-size:13px}.v{font-size:12px;font-weight:600}</style>',
        '<text x="24" y="32" class="h">Published Strong Baselines</text>',
    ]
    for column, task in enumerate(TASKS):
        svg.append(f'<text x="{left + column * cell_w + cell_w / 2}" y="62" text-anchor="middle" class="h">{task}</text>')
    for row_index, row in enumerate(rows):
        y = top + row_index * cell_h
        svg.append(f'<text x="24" y="{y + 29}" class="l">{html.escape(row[0])}</text>')
        for column, value in enumerate(row[1:]):
            x = left + column * cell_w
            fill = "#dbeafe" if value not in {"N/A", "pending", "training", "evaluating"} else "#f1f5f9"
            svg.append(f'<rect x="{x + 2}" y="{y + 2}" width="{cell_w - 4}" height="{cell_h - 4}" rx="6" fill="{fill}"/>')
            svg.append(f'<text x="{x + cell_w / 2}" y="{y + 29}" text-anchor="middle" class="v">{html.escape(value)}</text>')
    svg.append("</svg>")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "evaluation_matrix.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.out_dir / "evaluation_matrix.svg").write_text("\n".join(svg) + "\n", encoding="utf-8")
    (args.out_dir / "evaluation_matrix.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
