#!/usr/bin/env python3
"""Validate the T4 rescore, compute paired statistics, and rebuild the matrix."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rescore-dir", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    state = args.root / "results/taskonly_qwen_formal_eval_20260813"
    scored = next(args.rescore_dir.rglob("unified_t4_scored.jsonl"))
    receipt_path = next(args.rescore_dir.rglob("completion.json"))
    receipt = json.loads(receipt_path.read_text())
    rows = read_jsonl(scored)
    assert receipt["stage"] == "unified_t4_paired_rescore_v1" and receipt["status"] == "complete"
    assert receipt["contains_answer_text"] is False
    assert len(rows) == 4600 and len({row["example_id"] for row in rows}) == 4600
    assert all(isinstance(row.get("is_correct"), bool) for row in rows)
    assert not any("raw_output" in row or "target" in row for row in rows)
    assert sum(row["is_correct"] for row in rows) == int(receipt["correct"])
    # The rerun must reproduce the archived aggregate score exactly.
    matrix = json.loads((state / "formal_specialist_v3/formal_evaluation_matrix.json").read_text())
    unified = next(row for row in matrix["rows"] if row["model"] == "Unified Qwen3-32B QLoRA")
    assert abs(float(receipt["primary"]) - float(unified["T4"]["value"])) < 1e-15, (receipt, unified["T4"])

    taskonly = next((state / "T4/download").rglob("eval/T4/predictions.jsonl"))
    paired_t4 = state / "paired/T4.json"
    subprocess.run([
        "python", "scripts/compare_taskonly_unified_paired.py",
        "--task", "T4", "--unified", str(scored), "--taskonly", str(taskonly),
        "--output", str(paired_t4),
    ], cwd=args.root, check=True)

    paired_path = state / "paired/complete.json"
    paired = json.loads(paired_path.read_text())
    paired["tasks"]["T4"] = json.loads(paired_t4.read_text())
    paired_path.write_text(json.dumps(paired, indent=2, sort_keys=True) + "\n")
    subprocess.run([
        "python", "scripts/build_formal_evaluation_matrix.py",
        "--classification-root", "results/formal_specialist_v1/qwen_amlt",
        "--generation-root", "results/formal_specialist_v1/qwen_amlt",
        "--t3-root", "results/formal_specialist_v2/unrestricted_qwen_t3_v2",
        "--specialist-root", "results/formal_specialist_v1",
        "--taskonly-summary", str(state / "summary/summary.json"),
        "--paired-summary", str(paired_path),
        "--output-dir", str(state / "formal_specialist_v3"),
    ], cwd=args.root, check=True, stdout=subprocess.DEVNULL)
    (state / "unified_t4_paired_rescore_complete").touch()
    print(json.dumps(paired["tasks"]["T4"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
