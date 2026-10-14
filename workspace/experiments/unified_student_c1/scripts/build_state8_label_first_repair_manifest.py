#!/usr/bin/env python3
"""Create an audited 500-row suffix for label-first State8 repair."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from build_state8_interleaved_retention_manifest import (
        deficit_interleave,
        prefix_audit,
        read_jsonl,
        sha256,
        stratum,
    )
except ImportError:
    from scripts.build_state8_interleaved_retention_manifest import (
        deficit_interleave,
        prefix_audit,
        read_jsonl,
        sha256,
        stratum,
    )


STAGE = "state8_label_first_repair_index_manifest_v1"


def choose(rows: list[dict[str, Any]], source: dict[str, Any]) -> list[int]:
    excluded = {int(value) for value in source.get("indices") or []}
    start = max(excluded, default=-1) + 1
    quotas = {
        ("diagnose_misconception", "eedi_mining"): 200,
        ("diagnose_misconception", "malrule"): 50,
        ("predict_error_persistence", "foundationalassist", "persistent"): 75,
        ("predict_error_persistence", "foundationalassist", "transient"): 75,
        ("predict_correctness", "dbe_kt22"): 8,
        ("predict_correctness", "eedi_response"): 8,
        ("predict_correctness", "foundationalassist"): 8,
        ("predict_correctness", "malrule"): 8,
        ("predict_correctness", "xes3g5m"): 8,
        ("predict_prospective_error", "dbe_kt22"): 17,
        ("predict_prospective_error", "foundationalassist"): 18,
        ("predict_response", "dbe_kt22"): 6,
        ("predict_response", "eedi_response"): 6,
        ("predict_response", "foundationalassist"): 7,
        ("predict_response", "malrule"): 6,
    }
    selected: list[int] = []
    counts: Counter[tuple[str, ...]] = Counter()
    for index in list(range(start, len(rows))) + list(range(0, start)):
        if index in excluded:
            continue
        key = stratum(rows[index])
        if key in quotas and counts[key] < quotas[key]:
            selected.append(index)
            counts[key] += 1
        if all(counts[key] == count for key, count in quotas.items()):
            break
    if counts != Counter(quotas):
        missing = {"/".join(key): quotas[key] - counts[key] for key in quotas if counts[key] != quotas[key]}
        raise ValueError(f"Unable to fill label-first repair quotas: {missing}")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--formal-eval-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    rows = read_jsonl(args.train_jsonl)
    formal = read_jsonl(args.formal_eval_jsonl)
    if source.get("stage") != "state8_interleaved_retention_index_manifest_v1":
        raise ValueError("Expected the frozen interleaved-retention source manifest")
    if source.get("source_train_sha256") != sha256(args.train_jsonl):
        raise ValueError("Source train hash drift")
    if source.get("formal_eval_sha256") != sha256(args.formal_eval_jsonl):
        raise ValueError("Formal evaluation hash drift")

    selected = choose(rows, source)
    ordered, strata = deficit_interleave(selected, rows)
    selected_ids = [str(rows[index]["example_id"]) for index in ordered]
    formal_ids = {str(row["example_id"]) for row in formal}
    source_ids = {str(rows[int(index)]["example_id"]) for index in source["indices"]}
    if formal_ids.intersection(selected_ids):
        raise ValueError("Label-first repair overlaps formal evaluation")
    if source_ids.intersection(selected_ids):
        raise ValueError("Label-first repair reuses the previous training segment")

    value = {
        "stage": STAGE,
        "branch": "state_only_label_first_causal_repair",
        "curriculum_mode": "deterministic_stratum_deficit_interleave",
        "source_manifest_stage": source["stage"],
        "source_manifest_sha256": sha256(args.source_manifest),
        "source_train_sha256": sha256(args.train_jsonl),
        "formal_eval_sha256": sha256(args.formal_eval_jsonl),
        "segment_rows": len(ordered),
        "start_global_position": int(source.get("start_global_position", 0)) + int(source["segment_rows"]),
        "indices": ordered,
        "selected_example_ids_sha256": hashlib.sha256("\n".join(selected_ids).encode()).hexdigest(),
        "previous_segment_example_id_overlap": 0,
        "formal_example_id_overlap": 0,
        "strata_counts": strata,
        "task_counts": dict(sorted(Counter(str(rows[index]["task"]) for index in ordered).items())),
        "source_counts": dict(sorted(Counter(str(rows[index]["source"]) for index in ordered).items())),
        "prefix_audit": prefix_audit(ordered, rows, [20, 100, 250, 500]),
        "state_token_count": 8,
        "final_inference_route": "strict_state_only",
        "diagnosis_field_order": ["label", "diagnosis", "evidence"],
        "early_label_query_mode": "state_only",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "rows": len(ordered), "prefix_audit": value["prefix_audit"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
