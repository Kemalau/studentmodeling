#!/usr/bin/env python3
"""Combine disjoint formation and consolidation curricula for final promotion."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formation", type=Path, required=True)
    parser.add_argument("--consolidation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    formation = read(args.formation)
    consolidation = read(args.consolidation)
    if formation.get("start_global_position") != 12_000:
        raise ValueError("Formation boundary drift")
    if consolidation.get("start_global_position") != 13_000:
        raise ValueError("Consolidation boundary drift")
    first = [int(value) for value in formation["indices"]]
    second = [int(value) for value in consolidation["indices"]]
    indices = first + second
    if len(first) != 1_000 or len(second) != 1_000 or len(set(indices)) != 2_000:
        raise ValueError("Two-stage support is not disjoint 1000+1000")

    def sum_counts(key: str) -> dict[str, int]:
        total: Counter[str] = Counter()
        total.update({str(k): int(v) for k, v in formation[key].items()})
        total.update({str(k): int(v) for k, v in consolidation[key].items()})
        return dict(sorted(total.items()))

    t6 = Counter(formation["t6_label_counts"])
    t6.update(consolidation["t6_label_counts"])
    value = {
        "stage": "layerwise_task_retention_index_manifest_v1",
        "branch": "equal_t3_t6",
        "seed": 42,
        "source_train_sha256": formation["source_train_sha256"],
        "formal_eval_sha256": formation["formal_eval_sha256"],
        "formal_example_id_overlap": 0,
        "start_global_position": 12_000,
        "segment_rows": 2_000,
        "selected_example_ids_sha256": hashlib.sha256(
            (
                str(formation["selected_example_ids_sha256"])
                + ":"
                + str(consolidation["selected_example_ids_sha256"])
            ).encode()
        ).hexdigest(),
        "task_counts": sum_counts("task_counts"),
        "source_counts": sum_counts("source_counts"),
        "t6_label_counts": dict(sorted((str(k), int(v)) for k, v in t6.items())),
        "indices": indices,
        "curriculum_mode": "open_t3_formation_then_causal_task_consolidation",
        "curriculum_phases": [
            {"name": "open_t3_formation", "start_offset": 0, "end_offset": 999, "rows": 1000},
            {"name": "causal_task_consolidation", "start_offset": 1000, "end_offset": 1999, "rows": 1000},
        ],
        "formation_manifest": str(args.formation),
        "consolidation_manifest": str(args.consolidation),
    }
    if formation["source_train_sha256"] != consolidation["source_train_sha256"]:
        raise ValueError("Two-stage train hash drift")
    if formation["formal_eval_sha256"] != consolidation["formal_eval_sha256"]:
        raise ValueError("Two-stage formal hash drift")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: value[k] for k in ("segment_rows", "task_counts", "source_counts", "t6_label_counts", "curriculum_phases")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
