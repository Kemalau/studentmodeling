#!/usr/bin/env python3
"""Build deterministic train-only T1-retention diagnostic curricula."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SOURCE_SHA256 = "9961f746cb17e6c314e182d3453b6157022fb48b8de44d2e68bc8e87a1131ccd"
VALID_SHA256 = "78bfce75576dc3873cbb6459669a952adeedb8b77bda15f281af9519e2506d14"
FORMAL_SHA256 = "a368cefb0c49cb6a305ad1a892808361fba2d44d555a0a28ea64e56babe32525"
START_GLOBAL_POSITION = 12_000
SEGMENT_ROWS = 1_000

BRANCH_SPECS = {
    "t1_heavy": {
        "predict_correctness": {
            "dbe_kt22": 65,
            "eedi_response": 130,
            "foundationalassist": 130,
            "malrule": 65,
            "xes3g5m": 260,
        },
        "predict_response": {"dbe_kt22": 25, "eedi_response": 25, "foundationalassist": 25, "malrule": 25},
        "diagnose_misconception": {"eedi_mining": 75, "malrule": 75},
        "predict_prospective_error": {"dbe_kt22": 25, "foundationalassist": 25},
        "predict_error_persistence": {"dbe_kt22": 25, "foundationalassist": 25},
    },
    "balanced_all": {
        "predict_correctness": {
            "dbe_kt22": 40,
            "eedi_response": 80,
            "foundationalassist": 80,
            "malrule": 40,
            "xes3g5m": 160,
        },
        "predict_response": {"dbe_kt22": 15, "eedi_response": 20, "foundationalassist": 20, "malrule": 15},
        "diagnose_misconception": {"eedi_mining": 150, "malrule": 100},
        "predict_prospective_error": {"dbe_kt22": 15, "foundationalassist": 15},
        "predict_error_persistence": {"dbe_kt22": 100, "foundationalassist": 150},
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build(
    rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    formal_rows: list[dict[str, Any]],
    *,
    branch: str,
    seed: int,
) -> dict[str, Any]:
    spec = BRANCH_SPECS[branch]
    pools: dict[tuple[str, str, str | None], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        metadata = row.get("original_metadata") or {}
        if metadata.get("original_split") != "train":
            continue
        task, source = str(row["task"]), str(row["source"])
        label = str(row.get("target_label")) if task == "predict_correctness" else None
        pools[(task, source, label)].append(index)

    selected: list[int] = []
    used: set[int] = set()
    for task, sources in spec.items():
        for source, count in sources.items():
            if task == "predict_correctness":
                negative = count // 2
                positive = count - negative
                requests = (("incorrect", negative), ("correct", positive))
            else:
                requests = ((None, count),)
            for label, requested in requests:
                key = (task, source, label)
                candidates = [value for value in pools[key] if value not in used]
                random.Random(f"{seed}:{branch}:{key}").shuffle(candidates)
                if len(candidates) < requested:
                    raise ValueError(f"Insufficient pool {key}: {len(candidates)} < {requested}")
                chosen = candidates[:requested]
                selected.extend(chosen)
                used.update(chosen)
    if len(selected) != SEGMENT_ROWS or len(set(selected)) != SEGMENT_ROWS:
        raise AssertionError((branch, len(selected), len(set(selected))))
    random.Random(f"{seed}:{branch}:order").shuffle(selected)

    selected_ids = [str(rows[index]["example_id"]) for index in selected]
    validation_ids = {str(row["example_id"]) for row in validation_rows}
    formal_ids = {str(row["example_id"]) for row in formal_rows}
    if validation_ids.intersection(selected_ids):
        raise ValueError("Training curriculum overlaps validation")
    if formal_ids.intersection(selected_ids):
        raise ValueError("Training curriculum overlaps formal test")
    task_counts = Counter(str(rows[index]["task"]) for index in selected)
    source_counts = Counter(str(rows[index]["source"]) for index in selected)
    t1_labels = Counter(
        str(rows[index].get("target_label"))
        for index in selected
        if rows[index]["task"] == "predict_correctness"
    )
    return {
        "stage": "layerwise_t1_retention_index_manifest_v1",
        "branch": branch,
        "seed": seed,
        "source_train_sha256": SOURCE_SHA256,
        "validation_sha256": VALID_SHA256,
        "formal_eval_sha256": FORMAL_SHA256,
        "validation_example_id_overlap": 0,
        "formal_example_id_overlap": 0,
        "start_global_position": START_GLOBAL_POSITION,
        "segment_rows": SEGMENT_ROWS,
        "selected_example_ids_sha256": hashlib.sha256("\n".join(selected_ids).encode()).hexdigest(),
        "task_counts": dict(sorted(task_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "t1_label_counts": dict(sorted(t1_labels.items())),
        "indices": selected,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--validation-jsonl", type=Path, required=True)
    parser.add_argument("--formal-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    expected = ((args.train_jsonl, SOURCE_SHA256), (args.validation_jsonl, VALID_SHA256), (args.formal_jsonl, FORMAL_SHA256))
    for path, digest in expected:
        if sha256(path) != digest:
            raise ValueError(f"SHA256 drift: {path}")
    rows = read_jsonl(args.train_jsonl)
    validation = read_jsonl(args.validation_jsonl)
    formal = read_jsonl(args.formal_jsonl)
    if len(rows) != 259_845 or len(validation) != 15_514 or len(formal) != 21_676:
        raise ValueError("Support drift")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for branch in BRANCH_SPECS:
        value = build(rows, validation, formal, branch=branch, seed=args.seed)
        path = args.output_dir / f"layerwise_t1_retention_{branch}_seed42_v1.json"
        path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"path": str(path), "branch": branch, "task_counts": value["task_counts"], "source_counts": value["source_counts"], "t1_label_counts": value["t1_label_counts"]}, sort_keys=True))


if __name__ == "__main__":
    main()
