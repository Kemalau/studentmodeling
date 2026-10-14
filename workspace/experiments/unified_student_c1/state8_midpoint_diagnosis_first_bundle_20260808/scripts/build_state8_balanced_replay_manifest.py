#!/usr/bin/env python3
"""Build a deterministic 5k unified replay hedge for qualified State8."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SOURCE_SHA256 = "9961f746cb17e6c314e182d3453b6157022fb48b8de44d2e68bc8e87a1131ccd"
FORMAL_SHA256 = "a368cefb0c49cb6a305ad1a892808361fba2d44d555a0a28ea64e56babe32525"
SEED = 42

QUOTAS: dict[tuple[str, str, str | None], int] = {
    ("diagnose_misconception", "eedi_mining", None): 1_738,
    ("diagnose_misconception", "malrule", None): 762,
    ("predict_correctness", "dbe_kt22", None): 160,
    ("predict_correctness", "eedi_response", None): 160,
    ("predict_correctness", "foundationalassist", None): 160,
    ("predict_correctness", "malrule", None): 160,
    ("predict_correctness", "xes3g5m", None): 160,
    ("predict_response", "dbe_kt22", None): 150,
    ("predict_response", "eedi_response", None): 150,
    ("predict_response", "foundationalassist", None): 150,
    ("predict_response", "malrule", None): 150,
    ("predict_prospective_error", "dbe_kt22", None): 300,
    ("predict_prospective_error", "foundationalassist", None): 300,
    ("predict_error_persistence", "dbe_kt22", "persistent"): 15,
    ("predict_error_persistence", "dbe_kt22", "transient"): 15,
    ("predict_error_persistence", "foundationalassist", "persistent"): 235,
    ("predict_error_persistence", "foundationalassist", "transient"): 235,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def persistence_label(row: dict[str, Any]) -> str | None:
    if row.get("task") != "predict_error_persistence":
        return None
    target = str(row.get("target"))
    if target == "<persistence>persistent</persistence>":
        return "persistent"
    if target == "<persistence>transient</persistence>":
        return "transient"
    raise ValueError(f"Unexpected persistence target: {target}")


def formal_ids(path: Path) -> set[str]:
    values: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                values.add(str(value["example_id"]))
    if len(values) != 21_676:
        raise ValueError(f"Formal denominator drift: {len(values)}")
    return values


def build(source: Path, formal: Path, seed: int = SEED) -> dict[str, Any]:
    held_out = formal_ids(formal)
    pools: dict[tuple[str, str, str | None], list[tuple[int, str]]] = defaultdict(list)
    source_rows = 0
    with source.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line); source_rows += 1
            key = (str(row["task"]), str(row["source"]), persistence_label(row))
            if key in QUOTAS and str(row["example_id"]) not in held_out:
                pools[key].append((index, str(row["example_id"])))
    if source_rows != 259_845:
        raise ValueError(f"Source denominator drift: {source_rows}")
    selected: list[tuple[int, str]] = []
    for key, count in QUOTAS.items():
        candidates = list(pools[key])
        random.Random(f"state8-balanced:{seed}:{key}").shuffle(candidates)
        if len(candidates) < count:
            raise ValueError(f"Insufficient pool {key}: need={count} got={len(candidates)}")
        selected.extend(candidates[:count])
    if len(selected) != 5_000 or len({index for index, _ in selected}) != 5_000:
        raise AssertionError("Balanced replay support drift")
    random.Random(f"state8-balanced:{seed}:final-order").shuffle(selected)
    indices = [index for index, _ in selected]
    ids = [example_id for _, example_id in selected]
    task_counts = Counter(key[0] for key, count in QUOTAS.items() for _ in range(count))
    source_counts = Counter(key[1] for key, count in QUOTAS.items() for _ in range(count))
    t6_counts = Counter(
        key[2] for key, count in QUOTAS.items() for _ in range(count) if key[2] is not None
    )
    return {
        "stage": "state8_balanced_replay_index_manifest_v1",
        "branch": "qualified_teacher_balanced_replay",
        "seed": seed,
        "segment_rows": 5_000,
        "source_train_sha256": SOURCE_SHA256,
        "formal_eval_sha256": FORMAL_SHA256,
        "formal_example_id_overlap": 0,
        "selected_example_ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
        "task_counts": dict(sorted(task_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "t6_label_counts": dict(sorted(t6_counts.items())),
        "quota_counts": {"/".join(str(part) for part in key): count for key, count in sorted(QUOTAS.items())},
        "indices": indices,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-jsonl", type=Path, required=True)
    parser.add_argument("--formal-eval-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    if sha256(args.source_jsonl) != SOURCE_SHA256:
        raise ValueError("Source SHA256 drift")
    if sha256(args.formal_eval_jsonl) != FORMAL_SHA256:
        raise ValueError("Formal SHA256 drift")
    value = build(args.source_jsonl, args.formal_eval_jsonl, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value[key] for key in ("stage", "segment_rows", "task_counts", "source_counts", "t6_label_counts")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
