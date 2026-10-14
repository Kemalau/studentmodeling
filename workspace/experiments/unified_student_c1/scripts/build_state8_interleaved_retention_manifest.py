#!/usr/bin/env python3
"""Interleave a State8 curriculum without changing its example support."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any


STAGE = "state8_interleaved_retention_index_manifest_v1"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stratum(row: dict[str, Any]) -> tuple[str, ...]:
    task = str(row.get("task") or "")
    source = str(row.get("source") or "")
    if task == "predict_error_persistence":
        target = str(row.get("target") or "")
        if target == "<persistence>persistent</persistence>":
            label = "persistent"
        elif target == "<persistence>transient</persistence>":
            label = "transient"
        else:
            raise ValueError(f"Unexpected T6 target: {target}")
        return task, source, label
    return task, source


def deficit_interleave(
    indices: list[int], rows: list[dict[str, Any]]
) -> tuple[list[int], dict[str, int]]:
    pools: dict[tuple[str, ...], deque[int]] = defaultdict(deque)
    for index in indices:
        pools[stratum(rows[index])].append(index)
    totals = {key: len(values) for key, values in pools.items()}
    emitted = Counter()
    ordered: list[int] = []
    total = len(indices)
    keys = sorted(pools)
    while len(ordered) < total:
        position = len(ordered) + 1
        available = [key for key in keys if pools[key]]
        key = max(
            available,
            key=lambda value: (
                position * totals[value] / total - emitted[value],
                tuple(reversed(value)),
            ),
        )
        ordered.append(pools[key].popleft())
        emitted[key] += 1
    rendered = {"/".join(key): count for key, count in sorted(totals.items())}
    return ordered, rendered


def prefix_audit(
    indices: list[int], rows: list[dict[str, Any]], checkpoints: list[int]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for step in checkpoints:
        selected = [rows[index] for index in indices[:step]]
        tasks = Counter(str(row["task"]) for row in selected)
        t6 = Counter(
            stratum(row)[-1]
            for row in selected
            if row["task"] == "predict_error_persistence"
        )
        result[str(step)] = {
            "task_counts": dict(sorted(tasks.items())),
            "t6_label_counts": dict(sorted(t6.items())),
        }
    return result


def build(
    source: dict[str, Any], rows: list[dict[str, Any]], formal_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    indices = [int(value) for value in source.get("indices") or []]
    if len(indices) != int(source.get("segment_rows", -1)):
        raise ValueError("Source manifest cardinality drift")
    if len(indices) != len(set(indices)) or min(indices) < 0 or max(indices) >= len(rows):
        raise ValueError("Source manifest indices are invalid")
    ordered, strata = deficit_interleave(indices, rows)
    if set(ordered) != set(indices) or len(ordered) != len(indices):
        raise AssertionError("Interleaving changed source support")
    selected_ids = [str(rows[index]["example_id"]) for index in ordered]
    formal_ids = {str(row["example_id"]) for row in formal_rows}
    overlap = formal_ids.intersection(selected_ids)
    if overlap:
        raise ValueError(f"Interleaved train/test overlap: {sorted(overlap)[:5]}")
    output = dict(source)
    output.update(
        stage=STAGE,
        branch="interleaved_retention_early_label",
        curriculum_mode="deterministic_stratum_deficit_interleave",
        source_manifest_stage=source.get("stage"),
        source_manifest_branch=source.get("branch"),
        source_selected_example_ids_sha256=source.get("selected_example_ids_sha256"),
        selected_example_ids_sha256=hashlib.sha256(
            "\n".join(selected_ids).encode()
        ).hexdigest(),
        support_unchanged=True,
        order_changed=ordered != indices,
        formal_example_id_overlap=0,
        strata_counts=strata,
        prefix_audit=prefix_audit(ordered, rows, [20, 100, 240, 500, len(ordered)]),
        indices=ordered,
    )
    output.pop("curriculum_phases", None)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--formal-eval-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    if source.get("branch") != "staged_t3_t6" or int(source.get("segment_rows", -1)) != 1000:
        raise ValueError("Expected the frozen staged_t3_t6 source manifest")
    if source.get("source_train_sha256") != sha256(args.train_jsonl):
        raise ValueError("Source train hash drift")
    if source.get("formal_eval_sha256") != sha256(args.formal_eval_jsonl):
        raise ValueError("Formal evaluation hash drift")
    value = build(source, read_jsonl(args.train_jsonl), read_jsonl(args.formal_eval_jsonl))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "stage": value["stage"],
        "support_unchanged": value["support_unchanged"],
        "prefix_audit": value["prefix_audit"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
