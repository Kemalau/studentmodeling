#!/usr/bin/env python3
"""Build a deterministic, student-coherent small dataset for method iteration."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


Stratum = tuple[str, str]


SMALL_QUOTAS: dict[str, dict[Stratum, int]] = {
    "train": {
        ("predict_correctness", "dbe_kt22"): 500,
        ("predict_correctness", "eedi_response"): 700,
        ("predict_correctness", "foundationalassist"): 1000,
        ("predict_correctness", "malrule"): 700,
        ("predict_correctness", "xes3g5m"): 1200,
        ("predict_response", "dbe_kt22"): 500,
        ("predict_response", "eedi_response"): 700,
        ("predict_response", "foundationalassist"): 1000,
        ("predict_response", "malrule"): 700,
        ("diagnose_misconception", "eedi_mining"): 1000,
        ("diagnose_misconception", "malrule"): 700,
        ("predict_prospective_error", "dbe_kt22"): 500,
        ("predict_prospective_error", "foundationalassist"): 1800,
        ("predict_error_persistence", "dbe_kt22"): 114,
        ("predict_error_persistence", "foundationalassist"): 886,
    },
    "valid": {
        ("predict_correctness", "dbe_kt22"): 120,
        ("predict_correctness", "eedi_response"): 200,
        ("predict_correctness", "foundationalassist"): 300,
        ("predict_correctness", "malrule"): 100,
        ("predict_correctness", "xes3g5m"): 300,
        ("predict_response", "dbe_kt22"): 120,
        ("predict_response", "eedi_response"): 200,
        ("predict_response", "foundationalassist"): 300,
        ("predict_response", "malrule"): 100,
        ("diagnose_misconception", "eedi_mining"): 200,
        ("diagnose_misconception", "malrule"): 100,
        ("predict_prospective_error", "dbe_kt22"): 120,
        ("predict_prospective_error", "foundationalassist"): 500,
        ("predict_error_persistence", "dbe_kt22"): 15,
        ("predict_error_persistence", "foundationalassist"): 225,
    },
    "test": {
        ("predict_correctness", "dbe_kt22"): 200,
        ("predict_correctness", "eedi_response"): 200,
        ("predict_correctness", "foundationalassist"): 200,
        ("predict_correctness", "malrule"): 200,
        ("predict_correctness", "xes3g5m"): 200,
        ("predict_response", "dbe_kt22"): 150,
        ("predict_response", "eedi_response"): 150,
        ("predict_response", "foundationalassist"): 150,
        ("predict_response", "malrule"): 150,
        ("diagnose_misconception", "eedi_mining"): 300,
        ("diagnose_misconception", "malrule"): 200,
        ("predict_prospective_error", "dbe_kt22"): 150,
        ("predict_prospective_error", "foundationalassist"): 450,
        ("predict_error_persistence", "dbe_kt22"): 12,
        ("predict_error_persistence", "foundationalassist"): 188,
    },
}


def stable_hash(seed: int, *parts: object) -> int:
    payload = "\x1f".join([str(seed), *(str(part) for part in parts)]).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def row_source(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") or {}
    return str(metadata.get("mix_source") or row.get("dataset") or "unknown")


def row_stratum(row: dict[str, Any]) -> Stratum:
    return str(row["task"]), row_source(row)


def row_student(row: dict[str, Any]) -> str:
    return str(row.get("student_id") or f"NO_STUDENT::{row.get('example_id')}")


def original_phase(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") or {}
    split = str(metadata.get("original_split") or row.get("split") or "train")
    return "valid" if split in {"valid", "validation", "val"} else "train"


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from error


def per_student_cap(phase: str, task: str, source: str, quota: int) -> int:
    if source == "eedi_mining":
        return quota
    if source == "malrule":
        return 1
    if phase == "train":
        if task == "predict_prospective_error":
            return 8
        if task == "predict_error_persistence":
            return 4
        return 4
    if task == "predict_prospective_error":
        return 4
    if task == "predict_error_persistence":
        return 3
    return 2


def count_students(
    path: Path,
    quotas: dict[str, dict[Stratum, int]],
    fixed_phase: str | None,
) -> dict[tuple[str, Stratum], Counter[str]]:
    counts: dict[tuple[str, Stratum], Counter[str]] = defaultdict(Counter)
    for row in iter_jsonl(path):
        phase = fixed_phase or original_phase(row)
        stratum = row_stratum(row)
        if stratum[1] == "eedi_mining" and fixed_phase is None:
            continue
        if stratum not in quotas.get(phase, {}):
            continue
        counts[(phase, stratum)][row_student(row)] += 1
    return counts


def choose_students(
    counts: dict[tuple[str, Stratum], Counter[str]],
    quotas: dict[str, dict[Stratum, int]],
    seed: int,
) -> dict[tuple[str, Stratum], set[str]]:
    selected: dict[tuple[str, Stratum], set[str]] = {}
    for phase, phase_quotas in quotas.items():
        for stratum, quota in phase_quotas.items():
            if stratum[1] == "eedi_mining" and phase in {"train", "valid"}:
                continue
            available = counts.get((phase, stratum), Counter())
            cap = per_student_cap(phase, *stratum, quota)
            ranked = sorted(
                available,
                key=lambda student: (stable_hash(seed, "cohort", phase, stratum[1], student), student),
            )
            chosen: set[str] = set()
            capacity = 0
            for student in ranked:
                chosen.add(student)
                capacity += min(available[student], cap)
                if capacity >= quota:
                    break
            if capacity < quota:
                raise ValueError(
                    f"Insufficient rows for {phase}/{stratum}: quota={quota}, capacity={capacity}"
                )
            selected[(phase, stratum)] = chosen
    return selected


def push_smallest(
    heap: list[tuple[int, str, dict[str, Any]]],
    score: int,
    row: dict[str, Any],
    limit: int,
) -> None:
    example_id = str(row.get("example_id"))
    entry = (-score, example_id, row)
    if len(heap) < limit:
        heapq.heappush(heap, entry)
    elif score < -heap[0][0]:
        heapq.heapreplace(heap, entry)


def collect_selected_rows(
    path: Path,
    quotas: dict[str, dict[Stratum, int]],
    selected_students: dict[tuple[str, Stratum], set[str]],
    seed: int,
    fixed_phase: str | None,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    per_student: dict[tuple[str, Stratum, str], list[tuple[int, str, dict[str, Any]]]] = defaultdict(list)
    eedi_rows: list[dict[str, Any]] = []

    for row in iter_jsonl(path):
        phase = fixed_phase or original_phase(row)
        stratum = row_stratum(row)
        if stratum[1] == "eedi_mining" and fixed_phase is None:
            if stratum == ("diagnose_misconception", "eedi_mining"):
                eedi_rows.append(row)
            continue
        if stratum not in quotas.get(phase, {}):
            continue
        student = row_student(row)
        if student not in selected_students.get((phase, stratum), set()):
            continue
        quota = quotas[phase][stratum]
        cap = per_student_cap(phase, *stratum, quota)
        score = stable_hash(seed, "row", phase, row.get("example_id"))
        push_smallest(per_student[(phase, stratum, student)], score, row, cap)

    output: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for phase, phase_quotas in quotas.items():
        for stratum, quota in phase_quotas.items():
            if stratum[1] == "eedi_mining" and fixed_phase is None:
                continue
            candidates: list[tuple[int, int, str, dict[str, Any]]] = []
            for (candidate_phase, candidate_stratum, student), heap in per_student.items():
                if candidate_phase != phase or candidate_stratum != stratum:
                    continue
                cohort_score = stable_hash(seed, "cohort", phase, stratum[1], student)
                for neg_score, _, row in heap:
                    candidates.append((cohort_score, -neg_score, str(row.get("example_id")), row))
            candidates.sort(key=lambda item: item[:3])
            chosen = [item[3] for item in candidates[:quota]]
            if len(chosen) != quota:
                raise ValueError(f"Selected {len(chosen)} rows for {phase}/{stratum}, expected {quota}")
            output[phase].extend(chosen)
    return output, eedi_rows


def add_eedi_train_valid(
    output: dict[str, list[dict[str, Any]]],
    rows: list[dict[str, Any]],
    quotas: dict[str, dict[Stratum, int]],
    seed: int,
) -> None:
    stratum = ("diagnose_misconception", "eedi_mining")
    valid_quota = quotas["valid"][stratum]
    train_quota = quotas["train"][stratum]

    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_label[str(row.get("target_label") or row.get("target"))].append(row)
    for label_rows in by_label.values():
        label_rows.sort(key=lambda row: stable_hash(seed, "eedi-row", row.get("example_id")))
    labels = sorted(by_label, key=lambda label: stable_hash(seed, "eedi-label", label))
    shared_labels = [label for label in labels if len(by_label[label]) >= 2][:valid_quota]
    if len(shared_labels) < valid_quota:
        raise ValueError(
            f"Eedi-Mining has only {len(shared_labels)} labels with at least two rows; "
            f"need {valid_quota}"
        )

    valid_rows: list[dict[str, Any]] = []
    train_rows: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    for label in shared_labels:
        valid_row, train_row = by_label[label][:2]
        valid_rows.append(valid_row)
        train_rows.append(train_row)
        used_ids.update({str(valid_row.get("example_id")), str(train_row.get("example_id"))})

    depth = 0
    while len(train_rows) < train_quota:
        added = False
        for label in labels:
            if depth >= len(by_label[label]):
                continue
            row = by_label[label][depth]
            example_id = str(row.get("example_id"))
            if example_id in used_ids:
                continue
            train_rows.append(row)
            used_ids.add(example_id)
            added = True
            if len(train_rows) == train_quota:
                break
        if not added and depth >= max(len(label_rows) for label_rows in by_label.values()):
            break
        depth += 1
    if len(train_rows) < train_quota:
        raise ValueError("Insufficient Eedi-Mining rows")
    output["valid"].extend(valid_rows)
    output["train"].extend(train_rows)


def annotate_row(row: dict[str, Any], phase: str, seed: int) -> dict[str, Any]:
    copied = dict(row)
    metadata = dict(copied.get("metadata") or {})
    metadata.update(
        {
            "method_small_dataset": "method_small_v1",
            "method_small_phase": phase,
            "method_small_seed": seed,
            "method_small_original_split": copied.get("split"),
        }
    )
    copied["metadata"] = metadata
    copied["split"] = phase
    return copied


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def label_for_audit(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") or {}
    task = row.get("task")
    if task in {"predict_correctness", "predict_error_persistence"}:
        return str(row.get("target_label"))
    if task == "predict_prospective_error":
        return str(metadata.get("target_correctness"))
    if task == "diagnose_misconception":
        return str(row.get("target_label"))
    return "response"


def build_stats(outputs: dict[str, list[dict[str, Any]]], seed: int) -> dict[str, Any]:
    stats: dict[str, Any] = {"dataset": "method_small_v1", "seed": seed, "splits": {}}
    split_examples: dict[str, set[str]] = {}
    split_students: dict[str, dict[str, set[str]]] = {}
    for phase, rows in outputs.items():
        counts = Counter()
        labels: dict[str, Counter[str]] = defaultdict(Counter)
        students: dict[str, set[str]] = defaultdict(set)
        histories: Counter[str] = Counter()
        example_ids: set[str] = set()
        for row in rows:
            task, source = row_stratum(row)
            key = f"{task}/{source}"
            counts[key] += 1
            labels[key][label_for_audit(row)] += 1
            students[source].add(row_student(row))
            history = row.get("history")
            histories[str(len(history) if isinstance(history, list) else -1)] += 1
            example_ids.add(str(row.get("example_id")))
        split_examples[phase] = example_ids
        split_students[phase] = students
        stats["splits"][phase] = {
            "rows": len(rows),
            "counts": dict(sorted(counts.items())),
            "label_counts": {key: dict(value) for key, value in sorted(labels.items())},
            "unique_students_by_source": {key: len(value) for key, value in sorted(students.items())},
            "history_length_counts": dict(sorted(histories.items(), key=lambda item: int(item[0]))),
        }

    overlaps = {}
    phases = sorted(outputs)
    for index, left in enumerate(phases):
        for right in phases[index + 1 :]:
            overlaps[f"examples/{left}/{right}"] = len(split_examples[left] & split_examples[right])
            for source in sorted(set(split_students[left]) | set(split_students[right])):
                if source == "eedi_mining":
                    continue
                overlap = split_students[left].get(source, set()) & split_students[right].get(source, set())
                overlaps[f"students/{source}/{left}/{right}"] = len(overlap)
    stats["overlaps"] = overlaps
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-jsonl",
        type=Path,
        default=Path("data/processed/full_trainplusvalid_no_t4_strict_t6/train.jsonl"),
    )
    parser.add_argument(
        "--test-jsonl",
        type=Path,
        default=Path("data/processed/formal_evaluation_v1/test.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/method_small_v1"),
    )
    parser.add_argument("--seed", type=int, default=20260722)
    args = parser.parse_args()

    quotas = SMALL_QUOTAS
    train_valid_quotas = {phase: quotas[phase] for phase in ("train", "valid")}
    test_quotas = {"test": quotas["test"]}

    train_counts = count_students(args.train_jsonl, train_valid_quotas, fixed_phase=None)
    train_students = choose_students(train_counts, train_valid_quotas, args.seed)
    train_output, eedi_rows = collect_selected_rows(
        args.train_jsonl, train_valid_quotas, train_students, args.seed, fixed_phase=None
    )
    add_eedi_train_valid(train_output, eedi_rows, train_valid_quotas, args.seed)

    test_counts = count_students(args.test_jsonl, test_quotas, fixed_phase="test")
    test_students = choose_students(test_counts, test_quotas, args.seed)
    test_output, _ = collect_selected_rows(
        args.test_jsonl, test_quotas, test_students, args.seed, fixed_phase="test"
    )

    outputs = {
        "train": train_output["train"],
        "valid": train_output["valid"],
        "test": test_output["test"],
    }
    for phase, rows in outputs.items():
        rows.sort(
            key=lambda row: (
                stable_hash(args.seed, "output", phase, row_source(row), row_student(row)),
                stable_hash(args.seed, "output-row", phase, row.get("example_id")),
            )
        )
        outputs[phase] = [annotate_row(row, phase, args.seed) for row in rows]

    stats = build_stats(outputs, args.seed)
    if any(stats["overlaps"].values()):
        unexpected = {key: value for key, value in stats["overlaps"].items() if value}
        raise ValueError(f"Detected split overlap: {unexpected}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for phase, rows in outputs.items():
        write_jsonl(args.output_dir / f"{phase}.jsonl", rows)
    (args.output_dir / "dataset_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({phase: len(rows) for phase, rows in outputs.items()}, sort_keys=True))


if __name__ == "__main__":
    main()
