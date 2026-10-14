#!/usr/bin/env python3
"""Build deterministic train-only T3/T6 task-retention index curricula."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SOURCE_TRAIN_SHA256 = "9961f746cb17e6c314e182d3453b6157022fb48b8de44d2e68bc8e87a1131ccd"
FORMAL_EVAL_SHA256 = "a368cefb0c49cb6a305ad1a892808361fba2d44d555a0a28ea64e56babe32525"
START_GLOBAL_POSITION = 12_000
SEGMENT_ROWS = 1_000

ANCHOR_SPECS = {
    ("predict_correctness", "dbe_kt22"): 16,
    ("predict_correctness", "eedi_response"): 16,
    ("predict_correctness", "foundationalassist"): 16,
    ("predict_correctness", "malrule"): 16,
    ("predict_correctness", "xes3g5m"): 16,
    ("predict_response", "dbe_kt22"): 12,
    ("predict_response", "eedi_response"): 12,
    ("predict_response", "foundationalassist"): 13,
    ("predict_response", "malrule"): 13,
    ("predict_prospective_error", "dbe_kt22"): 35,
    ("predict_prospective_error", "foundationalassist"): 35,
}

BRANCH_SPECS = {
    "equal_t3_t6_consolidation": {
        # Phase-1 consumes every available DBE transient T6 row.  A disjoint
        # phase-2 therefore preserves task support and global label balance
        # with fresh FoundationalAssist persistence examples.
        "t3": {"eedi_mining": 240, "malrule": 160},
        "t6": {
            ("foundationalassist", "persistent"): 200,
            ("foundationalassist", "transient"): 200,
        },
    },
    "eedi_full": {
        # Formation-strength diagnostic: cover every available Eedi T3 row
        # once, add broad Malrule support, retain a balanced T6 panel, and
        # keep the same 200 cross-task anchors used by the bounded rehearsals.
        "segment_rows": 5000,
        "t6": {
            ("dbe_kt22", "persistent"): 15,
            ("dbe_kt22", "transient"): 15,
            ("foundationalassist", "persistent"): 135,
            ("foundationalassist", "transient"): 135,
        },
        # The 300 T6 rows and 200 anchors leave 4500 diagnosis rows. Keep all
        # Eedi rows and use Malrule for the remaining 1024 examples.
        "t3": {"eedi_mining": 3476, "malrule": 1024},
    },
    "eedi_dense": {
        "t3": {"eedi_mining": 600, "malrule": 100},
        "t6": {
            ("dbe_kt22", "persistent"): 10,
            ("dbe_kt22", "transient"): 10,
            ("foundationalassist", "persistent"): 40,
            ("foundationalassist", "transient"): 40,
        },
    },
    "equal_t3_t6": {
        "t3": {"eedi_mining": 240, "malrule": 160},
        "t6": {
            ("dbe_kt22", "persistent"): 30,
            ("dbe_kt22", "transient"): 15,
            ("foundationalassist", "persistent"): 170,
            ("foundationalassist", "transient"): 185,
        },
    },
    "t3_heavy": {
        "t3": {"eedi_mining": 360, "malrule": 240},
        "t6": {
            ("dbe_kt22", "persistent"): 15,
            ("dbe_kt22", "transient"): 15,
            ("foundationalassist", "persistent"): 85,
            ("foundationalassist", "transient"): 85,
        },
    },
    "staged_t3_t6": {
        "t3": {"eedi_mining": 400, "malrule": 100},
        "t6": {
            ("dbe_kt22", "persistent"): 15,
            ("dbe_kt22", "transient"): 15,
            ("foundationalassist", "persistent"): 135,
            ("foundationalassist", "transient"): 135,
        },
    },
}

# The mixed rehearsals above test task proportions.  This schedule tests the
# additional curriculum hypothesis without changing support: establish a
# diagnosis route first, introduce balanced persistence next, then emphasize
# persistence while continuing diagnosis and general-task rehearsal.  Counts
# are over disjoint train-only rows and sum to the exact branch totals above.
STAGED_T3_T6_PHASES = (
    {
        "name": "diagnosis_formation",
        "anchors": 50,
        "t3": {"eedi_mining": 300, "malrule": 50},
        "t6": {},
    },
    {
        "name": "balanced_transition",
        "anchors": 100,
        "t3": {"eedi_mining": 70, "malrule": 30},
        "t6": {
            ("dbe_kt22", "persistent"): 5,
            ("dbe_kt22", "transient"): 5,
            ("foundationalassist", "persistent"): 45,
            ("foundationalassist", "transient"): 45,
        },
    },
    {
        "name": "persistence_consolidation",
        "anchors": 50,
        "t3": {"eedi_mining": 30, "malrule": 20},
        "t6": {
            ("dbe_kt22", "persistent"): 10,
            ("dbe_kt22", "transient"): 10,
            ("foundationalassist", "persistent"): 90,
            ("foundationalassist", "transient"): 90,
        },
    },
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"Expected JSON objects in {path}")
                rows.append(value)
    return rows


def t6_label(row: dict[str, Any]) -> str:
    target = str(row.get("target", ""))
    if target == "<persistence>persistent</persistence>":
        return "persistent"
    if target == "<persistence>transient</persistence>":
        return "transient"
    raise ValueError(f"Unexpected T6 target: {target}")


def choose(
    pools: dict[Any, list[int]],
    key: Any,
    count: int,
    *,
    seed: int,
    used: set[int],
) -> list[int]:
    candidates = [index for index in pools[key] if index not in used]
    rng = random.Random(f"{seed}:{key}")
    rng.shuffle(candidates)
    if len(candidates) < count:
        raise ValueError(f"Insufficient pool for {key}: need={count} got={len(candidates)}")
    selected = candidates[:count]
    used.update(selected)
    return selected


def build(
    train_rows: list[dict[str, Any]],
    formal_rows: list[dict[str, Any]],
    *,
    branch: str,
    seed: int,
    preused: set[int] | None = None,
    start_global_position: int = START_GLOBAL_POSITION,
) -> dict[str, Any]:
    if branch not in BRANCH_SPECS:
        raise ValueError(f"Unknown branch: {branch}")
    by_task_source: dict[tuple[str, str], list[int]] = defaultdict(list)
    by_t6_source_label: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(train_rows):
        task = str(row["task"])
        source = str(row["source"])
        by_task_source[(task, source)].append(index)
        if task == "predict_error_persistence":
            by_t6_source_label[(source, t6_label(row))].append(index)

    used: set[int] = set(preused or ())
    excluded = set(used)
    selected: list[int] = []
    anchor_indices: list[int] = []
    for key, count in ANCHOR_SPECS.items():
        chosen = choose(by_task_source, key, count, seed=seed, used=used)
        anchor_indices.extend(chosen)
        selected.extend(chosen)
    spec = BRANCH_SPECS[branch]
    t3_indices: dict[str, list[int]] = {}
    for source, count in spec["t3"].items():
        chosen = choose(
            by_task_source,
            ("diagnose_misconception", source),
            count,
            seed=seed + 101,
            used=used,
        )
        t3_indices[source] = chosen
        selected.extend(chosen)
    t6_indices: dict[tuple[str, str], list[int]] = {}
    for key, count in spec["t6"].items():
        chosen = choose(by_t6_source_label, key, count, seed=seed + 211, used=used)
        t6_indices[key] = chosen
        selected.extend(chosen)
    segment_rows = int(spec.get("segment_rows", SEGMENT_ROWS))
    if len(selected) != segment_rows or len(set(selected)) != segment_rows:
        raise AssertionError((branch, len(selected), len(set(selected))))
    curriculum_phases: list[dict[str, Any]] = []
    if branch == "staged_t3_t6":
        offsets: dict[Any, int] = {"anchors": 0}
        offsets.update({("t3", key): 0 for key in t3_indices})
        offsets.update({("t6", key): 0 for key in t6_indices})
        ordered: list[int] = []
        for phase_index, phase in enumerate(STAGED_T3_T6_PHASES):
            phase_rows: list[int] = []
            start = offsets["anchors"]
            stop = start + int(phase["anchors"])
            phase_rows.extend(anchor_indices[start:stop])
            offsets["anchors"] = stop
            for source, count in phase["t3"].items():
                key = ("t3", source)
                start = offsets[key]
                stop = start + int(count)
                phase_rows.extend(t3_indices[source][start:stop])
                offsets[key] = stop
            for source_label, count in phase["t6"].items():
                key = ("t6", source_label)
                start = offsets[key]
                stop = start + int(count)
                phase_rows.extend(t6_indices[source_label][start:stop])
                offsets[key] = stop
            random.Random(f"{seed}:{branch}:phase:{phase_index}").shuffle(phase_rows)
            phase_start = len(ordered)
            ordered.extend(phase_rows)
            curriculum_phases.append(
                {
                    "name": phase["name"],
                    "start_offset": phase_start,
                    "end_offset": len(ordered) - 1,
                    "rows": len(phase_rows),
                }
            )
        if offsets["anchors"] != len(anchor_indices):
            raise AssertionError("Staged curriculum did not consume every anchor")
        if any(offsets[("t3", key)] != len(value) for key, value in t3_indices.items()):
            raise AssertionError("Staged curriculum did not consume every T3 row")
        if any(offsets[("t6", key)] != len(value) for key, value in t6_indices.items()):
            raise AssertionError("Staged curriculum did not consume every T6 row")
        if len(ordered) != segment_rows or set(ordered) != set(selected):
            raise AssertionError("Staged curriculum changed selected support")
        selected = ordered
    else:
        random.Random(f"{seed}:{branch}:final-order").shuffle(selected)

    formal_ids = {str(row["example_id"]) for row in formal_rows}
    selected_ids = [str(train_rows[index]["example_id"]) for index in selected]
    overlap = sorted(formal_ids.intersection(selected_ids))
    if overlap:
        raise ValueError(f"Task-retention train selection overlaps formal test: {overlap[:5]}")
    task_counts = Counter(str(train_rows[index]["task"]) for index in selected)
    source_counts = Counter(str(train_rows[index]["source"]) for index in selected)
    label_counts = Counter(
        t6_label(train_rows[index])
        for index in selected
        if train_rows[index]["task"] == "predict_error_persistence"
    )
    id_digest = hashlib.sha256("\n".join(selected_ids).encode("utf-8")).hexdigest()
    result = {
        "stage": "layerwise_task_retention_index_manifest_v1",
        "branch": branch,
        "seed": seed,
        "source_train_sha256": SOURCE_TRAIN_SHA256,
        "formal_eval_sha256": FORMAL_EVAL_SHA256,
        "formal_example_id_overlap": 0,
        "start_global_position": start_global_position,
        "segment_rows": segment_rows,
        "selected_example_ids_sha256": id_digest,
        "task_counts": dict(sorted(task_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "t6_label_counts": dict(sorted(label_counts.items())),
        "indices": selected,
    }
    if excluded:
        result.update(
            excluded_prior_rows=len(excluded),
            prior_segment_overlap=len(excluded.intersection(selected)),
        )
    # Preserve byte-for-byte compatibility of the three already submitted
    # manifests; staged metadata belongs only to the new condition.
    if branch == "staged_t3_t6":
        result.update(
            curriculum_mode="three_phase_diagnosis_then_persistence",
            curriculum_phases=curriculum_phases,
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--formal-eval-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if sha256(args.train_jsonl) != SOURCE_TRAIN_SHA256:
        raise ValueError("Task-retention source-train SHA256 drift")
    if sha256(args.formal_eval_jsonl) != FORMAL_EVAL_SHA256:
        raise ValueError("Task-retention formal-eval SHA256 drift")
    train_rows = read_jsonl(args.train_jsonl)
    formal_rows = read_jsonl(args.formal_eval_jsonl)
    if len(train_rows) != 259_845:
        raise ValueError(f"Expected 259845 source rows, got {len(train_rows)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for branch in BRANCH_SPECS:
        value = build(train_rows, formal_rows, branch=branch, seed=args.seed)
        path = args.output_dir / f"layerwise_task_retention_{branch}_seed42_v1.json"
        path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"path": str(path), **{k: value[k] for k in ("branch", "task_counts", "source_counts", "t6_label_counts")}}, sort_keys=True))


if __name__ == "__main__":
    main()
