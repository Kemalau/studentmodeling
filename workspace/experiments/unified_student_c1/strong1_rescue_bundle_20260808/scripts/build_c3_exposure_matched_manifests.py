#!/usr/bin/env python3
"""Build exposure-matched C3 transfer curricula.

Every arm contains the same 500 real Eedi-Mining diagnosis rows and the same
500 underlying Malrule examples.  Only the Malrule target changes among
correctness, response, and misconception identity.
"""

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
ARMS = {
    "malrule_correctness": "predict_correctness",
    "malrule_response": "predict_response",
    "malrule_misconception": "diagnose_misconception",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"Expected JSON object: {path}")
                rows.append(value)
    return rows


def raw_malrule_id(row: dict[str, Any]) -> str | None:
    metadata = row.get("original_metadata") or {}
    value = metadata.get("raw_example_id")
    return str(value) if value else None


def build(
    train_rows: list[dict[str, Any]],
    formal_rows: list[dict[str, Any]],
    *,
    seed: int,
    start_global_position: int,
    eedi_count: int = 500,
    auxiliary_count: int = 500,
) -> dict[str, dict[str, Any]]:
    if start_global_position <= 0 or start_global_position % 4:
        raise ValueError("C3 diagnostic start must be positive and four-rank aligned")
    formal_ids = {str(row["example_id"]) for row in formal_rows}

    eedi = [
        index
        for index, row in enumerate(train_rows)
        if row.get("task") == "diagnose_misconception"
        and row.get("source") == "eedi_mining"
        and str(row.get("example_id")) not in formal_ids
    ]
    random.Random(f"{seed}:eedi").shuffle(eedi)
    if len(eedi) < eedi_count:
        raise ValueError(f"Insufficient Eedi rows: {len(eedi)}")
    eedi = eedi[:eedi_count]

    paired: dict[str, dict[str, int]] = defaultdict(dict)
    for index, row in enumerate(train_rows):
        if row.get("source") != "malrule":
            continue
        task = str(row.get("task"))
        if task not in set(ARMS.values()):
            continue
        raw_id = raw_malrule_id(row)
        if raw_id is None:
            continue
        if task in paired[raw_id]:
            raise ValueError(f"Duplicate Malrule task row: {raw_id} {task}")
        paired[raw_id][task] = index
    eligible = sorted(
        raw_id for raw_id, tasks in paired.items() if set(tasks) == set(ARMS.values())
    )
    random.Random(f"{seed}:malrule").shuffle(eligible)
    if len(eligible) < auxiliary_count:
        raise ValueError(f"Insufficient paired Malrule rows: {len(eligible)}")
    selected_raw_ids = eligible[:auxiliary_count]

    # Freeze one role order. Every arm sees Eedi and Malrule examples in the
    # same slots; only the task-specific row chosen for each Malrule raw ID
    # differs.
    role_order = [("eedi", position) for position in range(eedi_count)] + [
        ("malrule", position) for position in range(auxiliary_count)
    ]
    random.Random(f"{seed}:role-order").shuffle(role_order)
    raw_digest = hashlib.sha256("\n".join(selected_raw_ids).encode()).hexdigest()
    eedi_ids = [str(train_rows[index]["example_id"]) for index in eedi]
    eedi_digest = hashlib.sha256("\n".join(eedi_ids).encode()).hexdigest()

    outputs: dict[str, dict[str, Any]] = {}
    for arm, auxiliary_task in ARMS.items():
        auxiliary = [paired[raw_id][auxiliary_task] for raw_id in selected_raw_ids]
        indices = [
            eedi[position] if role == "eedi" else auxiliary[position]
            for role, position in role_order
        ]
        if len(indices) != eedi_count + auxiliary_count or len(set(indices)) != len(indices):
            raise AssertionError(f"C3 row support drift: {arm}")
        example_ids = [str(train_rows[index]["example_id"]) for index in indices]
        overlap = set(example_ids).intersection(formal_ids)
        if overlap:
            raise ValueError(f"C3 formal overlap: {sorted(overlap)[:3]}")
        tasks = Counter(str(train_rows[index]["task"]) for index in indices)
        sources = Counter(str(train_rows[index]["source"]) for index in indices)
        outputs[arm] = {
            "stage": "layerwise_task_retention_index_manifest_v1",
            # The existing bounded continuation runner accepts this training
            # interface while C3-specific metadata below records the purpose.
            "branch": "eedi_full",
            "seed": seed,
            "source_train_sha256": SOURCE_TRAIN_SHA256,
            "formal_eval_sha256": FORMAL_EVAL_SHA256,
            "formal_example_id_overlap": 0,
            "start_global_position": start_global_position,
            "segment_rows": len(indices),
            "selected_example_ids_sha256": hashlib.sha256(
                "\n".join(example_ids).encode()
            ).hexdigest(),
            "task_counts": dict(sorted(tasks.items())),
            "source_counts": dict(sorted(sources.items())),
            "t6_label_counts": {},
            "indices": indices,
            "controlled_experiment": "C3_exposure_matched_auxiliary_supervision_v1",
            "controlled_factor": "malrule_auxiliary_target",
            "arm": arm,
            "auxiliary_task": auxiliary_task,
            "eedi_rows": eedi_count,
            "malrule_rows": auxiliary_count,
            "eedi_example_ids_sha256": eedi_digest,
            "paired_malrule_raw_ids_sha256": raw_digest,
            "role_order_sha256": hashlib.sha256(
                json.dumps(role_order, separators=(",", ":")).encode()
            ).hexdigest(),
            "same_eedi_rows_across_arms": True,
            "same_underlying_malrule_rows_across_arms": True,
            "same_role_order_across_arms": True,
            "diagnostic_only": True,
        }
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--formal-eval-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--start-global-position", type=int, default=24000)
    args = parser.parse_args()
    if sha256(args.train_jsonl) != SOURCE_TRAIN_SHA256:
        raise ValueError("Frozen C3 source train hash drift")
    if sha256(args.formal_eval_jsonl) != FORMAL_EVAL_SHA256:
        raise ValueError("Frozen C3 formal eval hash drift")
    outputs = build(
        read_jsonl(args.train_jsonl),
        read_jsonl(args.formal_eval_jsonl),
        seed=args.seed,
        start_global_position=args.start_global_position,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    receipt = {"stage": "C3_exposure_matched_manifest_bundle_v1", "arms": {}}
    for arm, value in sorted(outputs.items()):
        path = args.output_dir / f"c3_exposure_matched_{arm}_seed{args.seed}_v1.json"
        temporary = path.with_suffix(path.suffix + ".partial")
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        temporary.replace(path)
        receipt["arms"][arm] = {"path": str(path), "sha256": sha256(path)}
    receipt["eedi_example_ids_sha256"] = next(iter(outputs.values()))[
        "eedi_example_ids_sha256"
    ]
    receipt["paired_malrule_raw_ids_sha256"] = next(iter(outputs.values()))[
        "paired_malrule_raw_ids_sha256"
    ]
    receipt["role_order_sha256"] = next(iter(outputs.values()))["role_order_sha256"]
    path = args.output_dir / "c3_exposure_matched_bundle_receipt.json"
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
