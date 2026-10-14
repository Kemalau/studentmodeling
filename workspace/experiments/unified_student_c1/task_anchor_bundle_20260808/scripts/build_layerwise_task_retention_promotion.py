#!/usr/bin/env python3
"""Promote a diagnostic task-retention curriculum without repeating epoch rows."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Iterable

import torch


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
                    raise ValueError(f"Expected JSON object in {path}")
                rows.append(value)
    return rows


def retention_stratum(row: dict[str, Any]) -> tuple[str, str, str]:
    task = str(row["task"])
    source = str(row["source"])
    label = str(row.get("target", "")) if task == "predict_error_persistence" else ""
    return task, source, label


def promote_order(
    real_order: Iterable[int],
    train_rows: list[dict[str, Any]],
    requested_indices: Iterable[int],
    *,
    prefix_count: int,
    allow_task_only_replacements: bool = False,
) -> tuple[list[int], list[int], list[dict[str, Any]]]:
    """Keep the seen prefix fixed and move a matched unseen curriculum next."""

    order = list(real_order)
    requested = list(requested_indices)
    row_count = len(train_rows)
    if len(order) != row_count or set(order) != set(range(row_count)):
        raise ValueError("Source real-data order is not a complete permutation")
    if not 0 <= prefix_count < row_count:
        raise ValueError("Invalid completed-prefix length")
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("Requested curriculum must contain unique rows")
    if any(type(index) is not int or not 0 <= index < row_count for index in requested):
        raise ValueError("Requested curriculum contains an invalid row index")

    prefix = order[:prefix_count]
    suffix = order[prefix_count:]
    prefix_set = set(prefix)
    requested_set = set(requested)
    pools: dict[tuple[str, str, str], deque[int]] = defaultdict(deque)
    task_pools: dict[tuple[str, str], deque[int]] = defaultdict(deque)
    for index in suffix:
        if index not in requested_set:
            stratum = retention_stratum(train_rows[index])
            pools[stratum].append(index)
            task_pools[(stratum[0], stratum[2])].append(index)

    promoted: list[int] = []
    replacements: list[dict[str, Any]] = []
    replacement_used: set[int] = set()

    def take(pool: deque[int]) -> int | None:
        while pool and pool[0] in replacement_used:
            pool.popleft()
        if not pool:
            return None
        value = pool.popleft()
        replacement_used.add(value)
        return value

    for index in requested:
        if index not in prefix_set:
            promoted.append(index)
            continue
        key = retention_stratum(train_rows[index])
        replacement = take(pools[key])
        match_level = "task_source_label"
        if replacement is None and allow_task_only_replacements:
            replacement = take(task_pools[(key[0], key[2])])
            match_level = "task_label"
        if replacement is None:
            raise ValueError(f"No unseen replacement remains for stratum={key}")
        promoted.append(replacement)
        replacements.append(
            {
                "seen_index": index,
                "replacement_index": replacement,
                "stratum": list(key),
                "replacement_stratum": list(retention_stratum(train_rows[replacement])),
                "match_level": match_level,
            }
        )

    if len(promoted) != len(requested) or len(set(promoted)) != len(promoted):
        raise AssertionError("Promoted curriculum is not unique and support preserving")
    if prefix_set.intersection(promoted):
        raise AssertionError("Promoted curriculum repeats a completed-prefix row")
    promoted_set = set(promoted)
    result = prefix + promoted + [index for index in suffix if index not in promoted_set]
    if result[:prefix_count] != prefix:
        raise AssertionError("Completed prefix changed during promotion")
    if len(result) != row_count or set(result) != set(range(row_count)):
        raise AssertionError("Promoted order is not a complete one-epoch permutation")
    before = Counter(retention_stratum(train_rows[index]) for index in requested)
    after = Counter(retention_stratum(train_rows[index]) for index in promoted)
    if not allow_task_only_replacements and before != after:
        raise AssertionError("Promotion changed task/source/label curriculum support")
    before_task = Counter((key[0], key[2]) for key in before.elements())
    after_task = Counter((key[0], key[2]) for key in after.elements())
    if before_task != after_task:
        raise AssertionError("Promotion changed task/label curriculum support")
    return result, promoted, replacements


def summarize_counts(rows: list[dict[str, Any]], indices: list[int]) -> dict[str, Any]:
    task = Counter(str(rows[index]["task"]) for index in indices)
    source = Counter(str(rows[index]["source"]) for index in indices)
    t6 = Counter(
        str(rows[index].get("target", ""))
        for index in indices
        if str(rows[index]["task"]) == "predict_error_persistence"
    )
    return {
        "task_counts": dict(sorted(task.items())),
        "source_counts": dict(sorted(source.items())),
        "t6_target_counts": dict(sorted(t6.items())),
    }


def build(
    *,
    train_jsonl: Path,
    formal_eval_jsonl: Path,
    checkpoint: Path,
    diagnostic_manifest: Path,
    branch_override: str | None = None,
) -> dict[str, Any]:
    train_hash = sha256(train_jsonl)
    formal_hash = sha256(formal_eval_jsonl)
    manifest_hash = sha256(diagnostic_manifest)
    manifest = json.loads(diagnostic_manifest.read_text(encoding="utf-8"))
    if manifest.get("stage") != "layerwise_task_retention_index_manifest_v1":
        raise ValueError("Task-retention diagnostic manifest stage drift")
    if manifest.get("source_train_sha256") != train_hash:
        raise ValueError("Task-retention source-train SHA256 drift")
    if manifest.get("formal_eval_sha256") != formal_hash:
        raise ValueError("Task-retention formal-eval SHA256 drift")

    train_rows = read_jsonl(train_jsonl)
    formal_rows = read_jsonl(formal_eval_jsonl)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    if payload.get("distributed") is not True:
        raise ValueError("Promotion source must be a distributed checkpoint")
    round_index = int(payload.get("round", -1))
    world_size = int(payload.get("world_size", -1))
    if round_index <= 0 or world_size <= 1:
        raise ValueError("Promotion source checkpoint boundary is invalid")
    prefix_count = round_index * world_size
    if prefix_count != int(manifest.get("start_global_position", -1)):
        raise ValueError("Diagnostic manifest does not start at the checkpoint boundary")
    if (payload.get("config") or {}).get("train_sha256") != train_hash:
        raise ValueError("Checkpoint training corpus SHA256 drift")
    history = list(payload.get("history") or [])
    if len(history) != prefix_count:
        raise ValueError("Checkpoint history does not prove the completed prefix")

    requested = list(manifest.get("indices") or [])
    manifest_branch = str(manifest.get("branch"))
    branch = branch_override or manifest_branch
    if branch_override is not None:
        valid_override = (
            (branch_override == "eedi_full_open_t3" and manifest_branch == "eedi_full")
            or (
                branch_override == "equal_t3_t6_open_t3"
                and manifest_branch == "equal_t3_t6"
            )
            or (
                branch_override == "equal_t3_t6_open_t3_two_stage"
                and manifest_branch == "equal_t3_t6"
                and manifest.get("curriculum_mode")
                == "open_t3_formation_then_causal_task_consolidation"
            )
        )
        if not valid_override:
            raise ValueError("Unsupported promotion branch override")
    full_eedi_branch = branch in {"eedi_full", "eedi_full_open_t3"}
    order, promoted, replacements = promote_order(
        payload.get("real_order") or [],
        train_rows,
        requested,
        prefix_count=prefix_count,
        allow_task_only_replacements=full_eedi_branch,
    )
    formal_ids = {str(row["example_id"]) for row in formal_rows}
    promoted_ids = [str(train_rows[index]["example_id"]) for index in promoted]
    overlap = sorted(formal_ids.intersection(promoted_ids))
    if overlap:
        raise ValueError(f"Promoted curriculum overlaps formal evaluation: {overlap[:5]}")
    promoted_id_sha = hashlib.sha256("\n".join(promoted_ids).encode()).hexdigest()
    order_sha = hashlib.sha256("\n".join(map(str, order)).encode()).hexdigest()
    counts = summarize_counts(train_rows, promoted)
    if counts["task_counts"] != manifest.get("task_counts"):
        raise AssertionError("Promoted task counts differ from diagnostic manifest")
    if manifest.get("branch") != "eedi_full" and counts["source_counts"] != manifest.get("source_counts"):
        raise AssertionError("Promoted source counts differ from diagnostic manifest")
    if counts["t6_target_counts"] != summarize_counts(train_rows, requested)["t6_target_counts"]:
        raise AssertionError("Promoted T6 label counts differ from diagnostic manifest")
    if full_eedi_branch:
        eedi_total = sum(
            str(row["task"]) == "diagnose_misconception"
            and str(row["source"]) == "eedi_mining"
            for row in train_rows
        )
        eedi_prefix = sum(
            str(train_rows[index]["task"]) == "diagnose_misconception"
            and str(train_rows[index]["source"]) == "eedi_mining"
            for index in order[:prefix_count]
        )
        eedi_promoted = sum(
            str(train_rows[index]["task"]) == "diagnose_misconception"
            and str(train_rows[index]["source"]) == "eedi_mining"
            for index in promoted
        )
        if eedi_prefix + eedi_promoted != eedi_total:
            raise AssertionError("Eedi-full promotion does not cover every Eedi T3 row exactly once")
    curriculum_metadata = {}
    if manifest.get("curriculum_mode") is not None:
        curriculum_metadata = {
            "curriculum_mode": manifest["curriculum_mode"],
            "curriculum_phases": manifest.get("curriculum_phases"),
            "curriculum_source_manifest": str(diagnostic_manifest),
        }
    return {
        "stage": "layerwise_task_retention_promoted_order_v1",
        "branch": branch,
        "seed": manifest["seed"],
        "source_train_sha256": train_hash,
        "formal_eval_sha256": formal_hash,
        "diagnostic_manifest_sha256": manifest_hash,
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_round": round_index,
        "source_world_size": world_size,
        "completed_prefix_rows": prefix_count,
        "epoch_rows": len(order),
        "one_epoch_permutation": True,
        "completed_prefix_unchanged": True,
        "formal_example_id_overlap": 0,
        "requested_rows": len(requested),
        "promoted_rows": len(promoted),
        "seen_rows_replaced": len(replacements),
        "replacement_records": replacements,
        "source_rebalanced_for_seen_prefix": full_eedi_branch,
        "training_view": (
            "open_t3_selective_kd"
            if branch in {
                "equal_t3_t6_open_t3",
                "equal_t3_t6_open_t3_two_stage",
            }
            else (
                "open_t3_no_candidate_enumeration"
                if branch == "eedi_full_open_t3"
                else "standard"
            )
        ),
        "diagnostic_source_counts": manifest.get("source_counts"),
        "promoted_example_ids_sha256": promoted_id_sha,
        "real_order_sha256": order_sha,
        "promoted_indices": promoted,
        "real_order": order,
        **curriculum_metadata,
        **counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--formal-eval-jsonl", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--diagnostic-manifest", type=Path, required=True)
    parser.add_argument("--branch-override")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = build(
        train_jsonl=args.train_jsonl,
        formal_eval_jsonl=args.formal_eval_jsonl,
        checkpoint=args.checkpoint,
        diagnostic_manifest=args.diagnostic_manifest,
        branch_override=args.branch_override,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps({key: value[key] for key in (
        "branch", "epoch_rows", "completed_prefix_rows", "promoted_rows",
        "seen_rows_replaced", "real_order_sha256",
    )}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
