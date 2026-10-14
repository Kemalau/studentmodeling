#!/usr/bin/env python3
"""Freeze an XES3G5M confirmatory holdout for the revised State8 C4 claim.

DBE profile-test students have already been inspected during development.  They
remain useful for checkpoint selection, but they are no longer a clean
confirmatory set.  This builder applies the same chronological profile target
definition to the official XES3G5M student split.  The official test students
are retained only for one final, frozen-checkpoint evaluation; neither their
targets nor aggregate metrics may be used for model or readout selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

try:
    from build_dbe_student_profile_pilot import (
        build_profile_row,
        choose_kc_vocabulary,
        sha256,
    )
    from prepare_xes3g5m_t1_t6 import load_questions, read_sequences
    from student_task_utils import write_jsonl
except ModuleNotFoundError:  # Package imports in tests.
    from scripts.build_dbe_student_profile_pilot import (
        build_profile_row,
        choose_kc_vocabulary,
        sha256,
    )
    from scripts.prepare_xes3g5m_t1_t6 import load_questions, read_sequences
    from scripts.student_task_utils import write_jsonl


STAGE = "xes_state8_profile_confirmatory_holdout_v1"
PROTOCOL_VERSION = "2026-08-20.v1"


def stable_score(seed: int, value: str) -> int:
    payload = f"{seed}\x1f{value}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def add_interaction_ids(
    traces: dict[str, list[dict[str, Any]]], *, namespace: str
) -> None:
    """Supply stable identities required by the chronological leakage check."""
    for student_id, trace in traces.items():
        for position, event in enumerate(trace):
            event["interaction_id"] = (
                f"xes:{namespace}:{student_id}:{position}:"
                f"{event['timestamp']}:{event['question_id']}"
            )


def deterministic_cap(
    rows: list[dict[str, Any]], limit: int | None, seed: int
) -> list[dict[str, Any]]:
    if limit is None or limit <= 0 or len(rows) <= limit:
        return sorted(rows, key=lambda row: str(row["student_id"]))
    selected = sorted(
        rows,
        key=lambda row: (
            stable_score(seed, str(row["student_id"])),
            str(row["student_id"]),
        ),
    )[:limit]
    return sorted(selected, key=lambda row: str(row["student_id"]))


def build_confirmatory_rows(
    *,
    train_traces: dict[str, list[dict[str, Any]]],
    test_traces: dict[str, list[dict[str, Any]]],
    seed: int,
    top_kcs: int,
    max_history: int,
    min_future: int,
    min_eligible: int,
    max_train_students: int,
    max_valid_students: int,
) -> tuple[dict[str, list[dict[str, Any]]], list[str], dict[str, int]]:
    train_ids = sorted(train_traces)
    rng = random.Random(seed)
    rng.shuffle(train_ids)
    valid_count = max(1, int(0.1 * len(train_ids)))
    split_ids = {
        "train": set(train_ids[valid_count:]),
        "valid": set(train_ids[:valid_count]),
        "test": set(test_traces),
    }
    if split_ids["train"] & split_ids["valid"]:
        raise AssertionError("XES fitting students overlap")
    if (split_ids["train"] | split_ids["valid"]) & split_ids["test"]:
        raise AssertionError("Official XES test students overlap fitting students")

    vocabulary = choose_kc_vocabulary(
        train_traces, split_ids["train"], top_kcs
    )
    sources = {"train": train_traces, "valid": train_traces, "test": test_traces}
    qualified: dict[str, list[dict[str, Any]]] = {}
    eligible_counts: dict[str, int] = {}
    for split in ("train", "valid", "test"):
        rows = []
        for student_id in sorted(split_ids[split]):
            row = build_profile_row(
                split=split,
                student_id=student_id,
                trace=sources[split][student_id],
                vocabulary=vocabulary,
                max_history=max_history,
                min_future=min_future,
                min_eligible=min_eligible,
            )
            if row is None:
                continue
            row["dataset"] = "xes3g5m"
            row["example_id"] = f"xes_profile:{split}:{student_id}"
            row["target_provenance"]["source"] = "XES3G5M"
            row["target_provenance"]["official_test_student"] = split == "test"
            rows.append(row)
        eligible_counts[split] = len(rows)
        cap = (
            max_train_students
            if split == "train"
            else max_valid_students
            if split == "valid"
            else None
        )
        qualified[split] = deterministic_cap(rows, cap, seed + 1009)
    return qualified, vocabulary, eligible_counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw/xes3g5m/extracted/XES3G5M"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("manifests/xes_state8_profile_confirmatory_v1"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-kcs", type=int, default=30)
    parser.add_argument("--max-history", type=int, default=12)
    parser.add_argument("--min-future", type=int, default=5)
    parser.add_argument("--min-eligible", type=int, default=4)
    parser.add_argument("--max-train-students", type=int, default=800)
    parser.add_argument("--max-valid-students", type=int, default=200)
    args = parser.parse_args()

    questions = load_questions(args.raw_dir / "metadata")
    train_path = args.raw_dir / "question_level/train_valid_sequences_quelevel.csv"
    test_path = args.raw_dir / "question_level/test_quelevel.csv"
    train_traces = read_sequences(train_path, questions)
    test_traces = read_sequences(test_path, questions)
    add_interaction_ids(train_traces, namespace="official_train")
    add_interaction_ids(test_traces, namespace="official_test")
    rows, vocabulary, eligible_counts = build_confirmatory_rows(
        train_traces=train_traces,
        test_traces=test_traces,
        seed=args.seed,
        top_kcs=args.top_kcs,
        max_history=args.max_history,
        min_future=args.min_future,
        min_eligible=args.min_eligible,
        max_train_students=args.max_train_students,
        max_valid_students=args.max_valid_students,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    student_sets = {}
    for split in ("train", "valid", "test"):
        path = args.output_dir / f"{split}.jsonl"
        write_jsonl(path, rows[split])
        files[split] = {
            "path": str(path),
            "rows": len(rows[split]),
            "sha256": sha256(path),
        }
        student_sets[split] = {str(row["student_id"]) for row in rows[split]}
    if any(
        student_sets[left] & student_sets[right]
        for left, right in (("train", "valid"), ("train", "test"), ("valid", "test"))
    ):
        raise AssertionError("Materialized XES profile splits are not student-disjoint")

    manifest = {
        "stage": STAGE,
        "status": "frozen_before_confirmatory_evaluation",
        "protocol_version": PROTOCOL_VERSION,
        "created_utc_date": "2026-08-20",
        "source": "XES3G5M real longitudinal interactions",
        "official_test_students_are_confirmatory": True,
        "official_test_metrics_inspected_before_freeze": False,
        "student_disjoint": True,
        "raw_students": {
            "official_train_valid": len(train_traces),
            "official_test": len(test_traces),
        },
        "qualified_students_before_deterministic_fit_caps": eligible_counts,
        "files": files,
        "seed": args.seed,
        "chronological_cutoff_fraction": 0.5,
        "max_state_history": args.max_history,
        "min_future_attempts_per_kc": args.min_future,
        "min_eligible_kcs": args.min_eligible,
        "kc_vocabulary": vocabulary,
        "kc_vocabulary_selection": "top frequency from official-train prefixes only",
        "fit_caps": {
            "train": args.max_train_students,
            "valid": args.max_valid_students,
            "test": "all qualified official-test students",
        },
        "information_boundary": {
            "state_updater": (
                f"last {args.max_history} observed interactions before cutoff"
            ),
            "readout": "State8 only; no raw history or student ID",
            "target": "post-cutoff KC correctness only",
        },
        "confirmatory_use_policy": {
            "checkpoint": "one frozen checkpoint selected without XES profile-test metrics",
            "readout_and_alpha_selection": "official-train and validation students only",
            "official_test_forbidden_uses": [
                "checkpoint selection",
                "hyperparameter selection",
                "early stopping",
                "threshold selection",
                "prompt revision",
                "gate revision",
            ],
            "failed_gate_policy": (
                "Report failure as confirmatory evidence; any later repair is a new "
                "development cycle and requires a new untouched dataset."
            ),
        },
        "primary_claim": (
            "A frozen strict-state-only State8 supports student-specific future-KC "
            "prediction and readable weakness profiles on held-out XES students."
        ),
        "primary_gates": {
            "heldout_student_support": ">=80",
            "state_vs_population_rmse_bootstrap_probability": ">0.95",
            "state_vs_same_window_rmse_bootstrap_probability": ">0.95",
            "state_vs_population_weak_f1_bootstrap_probability": ">0.95",
            "state_vs_same_window_weak_f1_bootstrap_probability": ">0.95",
            "state_ownership_permutation_p_one_sided": "<0.05",
            "disjoint_history_prediction_stability_p_one_sided": "<0.05",
            "closed_information_route": True,
        },
        "diagnostic_only": [
            "cross-student literal text swap",
            "raw representation cosine similarity",
        ],
    }
    path = args.output_dir / "manifest.json"
    path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
