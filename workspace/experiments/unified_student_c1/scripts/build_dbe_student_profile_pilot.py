#!/usr/bin/env python3
"""Build a student-disjoint, future-validated DBE profile-readout pilot.

The state encoder sees only interactions before a chronological cutoff.  Profile
targets are computed only from interactions after that cutoff.  This keeps the
natural-language notion of a "weakness" tied to observable future behavior
rather than to an LLM-generated description.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    from prepare_dbe_kt22_t1_t6 import load_traces
    from student_task_utils import STATE_TOKEN, split_students, write_jsonl
except ModuleNotFoundError:  # Package import in tests.
    from scripts.prepare_dbe_kt22_t1_t6 import load_traces
    from scripts.student_task_utils import (
        STATE_TOKEN,
        split_students,
        write_jsonl,
    )


STAGE = "dbe_future_validated_student_profile_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def choose_kc_vocabulary(
    traces: dict[str, list[dict[str, Any]]], train_ids: set[str], top_k: int
) -> list[str]:
    """Select KCs using training prefixes only."""
    counts: Counter[str] = Counter()
    for student_id in train_ids:
        trace = traces[student_id]
        cutoff = len(trace) // 2
        for event in trace[:cutoff]:
            counts.update(str(value) for value in event["kc_names"])
    return [name for name, _count in counts.most_common(top_k)]


def kc_outcomes(
    events: list[dict[str, Any]], vocabulary: list[str]
) -> tuple[list[int], list[int]]:
    index = {name: position for position, name in enumerate(vocabulary)}
    counts = [0] * len(vocabulary)
    correct = [0] * len(vocabulary)
    for event in events:
        outcome = int(event["correctness"])
        for name in set(str(value) for value in event["kc_names"]):
            if name not in index:
                continue
            position = index[name]
            counts[position] += 1
            correct[position] += outcome
    return counts, correct


def ranked_indices(
    counts: list[int], correct: list[int], min_count: int, reverse: bool
) -> list[int]:
    eligible = [index for index, count in enumerate(counts) if count >= min_count]
    # Beta(1,1) smoothing avoids treating a 0/5 observation as infinitely
    # different from 1/6 while retaining the observed ordering.
    return sorted(
        eligible,
        key=lambda index: (
            (correct[index] + 1) / (counts[index] + 2),
            counts[index],
            index,
        ),
        reverse=reverse,
    )


def compact_text(value: Any, limit: int) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def compact_event(event: dict[str, Any], index: int) -> str:
    """Preserve educational evidence without retaining pathological long stems."""
    fields = [
        f"Trial {index}:",
        f"Knowledge components: {', '.join(event['kc_names'])}",
        f"Question: {compact_text(event['question_text'], 240)}",
        f"Outcome: {'correct' if event['correctness'] else 'incorrect'}",
    ]
    if event.get("student_response"):
        fields.append(f"Student response: {compact_text(event['student_response'], 120)}")
    if event.get("correct_response"):
        fields.append(f"Correct response: {compact_text(event['correct_response'], 120)}")
    return "\n".join(fields)


def profile_prompt(history: list[dict[str, Any]]) -> str:
    history_text = "\n\n".join(
        compact_event(event, index) for index, event in enumerate(history, start=1)
    )
    return (
        "You are a student modeling system.\n"
        "<task>encode_student_profile</task>\n"
        "<instruction>Compress the observed interactions into a reusable student "
        "state. The state will be read by several downstream educational tasks."
        "</instruction>\n"
        "<student_history>\n"
        f"{history_text}\n"
        "</student_history>\n"
        f"{STATE_TOKEN}\n"
        "<query>Represent the student's stable strengths and weaknesses.</query>\n"
        "Answer:"
    )


def build_profile_row(
    *,
    split: str,
    student_id: str,
    trace: list[dict[str, Any]],
    vocabulary: list[str],
    max_history: int,
    min_future: int,
    min_eligible: int,
) -> dict[str, Any] | None:
    cutoff = len(trace) // 2
    prefix = trace[:cutoff]
    future = trace[cutoff:]
    history = prefix[-max_history:]
    prefix_counts, prefix_correct = kc_outcomes(prefix, vocabulary)
    future_counts, future_correct = kc_outcomes(future, vocabulary)
    weak = ranked_indices(future_counts, future_correct, min_future, reverse=False)
    strong = ranked_indices(future_counts, future_correct, min_future, reverse=True)
    if len(weak) < min_eligible:
        return None
    history_ids = {str(event["interaction_id"]) for event in prefix}
    future_ids = {str(event["interaction_id"]) for event in future}
    if history_ids.intersection(future_ids):
        raise AssertionError("Chronological profile split leaked interactions")
    return {
        "example_id": f"dbe_profile:{split}:{student_id}",
        "dataset": "dbe_kt22",
        "split": split,
        "student_id": str(student_id),
        "cutoff_position": cutoff,
        "trace_length": len(trace),
        "state_history": history,
        "state_history_length": len(history),
        "prompt": profile_prompt(history),
        "kc_vocabulary": vocabulary,
        "prefix_counts": prefix_counts,
        "prefix_correct": prefix_correct,
        "future_counts": future_counts,
        "future_correct": future_correct,
        "future_smoothed_rates": [
            (right + 1) / (count + 2) if count else None
            for count, right in zip(future_counts, future_correct, strict=True)
        ],
        "future_eligible_mask": [count >= min_future for count in future_counts],
        "weak_kc_indices": weak[:2],
        "strong_kc_indices": strong[:2],
        "weak_kcs": [vocabulary[index] for index in weak[:2]],
        "strong_kcs": [vocabulary[index] for index in strong[:2]],
        "target_provenance": {
            "history_end_timestamp": prefix[-1]["timestamp"],
            "future_start_timestamp": future[0]["timestamp"],
            "profile_target_source": "post_cutoff_observed_correctness_by_kc",
            "min_future_attempts_per_kc": min_future,
            "beta_prior": [1, 1],
            "student_id_visible_to_encoder": False,
        },
    }


def build_rows(
    traces: dict[str, list[dict[str, Any]]],
    splits: dict[str, set[str]],
    vocabulary: list[str],
    max_history: int,
    min_future: int,
    min_eligible: int,
) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for split, student_ids in splits.items():
        rows = []
        for student_id in sorted(student_ids):
            row = build_profile_row(
                split=split,
                student_id=student_id,
                trace=traces[student_id],
                vocabulary=vocabulary,
                max_history=max_history,
                min_future=min_future,
                min_eligible=min_eligible,
            )
            if row is not None:
                rows.append(row)
        result[split] = rows
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/dbe_kt22"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("manifests/dbe_student_profile_v1")
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-kcs", type=int, default=30)
    parser.add_argument("--max-history", type=int, default=12)
    parser.add_argument("--min-future", type=int, default=5)
    parser.add_argument("--min-eligible", type=int, default=4)
    args = parser.parse_args()

    traces = load_traces(args.raw_dir)
    splits = split_students(sorted(traces), args.seed)
    vocabulary = choose_kc_vocabulary(traces, splits["train"], args.top_kcs)
    rows_by_split = build_rows(
        traces,
        splits,
        vocabulary,
        args.max_history,
        args.min_future,
        args.min_eligible,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for split, rows in rows_by_split.items():
        path = args.output_dir / f"{split}.jsonl"
        write_jsonl(path, rows)
        paths[split] = {"path": str(path), "rows": len(rows), "sha256": sha256(path)}
    manifest = {
        "stage": STAGE,
        "status": "complete",
        "seed": args.seed,
        "source": "DBE-KT22 real longitudinal interactions",
        "raw_students": len(traces),
        "raw_interactions": sum(len(trace) for trace in traces.values()),
        "student_disjoint": True,
        "chronological_cutoff_fraction": 0.5,
        "max_state_history": args.max_history,
        "min_future_attempts_per_kc": args.min_future,
        "min_eligible_kcs": args.min_eligible,
        "kc_vocabulary": vocabulary,
        "kc_vocabulary_selection": "top frequency from training prefixes only",
        "files": paths,
        "target_definition": (
            "Per-student KC correctness after the chronological cutoff with "
            "Beta(1,1) smoothing; weak/strong KCs are the bottom/top two among "
            "KCs with sufficient post-cutoff support."
        ),
        "information_boundary": {
            "encoder": "last max_state_history interactions before cutoff",
            "profile_target": "interactions after cutoff only",
            "student_id_visible_to_encoder": False,
        },
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
