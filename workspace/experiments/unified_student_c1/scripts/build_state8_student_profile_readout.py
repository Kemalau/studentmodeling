#!/usr/bin/env python3
"""Materialize a strict State8 student-profile task and causal transport pairs.

Profile supervision is computed only from the interactions that are supplied to
the recurrent updater.  The decoder query is constant, contains no student ID,
and never contains raw history.  Train/valid/test students are inherited from
the student-disjoint DBE profile manifest.
"""

from __future__ import annotations

import argparse
import heapq
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

try:
    from recurrent_student_model import STATE_TOKEN, build_update_text
except ImportError:  # package-style unit tests
    from scripts.recurrent_student_model import STATE_TOKEN, build_update_text


STAGE = "state8_future_student_profile_readout_v2"
PROFILE_TASK = "read_student_profile"
PROFILE_KIND = "student_profile_schema"
NEUTRAL_TARGET = (
    "<weakness>insufficient_evidence</weakness>\n"
    "<strength>insufficient_evidence</strength>\n"
    "<profile>The supplied latent state does not contain coherent evidence "
    "for a stable student profile.</profile>\n"
    "<evidence_confidence>none</evidence_confidence>"
)


def profile_label(profile: dict[str, Any]) -> str:
    """Canonical, order-invariant label used only by the auxiliary readout."""
    weaknesses = sorted(str(value).strip() for value in profile["weaknesses"])
    if len(weaknesses) != 2 or any(not value for value in weaknesses):
        raise ValueError("Profile label requires exactly two non-empty weaknesses")
    return " || ".join(weaknesses)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def profile_prompt(kc_vocabulary: list[str] | None = None) -> str:
    vocabulary = ""
    if kc_vocabulary:
        vocabulary = (
            "<knowledge_component_vocabulary>"
            + "; ".join(kc_vocabulary)
            + "</knowledge_component_vocabulary>\n"
        )
    return (
        "You are a student modeling system. Read the reusable student profile "
        "stored in the latent state. Return exactly one non-empty <weakness>, "
        "<strength>, <profile>, and <evidence_confidence> tag. Use semicolon-"
        "separated names from the fixed knowledge-component vocabulary in the "
        "first two tags. Predict performance after the observed history; do not "
        "merely repeat the last response.\n"
        f"{vocabulary}"
        f"{STATE_TOKEN}\n"
        "<query>What are this student's current weaknesses and strengths?</query>\n"
        "Answer:"
    )


def history_profile(row: dict[str, Any]) -> dict[str, Any]:
    vocabulary = [str(value) for value in row["kc_vocabulary"]]
    counts: Counter[str] = Counter()
    correct: Counter[str] = Counter()
    for event in row["state_history"]:
        outcome = int(event["correctness"])
        for name in set(str(value) for value in event["kc_names"]):
            if name in vocabulary:
                counts[name] += 1
                correct[name] += outcome
    scored = [
        ((correct[name] + 1) / (counts[name] + 2), -counts[name], name)
        for name in vocabulary
        if counts[name]
    ]
    weak = [value[2] for value in sorted(scored)[:2]]
    strong = [
        value[2]
        for value in sorted(scored, reverse=True)
        if value[2] not in set(weak)
    ][:2]
    support = sum(counts.values())
    confidence = "high" if support >= 10 else "medium" if support >= 6 else "low"
    weak_text = "; ".join(weak) if weak else "insufficient_evidence"
    strong_text = "; ".join(strong) if strong else "insufficient_evidence"
    summary = (
        f"The student currently needs support with {weak_text} while showing "
        f"relative strength in {strong_text}."
        if weak and strong
        else "The observed interactions are insufficient for a stable profile."
    )
    target = (
        f"<weakness>{weak_text}</weakness>\n"
        f"<strength>{strong_text}</strength>\n"
        f"<profile>{summary}</profile>\n"
        f"<evidence_confidence>{confidence}</evidence_confidence>"
    )
    return {
        "weaknesses": weak,
        "strengths": strong,
        "confidence": confidence,
        "visible_kc_attempts": support,
        "target": target,
    }


def future_profile(row: dict[str, Any]) -> dict[str, Any]:
    """Build the supervised post-cutoff profile hidden from the decoder.

    The recurrent updater sees only ``state_history``.  These labels are ordinary
    training/evaluation targets derived from later interactions, never prompt
    features.  This aligns training with the revised C4 prospective-state claim.
    """
    vocabulary = [str(value) for value in row["kc_vocabulary"]]
    weak = [str(value) for value in row["weak_kcs"]]
    strong = [str(value) for value in row["strong_kcs"]]
    if len(weak) != 2 or len(strong) != 2:
        raise ValueError("Future profile requires exactly two weak and strong KCs")
    if not set(weak + strong).issubset(set(vocabulary)):
        raise ValueError("Future profile contains a KC outside the fixed vocabulary")
    if set(weak) & set(strong):
        raise ValueError("Future weak and strong KCs must be disjoint")
    weak_text = "; ".join(weak)
    strong_text = "; ".join(strong)
    future_support = sum(
        int(count)
        for count, eligible in zip(
            row.get("future_counts", []), row.get("future_eligible_mask", []), strict=True
        )
        if eligible
    )
    target = (
        f"<weakness>{weak_text}</weakness>\n"
        f"<strength>{strong_text}</strength>\n"
        "<profile>The student's future performance is predicted to need support "
        f"with {weak_text} while remaining relatively strong in {strong_text}.</profile>\n"
        "<evidence_confidence>medium</evidence_confidence>"
    )
    return {
        "weaknesses": weak,
        "strengths": strong,
        "confidence": "medium",
        "visible_kc_attempts": future_support,
        "target": target,
        "target_provenance": "post_cutoff_observed_correctness_by_kc",
    }


def profile_row(row: dict[str, Any]) -> dict[str, Any]:
    observed_profile = history_profile(row)
    profile = future_profile(row)
    direct = str(row["prompt"])
    decode = profile_prompt([str(value) for value in row["kc_vocabulary"]])
    if decode.count(STATE_TOKEN) != 1 or "<student_history>" in decode.lower():
        raise AssertionError("Profile decoder is not closed-state")
    updates = [build_update_text(event) for event in row["state_history"]]
    vocabulary = [str(value) for value in row["kc_vocabulary"]]
    observed_names = {
        str(name)
        for event in row["state_history"]
        for name in event["kc_names"]
    }
    observed_kc_mask = [name in observed_names for name in vocabulary]
    return {
        "example_id": f"profile-readout:{row['split']}:{row['student_id']}",
        "student_id": str(row["student_id"]),
        "task": PROFILE_TASK,
        "source": "dbe_kt22",
        "sequence_semantics": "chronological_observed_profile",
        "observed_interactions": len(updates),
        "update_texts": updates,
        "decode_text": decode,
        "direct_text": direct,
        "full_context_text": direct,
        "target": profile["target"],
        "target_label": "|".join(profile["weaknesses"]),
        "candidate_labels": [],
        "candidate_targets": {},
        "history_length": len(updates),
        "profile": profile,
        "observed_profile": observed_profile,
        "future_weaknesses": list(row["weak_kcs"]),
        "future_strengths": list(row["strong_kcs"]),
        # Numeric post-cutoff targets are train-only auxiliary supervision for
        # the reusable-state objective.  They are never rendered into either
        # the updater text or the closed decoder prompt.
        "future_smoothed_rates": list(row["future_smoothed_rates"]),
        "future_eligible_mask": list(row["future_eligible_mask"]),
        # Input-side support only: this is derived from the exact interactions
        # already rendered to the updater.  It contains no future outcome or
        # future-support information and lets C4a separate identifiable
        # ownership from unseen-KC transfer.
        "observed_kc_mask": observed_kc_mask,
        "raw_history_visible_to_decoder": False,
        "student_id_visible_to_decoder": False,
    }


def attach_ownership_negatives(
    rows: list[dict[str, Any]],
    seed: int,
    reliable_kc_mask: list[bool] | None = None,
    negative_count: int = 1,
    ownership_scope: str = "all",
) -> None:
    """Attach a deterministic panel of mismatched-student targets to every row.

    Negatives are selected using only labels inside the current split.  They
    prioritize maximum jointly supported KCs and then future-rate disagreement,
    with stable hash tie breaking.  A panel prevents the ownership objective
    from fitting one fixed hardest student.  The first member is also written
    to the legacy scalar fields for backwards-compatible audits.  No negative
    history, student ID, or label is exposed to the updater/decoder; these
    fields are consumed only by the training loss.
    """
    if len(rows) < 2:
        raise ValueError("Ownership supervision requires at least two students")
    if negative_count <= 0:
        raise ValueError("Ownership negative count must be positive")
    if ownership_scope not in {"all", "seen"}:
        raise ValueError("Ownership scope must be all or seen")
    if reliable_kc_mask is None:
        reliable_kc_mask = [True] * len(rows[0]["future_eligible_mask"])
    if len(reliable_kc_mask) != len(rows[0]["future_eligible_mask"]):
        raise ValueError("Ownership reliable-KC mask width mismatch")
    if not any(reliable_kc_mask):
        raise ValueError("Ownership supervision has no train-reliable KC")
    for row in rows:
        observed = row.get("observed_kc_mask")
        if ownership_scope == "seen":
            if not isinstance(observed, list) or len(observed) != len(reliable_kc_mask):
                raise ValueError("Seen ownership requires an aligned observed-KC mask")
        else:
            observed = [True] * len(reliable_kc_mask)
        mask = [
            bool(value) and reliable and bool(seen)
            for value, reliable, seen in zip(
                row["future_eligible_mask"], reliable_kc_mask, observed, strict=True
            )
        ]
        row["ownership_training_mask"] = mask
        row["ownership_training_scope"] = ownership_scope
        rates = row["future_smoothed_rates"]
        candidates: list[tuple[int, float, int, dict[str, Any], list[bool]]] = []
        for other in rows:
            if other["student_id"] == row["student_id"]:
                continue
            overlap = [
                left and bool(right) and reliable
                for left, right, reliable in zip(
                    mask,
                    other["future_eligible_mask"],
                    reliable_kc_mask,
                    strict=True,
                )
            ]
            overlap_count = sum(overlap)
            if overlap_count <= 0:
                continue
            distance = sum(
                (float(left) - float(right)) ** 2
                for left, right, keep in zip(
                    rates, other["future_smoothed_rates"], overlap, strict=True
                )
                if keep
            ) / overlap_count
            tie = stable_score(
                seed,
                f"{row['student_id']}\x1f{other['student_id']}",
            )
            candidates.append((overlap_count, float(distance), tie, other, overlap))
        if not candidates:
            if ownership_scope == "seen" and not any(mask):
                # The profile row remains useful for language/retention
                # training, but contributes no C4a numeric gradient.
                row["ownership_training_eligible"] = False
                row["ownership_negative_panel_size"] = 0
                row["ownership_supervision_visible_at_inference"] = False
                continue
            raise ValueError(
                f"No ownership negative with a shared supported KC for {row['student_id']}"
            )
        panel = sorted(
            candidates,
            key=lambda value: (value[0], value[1], value[2]),
            reverse=True,
        )[:negative_count]
        _count, distance, _tie, negative, overlap = panel[0]
        row["ownership_negative_student_id"] = str(negative["student_id"])
        row["ownership_negative_future_smoothed_rates"] = list(
            negative["future_smoothed_rates"]
        )
        row["ownership_overlap_mask"] = overlap
        row["ownership_target_mse"] = distance
        row["ownership_negative_student_ids"] = [
            str(value[3]["student_id"]) for value in panel
        ]
        row["ownership_negative_future_smoothed_rates_panel"] = [
            list(value[3]["future_smoothed_rates"]) for value in panel
        ]
        row["ownership_overlap_mask_panel"] = [list(value[4]) for value in panel]
        row["ownership_target_mse_panel"] = [float(value[1]) for value in panel]
        row["ownership_negative_panel_size"] = len(panel)
        row["ownership_training_eligible"] = True
        row["ownership_supervision_visible_at_inference"] = False


def stable_score(seed: int, value: str) -> int:
    payload = f"{seed}\x1f{value}".encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def retention_rows(path: Path, limit: int, seed: int) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    # The unified recurrent corpus is multi-GB.  Keep only a bounded hash-top-k
    # heap per task instead of materializing the corpus on the submission host.
    per_task_capacity = limit
    heaps: dict[str, list[tuple[int, int, dict[str, Any]]]] = defaultdict(list)
    for row in iter_jsonl(path):
        task = str(row.get("task") or "")
        if task == PROFILE_TASK:
            continue
        example_id = str(row["example_id"])
        score = stable_score(seed, example_id)
        tie = stable_score(seed + 10_003, example_id)
        item = (-score, -tie, row)
        heap = heaps[task]
        if len(heap) < per_task_capacity:
            heapq.heappush(heap, item)
        elif item > heap[0]:
            heapq.heapreplace(heap, item)
    if not heaps:
        raise ValueError("Retention corpus is empty")
    by_task = {task: [item[2] for item in heap] for task, heap in heaps.items()}
    tasks = sorted(by_task)
    base, remainder = divmod(limit, len(tasks))
    selected: list[dict[str, Any]] = []
    for index, task in enumerate(tasks):
        count = base + int(index < remainder)
        rows = sorted(
            by_task[task],
            key=lambda row: (stable_score(seed, str(row["example_id"])), str(row["example_id"])),
        )
        if len(rows) < count:
            raise ValueError(f"Retention task {task} has only {len(rows)} rows")
        selected.extend(rows[:count])
    return sorted(
        selected,
        key=lambda row: (stable_score(seed + 1, str(row["example_id"])), str(row["example_id"])),
    )


def pair_distance(left: dict[str, Any], right: dict[str, Any]) -> int:
    return len(
        set(left["profile"]["weaknesses"]).symmetric_difference(
            right["profile"]["weaknesses"]
        )
    )


def build_pairs(
    rows: list[dict[str, Any]], count: int, seed: int
) -> list[dict[str, Any]]:
    base_qualified = [
        row
        for row in rows
        if row["profile"]["weaknesses"]
        and row["profile"]["strengths"]
        and row["profile"]["visible_kc_attempts"] >= 4
    ]
    qualified = [
        row
        for row in base_qualified
        if any(
            other["student_id"] != row["student_id"]
            and other["target"] == row["target"]
            for other in base_qualified
        )
    ]
    if len(qualified) < 3:
        raise ValueError("Too few profile rows for transport pairs")
    rng = random.Random(seed)
    result = []
    used: set[tuple[str, str]] = set()
    for pair_index in range(count):
        native = qualified[pair_index % len(qualified)]
        donors = [
            row
            for row in base_qualified
            if row["student_id"] != native["student_id"]
            and pair_distance(native, row) >= 2
        ]
        if not donors:
            raise ValueError("No different-profile donor")
        donor = donors[rng.randrange(len(donors))]
        key = (native["student_id"], donor["student_id"])
        attempts = 0
        while key in used and attempts < len(donors):
            donor = donors[(donors.index(donor) + 1) % len(donors)]
            key = (native["student_id"], donor["student_id"])
            attempts += 1
        used.add(key)
        same = next(
            row
            for row in base_qualified
            if row["student_id"] != native["student_id"]
            and row["target"] == native["target"]
        )
        decode_text = str(native.get("decode_text") or profile_prompt())
        if decode_text != str(donor.get("decode_text") or profile_prompt()) or decode_text != str(
            same.get("decode_text") or profile_prompt()
        ):
            raise ValueError("Profile transport requires one fixed decoder vocabulary")
        result.append(
            {
                "pair_id": f"state8-profile:{pair_index:05d}",
                "task": PROFILE_TASK,
                "source": "dbe_kt22",
                "causal_target_kind": PROFILE_KIND,
                "decode_text": decode_text,
                "native_update_texts": native["update_texts"],
                "recipient_native_update_texts": native["update_texts"],
                "donor_update_texts": donor["update_texts"],
                "same_rule_control_update_texts": same["update_texts"],
                "native_target": native["target"],
                "donor_target": donor["target"],
                "neutral_target": NEUTRAL_TARGET,
                "native_profile": native["profile"],
                "donor_profile": donor["profile"],
                "same_profile": same["profile"],
                "native_target_label": profile_label(native["profile"]),
                "donor_target_label": profile_label(donor["profile"]),
                "native_control_role": "recipient_native",
                "recipient_id": native["student_id"],
                "donor_id": donor["student_id"],
                "same_profile_control_id": same["student_id"],
                "query_held_fixed": True,
                "only_state_identity_changes": True,
                "raw_history_visible_to_decoder": False,
                "student_id_visible_to_decoder": False,
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-manifest-dir", type=Path, required=True)
    parser.add_argument("--retention-train-jsonl", type=Path, required=True)
    parser.add_argument("--retention-valid-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--retention-train", type=int, default=337)
    parser.add_argument("--retention-valid", type=int, default=100)
    parser.add_argument("--causal-train-pairs", type=int, default=300)
    parser.add_argument("--causal-valid-pairs", type=int, default=50)
    parser.add_argument("--ownership-min-train-kc-support", type=int, default=1)
    parser.add_argument("--ownership-negative-count", type=int, default=1)
    parser.add_argument("--ownership-scope", choices=("all", "seen"), default="all")
    parser.add_argument("--seed", type=int, default=20260817)
    args = parser.parse_args()

    raw = {
        split: read_jsonl(args.profile_manifest_dir / f"{split}.jsonl")
        for split in ("train", "valid", "test")
    }
    students = {split: {str(row["student_id"]) for row in values} for split, values in raw.items()}
    if any(students[a] & students[b] for a, b in (("train", "valid"), ("train", "test"), ("valid", "test"))):
        raise ValueError("Profile student split overlap")
    profiles = {split: [profile_row(row) for row in values] for split, values in raw.items()}
    if args.ownership_min_train_kc_support <= 0:
        raise ValueError("Ownership minimum train-KC support must be positive")
    kc_width = len(profiles["train"][0]["future_eligible_mask"])
    train_kc_support = [0] * kc_width
    for row in profiles["train"]:
        for index, eligible in enumerate(row["future_eligible_mask"]):
            train_kc_support[index] += int(bool(eligible))
    reliable_kc_mask = [
        count >= args.ownership_min_train_kc_support for count in train_kc_support
    ]
    if not any(reliable_kc_mask):
        raise ValueError("No KC meets the ownership train-support threshold")
    for split_index, split in enumerate(("train", "valid", "test")):
        attach_ownership_negatives(
            profiles[split],
            args.seed + 10_000 + split_index,
            reliable_kc_mask,
            args.ownership_negative_count,
            args.ownership_scope,
        )
    train = profiles["train"] + retention_rows(
        args.retention_train_jsonl, args.retention_train, args.seed
    )
    valid = profiles["valid"] + retention_rows(
        args.retention_valid_jsonl, args.retention_valid, args.seed + 1
    )
    rng = random.Random(args.seed)
    rng.shuffle(train)
    rng.shuffle(valid)
    causal_train = build_pairs(profiles["train"], args.causal_train_pairs, args.seed + 2)
    causal_valid = build_pairs(profiles["valid"], args.causal_valid_pairs, args.seed + 3)

    outputs = {
        "train.jsonl": train,
        "valid.jsonl": valid,
        "test_profiles.jsonl": profiles["test"],
        "causal_train.jsonl": causal_train,
        "causal_valid.jsonl": causal_valid,
    }
    for name, values in outputs.items():
        write_jsonl(args.output_dir / name, values)
    receipt = {
        "stage": STAGE,
        "status": "complete",
        "profile_students": {split: len(values) for split, values in profiles.items()},
        "student_disjoint": True,
        "chronological_future_targets": True,
        "profile_supervision_scope": "post_cutoff_targets_from_student_disjoint_training_students",
        "future_targets_used_for_training": True,
        "future_targets_visible_to_decoder": False,
        "raw_history_visible_to_decoder": False,
        "student_id_visible_to_decoder": False,
        "ownership_auxiliary": {
            "target": "post_cutoff_smoothed_KC_correctness",
            "negative": "same_split_hard_mismatched_student_panel",
            "negative_selection": "maximize_shared_support_then_top_rate_disagreement",
            "negative_count_requested": args.ownership_negative_count,
            "scope": args.ownership_scope,
            "minimum_shared_supported_kcs": 1,
            "minimum_training_students_per_kc": args.ownership_min_train_kc_support,
            "train_kc_support": train_kc_support,
            "reliable_kc_indices": [
                index for index, keep in enumerate(reliable_kc_mask) if keep
            ],
            "visible_to_updater": False,
            "visible_to_decoder": False,
            "visible_at_inference": False,
        },
        "constant_profile_query": profile_prompt(),
        "rows": {name: len(values) for name, values in outputs.items()},
        "sha256": {name: sha256(args.output_dir / name) for name in outputs},
    }
    (args.output_dir / "receipt.json").write_text(
        json.dumps(receipt, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
