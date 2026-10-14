#!/usr/bin/env python3
"""Build natural Eedi same-question state-swap diagnosis twins.

Different incorrect options for the same Eedi question often carry different
misconception labels.  The decoder query is held byte-identical while native
and donor updater trajectories contain the respective wrong responses.  A
same-label state from another question supplies the no-change control.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

try:
    from augment_recurrent_causal_diagnosis import diagnosis_decode_text
    from state8_current_query_routing import apply_current_query_routing
    from state8_natural_curriculum_view import natural_target
    from t3_judge_utils import natural_schema_valid
except ImportError:
    from scripts.augment_recurrent_causal_diagnosis import diagnosis_decode_text
    from scripts.state8_current_query_routing import apply_current_query_routing
    from scripts.state8_natural_curriculum_view import natural_target
    from scripts.t3_judge_utils import natural_schema_valid


OPTION_SUFFIX = re.compile(r"_[A-D]$")


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def is_eedi(row: dict[str, Any]) -> bool:
    return (
        str(row.get("task") or "") == "diagnose_misconception"
        and str(row.get("source") or "") == "eedi_mining"
        and bool(row.get("update_texts"))
    )


def question_id(row: dict[str, Any]) -> str:
    metadata = row.get("original_metadata") or {}
    value = str(metadata.get("question_id") or metadata.get("original_example_id") or "")
    if not value:
        raise ValueError(f"Eedi row lacks question identity: {row.get('example_id')}")
    return value


def base_question(row: dict[str, Any]) -> str:
    return OPTION_SUFFIX.sub("", question_id(row))


def stable_order(values: Iterable[str], key: str) -> list[str]:
    return sorted(
        values,
        key=lambda value: hashlib.sha256(f"{key}:{value}".encode()).hexdigest(),
    )


def routed_view(row: dict[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(row)
    audit = apply_current_query_routing([value], sources={"eedi_mining"})
    if audit.get("current_query_routed_rows") != 1:
        raise ValueError(f"Eedi row was not query-routed: {row.get('example_id')}")
    return value


def target(row: dict[str, Any]) -> str:
    value = natural_target(row)
    if not natural_schema_valid(value):
        raise ValueError(f"Invalid Eedi natural target: {row.get('example_id')}")
    return value


def choose_control(
    by_label: dict[str, list[dict[str, Any]]],
    *,
    label: str,
    excluded_ids: set[str],
    excluded_base: str,
    key: str,
) -> dict[str, Any] | None:
    candidates = [
        row
        for row in by_label.get(label, [])
        if str(row["example_id"]) not in excluded_ids
        and base_question(row) != excluded_base
    ]
    if not candidates:
        return None
    ordered = stable_order((str(row["example_id"]) for row in candidates), key)
    by_id = {str(row["example_id"]): row for row in candidates}
    return by_id[ordered[0]]


def exclusion_sets(
    rows: Iterable[dict[str, Any]],
) -> tuple[set[str], set[str]]:
    """Return every source entity and question family used by frozen twins."""
    entity_ids: set[str] = set()
    base_ids: set[str] = set()
    for row in rows:
        for key in ("recipient_id", "donor_id", "same_rule_control_id"):
            value = str(row.get(key) or "")
            if value:
                entity_ids.add(value)
        for key in ("same_question_base_id", "same_rule_control_base_id"):
            value = str(row.get(key) or "")
            if value:
                base_ids.add(value)
    return entity_ids, base_ids


def build_twins(
    rows: list[dict[str, Any]],
    *,
    twins: int,
    seed: int,
    offset: int = 0,
    excluded_entity_ids: set[str] | None = None,
    excluded_base_ids: set[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if twins <= 0 or offset < 0:
        raise ValueError("twins must be positive and offset non-negative")
    forbidden_entities = set(excluded_entity_ids or ())
    forbidden_bases = set(excluded_base_ids or ())
    eedi = [
        row
        for row in rows
        if is_eedi(row)
        and str(row["example_id"]) not in forbidden_entities
        and base_question(row) not in forbidden_bases
    ]
    by_question: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in eedi:
        by_question[base_question(row)].append(row)
        by_label[str(row["target_label"])].append(row)

    candidates: list[tuple[str, dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    for base, group in by_question.items():
        if base in forbidden_bases:
            continue
        by_id = {str(row["example_id"]): row for row in group}
        ordered_ids = stable_order(by_id, f"{seed}:question:{base}")
        selected_triplet = None
        for recipient_id in ordered_ids:
            recipient = by_id[recipient_id]
            recipient_label = str(recipient["target_label"])
            for donor_id in ordered_ids:
                donor = by_id[donor_id]
                if donor_id != recipient_id and str(donor["target_label"]) != recipient_label:
                    excluded = {str(recipient["example_id"]), str(donor["example_id"])}
                    control = choose_control(
                        by_label,
                        label=recipient_label,
                        excluded_ids=excluded,
                        excluded_base=base,
                        key=f"{seed}:control:{base}:{recipient_id}",
                    )
                    if control is not None:
                        selected_triplet = (recipient, donor, control)
                        break
            if selected_triplet is not None:
                break
        if selected_triplet is not None:
            candidates.append((base, *selected_triplet))

    ordered = sorted(
        candidates,
        key=lambda value: hashlib.sha256(f"{seed}:twin:{value[0]}".encode()).hexdigest(),
    )
    if len(ordered) < offset + twins:
        raise ValueError(
            f"Need offset {offset} plus {twins} Eedi twins, found {len(ordered)}"
        )

    output: list[dict[str, Any]] = []
    for index, (base, recipient, donor, control) in enumerate(
        ordered[offset : offset + twins]
    ):
        recipient_view = routed_view(recipient)
        donor_view = routed_view(donor)
        control_view = routed_view(control)
        recipient_decode = diagnosis_decode_text({"decode_text": recipient_view["decode_text"]})
        donor_decode = diagnosis_decode_text({"decode_text": donor_view["decode_text"]})
        if recipient_decode != donor_decode:
            raise ValueError(f"Same-question Eedi query drift: {base}")
        native_label = str(recipient["target_label"])
        donor_label = str(donor["target_label"])
        native_target = target(recipient)
        donor_target = target(donor)
        common = {
            "donor_id": str(donor["example_id"]),
            "recipient_id": str(recipient["example_id"]),
            "same_rule_control_id": str(control["example_id"]),
            "same_rule_control_base_id": base_question(control),
            "same_rule_recipient_id": str(recipient["example_id"]),
            "donor_label": donor_label,
            "recipient_label": native_label,
            "family": "eedi",
            "source": "eedi_mining",
            "task": "diagnose_misconception",
            "causal_target_kind": "diagnosis_natural_schema",
            "decode_text": recipient_decode,
            "native_direct_text": str(recipient_view["direct_text"]),
            "donor_direct_text": str(donor_view["direct_text"]),
            "native_target": native_target,
            "donor_target": donor_target,
            "native_target_label": native_label,
            "donor_target_label": donor_label,
            "candidate_labels": [native_label, donor_label],
            "candidate_targets": {native_label: native_target, donor_label: donor_target},
            "native_update_texts": list(recipient["update_texts"]),
            "donor_update_texts": list(donor["update_texts"]),
            "recipient_native_update_texts": list(recipient["update_texts"]),
            "same_rule_control_update_texts": list(control["update_texts"]),
            "same_question_base_id": base,
            "same_question_query_identical": True,
            "current_item_visible_to_decoder": True,
            "student_response_visible_to_decoder": False,
            "raw_history_visible_to_decoder": False,
            "output_schema": "label_diagnosis_evidence_v1",
            "canonical_label_auxiliary_supervision": True,
            "information_route": "same_eedi_question_plus_swapped_student_state",
        }
        identity = hashlib.sha256(
            (
                f"{seed}:{base}:{recipient['example_id']}:"
                f"{donor['example_id']}:{control['example_id']}"
            ).encode("utf-8")
        ).hexdigest()[:20]
        native = dict(common)
        native.update(
            pair_id=f"state8-eedi-same-question:{identity}:native",
            native_control_role="recipient_native",
            native_state_source_id=str(recipient["example_id"]),
        )
        same_rule = dict(common)
        same_rule.update(
            pair_id=f"state8-eedi-same-question:{identity}:same-rule",
            native_update_texts=list(control["update_texts"]),
            native_direct_text=str(control_view["direct_text"]),
            native_control_role="same_rule_native",
            native_state_source_id=str(control["example_id"]),
        )
        output.extend((native, same_rule))

    roles = Counter(str(row["native_control_role"]) for row in output)
    labels = Counter(str(row["recipient_label"]) for row in output[::2])
    return output, {
        "stage": "state8_eedi_same_question_causal_twins_v1",
        "status": "passed",
        "source_eedi_rows": len(eedi),
        "qualified_question_groups": len(candidates),
        "twins": twins,
        "output_pairs": len(output),
        "role_counts": dict(sorted(roles.items())),
        "recipient_label_counts": dict(sorted(labels.items())),
        "same_question_query_identical": True,
        "current_item_visible_to_decoder": True,
        "student_response_visible_to_decoder": False,
        "raw_history_visible_to_decoder": False,
        "excluded_entity_ids": len(forbidden_entities),
        "excluded_base_ids": len(forbidden_bases),
        "seed": seed,
        "offset": offset,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--twins", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument(
        "--exclude-jsonl",
        type=Path,
        action="append",
        default=[],
        help="Previously frozen twin panels whose entities and questions are excluded.",
    )
    args = parser.parse_args()
    excluded_rows = [
        row for path in args.exclude_jsonl for row in read_jsonl(path)
    ]
    excluded_entities, excluded_bases = exclusion_sets(excluded_rows)
    rows, receipt = build_twins(
        read_jsonl(args.source_jsonl),
        twins=args.twins,
        seed=args.seed,
        offset=args.offset,
        excluded_entity_ids=excluded_entities,
        excluded_base_ids=excluded_bases,
    )
    write_jsonl(args.output_jsonl, rows)
    receipt.update(
        source_jsonl=str(args.source_jsonl), source_sha256=sha256(args.source_jsonl),
        output_jsonl=str(args.output_jsonl), output_sha256=sha256(args.output_jsonl),
        exclusion_jsonl=[str(path) for path in args.exclude_jsonl],
        exclusion_sha256=[sha256(path) for path in args.exclude_jsonl],
    )
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
