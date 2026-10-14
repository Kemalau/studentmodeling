#!/usr/bin/env python3
"""Exploratory natural-language diagnosis intervention on a closed State8 state.

This complements the frozen answer-level C4 gate.  It reuses the same
donor/recipient/same-rule pairs, but asks the decoder to emit a canonical label,
natural-language diagnosis, and evidence after intervening on the recurrent
state.  Exact label movement is only an auxiliary readout; the emitted judge
panel is scored separately by the independent semantic T3 judge.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch

try:
    from evaluate_recurrent_causal_gate import random_state_like, state_for_row
    from evaluate_recurrent_formal_protocols import generate_one, load_wrapper
    from state8_natural_curriculum_view import malrule_description
    from state_swap_misconception_smoke import (
        family_of,
        label_of,
        load_pair_manifest,
        read_jsonl,
    )
    from t3_judge_utils import extract_tag, natural_schema_valid, write_jsonl
except ImportError:  # package-style unit tests
    from scripts.evaluate_recurrent_causal_gate import random_state_like, state_for_row
    from scripts.evaluate_recurrent_formal_protocols import generate_one, load_wrapper
    from scripts.state8_natural_curriculum_view import malrule_description
    from scripts.state_swap_misconception_smoke import (
        family_of,
        label_of,
        load_pair_manifest,
        read_jsonl,
    )
    from scripts.t3_judge_utils import extract_tag, natural_schema_valid, write_jsonl


CONDITIONS = ("native", "noop", "samemal", "random", "swap")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def diagnosis_text(label: str) -> str:
    return malrule_description(label) or label.replace(".", " ").replace("_", " ")


def diagnosis_target(label: str) -> str:
    return (
        f"<label>{label}</label>\n"
        f"<diagnosis>{diagnosis_text(label)}</diagnosis>\n"
        "<evidence>The student's latent state supports this systematic error mechanism.</evidence>"
    )


def diagnosis_prompt(row: dict[str, Any]) -> str:
    query = row.get("query") or {}
    problem = str(query.get("problem_text") or "")
    correct = str(query.get("correct_answer") or "")
    if not problem or not correct:
        raise ValueError(f"C4 row lacks a complete query: {row.get('example_id')}")
    return (
        "You are modeling one student's mathematical behavior. Diagnose the specific "
        "systematic misconception encoded in the student's latent state and explain "
        "how it bears on the current problem. Return exactly one non-empty <label>, "
        "<diagnosis>, and <evidence> tag.\n"
        "<STUDENT_STATE>\n"
        "<query>\n"
        f"Problem: {problem}\n"
        f"Correct answer: {correct}\n"
        "</query>\n"
        "Answer:"
    )


def category(predicted_label: str | None, donor_label: str, recipient_label: str) -> str:
    if predicted_label == donor_label:
        return "donor"
    if predicted_label == recipient_label:
        return "recipient"
    return "other"


def summarize_exact(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_pair: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        condition = str(record["condition"])
        grouped[condition].append(record)
        by_pair[int(record["pair_idx"])][condition] = record
    if set(grouped) != set(CONDITIONS):
        raise ValueError(f"Diagnosis-C4 condition drift: {sorted(grouped)}")
    result: dict[str, Any] = {"by_condition": {}}
    for condition in CONDITIONS:
        rows = grouped[condition]
        counts = Counter(str(row["category"]) for row in rows)
        result["by_condition"][condition] = {
            "n": len(rows),
            "natural_schema_valid_rate": sum(
                bool(row["natural_schema_valid"]) for row in rows
            )
            / len(rows),
            "exact_label_to_donor_rate": counts["donor"] / len(rows),
            "exact_label_to_recipient_rate": counts["recipient"] / len(rows),
            "exact_label_other_rate": counts["other"] / len(rows),
        }
    result["noop_output_identical_to_native_rate"] = sum(
        conditions["noop"]["raw_output"] == conditions["native"]["raw_output"]
        for conditions in by_pair.values()
    ) / len(by_pair)
    result["exact_label_D_specific"] = (
        result["by_condition"]["swap"]["exact_label_to_donor_rate"]
        - result["by_condition"]["random"]["exact_label_to_donor_rate"]
    )
    result["exploratory_only"] = True
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--recurrent-state", type=Path, required=True)
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument("--pair-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--expected-adapter-sha256",
        help="Fail closed unless the evaluated adapter matches this immutable receipt.",
    )
    parser.add_argument(
        "--expected-recurrent-state-sha256",
        help="Fail closed unless the evaluated state matches this immutable receipt.",
    )
    parser.add_argument("--state-token-alias", default="<|vision_pad|>")
    parser.add_argument("--state-architecture", choices=("scalar", "layerwise"), default="layerwise")
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--num-pairs", type=int, default=20)
    parser.add_argument(
        "--pair-start",
        type=int,
        default=0,
        help="Zero-based first pair in the frozen manifest prefix.",
    )
    parser.add_argument(
        "--pair-count",
        type=int,
        help="Number of consecutive pairs to evaluate; defaults to the remainder.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.num_pairs <= 0:
        raise ValueError("--num-pairs must be positive")
    if args.pair_start < 0 or args.pair_start >= args.num_pairs:
        raise ValueError("--pair-start must select a pair in the requested prefix")
    pair_count = args.pair_count or (args.num_pairs - args.pair_start)
    if pair_count <= 0 or args.pair_start + pair_count > args.num_pairs:
        raise ValueError("--pair-count extends beyond the requested manifest prefix")
    observed_artifacts = {
        "adapter_model": file_sha256(args.adapter / "adapter_model.safetensors"),
        "recurrent_state": file_sha256(args.recurrent_state),
    }
    expected_artifacts = {
        "adapter_model": args.expected_adapter_sha256,
        "recurrent_state": args.expected_recurrent_state_sha256,
    }
    for key, expected in expected_artifacts.items():
        if expected is not None and observed_artifacts[key] != expected:
            raise ValueError(
                f"Diagnosis-C4 {key} hash drift: "
                f"expected={expected} observed={observed_artifacts[key]}"
            )

    rows = read_jsonl(args.jsonl, "malrule_next_response")
    all_pairs, all_controls = load_pair_manifest(
        args.pair_manifest,
        rows,
        "malrule_next_response",
        "response",
        args.num_pairs,
        allow_prefix=True,
    )
    pair_stop = args.pair_start + pair_count
    pairs = all_pairs[args.pair_start : pair_stop]
    controls = all_controls[args.pair_start : pair_stop]
    tokenizer, wrapper = load_wrapper(
        args.model,
        str(args.adapter),
        args.recurrent_state,
        args.state_token_alias,
        args.state_architecture,
    )
    alias = wrapper.state_token_rendered_alias
    unique: dict[str, dict[str, Any]] = {}
    for donor, recipient in pairs:
        unique[str(donor["example_id"])] = donor
        unique[str(recipient["example_id"])] = recipient
    for control in controls:
        if control is not None:
            unique[str(control["example_id"])] = control
    state_cache: dict[str, torch.Tensor] = {}
    for index, (example_id, row) in enumerate(unique.items(), start=1):
        state_cache[example_id] = state_for_row(
            wrapper, tokenizer, row, alias, args.max_length
        ).detach().cpu()
        if index == 1 or index % 10 == 0 or index == len(unique):
            print(json.dumps({"states": index, "total": len(unique)}), flush=True)

    records: list[dict[str, Any]] = []
    judge_predictions: list[dict[str, Any]] = []
    judge_eval: list[dict[str, Any]] = []
    device = next(wrapper.parameters()).device
    for pair_index, ((donor, recipient), same_control) in enumerate(
        zip(pairs, controls, strict=True), start=args.pair_start + 1
    ):
        donor_label = label_of(donor)
        recipient_label = label_of(recipient)
        if donor_label == recipient_label or family_of(donor) != family_of(recipient):
            raise ValueError(f"Invalid diagnosis-C4 pair {pair_index}")
        native_state = state_cache[str(recipient["example_id"])].to(device)
        condition_states = {
            "native": native_state,
            "noop": native_state.clone(),
            "samemal": state_cache[str(same_control["example_id"])].to(device),
            "random": random_state_like(
                native_state, args.seed + 1_000_003 * pair_index
            ),
            "swap": state_cache[str(donor["example_id"])].to(device),
        }
        prompt = diagnosis_prompt(recipient)
        for condition in CONDITIONS:
            output = generate_one(
                wrapper,
                tokenizer,
                condition_states[condition],
                prompt,
                alias,
                args.max_length,
                args.max_new_tokens,
            )
            predicted_label = extract_tag(output, "label")
            record = {
                "pair_idx": pair_index,
                "condition": condition,
                "donor_id": donor["example_id"],
                "recipient_id": recipient["example_id"],
                "samemal_id": same_control["example_id"],
                "family": family_of(recipient),
                "donor_label": donor_label,
                "recipient_label": recipient_label,
                "predicted_label": predicted_label,
                "category": category(predicted_label, donor_label, recipient_label),
                "natural_schema_valid": natural_schema_valid(output),
                "raw_output": output,
                "decode_prompt": prompt,
            }
            records.append(record)
            for reference_role, reference_label in (
                ("donor", donor_label),
                ("recipient", recipient_label),
            ):
                example_id = f"diagnosis-c4:{pair_index:04d}:{condition}:{reference_role}"
                judge_predictions.append(
                    {
                        "example_id": example_id,
                        "task": "diagnose_misconception",
                        "source": "malrule",
                        "protocol": "closed_state_diagnosis_c4",
                        "condition": condition,
                        "reference_role": reference_role,
                        "pair_idx": pair_index,
                        "target": diagnosis_target(reference_label),
                        "raw_output": output,
                    }
                )
                judge_eval.append(
                    {
                        "example_id": example_id,
                        "task": "diagnose_misconception",
                        "direct_text": prompt,
                    }
                )
        print(
            json.dumps(
                {
                    "pairs": pair_index - args.pair_start,
                    "total": len(pairs),
                    "absolute_pair_idx": pair_index,
                }
            ),
            flush=True,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "predictions.jsonl", records)
    write_jsonl(args.output_dir / "judge_predictions.jsonl", judge_predictions)
    write_jsonl(args.output_dir / "judge_eval.jsonl", judge_eval)
    summary = {
        "stage": "state8_closed_diagnosis_c4_smoke_v1",
        "num_pairs": len(pairs),
        "manifest_prefix_pairs": args.num_pairs,
        "pair_start": args.pair_start,
        "pair_stop": pair_stop,
        "pair_seed": args.seed,
        "conditions": list(CONDITIONS),
        "state_architecture": args.state_architecture,
        "state_shape": list(next(iter(state_cache.values())).shape[1:]),
        "raw_history_visible_to_decoder": False,
        "persistent_history_kv_cache": False,
        "artifact_sha256": observed_artifacts,
        "exact_label_auxiliary": summarize_exact(records),
        "semantic_judgment_pending": True,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
