#!/usr/bin/env python3
"""Diagnose the State8-to-decoder interface with canonical-label readout.

This is a read-only diagnostic, not an authoritative task metric.  It reuses
the frozen identical-query Eedi C4 twins and the production recurrent states,
but asks the same decoder to emit only the canonical misconception identifier.
The short target removes natural-language teacher-forcing continuations and
therefore reveals whether native/donor identity is already decodable from the
state before free-form explanation begins.

TRAPI semantic judgment of natural-language generations remains authoritative.
Exact identifiers here are used only to decide whether the next repair belongs
in the state updater or in the state-to-explanation decoder interface.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_recurrent_causal_gate import random_state_like
from evaluate_recurrent_formal_protocols import generate_one, load_wrapper
from evaluate_state8_eedi_same_question_c4 import (
    file_sha256,
    read_jsonl,
    select_recipient_twins,
)
from t3_judge_utils import write_jsonl
from train_recurrent_student_m4 import rollout_state
from train_state8_semantic_causal_repair import early_label_decode_text


CONDITIONS = ("native", "noop", "samemal", "random", "swap")
LABEL = re.compile(r"^[A-Za-z0-9_.:-]+$")


def normalize_identifier(value: str) -> str:
    """Accept exactly one bare identifier and reject explanatory spillover."""
    text = str(value).strip()
    if text.startswith("<label>") and text.endswith("</label>"):
        text = text[len("<label>") : -len("</label>")].strip()
    first = text.splitlines()[0].strip() if text else ""
    if first.endswith(".") and first[:-1].isdigit():
        first = first[:-1]
    return first if LABEL.fullmatch(first) else ""


def categorize(identifier: str, recipient: str, donor: str) -> str:
    if identifier == donor:
        return "donor"
    if identifier == recipient:
        return "recipient"
    return "other"


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_pair: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in records:
        condition = str(row["condition"])
        grouped[condition].append(row)
        by_pair[int(row["pair_idx"])][condition] = row
    if set(grouped) != set(CONDITIONS):
        raise ValueError(f"Label-readout condition drift: {sorted(grouped)}")
    if any(set(rows) != set(CONDITIONS) for rows in by_pair.values()):
        raise ValueError("Label-readout panel has an incomplete pair")

    by_condition: dict[str, Any] = {}
    for condition in CONDITIONS:
        rows = grouped[condition]
        counts = Counter(str(row["category"]) for row in rows)
        by_condition[condition] = {
            "n": len(rows),
            "to_donor_rate": counts["donor"] / len(rows),
            "to_recipient_rate": counts["recipient"] / len(rows),
            "other_rate": counts["other"] / len(rows),
            "valid_identifier_rate": sum(bool(row["predicted_identifier"]) for row in rows)
            / len(rows),
        }

    random_any = (
        by_condition["random"]["to_donor_rate"]
        + by_condition["random"]["to_recipient_rate"]
    )
    result = {
        "stage": "state8_eedi_label_readout_c4_v1",
        "status": "complete",
        "diagnostic_only": True,
        "semantic_judge_authoritative": True,
        "exact_identifier_authoritative": False,
        "by_condition": by_condition,
        "native_recipient_retention": by_condition["native"]["to_recipient_rate"],
        "noop_recipient_retention": by_condition["noop"]["to_recipient_rate"],
        "same_rule_recipient_retention": by_condition["samemal"]["to_recipient_rate"],
        "swap_to_donor_rate": by_condition["swap"]["to_donor_rate"],
        "random_matches_either_target_rate": random_any,
        "D_specific": (
            by_condition["swap"]["to_donor_rate"]
            - by_condition["random"]["to_donor_rate"]
        ),
        "noop_output_identical_to_native_rate": sum(
            rows["noop"]["raw_output"] == rows["native"]["raw_output"]
            for rows in by_pair.values()
        )
        / len(by_pair),
    }
    result["diagnostic_pass"] = (
        result["native_recipient_retention"] >= 0.65
        and result["noop_recipient_retention"] >= 0.65
        and result["same_rule_recipient_retention"] >= 0.65
        and result["swap_to_donor_rate"] >= 0.50
        and result["random_matches_either_target_rate"] <= 0.25
        and result["D_specific"] >= 0.15
    )
    return result


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--recurrent-state", type=Path, required=True)
    parser.add_argument("--causal-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-adapter-sha256", required=True)
    parser.add_argument("--expected-recurrent-state-sha256", required=True)
    parser.add_argument("--state-token-alias", default="<|vision_pad|>")
    parser.add_argument("--state-architecture", choices=("scalar", "layerwise"), default="layerwise")
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--num-pairs", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260811)
    args = parser.parse_args()
    if args.num_pairs <= 0 or args.max_new_tokens <= 0:
        raise ValueError("Pair and generation limits must be positive")

    artifacts = {
        "adapter_model": file_sha256(args.adapter / "adapter_model.safetensors"),
        "recurrent_state": file_sha256(args.recurrent_state),
    }
    expected = {
        "adapter_model": args.expected_adapter_sha256,
        "recurrent_state": args.expected_recurrent_state_sha256,
    }
    if artifacts != expected:
        raise ValueError(f"Label-readout artifact drift: expected={expected} observed={artifacts}")

    rows = select_recipient_twins(read_jsonl(args.causal_jsonl), args.num_pairs)
    if len(rows) != args.num_pairs:
        raise ValueError(f"Requested {args.num_pairs} pairs but found {len(rows)}")
    tokenizer, wrapper = load_wrapper(
        args.model,
        str(args.adapter),
        args.recurrent_state,
        args.state_token_alias,
        args.state_architecture,
    )
    alias = wrapper.state_token_rendered_alias
    device = next(wrapper.parameters()).device
    records: list[dict[str, Any]] = []
    state_shape: list[int] | None = None
    for pair_index, row in enumerate(rows, start=1):
        native = rollout_state(
            wrapper, tokenizer, list(row["native_update_texts"]), alias, args.max_length, device
        )
        donor = rollout_state(
            wrapper, tokenizer, list(row["donor_update_texts"]), alias, args.max_length, device
        )
        same = rollout_state(
            wrapper,
            tokenizer,
            list(row["same_rule_control_update_texts"]),
            alias,
            args.max_length,
            device,
        )
        states = {
            "native": native,
            "noop": native.clone(),
            "samemal": same,
            "random": random_state_like(native, args.seed + 1_000_003 * pair_index),
            "swap": donor,
        }
        prompt = early_label_decode_text(str(row["decode_text"]), "state_only")
        recipient_label = str(
            row.get("native_semantic_label") or row["native_target_label"]
        ).strip()
        donor_label = str(
            row.get("donor_semantic_label") or row["donor_target_label"]
        ).strip()
        state_shape = list(native.shape[1:])
        for condition in CONDITIONS:
            output = generate_one(
                wrapper,
                tokenizer,
                states[condition],
                prompt,
                alias,
                args.max_length,
                args.max_new_tokens,
            )
            identifier = normalize_identifier(output)
            records.append(
                {
                    "pair_idx": pair_index,
                    "pair_id": row["pair_id"],
                    "condition": condition,
                    "recipient_label": recipient_label,
                    "donor_label": donor_label,
                    "predicted_identifier": identifier,
                    "category": categorize(identifier, recipient_label, donor_label),
                    "raw_output": output,
                }
            )
        print(json.dumps({"pairs": pair_index, "total": len(rows)}), flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "predictions.jsonl", records)
    metrics = summarize(records)
    metrics.update(
        num_pairs=len(rows),
        conditions=list(CONDITIONS),
        state_architecture=args.state_architecture,
        state_shape=state_shape,
        raw_history_visible_to_decoder=False,
        persistent_history_kv_cache=False,
        same_question_query_identical=True,
        current_item_visible_to_decoder=False,
        label_readout_query_mode="state_only",
        student_response_visible_to_decoder=False,
        target_candidates_visible_to_decoder=False,
        label_readout_target=(
            "semantic_identifier"
            if any(row.get("native_semantic_label") for row in rows)
            else "source_taxonomy_identifier"
        ),
        artifact_sha256=artifacts,
        causal_jsonl_sha256=file_sha256(args.causal_jsonl),
    )
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
