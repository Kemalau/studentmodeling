#!/usr/bin/env python3
"""Evaluate State8 interventions on identical-query Eedi diagnosis twins.

Only ``recipient_native`` rows are evaluated.  Their decoder query is held
fixed while the recurrent trajectory is replaced by a same-label control,
norm-matched random state, or a different-label donor from the same question.
The emitted natural-language panel is judged separately by the semantic T3
judge; canonical-label exact match remains an auxiliary diagnostic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    from evaluate_recurrent_causal_gate import (
        atomic_json,
        flush_progress,
        install_termination_flush,
        random_state_like,
    )
    from evaluate_recurrent_diagnosis_c4 import summarize_exact
    from evaluate_recurrent_formal_protocols import generate_one, load_wrapper
    from t3_judge_utils import extract_tag, natural_schema_valid, write_jsonl
    from train_recurrent_student_m4 import rollout_state
    from train_state8_diagnosis_first_causal_repair import (
        diagnosis_first_prompt,
        reorder_target,
    )
except ImportError:  # package-style unit tests
    from scripts.evaluate_recurrent_causal_gate import (
        atomic_json,
        flush_progress,
        install_termination_flush,
        random_state_like,
    )
    from scripts.evaluate_recurrent_diagnosis_c4 import summarize_exact
    from scripts.evaluate_recurrent_formal_protocols import generate_one, load_wrapper
    from scripts.t3_judge_utils import extract_tag, natural_schema_valid, write_jsonl
    from scripts.train_recurrent_student_m4 import rollout_state
    from scripts.train_state8_diagnosis_first_causal_repair import (
        diagnosis_first_prompt,
        reorder_target,
    )


CONDITIONS = ("native", "noop", "samemal", "random", "swap")


def progress_fingerprint(args: argparse.Namespace, artifacts: dict[str, str]) -> dict[str, Any]:
    return {
        "schema": 1,
        "progress_key": str(args.progress_key),
        "artifact_sha256": artifacts,
        "causal_jsonl_sha256": file_sha256(args.causal_jsonl),
        "state_architecture": str(args.state_architecture),
        "state_token_alias": str(args.state_token_alias),
        "max_length": int(args.max_length),
        "max_new_tokens": int(args.max_new_tokens),
        "num_pairs": int(args.num_pairs),
        "seed": int(args.seed),
    }


def validate_progress_bundle(
    bundle: dict[str, Any], fingerprint: dict[str, Any], num_pairs: int
) -> set[int]:
    if bundle.get("fingerprint") != fingerprint:
        raise ValueError("Eedi C4 resumable progress fingerprint mismatch")
    records = bundle.get("records")
    judge_predictions = bundle.get("judge_predictions")
    judge_eval = bundle.get("judge_eval")
    if not all(isinstance(value, list) for value in (records, judge_predictions, judge_eval)):
        raise ValueError("Eedi C4 resumable progress payload is invalid")
    by_pair: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        pair_idx = int(record.get("pair_idx", -1))
        if not 1 <= pair_idx <= num_pairs:
            raise ValueError(f"Eedi C4 resumable pair index out of range: {pair_idx}")
        by_pair[pair_idx].append(record)
    completed: set[int] = set()
    for pair_idx, rows in by_pair.items():
        conditions = [str(row.get("condition")) for row in rows]
        if len(rows) != len(CONDITIONS) or set(conditions) != set(CONDITIONS):
            raise ValueError(f"Incomplete Eedi C4 resumable pair: {pair_idx}")
        completed.add(pair_idx)
    for name, values in (("judge_predictions", judge_predictions), ("judge_eval", judge_eval)):
        counts: dict[int, int] = defaultdict(int)
        for row in values:
            pair_idx = int(row.get("pair_idx", -1))
            if pair_idx < 0 and name == "judge_eval":
                parts = str(row.get("example_id") or "").split(":")
                if len(parts) == 4 and parts[0] == "eedi-same-question-c4":
                    pair_idx = int(parts[1])
            if pair_idx not in completed:
                raise ValueError(f"{name} contains an incomplete pair: {pair_idx}")
            counts[pair_idx] += 1
        if any(counts[pair_idx] != 2 * len(CONDITIONS) for pair_idx in completed):
            raise ValueError(f"{name} resumable cardinality drift")
    return completed


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def select_recipient_twins(
    rows: list[dict[str, Any]], limit: int | None = None
) -> list[dict[str, Any]]:
    selected = [
        row
        for row in rows
        if str(row.get("native_control_role")) == "recipient_native"
    ]
    if limit is not None:
        selected = selected[:limit]
    if not selected:
        raise ValueError("Eedi C4 has no recipient-native twins")
    for row in selected:
        if str(row.get("source")) != "eedi_mining":
            raise ValueError("Eedi C4 row has the wrong source")
        if not bool(row.get("same_question_query_identical")):
            raise ValueError("Eedi C4 row does not freeze the query")
        if bool(row.get("raw_history_visible_to_decoder")):
            raise ValueError("Eedi C4 decoder may not see raw history")
        if bool(row.get("student_response_visible_to_decoder")):
            raise ValueError("Eedi C4 decoder query leaked the student response")
        if str(row["native_target_label"]) == str(row["donor_target_label"]):
            raise ValueError("Eedi C4 donor and recipient labels must differ")
    return selected


def semantic_panel_rows(
    *,
    pair_index: int,
    row: dict[str, Any],
    condition: str,
    output: str,
    prompt: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    predictions: list[dict[str, Any]] = []
    evaluation: list[dict[str, Any]] = []
    references = (
        ("donor", reorder_target(str(row["donor_target"]))),
        ("recipient", reorder_target(str(row["native_target"]))),
    )
    for role, target in references:
        example_id = f"eedi-same-question-c4:{pair_index:04d}:{condition}:{role}"
        predictions.append(
            {
                "example_id": example_id,
                "task": "diagnose_misconception",
                "source": "eedi_mining",
                "protocol": "closed_state_eedi_same_question_diagnosis_c4",
                "condition": condition,
                "reference_role": role,
                "pair_idx": pair_index,
                "target": target,
                "raw_output": output,
            }
        )
        evaluation.append(
            {
                "example_id": example_id,
                "task": "diagnose_misconception",
                "source": "eedi_mining",
                "condition": condition,
                "reference_role": role,
                "pair_idx": pair_index,
                "direct_text": prompt,
            }
        )
    return predictions, evaluation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--recurrent-state", type=Path, required=True)
    parser.add_argument("--causal-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-adapter-sha256")
    parser.add_argument("--expected-recurrent-state-sha256")
    parser.add_argument("--state-token-alias", default="<|vision_pad|>")
    parser.add_argument(
        "--state-architecture", choices=("scalar", "layerwise"), default="layerwise"
    )
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--num-pairs", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--progress-dir", type=Path)
    parser.add_argument("--progress-key")
    args = parser.parse_args()
    if args.num_pairs <= 0:
        raise ValueError("--num-pairs must be positive")
    if (args.progress_dir is None) != (args.progress_key is None):
        raise ValueError("--progress-dir and --progress-key must be provided together")
    install_termination_flush(args.progress_dir is not None)

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
                f"Eedi C4 {key} hash drift: expected={expected} "
                f"observed={observed_artifacts[key]}"
            )

    fingerprint = progress_fingerprint(args, observed_artifacts) if args.progress_dir else None
    progress_path = args.progress_dir / "progress.json" if args.progress_dir else None
    progress_bundle: dict[str, Any] = {
        "fingerprint": fingerprint,
        "records": [],
        "judge_predictions": [],
        "judge_eval": [],
    }
    completed_pairs: set[int] = set()
    if progress_path and progress_path.is_file():
        progress_bundle = json.loads(progress_path.read_text(encoding="utf-8"))
        completed_pairs = validate_progress_bundle(progress_bundle, fingerprint, args.num_pairs)
        print(json.dumps({"resumed_pairs": len(completed_pairs), "total": args.num_pairs}), flush=True)

    rows = select_recipient_twins(read_jsonl(args.causal_jsonl), args.num_pairs)
    if len(rows) != args.num_pairs:
        raise ValueError(
            f"Eedi C4 requested {args.num_pairs} pairs but found {len(rows)}"
        )
    tokenizer, wrapper = load_wrapper(
        args.model,
        str(args.adapter),
        args.recurrent_state,
        args.state_token_alias,
        args.state_architecture,
    )
    alias = wrapper.state_token_rendered_alias
    device = next(wrapper.parameters()).device
    records: list[dict[str, Any]] = list(progress_bundle["records"])
    judge_predictions: list[dict[str, Any]] = list(progress_bundle["judge_predictions"])
    judge_eval: list[dict[str, Any]] = list(progress_bundle["judge_eval"])
    state_shape: list[int] | None = (
        [wrapper.hidden_size]
        if args.state_architecture == "scalar"
        else (
            [wrapper.layer_count, wrapper.hidden_size]
            if wrapper.state_token_count == 1
            else [wrapper.layer_count, wrapper.state_token_count, wrapper.hidden_size]
        )
    )
    for pair_index, row in enumerate(rows, start=1):
        if pair_index in completed_pairs:
            continue
        native_state = rollout_state(
            wrapper,
            tokenizer,
            list(row["native_update_texts"]),
            alias,
            args.max_length,
            device,
        )
        donor_state = rollout_state(
            wrapper,
            tokenizer,
            list(row["donor_update_texts"]),
            alias,
            args.max_length,
            device,
        )
        same_rule_state = rollout_state(
            wrapper,
            tokenizer,
            list(row["same_rule_control_update_texts"]),
            alias,
            args.max_length,
            device,
        )
        condition_states = {
            "native": native_state,
            "noop": native_state.clone(),
            "samemal": same_rule_state,
            "random": random_state_like(
                native_state, args.seed + 1_000_003 * pair_index
            ),
            "swap": donor_state,
        }
        prompt = diagnosis_first_prompt(str(row["decode_text"]))
        native_label = str(row["native_target_label"])
        donor_label = str(row["donor_target_label"])
        state_shape = list(native_state.shape[1:])
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
            category = (
                "donor"
                if predicted_label == donor_label
                else "recipient"
                if predicted_label == native_label
                else "other"
            )
            records.append(
                {
                    "pair_idx": pair_index,
                    "condition": condition,
                    "pair_id": row["pair_id"],
                    "donor_id": row["donor_id"],
                    "recipient_id": row["recipient_id"],
                    "samemal_id": row["same_rule_control_id"],
                    "donor_label": donor_label,
                    "recipient_label": native_label,
                    "predicted_label": predicted_label,
                    "category": category,
                    "natural_schema_valid": natural_schema_valid(output),
                    "raw_output": output,
                    "decode_prompt": prompt,
                }
            )
            predictions, evaluation = semantic_panel_rows(
                pair_index=pair_index,
                row=row,
                condition=condition,
                output=output,
                prompt=prompt,
            )
            judge_predictions.extend(predictions)
            judge_eval.extend(evaluation)
        if progress_path:
            atomic_json(
                progress_path,
                {
                    "fingerprint": fingerprint,
                    "records": records,
                    "judge_predictions": judge_predictions,
                    "judge_eval": judge_eval,
                },
            )
            flush_progress()
        print(json.dumps({"pairs": pair_index, "total": len(rows)}), flush=True)

    if progress_path:
        completed_pairs = validate_progress_bundle(
            {
                "fingerprint": fingerprint,
                "records": records,
                "judge_predictions": judge_predictions,
                "judge_eval": judge_eval,
            },
            fingerprint,
            args.num_pairs,
        )
        if len(completed_pairs) != args.num_pairs:
            raise ValueError(f"Eedi C4 resumable support drift: {len(completed_pairs)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "predictions.jsonl", records)
    write_jsonl(args.output_dir / "judge_predictions.jsonl", judge_predictions)
    write_jsonl(args.output_dir / "judge_eval.jsonl", judge_eval)
    summary = {
        "stage": "state8_eedi_same_question_diagnosis_c4_v1",
        "num_pairs": len(rows),
        "conditions": list(CONDITIONS),
        "state_architecture": args.state_architecture,
        "state_shape": state_shape,
        "same_question_query_identical": True,
        "current_item_visible_to_decoder": True,
        "student_response_visible_to_decoder": False,
        "raw_history_visible_to_decoder": False,
        "persistent_history_kv_cache": False,
        "artifact_sha256": observed_artifacts,
        "causal_jsonl_sha256": file_sha256(args.causal_jsonl),
        "exact_label_auxiliary": summarize_exact(records),
        "semantic_judgment_pending": True,
        "resumable_progress": bool(args.progress_dir),
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
