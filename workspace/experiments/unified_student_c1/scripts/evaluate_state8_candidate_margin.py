#!/usr/bin/env python3
"""Read-only candidate-margin audit for the State8 diagnosis interface.

The free-generation label diagnostic can fail because the decoder emits prose
instead of a bare identifier.  This audit teacher-forces the two labels from
each frozen identical-question Eedi twin and compares their complete-sequence
log probabilities.  It separately measures recurrent student states and
full-history direct states, so updater failure is distinguishable from a
state-to-decoder readout failure.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_c1 import load_model
from evaluate_recurrent_formal_protocols import (
    load_wrapper,
    prepare_recurrent_modules_for_inference,
)
from evaluate_state8_eedi_same_question_c4 import (
    file_sha256,
    read_jsonl,
    select_recipient_twins,
)
from layerwise_recurrent_student_model import LayerwiseRecurrentStudentModel
from recurrent_student_model import expand_state_token_alias
from t3_judge_utils import write_jsonl
from train_recurrent_student_diagnostic import direct_supervised_batch
from train_recurrent_student_diagnostic import encode_prompt_target, move
from train_recurrent_student_m4 import (
    causal_lm_loss_per_sequence,
    decode_candidate_losses,
    pad_supervised_batches,
    rollout_state_pair,
)
from train_state8_semantic_causal_repair import (
    early_label_decode_text,
    early_label_target,
)
from train_state8_diagnosis_first_causal_repair import (
    diagnosis_first_prompt,
    reorder_target,
)
from state8_natural_curriculum_view import natural_prompt_text
from train_recurrent_student_m4_external_state_teacher import (
    STUDENT_ADAPTER_NAME,
    TEACHER_ADAPTER_NAME,
    capture_from_external_teacher,
    load_external_teacher,
)


def state_pair_geometry(left: torch.Tensor, right: torch.Tensor) -> dict[str, float]:
    left_float = left.float()
    right_float = right.float()
    cosine = F.cosine_similarity(left_float, right_float, dim=-1).mean()
    delta = torch.linalg.vector_norm(right_float - left_float, dim=-1).mean()
    scale = 0.5 * (
        torch.linalg.vector_norm(left_float, dim=-1).mean()
        + torch.linalg.vector_norm(right_float, dim=-1).mean()
    )
    return {
        "state_cosine": float(cosine),
        "state_delta_norm": float(delta),
        "state_relative_delta": float(delta / scale.clamp_min(1e-12)),
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("Candidate-margin audit has no records")
    paths = sorted({str(row["state_path"]) for row in records})
    result: dict[str, Any] = {}
    for path in paths:
        rows = [row for row in records if row["state_path"] == path]
        pair_ids = {str(row["pair_id"]) for row in rows}
        if len(rows) != 2 * len(pair_ids):
            raise ValueError(f"State path {path} lacks native/donor records")
        by_state = {role: [row for row in rows if row["state_role"] == role]
                    for role in ("native", "donor")}
        if any(len(values) != len(pair_ids) for values in by_state.values()):
            raise ValueError(f"State path {path} has an imbalanced role panel")
        native_acc = sum(row["preferred_label"] == row["native_label"]
                         for row in by_state["native"]) / len(pair_ids)
        donor_acc = sum(row["preferred_label"] == row["donor_label"]
                        for row in by_state["donor"]) / len(pair_ids)
        pair_correct = 0
        margins: list[float] = []
        for pair_id in pair_ids:
            pair = {row["state_role"]: row for row in rows if row["pair_id"] == pair_id}
            pair_correct += int(
                pair["native"]["preferred_label"] == pair["native"]["native_label"]
                and pair["donor"]["preferred_label"] == pair["donor"]["donor_label"]
            )
            margins.extend(
                [float(pair["native"]["correct_margin_mean_nll"]),
                 float(pair["donor"]["correct_margin_mean_nll"])]
            )
        result[path] = {
            "pairs": len(pair_ids),
            "native_accuracy": native_acc,
            "donor_accuracy": donor_acc,
            "state_accuracy": 0.5 * (native_acc + donor_acc),
            "pair_accuracy": pair_correct / len(pair_ids),
            "mean_correct_margin_mean_nll": sum(margins) / len(margins),
            "positive_margin_rate": sum(value > 0 for value in margins) / len(margins),
        }
    return result


def direct_state(
    wrapper,
    tokenizer,
    direct_text: str,
    label: str,
    alias: str,
    max_length: int,
    device: torch.device,
) -> torch.Tensor:
    batch = direct_supervised_batch(
        tokenizer,
        direct_text,
        early_label_target(label),
        alias,
        max_length,
        device,
    )
    batch.pop("prompt_ends")
    batch.pop("labels")
    batch["attention_mask"] = batch["attention_mask"].to(
        dtype=wrapper.backbone.get_input_embeddings().weight.dtype
    )
    return wrapper.capture_direct_state(**batch).state.detach()


def score_state(
    wrapper,
    tokenizer,
    state: torch.Tensor,
    prompt: str,
    native_label: str,
    donor_label: str,
    alias: str,
    max_length: int,
    native_target: str | None = None,
    donor_target: str | None = None,
) -> dict[str, Any]:
    device = next(wrapper.parameters()).device
    labels = [
        str(native_target) if native_target is not None else early_label_target(native_label),
        str(donor_target) if donor_target is not None else early_label_target(donor_label),
    ]
    states = [state, state]
    mean_losses, sum_losses = decode_candidate_losses(
        wrapper,
        tokenizer,
        states,
        prompt,
        labels,
        alias,
        max_length,
        device,
        "sum",
    )
    mean_values = [float(value) for value in mean_losses]
    sum_values = [float(value) for value in sum_losses]
    preferred = labels[0] if mean_values[0] < mean_values[1] else labels[1]
    return {
        "native_mean_nll": mean_values[0],
        "donor_mean_nll": mean_values[1],
        "native_sum_nll": sum_values[0],
        "donor_sum_nll": sum_values[1],
        "native_minus_donor_mean_logprob": mean_values[1] - mean_values[0],
        "native_minus_donor_sum_logprob": sum_values[1] - sum_values[0],
        "preferred_label": preferred,
    }


def full_history_label_prompt(direct_text: str) -> str:
    """Keep the observed history and current item, but request only one label."""
    text = str(direct_text)
    history_open = "<student_history>"
    history_close = "</student_history>"
    query_open = "<query>"
    query_close = "</query>"
    if any(text.count(marker) != 1 for marker in (
        history_open, history_close, query_open, query_close
    )):
        raise ValueError("Full-history label prompt has an ambiguous history/query schema")
    history = text[text.index(history_open) : text.index(history_close) + len(history_close)]
    query = text[text.index(query_open) : text.index(query_close) + len(query_close)]
    return (
        "You are performing an auxiliary diagnostic readout.\n"
        f"{history}\n{query}\n"
        "Return only the exact canonical misconception label identifier. "
        "Do not emit XML tags, a diagnosis, evidence, or alternatives.\n"
        "Canonical misconception label:"
    )


def score_full_context(
    wrapper,
    tokenizer,
    prompt: str,
    native_label: str,
    donor_label: str,
    max_length: int,
    native_target: str | None = None,
    donor_target: str | None = None,
) -> dict[str, Any]:
    device = next(wrapper.parameters()).device
    labels = [
        str(native_target) if native_target is not None else early_label_target(native_label),
        str(donor_target) if donor_target is not None else early_label_target(donor_label),
    ]
    batches = [
        move(encode_prompt_target(tokenizer, prompt, label, max_length), device)
        for label in labels
    ]
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    combined = pad_supervised_batches(batches, int(pad_id))
    combined.pop("prompt_ends")
    output = wrapper.backbone(
        input_ids=combined["input_ids"],
        attention_mask=combined["attention_mask"],
        use_cache=False,
        return_dict=True,
    )
    mean_losses = causal_lm_loss_per_sequence(
        output.logits, combined["labels"], reduction="mean"
    )
    sum_losses = causal_lm_loss_per_sequence(
        output.logits, combined["labels"], reduction="sum"
    )
    mean_values = [float(value) for value in mean_losses]
    sum_values = [float(value) for value in sum_losses]
    preferred = labels[0] if mean_values[0] < mean_values[1] else labels[1]
    return {
        "native_mean_nll": mean_values[0],
        "donor_mean_nll": mean_values[1],
        "native_sum_nll": sum_values[0],
        "donor_sum_nll": sum_values[1],
        "native_minus_donor_mean_logprob": mean_values[1] - mean_values[0],
        "native_minus_donor_sum_logprob": sum_values[1] - sum_values[0],
        "preferred_label": preferred,
    }


def build_direct_wrapper(model: str, adapter: Path, alias: str, state_count: int):
    tokenizer, backbone = load_model(
        model, str(adapter), bottleneck_mask=True, load_4bit=True
    )
    ids = tokenizer(alias, add_special_tokens=False)["input_ids"]
    if len(ids) != 1:
        raise ValueError(f"State alias must be one token, got {ids}")
    wrapper = LayerwiseRecurrentStudentModel(
        backbone, int(ids[0]), transition_rank=0, state_token_count=state_count
    )
    prepare_recurrent_modules_for_inference(wrapper, backbone)
    wrapper.state_token_rendered_alias = expand_state_token_alias(alias, state_count)
    wrapper.eval()
    return tokenizer, wrapper


def structured_teacher_prompt(direct_text: str) -> str:
    text = diagnosis_first_prompt(natural_prompt_text(str(direct_text)))
    text = text.replace("<STUDENT_STATE>", "")
    if "<student_history>" not in text.lower():
        raise ValueError("Structured teacher prompt lost full history")
    return text


def external_teacher_direct_state(
    wrapper,
    tokenizer,
    direct_text: str,
    target: str,
    alias: str,
    max_length: int,
    device: torch.device,
) -> torch.Tensor:
    batch = direct_supervised_batch(
        tokenizer, direct_text, target, alias, max_length, device
    )
    batch.pop("prompt_ends")
    batch.pop("labels")
    batch["attention_mask"] = batch["attention_mask"].to(
        dtype=wrapper.backbone.get_input_embeddings().weight.dtype
    )
    captured = capture_from_external_teacher(
        wrapper, LayerwiseRecurrentStudentModel.capture_direct_state, **batch
    )
    return captured.state.detach()


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--recurrent-state", type=Path)
    parser.add_argument("--causal-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-adapter-sha256", required=True)
    parser.add_argument("--expected-recurrent-state-sha256")
    parser.add_argument("--mode", choices=("student", "teacher"), required=True)
    parser.add_argument(
        "--target-protocol", choices=("label_only", "structured"), default="label_only"
    )
    parser.add_argument("--external-teacher-adapter", type=Path)
    parser.add_argument("--expected-external-teacher-sha256")
    parser.add_argument("--state-token-alias", default="<|vision_pad|>")
    parser.add_argument("--state-token-count", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--num-pairs", type=int, default=25)
    args = parser.parse_args()

    adapter_digest = file_sha256(args.adapter / "adapter_model.safetensors")
    if adapter_digest != args.expected_adapter_sha256:
        raise ValueError(
            f"Adapter drift: expected={args.expected_adapter_sha256} observed={adapter_digest}"
        )
    recurrent_digest = None
    if args.mode == "student":
        if args.recurrent_state is None or not args.expected_recurrent_state_sha256:
            raise ValueError("Student mode requires a recurrent-state artifact and digest")
        recurrent_digest = file_sha256(args.recurrent_state)
        if recurrent_digest != args.expected_recurrent_state_sha256:
            raise ValueError("Recurrent-state digest drift")
        tokenizer, wrapper = load_wrapper(
            args.model,
            str(args.adapter),
            args.recurrent_state,
            args.state_token_alias,
            "layerwise",
        )
        paths = ["student_recurrent", "student_direct"]
        if args.target_protocol == "structured":
            if (
                args.external_teacher_adapter is None
                or not args.expected_external_teacher_sha256
            ):
                raise ValueError(
                    "Structured student audit requires the frozen external teacher"
                )
            observed_teacher = file_sha256(
                args.external_teacher_adapter / "adapter_model.safetensors"
            )
            if observed_teacher != args.expected_external_teacher_sha256:
                raise ValueError("External teacher adapter digest drift")
            load_external_teacher(wrapper.backbone, args.external_teacher_adapter)
            paths.extend(
                ["external_teacher_direct_to_student_decoder", "teacher_full_context"]
            )
    else:
        tokenizer, wrapper = build_direct_wrapper(
            args.model, args.adapter, args.state_token_alias, args.state_token_count
        )
        paths = ("teacher_full_context", "teacher_direct")

    alias = wrapper.state_token_rendered_alias
    device = next(wrapper.parameters()).device
    rows = select_recipient_twins(read_jsonl(args.causal_jsonl), args.num_pairs)
    if len(rows) != args.num_pairs:
        raise ValueError(f"Requested {args.num_pairs} pairs but found {len(rows)}")

    records: list[dict[str, Any]] = []
    geometries: list[dict[str, Any]] = []
    for pair_index, row in enumerate(rows, start=1):
        native_label = str(row["native_target_label"]).strip()
        donor_label = str(row["donor_target_label"]).strip()
        if args.target_protocol == "structured":
            native_target = reorder_target(str(row["native_target"]))
            donor_target = reorder_target(str(row["donor_target"]))
            state_prompt = diagnosis_first_prompt(str(row["decode_text"]))
        else:
            native_target = early_label_target(native_label)
            donor_target = early_label_target(donor_label)
            state_prompt = early_label_decode_text(str(row["decode_text"]))
        state_pairs: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        if args.mode == "student":
            recurrent_pair = rollout_state_pair(
                wrapper,
                tokenizer,
                list(row["native_update_texts"]),
                list(row["donor_update_texts"]),
                alias,
                args.max_length,
                device,
            )
            state_pairs["student_recurrent"] = recurrent_pair
            state_pairs["student_direct"] = (
                direct_state(
                    wrapper, tokenizer, str(row["native_direct_text"]), native_label,
                    alias, args.max_length, device
                ),
                direct_state(
                    wrapper, tokenizer, str(row["donor_direct_text"]), donor_label,
                    alias, args.max_length, device
                ),
            )
            if args.target_protocol == "structured":
                state_pairs["external_teacher_direct_to_student_decoder"] = (
                    external_teacher_direct_state(
                        wrapper, tokenizer, str(row["native_direct_text"]),
                        native_target, alias, args.max_length, device
                    ),
                    external_teacher_direct_state(
                        wrapper, tokenizer, str(row["donor_direct_text"]),
                        donor_target, alias, args.max_length, device
                    ),
                )
        else:
            state_pairs["teacher_direct"] = (
                direct_state(
                    wrapper, tokenizer, str(row["native_direct_text"]), native_label,
                    alias, args.max_length, device
                ),
                direct_state(
                    wrapper, tokenizer, str(row["donor_direct_text"]), donor_label,
                    alias, args.max_length, device
                ),
            )

        if args.mode == "teacher":
            for role, direct_key in (
                ("native", "native_direct_text"),
                ("donor", "donor_direct_text"),
            ):
                score = score_full_context(
                    wrapper,
                    tokenizer,
                    full_history_label_prompt(str(row[direct_key])),
                    native_label,
                    donor_label,
                    args.max_length,
                    native_target=native_target,
                    donor_target=donor_target,
                )
                correct_margin = (
                    score["native_minus_donor_mean_logprob"]
                    if role == "native"
                    else -score["native_minus_donor_mean_logprob"]
                )
                records.append(
                    {
                        "pair_idx": pair_index,
                        "pair_id": row["pair_id"],
                        "state_path": "teacher_full_context",
                        "state_role": role,
                        "native_label": native_label,
                        "donor_label": donor_label,
                        "correct_margin_mean_nll": correct_margin,
                        **score,
                    }
                )

        if args.mode == "student" and args.target_protocol == "structured":
            backbone = wrapper.backbone
            if backbone.active_adapter != STUDENT_ADAPTER_NAME:
                raise ValueError("Expected active student adapter before teacher scoring")
            backbone.set_adapter(TEACHER_ADAPTER_NAME, inference_mode=True)
            backbone.eval()
            try:
                for role, direct_key in (
                    ("native", "native_direct_text"),
                    ("donor", "donor_direct_text"),
                ):
                    score = score_full_context(
                        wrapper,
                        tokenizer,
                        structured_teacher_prompt(str(row[direct_key])),
                        native_label,
                        donor_label,
                        args.max_length,
                        native_target=native_target,
                        donor_target=donor_target,
                    )
                    correct_margin = (
                        score["native_minus_donor_mean_logprob"]
                        if role == "native"
                        else -score["native_minus_donor_mean_logprob"]
                    )
                    records.append(
                        {
                            "pair_idx": pair_index,
                            "pair_id": row["pair_id"],
                            "state_path": "teacher_full_context",
                            "state_role": role,
                            "native_label": native_label,
                            "donor_label": donor_label,
                            "correct_margin_mean_nll": correct_margin,
                            **score,
                        }
                    )
            finally:
                backbone.set_adapter(STUDENT_ADAPTER_NAME, inference_mode=False)
                backbone.eval()

        for state_path in state_pairs:
            native_state, donor_state = state_pairs[state_path]
            geometries.append(
                {
                    "pair_idx": pair_index,
                    "pair_id": row["pair_id"],
                    "state_path": state_path,
                    **state_pair_geometry(native_state, donor_state),
                }
            )
            for role, state in (("native", native_state), ("donor", donor_state)):
                score = score_state(
                    wrapper,
                    tokenizer,
                    state,
                    state_prompt,
                    native_label,
                    donor_label,
                    alias,
                    args.max_length,
                    native_target=native_target,
                    donor_target=donor_target,
                )
                correct_margin = (
                    score["native_minus_donor_mean_logprob"]
                    if role == "native"
                    else -score["native_minus_donor_mean_logprob"]
                )
                records.append(
                    {
                        "pair_idx": pair_index,
                        "pair_id": row["pair_id"],
                        "state_path": state_path,
                        "state_role": role,
                        "native_label": native_label,
                        "donor_label": donor_label,
                        "correct_margin_mean_nll": correct_margin,
                        **score,
                    }
                )
        print(json.dumps({"pairs": pair_index, "total": len(rows)}), flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "predictions.jsonl", records)
    write_jsonl(args.output_dir / "state_geometry.jsonl", geometries)
    metrics = {
        "stage": "state8_eedi_candidate_margin_audit_v1",
        "status": "complete",
        "diagnostic_only": True,
        "mode": args.mode,
        "num_pairs": len(rows),
        "candidate_protocol": (
            "pairwise_complete_structured_target_mean_token_nll"
            if args.target_protocol == "structured"
            else "pairwise_complete_label_mean_token_nll"
        ),
        "target_protocol": args.target_protocol,
        "unnormalized_sum_role": "diagnostic_only",
        "raw_history_visible_to_final_decoder": False,
        "student_response_visible_to_final_decoder": False,
        "state_paths": summarize(records),
        "artifact_sha256": {
            "adapter_model": adapter_digest,
            "recurrent_state": recurrent_digest,
            "causal_jsonl": file_sha256(args.causal_jsonl),
            "external_teacher_adapter": (
                args.expected_external_teacher_sha256
                if args.external_teacher_adapter is not None
                else None
            ),
        },
    }
    if not all(math.isfinite(float(row["correct_margin_mean_nll"])) for row in records):
        raise ValueError("Candidate-margin audit produced a non-finite margin")
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
