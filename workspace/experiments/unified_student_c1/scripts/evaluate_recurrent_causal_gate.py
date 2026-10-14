#!/usr/bin/env python3
"""Frozen C4 gate for the closed recurrent student state.

Unlike the legacy one-shot layer-patch test, this evaluator intervenes on the
actual cached recurrent state passed to the decoder.  It retains the same
donor/recipient/same-rule 150-pair manifest and reports directed, random,
same-rule, and no-op controls on one clean subset.
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
import signal
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch

from evaluate_recurrent_formal_protocols import load_wrapper
from recurrent_student_model import build_update_text, extract_direct_decode_text
from state_swap_misconception_smoke import (
    family_of,
    label_of,
    load_pair_manifest,
    pair_targets,
    read_jsonl,
)
from train_recurrent_student_diagnostic import (
    alias_text,
    encode_prompt,
    encode_prompt_target,
    move,
    sequence_logprob,
)


CONDITIONS = ("noop", "samemal", "random", "swap")
ALL_CONDITIONS = ("native", *CONDITIONS)


def fsync_parent(path: Path) -> None:
    """Persist the directory entry after an atomic rename on mounted storage."""
    descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        try:
            os.fsync(descriptor)
        except OSError as error:
            if error.errno not in (errno.EINVAL, errno.ENOTSUP, errno.EROFS):
                raise
            # Some FUSE implementations do not support directory fsync.  A
            # filesystem-wide sync still gives the cluster termination handler
            # a durable fallback without weakening progress validation.
            os.sync()
    finally:
        os.close(descriptor)


def install_termination_flush(enabled: bool) -> None:
    """Flush completed progress when the cluster forwards a termination signal."""
    if not enabled:
        return

    def flush_and_exit(signum: int, _frame: Any) -> None:
        os.sync()
        raise SystemExit(128 + int(signum))

    signal.signal(signal.SIGTERM, flush_and_exit)
    signal.signal(signal.SIGINT, flush_and_exit)


def flush_progress() -> None:
    """Force closed progress files through delayed FUSE writeback."""
    os.sync()


def file_sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def progress_fingerprint(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema": 1,
        "progress_key": str(args.progress_key),
        "jsonl_sha256": file_sha256(args.jsonl),
        "pair_manifest_sha256": file_sha256(args.pair_manifest),
        "state_architecture": str(args.state_architecture),
        "state_token_alias": str(args.state_token_alias),
        "max_length": int(args.max_length),
        "num_pairs": int(args.num_pairs),
        "seed": int(args.seed),
    }


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    fsync_parent(path)


def atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    torch.save(value, temporary)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    temporary.replace(path)
    fsync_parent(path)


def validate_resumable_records(
    records: list[dict[str, Any]], num_pairs: int
) -> set[int]:
    by_pair: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        pair_idx = int(record.get("pair_idx", -1))
        if not 1 <= pair_idx <= num_pairs:
            raise ValueError(f"Resumable C4 pair index out of range: {pair_idx}")
        by_pair[pair_idx].append(record)
    completed: set[int] = set()
    for pair_idx, rows in by_pair.items():
        conditions = [str(row.get("condition")) for row in rows]
        if len(rows) != len(ALL_CONDITIONS) or set(conditions) != set(ALL_CONDITIONS):
            raise ValueError(f"Incomplete resumable C4 pair: {pair_idx}")
        completed.add(pair_idx)
    return completed


def random_state_like(reference: torch.Tensor, seed: int) -> torch.Tensor:
    """Isotropic norm-matched random state generated reproducibly on CPU."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    noise = torch.randn(reference.shape, generator=generator, dtype=torch.float32)
    reference_float = reference.detach().float().cpu()
    noise_norm = torch.linalg.vector_norm(noise, dim=-1, keepdim=True).clamp_min(1e-12)
    reference_norm = torch.linalg.vector_norm(
        reference_float, dim=-1, keepdim=True
    )
    return (noise * (reference_norm / noise_norm)).to(
        device=reference.device, dtype=reference.dtype
    )


@torch.inference_mode()
def state_for_row(wrapper, tokenizer, row: dict[str, Any], alias: str, max_length: int):
    device = next(wrapper.parameters()).device
    state = wrapper.initial_state(1, device=device)
    history = list(row.get("history") or [])
    if not history:
        raise ValueError(f"C4 row has no observed history: {row.get('example_id')}")
    for interaction in history:
        update = move(
            encode_prompt(
                tokenizer,
                alias_text(build_update_text(interaction), alias),
                max_length,
            ),
            device,
        )
        state = wrapper.update_state(state, **update).state
    return state


@torch.inference_mode()
def score_state(wrapper, tokenizer, state, row, alias: str, max_length: int) -> dict:
    text = extract_direct_decode_text(str(row["prompt"]))
    if "<student_history>" in text.lower():
        raise AssertionError("Recurrent C4 decoder leaked raw history")
    device = state.device
    scores = {}
    for label_value in row["candidate_labels"]:
        label = str(label_value)
        encoded = move(
            encode_prompt_target(
                tokenizer,
                alias_text(text, alias),
                str(row["candidate_targets"][label]),
                max_length,
            ),
            device,
        )
        prompt_end = encoded.pop("prompt_ends")
        output = wrapper.decode_from_state(
            state, **encoded, prompt_ends=prompt_end
        )
        scores[label] = sequence_logprob(
            output.logits, encoded["input_ids"], int(prompt_end.item())
        )
    labels = list(scores)
    probabilities = torch.softmax(
        torch.tensor([scores[label] for label in labels]), dim=0
    ).tolist()
    return {
        "prediction": max(scores, key=scores.get),
        "scores": scores,
        "probabilities": dict(zip(labels, probabilities)),
    }


def categorize(prediction: str, donor_target: str, recipient_target: str) -> str:
    if prediction == donor_target:
        return "flip_to_A"
    if prediction == recipient_target:
        return "stay_B"
    return "other"


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_pair: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        by_pair[int(record["pair_idx"])][str(record["condition"])] = record
    required = set(ALL_CONDITIONS)
    incomplete = [
        pair_idx
        for pair_idx, conditions in by_pair.items()
        if set(conditions) != required
    ]
    if incomplete:
        raise ValueError(f"Incomplete recurrent C4 conditions: {incomplete[:5]}")
    clean_ids = sorted(
        pair_idx
        for pair_idx, conditions in by_pair.items()
        if conditions["native"]["category"] == "stay_B"
    )
    if not clean_ids:
        raise ValueError("Recurrent C4 gate produced no clean native=B pairs")
    summary: dict[str, Any] = {
        "clean_definition": "native.category == stay_B",
        "all_pairs_n": len(by_pair),
        "clean_n": len(clean_ids),
        "clean": {},
    }
    native = [by_pair[pair_idx]["native"] for pair_idx in clean_ids]
    for condition in CONDITIONS:
        rows = [by_pair[pair_idx][condition] for pair_idx in clean_ids]
        counts = Counter(row["category"] for row in rows)
        n = len(rows)
        summary["clean"][condition] = {
            "n": n,
            "category_counts": dict(sorted(counts.items())),
            "flip_to_A_rate": counts["flip_to_A"] / n,
            "stay_B_rate": counts["stay_B"] / n,
            "other_rate": counts["other"] / n,
            "changed_top1_rate": sum(
                row["prediction"] != base["prediction"]
                for row, base in zip(rows, native, strict=True)
            )
            / n,
            "mean_delta_p_A": sum(
                row["p_A"] - base["p_A"]
                for row, base in zip(rows, native, strict=True)
            )
            / n,
            "mean_delta_margin_A_minus_B": sum(
                row["margin_A_minus_B"] - base["margin_A_minus_B"]
                for row, base in zip(rows, native, strict=True)
            )
            / n,
        }
    swap = summary["clean"]["swap"]["flip_to_A_rate"]
    random = summary["clean"]["random"]["flip_to_A_rate"]
    noop = summary["clean"]["noop"]["flip_to_A_rate"]
    summary["D_specific"] = swap - random
    summary["D_damage"] = random - noop
    checks = {
        "candidate_support": len(clean_ids) >= 50,
        "specificity": summary["D_specific"] >= 0.15,
        # A norm-matched random state must not itself become a stable donor
        # address.  The specificity contrast alone is insufficient: a large
        # swap effect can coexist with an unacceptably large random effect.
        "random_control": random <= 0.25,
        "noop_control": noop <= 0.05,
        "same_rule_control": summary["clean"]["samemal"]["flip_to_A_rate"] <= 0.10,
    }
    summary["checks"] = checks
    summary["scientific_gate_pass"] = all(checks.values())
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--recurrent-state", type=Path, required=True)
    parser.add_argument(
        "--state-architecture",
        choices=("scalar", "layerwise"),
        default="scalar",
    )
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument("--pair-manifest", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--state-token-alias", default="<|vision_pad|>")
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--num-pairs", type=int, default=150)
    parser.add_argument(
        "--allow-manifest-prefix",
        action="store_true",
        help=(
            "Evaluate the first --num-pairs entries of a larger immutable pair "
            "manifest. Without this flag the manifest cardinality must match exactly."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--progress-dir",
        type=Path,
        help="Optional shared directory for fail-closed state/pair-level resume.",
    )
    parser.add_argument(
        "--progress-key",
        help="Immutable checkpoint identity required when --progress-dir is used.",
    )
    args = parser.parse_args()
    if (args.progress_dir is None) != (args.progress_key is None):
        raise ValueError("--progress-dir and --progress-key must be provided together")
    install_termination_flush(args.progress_dir is not None)

    fingerprint = progress_fingerprint(args) if args.progress_dir else None
    state_progress = args.progress_dir / "state_cache.pt" if args.progress_dir else None
    record_progress = args.progress_dir / "records.json" if args.progress_dir else None
    metadata_progress = args.progress_dir / "metadata.json" if args.progress_dir else None
    if metadata_progress:
        if metadata_progress.is_file():
            observed = json.loads(metadata_progress.read_text(encoding="utf-8"))
            if observed != fingerprint:
                raise ValueError("Resumable C4 progress fingerprint mismatch")
        else:
            atomic_json(metadata_progress, fingerprint)
            flush_progress()

    rows = read_jsonl(args.jsonl, "malrule_next_response")
    pairs, controls = load_pair_manifest(
        args.pair_manifest,
        rows,
        "malrule_next_response",
        "response",
        args.num_pairs,
        allow_prefix=args.allow_manifest_prefix,
    )
    tokenizer, wrapper = load_wrapper(
        args.model,
        args.adapter,
        args.recurrent_state,
        args.state_token_alias,
        args.state_architecture,
    )
    rendered_alias = wrapper.state_token_rendered_alias
    unique = {}
    for donor, recipient in pairs:
        unique[str(donor["example_id"])] = donor
        unique[str(recipient["example_id"])] = recipient
    for control in controls:
        if control is not None:
            unique[str(control["example_id"])] = control
    state_cache: dict[str, torch.Tensor] = {}
    if state_progress and state_progress.is_file():
        payload = torch.load(state_progress, map_location="cpu", weights_only=False)
        if payload.get("fingerprint") != fingerprint:
            raise ValueError("Resumable C4 state-cache fingerprint mismatch")
        raw_states = payload.get("states")
        if not isinstance(raw_states, dict):
            raise ValueError("Resumable C4 state cache is invalid")
        state_cache = {str(key): value.cpu() for key, value in raw_states.items()}
        unknown = sorted(set(state_cache) - set(unique))
        if unknown:
            raise ValueError(f"Resumable C4 state cache has unknown rows: {unknown[:3]}")
        print(
            json.dumps({"resumed_states": len(state_cache), "total": len(unique)}),
            flush=True,
        )
    for index, (example_id, row) in enumerate(unique.items(), start=1):
        if example_id not in state_cache:
            state_cache[example_id] = state_for_row(
                wrapper, tokenizer, row, rendered_alias, args.max_length
            ).detach().cpu()
        if state_progress and (
            index == 1 or index % 10 == 0 or index == len(unique)
        ):
            atomic_torch_save(
                state_progress,
                {"fingerprint": fingerprint, "states": state_cache},
            )
            flush_progress()
        if index == 1 or index % 10 == 0 or index == len(unique):
            print(json.dumps({"states": index, "total": len(unique)}), flush=True)

    records: list[dict[str, Any]] = []
    if record_progress and record_progress.is_file():
        payload = json.loads(record_progress.read_text(encoding="utf-8"))
        if payload.get("fingerprint") != fingerprint:
            raise ValueError("Resumable C4 record fingerprint mismatch")
        raw_records = payload.get("records")
        if not isinstance(raw_records, list):
            raise ValueError("Resumable C4 records are invalid")
        records = raw_records
    completed_pairs = validate_resumable_records(records, len(pairs))
    if completed_pairs:
        print(
            json.dumps({"resumed_pairs": len(completed_pairs), "total": len(pairs)}),
            flush=True,
        )
    for pair_index, ((donor, recipient), same_control) in enumerate(
        zip(pairs, controls, strict=True), start=1
    ):
        if pair_index in completed_pairs:
            continue
        donor_target, recipient_target, _ = pair_targets(
            donor, recipient, "response"
        )
        device = next(wrapper.parameters()).device
        native_state = state_cache[str(recipient["example_id"])].to(device)
        conditions = {
            "native": native_state,
            "noop": native_state.clone(),
            "swap": state_cache[str(donor["example_id"])].to(device),
            "random": random_state_like(
                native_state, args.seed + 1_000_003 * pair_index
            ),
            "samemal": state_cache[str(same_control["example_id"])].to(device),
        }
        for condition, state in conditions.items():
            scored = score_state(
                wrapper,
                tokenizer,
                state,
                recipient,
                rendered_alias,
                args.max_length,
            )
            p_a = float(scored["probabilities"].get(donor_target, 0.0))
            p_b = float(scored["probabilities"].get(recipient_target, 0.0))
            score_a = float(scored["scores"].get(donor_target, float("-inf")))
            score_b = float(scored["scores"].get(recipient_target, float("-inf")))
            records.append(
                {
                    "pair_idx": pair_index,
                    "condition": condition,
                    "category": categorize(
                        str(scored["prediction"]), donor_target, recipient_target
                    ),
                    "prediction": scored["prediction"],
                    "donor_id": donor["example_id"],
                    "recipient_id": recipient["example_id"],
                    "samemal_id": same_control["example_id"],
                    "donor_label": label_of(donor),
                    "recipient_label": label_of(recipient),
                    "family": family_of(recipient),
                    "donor_target": donor_target,
                    "recipient_target": recipient_target,
                    "p_A": p_a,
                    "p_B": p_b,
                    "margin_A_minus_B": score_a - score_b,
                }
            )
        if record_progress:
            atomic_json(
                record_progress,
                {"fingerprint": fingerprint, "records": records},
            )
            if pair_index % 10 == 0 or pair_index == len(pairs):
                flush_progress()
        if pair_index == 1 or pair_index % 10 == 0:
            print(json.dumps({"pairs": pair_index, "total": len(pairs)}), flush=True)
    result = {
        "model": args.model,
        "adapter": args.adapter,
        "recurrent_state": str(args.recurrent_state),
        "artifact_sha256": {
            "adapter_model": file_sha256(Path(args.adapter) / "adapter_model.safetensors"),
            "recurrent_state": file_sha256(args.recurrent_state),
        },
        "jsonl": str(args.jsonl),
        "pair_manifest": str(args.pair_manifest),
        "pair_seed": args.seed,
        "state_architecture": args.state_architecture,
        "state_shape": list(state_cache[next(iter(state_cache))].shape[1:]),
        "intervention_target": (
            "final_layerwise_recurrent_state"
            if args.state_architecture == "layerwise"
            else "final_continuous_recurrent_state"
        ),
        "raw_history_visible_to_decoder": False,
        "persistent_history_kv_cache": False,
        "random_control": "isotropic_native_state_norm_matched",
        "resumable_progress": bool(args.progress_dir),
        "summary": summarize_records(records),
        "records": records,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["summary"], indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
