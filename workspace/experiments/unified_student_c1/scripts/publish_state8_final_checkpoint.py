#!/usr/bin/env python3
"""Atomically publish the selected State8 final fit for downstream evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any


STAGE = "state8_canonical_final_checkpoint_publish_v1"
REQUIRED_FILES = (
    "completion.json",
    "state8_final_fit_receipt.json",
    "run_config.json",
    "training_metrics.json",
    "validation_metrics.json",
    "causal_validation_metrics.json",
)
STRICT_TAXONOMY_VARIANTS = {
    "highcausal_taxonomy",
    "taxonomy_curriculum_t3heavy",
    "taxonomy_curriculum_balanced",
    "taxonomy_curriculum_t3heavy_diag",
    "taxonomy_curriculum_balanced_diag",
    "strong1_widen8_lr5e7_seed58",
    "strong1_widen8_lr1e6_seed59",
    "strong1_widen8_single_lr5e7_seed60",
    "strong1_rescue_taxonomy_joint_seed61",
    "strong1_rescue_taskanchor_c4strong_seed62",
    "strong1_midpoint_diagnosis_first_seed63",
    "strong1_exact_diagnosis_curriculum_seed65",
    "strong1_exact_rescue_semantic_preserve_seed66",
    "strong1_exact_rescue_causal_taskanchor_seed67",
    "strong1_exact_rescue_retention_preserve_seed68",
    "diagnosis_first_matched_seed67",
    "same_question_updater_mse1_seed3072",
    "same_question_full_mse1_seed3071",
    "same_question_repair_promoted",
    "same_question_repair_checkpoint_promoted",
    "same_question_curriculum_promoted",
}
TASK_ANCHOR_VARIANTS = {
    "task_anchor_base",
    "task_anchor_c4strong",
    "strong1_rescue_taskanchor_c4strong_seed62",
}
DIAGNOSIS_FIRST_VARIANTS = {
    "strong1_rescue_taxonomy_joint_seed61",
    "strong1_midpoint_diagnosis_first_seed63",
    "strong1_exact_diagnosis_curriculum_seed65",
    "strong1_exact_rescue_semantic_preserve_seed66",
    "strong1_exact_rescue_causal_taskanchor_seed67",
    "strong1_exact_rescue_retention_preserve_seed68",
    "diagnosis_first_matched_seed67",
    "same_question_updater_mse1_seed3072",
    "same_question_full_mse1_seed3071",
    "same_question_repair_promoted",
    "same_question_repair_checkpoint_promoted",
    "same_question_curriculum_promoted",
}
SAME_QUESTION_VARIANTS = {
    "same_question_updater_mse1_seed3072",
    "same_question_full_mse1_seed3071",
    "same_question_repair_promoted",
    "same_question_repair_checkpoint_promoted",
    "same_question_curriculum_promoted",
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def payload_identity(root: Path) -> dict[str, str]:
    receipt = read_object(root / "state8_final_fit_receipt.json")
    view_receipt = (
        "state8_stateonly_teacher_repair.json"
        if receipt.get("history_training_mode") == "strict_state_only"
        else "state8_external_teacher_curriculum.json"
    )
    variant = receipt.get("selected_replay_variant")
    auxiliary = ()
    if variant in STRICT_TAXONOMY_VARIANTS:
        auxiliary = (Path("taxonomy_causal_repair.json"),)
    if variant in TASK_ANCHOR_VARIANTS:
        auxiliary = (
            Path("taxonomy_causal_repair.json"),
            Path("task_anchored_causal_repair.json"),
        )
    if variant in DIAGNOSIS_FIRST_VARIANTS:
        auxiliary = (*auxiliary, Path("diagnosis_first_causal_repair.json"))
    if variant == "strong1_exact_rescue_causal_taskanchor_seed67":
        auxiliary = (
            *auxiliary,
            Path("diagnosis_first_task_anchored_causal_repair.json"),
        )
    if variant == "strong1_exact_rescue_retention_preserve_seed68":
        auxiliary = (
            *auxiliary,
            Path("diagnosis_first_retention_anchored_causal_repair.json"),
        )
    if variant == "diagnosis_first_matched_seed67":
        auxiliary = (*auxiliary, Path("diagnosis_first_matched_finalfit.json"))
    if variant in SAME_QUESTION_VARIANTS:
        auxiliary = (
            *auxiliary,
            Path("diagnosis_first_source_anchored_repair.json"),
            Path("eedi_same_question_finalfit.json"),
        )
    paths = (
        Path("final_adapter/adapter_model.safetensors"),
        Path("recurrent_state.pt"),
        *(Path(name) for name in REQUIRED_FILES),
        Path(view_receipt),
        *auxiliary,
    )
    return {str(path): digest(root / path) for path in paths}


def source_identity(
    source: Path,
) -> tuple[dict[str, Any], dict[str, str], dict[str, str]]:
    adapter = source / "final_adapter/adapter_model.safetensors"
    state = source / "recurrent_state.pt"
    completion = read_object(source / "completion.json")
    receipt = read_object(source / "state8_final_fit_receipt.json")
    view_receipt = (
        "state8_stateonly_teacher_repair.json"
        if receipt.get("history_training_mode") == "strict_state_only"
        else "state8_external_teacher_curriculum.json"
    )
    for path in (
        adapter,
        state,
        *(source / name for name in REQUIRED_FILES),
        source / view_receipt,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    artifacts = {
        "adapter_model": digest(adapter),
        "recurrent_state": digest(state),
    }
    if completion.get("status") != "complete" or completion.get("micro_steps") != 259845:
        raise ValueError("Source completion is not the exact final fit")
    if completion.get("exact_history_support") is not True:
        raise ValueError("Source completion lacks exact history support")
    if receipt.get("stage") != "state8_qualified_unified_final_fit_v1":
        raise ValueError("Source final-fit receipt stage drift")
    if receipt.get("status") != "final_fit_complete" or receipt.get("one_epoch_exact") is not True:
        raise ValueError("Source final-fit receipt is incomplete")
    if receipt.get("selected_replay_variant") not in {
        "t3_heavy",
        "highkd",
        "balanced",
        "repair_logits",
        "repair_trajectory01",
        "highcausal_joint",
        "highcausal_updater",
        "highcausal_mid",
        "highcausal_eedi",
        "highcausal_taxonomy",
        *STRICT_TAXONOMY_VARIANTS,
        *TASK_ANCHOR_VARIANTS,
    }:
        raise ValueError("Source final-fit variant drift")
    if receipt.get("state_shape") != [64, 8, 5120]:
        raise ValueError("Source State8 shape drift")
    if receipt.get("raw_history_visible_to_decoder") is not False:
        raise ValueError("Source decoder is not history-free")
    if receipt.get("persistent_history_kv_cache") is not False:
        raise ValueError("Source persists a history KV cache")
    if receipt.get("history_training_mode") == "strict_state_only":
        if receipt.get("history_dropout_counts") != {"1.0": 259845}:
            raise ValueError("Strict final fit did not keep history fully dropped")
        if float(receipt.get("online_kd_weight", -1)) != 1.0:
            raise ValueError("Strict final fit did not keep teacher KD enabled")
        expected_label_weight = 1.0 if receipt.get(
            "selected_replay_variant"
        ) in DIAGNOSIS_FIRST_VARIANTS else 4.0
        if float(receipt.get("label_auxiliary_weight", -1)) != expected_label_weight:
            raise ValueError("Strict final fit lost canonical-label weighting")
        if receipt.get("selected_replay_variant") in DIAGNOSIS_FIRST_VARIANTS:
            if receipt.get("diagnosis_field_order") != ["diagnosis", "evidence", "label"]:
                raise ValueError("Diagnosis-first final fit lost semantic field order")
            if receipt.get("canonical_label_role") != "auxiliary_ordinary_weight":
                raise ValueError("Diagnosis-first final fit changed canonical-label role")
    if completion.get("sha256", {}).get("adapter_model") != artifacts["adapter_model"]:
        raise ValueError("Source adapter hash disagrees with completion")
    if completion.get("sha256", {}).get("recurrent_state") != artifacts["recurrent_state"]:
        raise ValueError("Source state hash disagrees with completion")
    if receipt.get("adapter_sha256") != artifacts["adapter_model"]:
        raise ValueError("Source adapter hash disagrees with final-fit receipt")
    if receipt.get("recurrent_state_sha256") != artifacts["recurrent_state"]:
        raise ValueError("Source state hash disagrees with final-fit receipt")
    return receipt, artifacts, payload_identity(source)


def validate_existing(
    destination: Path,
    receipt: dict[str, Any],
    artifacts: dict[str, str],
    payloads: dict[str, str],
) -> dict[str, Any]:
    publish_path = destination / "canonical_publish_receipt.json"
    if not publish_path.is_file():
        raise ValueError(f"Canonical destination exists without a publish receipt: {destination}")
    published = read_object(publish_path)
    if published.get("stage") != STAGE or published.get("status") != "complete":
        raise ValueError("Canonical publish receipt is incomplete")
    if published.get("artifact_sha256") != artifacts:
        raise ValueError("Canonical destination belongs to different artifacts")
    if published.get("payload_sha256") != payloads:
        raise ValueError("Canonical destination belongs to a different payload manifest")
    if published.get("selected_replay_variant") != receipt["selected_replay_variant"]:
        raise ValueError("Canonical destination belongs to a different selected variant")
    observed = {
        "adapter_model": digest(destination / "final_adapter/adapter_model.safetensors"),
        "recurrent_state": digest(destination / "recurrent_state.pt"),
    }
    if observed != artifacts:
        raise ValueError("Canonical payload hashes do not match its publish receipt")
    try:
        observed_payloads = payload_identity(destination)
    except OSError as error:
        raise ValueError("Canonical payload manifest is incomplete") from error
    if observed_payloads != payloads:
        raise ValueError("Canonical payload manifest hashes do not match its publish receipt")
    return published


def publish_receipt(destination: Path, value: dict[str, Any]) -> None:
    final = destination / "canonical_publish_receipt.json"
    temporary = destination / "canonical_publish_receipt.json.partial"
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, final)


def publish(source: Path, destination: Path) -> dict[str, Any]:
    source = source.resolve()
    destination = destination.resolve()
    if source == destination:
        raise ValueError("Source and canonical destination must differ")
    receipt, artifacts, payloads = source_identity(source)
    published = {
        "stage": STAGE,
        "status": "complete",
        "selected_replay_variant": receipt["selected_replay_variant"],
        "source_run_dir": str(source),
        "canonical_run_dir": str(destination),
        "artifact_sha256": artifacts,
        "payload_sha256": payloads,
        "final_fit_receipt_sha256": digest(source / "state8_final_fit_receipt.json"),
    }
    if destination.exists():
        publish_path = destination / "canonical_publish_receipt.json"
        if publish_path.is_file():
            return validate_existing(destination, receipt, artifacts, payloads)
        # Recover only the precise crash window after the payload rename and
        # before receipt-last publication.  Missing or mismatched payloads are
        # never repaired by overwriting an existing canonical destination.
        try:
            stranded = payload_identity(destination)
        except OSError as error:
            raise ValueError(
                f"Incomplete canonical destination without publish receipt: {destination}"
            ) from error
        if stranded != payloads:
            raise ValueError("Canonical destination without receipt has different artifacts")
        publish_receipt(destination, published)
        return validate_existing(destination, receipt, artifacts, payloads)

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(
        f"{destination.name}.staging-{artifacts['adapter_model'][:12]}-{artifacts['recurrent_state'][:12]}"
    )
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=False)
    try:
        shutil.copytree(source / "final_adapter", staging / "final_adapter")
        shutil.copy2(source / "recurrent_state.pt", staging / "recurrent_state.pt")
        for name in REQUIRED_FILES:
            shutil.copy2(source / name, staging / name)
        view_receipt = (
            "state8_stateonly_teacher_repair.json"
            if receipt.get("history_training_mode") == "strict_state_only"
            else "state8_external_teacher_curriculum.json"
        )
        shutil.copy2(source / view_receipt, staging / view_receipt)
        variant = receipt.get("selected_replay_variant")
        if variant in STRICT_TAXONOMY_VARIANTS or variant in TASK_ANCHOR_VARIANTS:
            shutil.copy2(
                source / "taxonomy_causal_repair.json",
                staging / "taxonomy_causal_repair.json",
            )
        if variant in TASK_ANCHOR_VARIANTS:
            shutil.copy2(
                source / "task_anchored_causal_repair.json",
                staging / "task_anchored_causal_repair.json",
            )
        if variant in DIAGNOSIS_FIRST_VARIANTS:
            shutil.copy2(
                source / "diagnosis_first_causal_repair.json",
                staging / "diagnosis_first_causal_repair.json",
            )
        if variant == "strong1_exact_rescue_causal_taskanchor_seed67":
            shutil.copy2(
                source / "diagnosis_first_task_anchored_causal_repair.json",
                staging / "diagnosis_first_task_anchored_causal_repair.json",
            )
        if variant == "strong1_exact_rescue_retention_preserve_seed68":
            shutil.copy2(
                source / "diagnosis_first_retention_anchored_causal_repair.json",
                staging / "diagnosis_first_retention_anchored_causal_repair.json",
            )
        if variant == "diagnosis_first_matched_seed67":
            shutil.copy2(
                source / "diagnosis_first_matched_finalfit.json",
                staging / "diagnosis_first_matched_finalfit.json",
            )
        if variant in SAME_QUESTION_VARIANTS:
            shutil.copy2(
                source / "diagnosis_first_source_anchored_repair.json",
                staging / "diagnosis_first_source_anchored_repair.json",
            )
            shutil.copy2(
                source / "eedi_same_question_finalfit.json",
                staging / "eedi_same_question_finalfit.json",
            )
        observed = {
            "adapter_model": digest(staging / "final_adapter/adapter_model.safetensors"),
            "recurrent_state": digest(staging / "recurrent_state.pt"),
        }
        if observed != artifacts:
            raise ValueError("Staged canonical payload failed hash verification")
        # The canonical directory becomes visible without an authoritative
        # receipt.  Consumers require the receipt, which is written only after
        # the moved payload has been re-hashed at its final path.
        os.rename(staging, destination)
        moved = {
            "adapter_model": digest(destination / "final_adapter/adapter_model.safetensors"),
            "recurrent_state": digest(destination / "recurrent_state.pt"),
        }
        if moved != artifacts:
            raise ValueError("Canonical payload failed post-move hash verification")
        publish_receipt(destination, published)
        return validate_existing(destination, receipt, artifacts, payloads)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    value = publish(args.source, args.destination)
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
