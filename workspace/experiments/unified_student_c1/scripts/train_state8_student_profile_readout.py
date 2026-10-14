#!/usr/bin/env python3
"""State-only profile continuation with random-neutral causal transport."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import train_recurrent_student_m4 as trainer
import train_state8_semantic_causal_repair as semantic
from build_state8_student_profile_readout import NEUTRAL_TARGET, PROFILE_KIND, PROFILE_TASK


def main() -> None:
    # Reuse the already audited semantic-geometry and prefix-free label-binding
    # objectives.  This profile entrypoint only supplies the profile schema and
    # makes their activation fail closed in its own receipt.
    os.environ.setdefault("STATE8_GEOMETRY_WEIGHT", "0.10")
    os.environ.setdefault("STATE8_DIFFERENT_MAX_COSINE", "0.90")
    os.environ.setdefault("STATE8_EARLY_LABEL_QUERY_MODE", "state_only")
    os.environ.setdefault("STATE8_EARLY_LABEL_BINDING_WEIGHT", "1.0")
    os.environ.setdefault("STATE8_EARLY_LABEL_RANDOM_WEIGHT", "0.5")
    os.environ.setdefault("STATE8_EARLY_LABEL_SAME_RULE_WEIGHT", "0.5")
    os.environ.setdefault("STATE8_PROFILE_LEGACY_CAUSAL_ENABLED", "1")
    output_dir: Path | None = None
    original_parse = argparse.ArgumentParser.parse_args
    original_pair = semantic.is_diagnosis_pair
    original_neutral = semantic.NEUTRAL_TARGET
    original_specificity = trainer.backward_specificity_triplet_objective
    original_objective = trainer.backward_training_objective
    geometry_weight = float(os.environ["STATE8_GEOMETRY_WEIGHT"])
    different_max_cosine = float(os.environ["STATE8_DIFFERENT_MAX_COSINE"])
    label_weight = float(os.environ["STATE8_EARLY_LABEL_BINDING_WEIGHT"])
    label_random_weight = float(os.environ["STATE8_EARLY_LABEL_RANDOM_WEIGHT"])
    label_same_weight = float(os.environ["STATE8_EARLY_LABEL_SAME_RULE_WEIGHT"])
    legacy_causal_enabled = os.environ["STATE8_PROFILE_LEGACY_CAUSAL_ENABLED"] == "1"

    def parse(parser, *args, **kwargs):
        nonlocal output_dir
        namespace = original_parse(parser, *args, **kwargs)
        output_dir = Path(namespace.output_dir)
        namespace.profile_readout_task = PROFILE_TASK
        namespace.profile_readout_schema = PROFILE_KIND
        namespace.profile_readout_query = "constant_state_only"
        namespace.profile_future_labels_visible_to_training = True
        namespace.profile_future_labels_visible_to_decoder = False
        namespace.profile_student_id_visible_to_decoder = False
        namespace.profile_raw_history_visible_to_decoder = False
        namespace.profile_causal_transport = "fixed_query_only_state_identity_changes"
        namespace.profile_legacy_causal_enabled = legacy_causal_enabled
        namespace.final_inference_route = "strict_state_only"
        namespace.raw_history_visible_to_decoder = False
        namespace.student_id_visible_to_decoder = False
        return namespace

    def is_profile_or_diagnosis(row):
        return bool(
            original_pair(row)
            or str((row or {}).get("causal_target_kind") or "") == PROFILE_KIND
        )

    def objective(
        wrapper,
        tokenizer,
        row,
        causal_row,
        cache,
        online_kd_cache,
        namespace,
        device,
        state_token_id,
        loss_scale,
        training_progress,
    ):
        record = original_objective(
            wrapper,
            tokenizer,
            row,
            causal_row,
            cache,
            online_kd_cache,
            namespace,
            device,
            state_token_id,
            loss_scale,
            training_progress,
        )
        record["semantic_geometry_loss"] = 0.0
        record["early_label_binding_loss"] = 0.0
        record["early_label_control_loss"] = 0.0
        if causal_row is None or str(causal_row.get("causal_target_kind")) != PROFILE_KIND:
            return record
        if not legacy_causal_enabled:
            record["profile_legacy_causal_suppressed"] = True
            return record
        alias = getattr(
            namespace, "state_token_rendered_alias", namespace.state_token_alias
        )
        exposure = float(namespace.causal_exposure_multiplier)
        geometry_effective = geometry_weight * exposure
        geometry_value, geometry_diagnostics = semantic.backward_semantic_state_geometry(
            wrapper,
            tokenizer,
            causal_row,
            alias,
            namespace.max_length,
            device,
            different_max_cosine=different_max_cosine,
            weight=geometry_effective,
            loss_scale=loss_scale,
        )
        record["semantic_geometry_loss"] = geometry_value
        record["semantic_geometry_effective_weight"] = geometry_effective
        record["loss"] += geometry_effective * geometry_value
        record.update(geometry_diagnostics)

        label_effective = label_weight * exposure
        label_value, margin_value, label_diagnostics = semantic.backward_early_label_binding(
            wrapper,
            tokenizer,
            causal_row,
            alias,
            namespace.max_length,
            device,
            namespace.causal_margin,
            label_effective,
            loss_scale,
        )
        record["early_label_binding_loss"] = label_value
        record["early_label_binding_margin_loss"] = margin_value
        record["early_label_binding_effective_weight"] = label_effective
        record["early_label_binding_source"] = PROFILE_TASK
        record["loss"] += label_effective * label_value
        record.update(label_diagnostics)
        control_value, control_diagnostics = semantic.backward_early_label_controls(
            wrapper,
            tokenizer,
            causal_row,
            alias,
            namespace.max_length,
            device,
            namespace.causal_margin,
            int(namespace.seed) + int(record.get("global_position", 0)),
            label_effective,
            loss_scale,
            random_weight=label_random_weight,
            same_rule_weight=label_same_weight,
        )
        record["early_label_control_loss"] = control_value
        record["loss"] += label_effective * control_value
        record.update(control_diagnostics)
        return record

    argparse.ArgumentParser.parse_args = parse
    semantic.is_diagnosis_pair = is_profile_or_diagnosis
    semantic.NEUTRAL_TARGET = NEUTRAL_TARGET
    os.environ["STATE8_EARLY_LABEL_QUERY_MODE"] = "state_only"
    trainer.backward_specificity_triplet_objective = (
        semantic.backward_random_neutral_specificity
    )
    trainer.backward_training_objective = objective
    try:
        trainer.main()
        if int(os.environ.get("OMPI_COMM_WORLD_RANK", "0")) == 0:
            if output_dir is None:
                raise RuntimeError("Profile trainer did not observe output directory")
            history = json.loads(
                (output_dir / "trainer_history.json").read_text(encoding="utf-8")
            )
            run_config = json.loads(
                (output_dir / "run_config.json").read_text(encoding="utf-8")
            )
            diagnostic = bool(
                run_config.get("diagnostic_max_micro_steps") is not None
                or run_config.get("diagnostic_stop_after_round") is not None
                or run_config.get("diagnostic_weight_only_warm_start") is True
            )
            profile_rows = [row for row in history if row.get("task") == PROFILE_TASK]
            profile_pairs = [
                row
                for row in history
                if str(row.get("causal_pair_id") or "").startswith("state8-profile:")
            ]
            if not profile_rows or not profile_pairs:
                raise RuntimeError("Profile objectives were not exercised")
            geometry, _ = semantic.aggregate_geometry_counts(
                history,
                geometry_weight=geometry_weight if legacy_causal_enabled else 0.0,
                eedi_geometry_weight=0.0,
            )
            early_label = semantic.aggregate_early_label_counts(
                history, label_weight if legacy_causal_enabled else 0.0
            )
            if legacy_causal_enabled and (
                geometry["applied"] <= 0 or early_label["applied"] <= 0
            ):
                raise RuntimeError("Profile ownership objectives were not exercised")
            receipt = {
                "stage": "state8_student_profile_readout_repair_v1",
                "status": "diagnostic_complete" if diagnostic else "complete",
                "diagnostic": diagnostic,
                "state_token_count": 8,
                "profile_rows": len(profile_rows),
                "profile_causal_exposures": len(profile_pairs),
                "query_held_fixed": True,
                "only_state_identity_changes": True,
                "random_target": "insufficient_evidence_profile",
                "future_labels_visible_to_training": True,
                "future_labels_visible_to_decoder": False,
                "raw_history_visible_to_decoder": False,
                "student_id_visible_to_decoder": False,
                "final_inference_route": "strict_state_only",
                "ownership_geometry_weight": geometry_weight,
                "ownership_geometry_exposures": geometry["applied"],
                "prefix_free_profile_label_weight": label_weight,
                "prefix_free_profile_label_exposures": early_label["applied"],
                "profile_label_query_mode": "state_only",
                "profile_label_visible_at_inference": False,
                "legacy_profile_causal_enabled": legacy_causal_enabled,
                "legacy_profile_causal_exposures_suppressed": sum(
                    bool(row.get("profile_legacy_causal_suppressed"))
                    for row in history
                ),
                "entrypoint_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            }
            (output_dir / "student_profile_readout_repair.json").write_text(
                json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    finally:
        argparse.ArgumentParser.parse_args = original_parse
        semantic.is_diagnosis_pair = original_pair
        semantic.NEUTRAL_TARGET = original_neutral
        trainer.backward_specificity_triplet_objective = original_specificity
        trainer.backward_training_objective = original_objective


if __name__ == "__main__":
    main()
