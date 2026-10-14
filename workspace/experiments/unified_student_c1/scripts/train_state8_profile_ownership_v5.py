#!/usr/bin/env python3
"""V5 State8 ownership training with an evaluator-aligned loss budget.

V4 applied long profile-text transport, semantic geometry, early-label control,
and the numeric ownership loss to the same sparse profile exposures.  The text
branches were roughly three orders of magnitude larger and the auxiliary head
could fit one deterministic hardest negative.  V5 keeps the state-only natural
language profile objective at a small explicit coefficient, disables every
legacy profile causal branch, and trains the numeric objective against a
deterministic multi-student panel.  Non-profile rows retain the unified-task
objective unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import train_recurrent_student_m4 as trainer
import train_state8_profile_ownership_repair as ownership
import train_state8_student_profile_readout as profile_trainer
from build_state8_student_profile_readout import PROFILE_KIND, PROFILE_TASK


STAGE = "state8_profile_ownership_v5_loss_fix"


def validate_v5_weights(profile_generation_weight: float) -> None:
    if not 0.0 <= profile_generation_weight <= 1.0:
        raise ValueError("V5 profile generation weight must be in [0, 1]")


def is_profile_pair(row: dict[str, Any] | None) -> bool:
    return str((row or {}).get("causal_target_kind") or "") == PROFILE_KIND


def masked_profile_causal_row(
    row: dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, bool]:
    """Remove only the legacy profile-pair gradient, never retention C4."""
    if is_profile_pair(row):
        return None, True
    return row, False


def apply_profile_loss_budget(namespace, row: dict[str, Any], weight: float):
    """Temporarily scale generation on profile rows without mutating config."""
    if str(row.get("task")) != PROFILE_TASK:
        return namespace.generation_weight
    previous = namespace.generation_weight
    namespace.generation_weight = weight
    return previous


def main() -> None:
    profile_generation_weight = float(
        os.environ.get("STATE8_V5_PROFILE_GENERATION_WEIGHT", "0.1")
    )
    validate_v5_weights(profile_generation_weight)
    os.environ.setdefault("STATE8_PROFILE_OWNERSHIP_WEIGHT", "1.0")
    os.environ.setdefault("STATE8_PROFILE_OWNERSHIP_MARGIN", "0.02")
    os.environ.setdefault("STATE8_PROFILE_OWNERSHIP_HEAD_LEARNING_RATE", "3e-4")
    os.environ.setdefault("STATE8_PROFILE_OWNERSHIP_RESET_HEAD", "1")
    os.environ["STATE8_PROFILE_OWNERSHIP_FREEZE_HEAD"] = "1"
    os.environ.setdefault("STATE8_PROFILE_OWNERSHIP_FEATURE_NORMALIZATION", "layernorm")
    os.environ.setdefault("STATE8_PROFILE_OWNERSHIP_MIN_KC_TRAIN_SUPPORT", "4")
    os.environ.setdefault("STATE8_PROFILE_OWNERSHIP_BIAS_PRIOR_STRENGTH", "8")
    os.environ["STATE8_PROFILE_LEGACY_CAUSAL_ENABLED"] = "0"

    original_parse = argparse.ArgumentParser.parse_args
    original_objective = trainer.backward_training_objective
    original_masked_loss = ownership.masked_profile_loss
    original_row_targets = ownership.row_targets
    output_dir: Path | None = None

    def parse(parser, *args, **kwargs):
        nonlocal output_dir
        namespace = original_parse(parser, *args, **kwargs)
        output_dir = Path(namespace.output_dir)
        namespace.state8_v5_loss_fix = True
        namespace.state8_v5_profile_generation_weight = profile_generation_weight
        namespace.state8_v5_profile_causal_objectives = "disabled"
        namespace.state8_v5_nonprofile_objectives = "unchanged"
        namespace.state8_v5_numeric_negative_policy = "deterministic_topk_panel_mean"
        namespace.final_inference_route = "strict_state_only"
        namespace.raw_history_visible_to_decoder = False
        namespace.student_id_visible_to_decoder = False
        return namespace

    def panel_row_targets(row: dict[str, Any], device: torch.device):
        target, mask, negative_panel, overlap_panel = ownership.row_target_panel(
            row, device
        )
        # The ownership wrapper expects [B, KC].  It calls the function below,
        # which accepts the panel tensors saved here as [B, N, KC].
        return target, mask, negative_panel, overlap_panel

    def panel_masked_loss(
        prediction,
        target,
        mask,
        negative_target,
        overlap_mask,
        *,
        ownership_margin,
        kc_weight=None,
    ):
        return ownership.masked_profile_panel_loss(
            prediction,
            target,
            mask,
            negative_target,
            overlap_mask,
            ownership_margin=ownership_margin,
            kc_weight=kc_weight,
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
        state_auxiliary_objective=None,
    ):
        filtered_causal, profile_pair_suppressed = masked_profile_causal_row(causal_row)
        previous_generation_weight = apply_profile_loss_budget(
            namespace, row, profile_generation_weight
        )
        try:
            record = original_objective(
                wrapper,
                tokenizer,
                row,
                filtered_causal,
                cache,
                online_kd_cache,
                namespace,
                device,
                state_token_id,
                loss_scale,
                training_progress,
                state_auxiliary_objective=state_auxiliary_objective,
            )
        finally:
            namespace.generation_weight = previous_generation_weight
        is_profile = str(row.get("task")) == PROFILE_TASK
        if profile_pair_suppressed:
            # Preserve the scheduled-pair identity as an exposure receipt.  The
            # base completion audit requires every preregistered pair exactly
            # once even when V5 intentionally assigns that legacy branch zero
            # gradient.  ``causal_effective_weight=0`` makes the distinction
            # explicit and prevents this metadata from being read as a trained
            # text-transport objective.
            record["causal_pair_id"] = str(causal_row["pair_id"])
            record["causal_effective_weight"] = 0.0
            record["causal_objective_phase"] = "v5_profile_legacy_suppressed"
        record.update({
            "state8_v5_loss_fix": True,
            "state8_v5_profile_row": is_profile,
            "state8_v5_profile_generation_weight": (
                profile_generation_weight if is_profile else None
            ),
            "state8_v5_profile_causal_suppressed": profile_pair_suppressed,
            "state8_v5_legacy_geometry_effective_weight": 0.0,
            "state8_v5_legacy_early_label_effective_weight": 0.0,
            "state8_v5_numeric_panel_size": float(
                record.get("profile_ownership_panel_size", 0.0) or 0.0
            ),
        })
        return record

    argparse.ArgumentParser.parse_args = parse
    trainer.backward_training_objective = objective
    ownership.masked_profile_loss = panel_masked_loss
    ownership.row_targets = panel_row_targets
    try:
        ownership.main()
        if int(os.environ.get("OMPI_COMM_WORLD_RANK", "0")) != 0:
            return
        if output_dir is None:
            raise RuntimeError("V5 trainer did not observe output directory")
        history = json.loads(
            (output_dir / "trainer_history.json").read_text(encoding="utf-8")
        )
        profile_rows = [row for row in history if row.get("state8_v5_profile_row")]
        if not profile_rows:
            raise RuntimeError("V5 never exercised a profile row")
        panel_rows = [
            row for row in profile_rows
            if float(row.get("state8_v5_numeric_panel_size", 0.0)) > 1.0
        ]
        if not panel_rows:
            raise RuntimeError("V5 never exercised a multi-student negative panel")
        if any(
            float(row.get("state8_v5_legacy_geometry_effective_weight", -1)) != 0.0
            or float(row.get("state8_v5_legacy_early_label_effective_weight", -1)) != 0.0
            for row in profile_rows
        ):
            raise RuntimeError("V5 legacy profile objectives were not fully disabled")
        suppressed = sum(
            bool(row.get("state8_v5_profile_causal_suppressed")) for row in history
        )
        run_config = json.loads(
            (output_dir / "run_config.json").read_text(encoding="utf-8")
        )
        diagnostic = bool(
            run_config.get("diagnostic_max_micro_steps") is not None
            or run_config.get("diagnostic_stop_after_round") is not None
            or run_config.get("diagnostic_weight_only_warm_start") is True
        )
        receipt = {
            "stage": STAGE,
            "status": "diagnostic_complete" if diagnostic else "complete",
            "diagnostic": diagnostic,
            "profile_rows": len(profile_rows),
            "multi_negative_profile_rows": len(panel_rows),
            "profile_causal_exposures_suppressed": suppressed,
            "profile_generation_weight": profile_generation_weight,
            "profile_numeric_ownership_weight": float(
                os.environ["STATE8_PROFILE_OWNERSHIP_WEIGHT"]
            ),
            "profile_numeric_head_frozen": True,
            "profile_legacy_text_transport_weight": 0.0,
            "profile_legacy_semantic_geometry_weight": 0.0,
            "profile_legacy_early_label_weight": 0.0,
            "nonprofile_retention_objective": "unchanged",
            "negative_policy": "deterministic_topk_panel_mean",
            "future_labels_visible_to_training": True,
            "future_labels_visible_to_decoder": False,
            "raw_history_visible_to_decoder": False,
            "student_id_visible_to_decoder": False,
            "final_inference_route": "strict_state_only",
            "entrypoint_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
        (output_dir / "profile_ownership_v5.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    finally:
        argparse.ArgumentParser.parse_args = original_parse
        trainer.backward_training_objective = original_objective
        ownership.masked_profile_loss = original_masked_loss
        ownership.row_targets = original_row_targets


if __name__ == "__main__":
    main()
