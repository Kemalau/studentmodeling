#!/usr/bin/env python3
"""Diagnosis-first State8 repair with source-aware trajectory anchoring.

The existing task anchor scales a cosine-only trajectory loss uniformly across
all diagnosis sources.  That is insufficient for Eedi: a recurrent state can
share the teacher direction while losing the layer/slot magnitudes that
distinguish a large open taxonomy.  This entrypoint keeps the strict state-only
decoder and all diagnosis C4 objectives, but adds two changes that are audited
per training example:

* source-aware diagnosis schedules (Eedi and Malrule are controlled
  independently), plus an increasing persistence schedule for T6 retention;
* an explicit relative-MSE component in the layerwise teacher-state loss;
* optional native/donor causal-pair trajectory binding to the same frozen
  full-history teacher used by the ordinary State8 anchor.  This closes the
  gap between merely separating two recurrent states and placing each state
  on a decoder-interpretable misconception trajectory.

No source, label, history, or teacher state is exposed to the final decoder.
The source is used only to choose a training-time loss coefficient.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import train_recurrent_student_m4 as trainer
import train_state8_diagnosis_first_causal_repair as diagnosis_first
import train_state8_taxonomy_causal_repair as taxonomy
from state8_current_query_routing import apply_current_query_routing
from train_state8_task_anchored_causal_repair import (
    anchor_phase,
    cli_value,
    parse_schedule,
)


def task_anchor_kind(row: dict[str, Any]) -> str:
    task = str(row.get("task") or "")
    source = str(row.get("source") or "")
    if task == "diagnose_misconception":
        if source == "eedi_mining":
            return "diagnosis_eedi"
        if source == "malrule":
            return "diagnosis_malrule"
        return "diagnosis_other"
    if task == "predict_error_persistence":
        return "persistence"
    return "other"


def effective_anchor_weight(
    row: dict[str, Any],
    progress: float,
    *,
    eedi_schedule: tuple[float, float, float],
    malrule_schedule: tuple[float, float, float],
    diagnosis_other_schedule: tuple[float, float, float],
    persistence_schedule: tuple[float, float, float],
    other_weight: float,
) -> tuple[int, str, float]:
    phase = anchor_phase(float(progress))
    kind = task_anchor_kind(row)
    schedules = {
        "diagnosis_eedi": eedi_schedule,
        "diagnosis_malrule": malrule_schedule,
        "diagnosis_other": diagnosis_other_schedule,
        "persistence": persistence_schedule,
    }
    weight = schedules[kind][phase] if kind in schedules else float(other_weight)
    return phase, kind, float(weight)


def _positive_schedule(name: str, default: str) -> tuple[float, float, float]:
    # The existing parser requires a non-increasing curriculum.  Persistence
    # deliberately increases, so validate its three values separately below.
    if name == "STATE8_T6_ANCHOR_SCHEDULE":
        pieces = tuple(float(piece.strip()) for piece in os.environ.get(name, default).split(","))
        if len(pieces) != 3 or any(weight <= 0 for weight in pieces):
            raise ValueError(f"{name} must contain three positive weights")
        if not (pieces[0] <= pieces[1] <= pieces[2]):
            raise ValueError(f"{name} must be non-decreasing")
        return pieces
    return parse_schedule(os.environ.get(name, default))


def env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean flag, got {value!r}")


def current_query_mode() -> str:
    value = os.environ.get("STATE8_CURRENT_QUERY_ROUTING", "0").strip().lower()
    aliases = {
        "0": "off", "false": "off", "no": "off", "off": "off",
        "1": "all", "true": "all", "yes": "all", "on": "all", "all": "all",
        "eedi": "eedi_only", "eedi_only": "eedi_only",
    }
    if value not in aliases:
        raise ValueError(
            "STATE8_CURRENT_QUERY_ROUTING must be off, all, or eedi_only"
        )
    return aliases[value]


def is_bindable_causal_pair(row: dict[str, Any] | None) -> bool:
    """Select natural diagnosis twins with two explicit direct-state teachers."""
    return bool(
        row
        and str(row.get("causal_target_kind") or "")
        == "diagnosis_natural_schema"
        and row.get("native_update_texts")
        and row.get("donor_update_texts")
        and row.get("native_direct_text")
        and row.get("donor_direct_text")
        and row.get("native_target")
        and row.get("donor_target")
    )


def aggregate_causal_teacher_binding(
    history: list[dict[str, Any]], configured_weight: float
) -> dict[str, Any]:
    """Build a fail-closed receipt from globally gathered trainer history."""
    applied = [
        row
        for row in history
        if float(row.get("causal_teacher_binding_effective_weight", 0.0)) > 0.0
    ]
    if configured_weight > 0 and not applied:
        raise RuntimeError(
            "Positive causal teacher-binding weight produced no history records"
        )
    by_source = Counter(str(row.get("causal_teacher_binding_source") or "unknown") for row in applied)
    by_rank = Counter(str(row.get("rank", "single")) for row in applied)
    return {
        "eligible": len(applied),
        "applied": len(applied),
        "by_source": dict(sorted(by_source.items())),
        "by_rank": dict(sorted(by_rank.items())),
        "audit_source": "global_trainer_history",
    }


def main() -> None:
    diagnosis_field_order = list(diagnosis_first.configured_field_order())
    routing_mode = current_query_mode()
    current_query_routing = routing_mode != "off"
    routing_sources = {"eedi_mining"} if routing_mode == "eedi_only" else None
    eedi_schedule = _positive_schedule(
        "STATE8_EEDI_ANCHOR_SCHEDULE", "1.0,1.0,1.0"
    )
    malrule_schedule = _positive_schedule(
        "STATE8_MALRULE_ANCHOR_SCHEDULE", "1.0,1.0,1.0"
    )
    diagnosis_other_schedule = _positive_schedule(
        "STATE8_OTHER_DIAGNOSIS_ANCHOR_SCHEDULE", "1.0,0.5,0.25"
    )
    persistence_schedule = _positive_schedule(
        "STATE8_T6_ANCHOR_SCHEDULE", "0.5,0.75,1.0"
    )
    other_weight = float(os.environ.get("STATE8_OTHER_ANCHOR_WEIGHT", "0.1"))
    cosine_weight = float(os.environ.get("STATE8_TRAJECTORY_COSINE_WEIGHT", "1.0"))
    relative_mse_weight = float(
        os.environ.get("STATE8_TRAJECTORY_RELATIVE_MSE_WEIGHT", "1.0")
    )
    configured_weight = float(
        os.environ.get("STATE8_REPAIR_TRAJECTORY_WEIGHT", "1.0")
    )
    causal_teacher_weight = float(
        os.environ.get("STATE8_CAUSAL_TEACHER_TRAJECTORY_WEIGHT", "0")
    )
    causal_teacher_cosine_weight = float(
        os.environ.get("STATE8_CAUSAL_TEACHER_COSINE_WEIGHT", "1.0")
    )
    causal_teacher_relative_mse_weight = float(
        os.environ.get("STATE8_CAUSAL_TEACHER_RELATIVE_MSE_WEIGHT", "1.0")
    )
    schedules = (
        eedi_schedule,
        malrule_schedule,
        diagnosis_other_schedule,
        persistence_schedule,
    )
    maximum = max(weight for schedule in schedules for weight in schedule)
    if other_weight < 0:
        raise ValueError("STATE8_OTHER_ANCHOR_WEIGHT must be non-negative")
    if cosine_weight < 0 or relative_mse_weight <= 0:
        raise ValueError(
            "Source-aware repair requires non-negative cosine and positive relative MSE"
        )
    if cosine_weight + relative_mse_weight <= 0:
        raise ValueError("At least one trajectory component must be positive")
    if causal_teacher_weight < 0:
        raise ValueError("Causal teacher-trajectory weight must be non-negative")
    if (
        causal_teacher_cosine_weight < 0
        or causal_teacher_relative_mse_weight < 0
        or causal_teacher_cosine_weight + causal_teacher_relative_mse_weight <= 0
    ):
        raise ValueError(
            "Causal teacher binding requires non-negative, nonzero trajectory components"
        )
    if abs(configured_weight - maximum) > 1e-12:
        raise ValueError(
            "Configured trajectory weight must equal the largest source/task anchor"
        )

    output_dir = Path(cli_value("--output-dir")).resolve()
    original_parse = argparse.ArgumentParser.parse_args
    original_objective = trainer.backward_training_objective
    original_natural_view = taxonomy.apply_state8_natural_curriculum_view
    routing_audits: list[dict[str, int]] = []

    def parse(parser, *args, **kwargs):
        namespace = original_parse(parser, *args, **kwargs)
        namespace.source_anchor_eedi_schedule = list(eedi_schedule)
        namespace.source_anchor_malrule_schedule = list(malrule_schedule)
        namespace.source_anchor_other_diagnosis_schedule = list(
            diagnosis_other_schedule
        )
        namespace.persistence_anchor_schedule = list(persistence_schedule)
        namespace.source_anchor_other_weight = other_weight
        namespace.source_anchor_boundaries = [0.40, 0.70]
        namespace.trajectory_cosine_weight = cosine_weight
        namespace.trajectory_relative_mse_weight = relative_mse_weight
        namespace.source_anchor_train_only = True
        namespace.source_visible_at_inference = False
        namespace.raw_history_visible_to_decoder = False
        namespace.final_inference_route = "strict_state_only"
        namespace.current_query_routing = (
            f"current_item_visible_response_state_only_v1:{routing_mode}"
            if current_query_routing
            else "legacy_generic_query"
        )
        namespace.current_item_visible_to_decoder = current_query_routing
        namespace.student_response_visible_to_decoder = False
        namespace.causal_teacher_trajectory_weight = causal_teacher_weight
        namespace.causal_teacher_trajectory_cosine_weight = (
            causal_teacher_cosine_weight
        )
        namespace.causal_teacher_trajectory_relative_mse_weight = (
            causal_teacher_relative_mse_weight
        )
        namespace.causal_teacher_trajectory_scope = (
            "native_and_donor_diagnosis_pairs_train_only"
            if causal_teacher_weight > 0
            else "disabled"
        )
        return namespace

    def routed_natural_view(rows):
        audit = dict(original_natural_view(rows))
        if current_query_routing:
            routed = apply_current_query_routing(rows, sources=routing_sources)
            audit.update(routed)
            if int(routed["current_query_routed_rows"]) > 0:
                routing_audits.append(routed)
        return audit

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
        phase, kind, weight = effective_anchor_weight(
            row,
            float(training_progress),
            eedi_schedule=eedi_schedule,
            malrule_schedule=malrule_schedule,
            diagnosis_other_schedule=diagnosis_other_schedule,
            persistence_schedule=persistence_schedule,
            other_weight=other_weight,
        )
        before = float(namespace.consistency_weight)
        namespace.consistency_weight = weight
        try:
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
        finally:
            namespace.consistency_weight = before
        record["source_anchor_phase"] = phase
        record["source_anchor_kind"] = kind
        record["source_anchor_effective_weight"] = weight
        record["source_anchor_cosine_weight"] = cosine_weight
        record["source_anchor_relative_mse_weight"] = relative_mse_weight
        record["source_anchor_train_only"] = True
        record["causal_teacher_binding_loss"] = 0.0
        if causal_teacher_weight > 0 and is_bindable_causal_pair(causal_row):
            effective_weight = causal_teacher_weight * float(
                namespace.causal_exposure_multiplier
            )
            value, diagnostics = trainer.backward_causal_trajectory_objective(
                wrapper,
                tokenizer,
                causal_row,
                getattr(
                    namespace,
                    "state_token_rendered_alias",
                    namespace.state_token_alias,
                ),
                namespace.max_length,
                device,
                cosine_weight=causal_teacher_cosine_weight,
                relative_mse_weight=causal_teacher_relative_mse_weight,
                weight=effective_weight,
                loss_scale=loss_scale,
            )
            record["causal_teacher_binding_loss"] = value
            record["causal_teacher_binding_effective_weight"] = effective_weight
            record["causal_teacher_binding_source"] = str(
                causal_row.get("source") or "unknown"
            )
            record["loss"] += effective_weight * value
            record.update(
                {f"causal_teacher_{key}": metric for key, metric in diagnostics.items()}
            )
        return record

    argparse.ArgumentParser.parse_args = parse
    trainer.backward_training_objective = objective
    taxonomy.apply_state8_natural_curriculum_view = routed_natural_view
    try:
        diagnosis_first.main()
        if int(os.environ.get("OMPI_COMM_WORLD_RANK", "0")) == 0:
            history = json.loads(
                (output_dir / "trainer_history.json").read_text(encoding="utf-8")
            )
            diagnosis_receipt = json.loads(
                (output_dir / "diagnosis_first_causal_repair.json").read_text(
                    encoding="utf-8"
                )
            )
            if not isinstance(history, list) or not history:
                raise ValueError("Source-anchored training history is empty")
            kind_counts = Counter(str(row["source_anchor_kind"]) for row in history)
            phase_counts = Counter(int(row["source_anchor_phase"]) for row in history)
            kind_weight_counts: dict[str, Counter[float]] = defaultdict(Counter)
            for row in history:
                phase, kind, expected = effective_anchor_weight(
                    row,
                    float(row.get("global_position", 0)) / max(1, len(history) - 1),
                    eedi_schedule=eedi_schedule,
                    malrule_schedule=malrule_schedule,
                    diagnosis_other_schedule=diagnosis_other_schedule,
                    persistence_schedule=persistence_schedule,
                    other_weight=other_weight,
                )
                # The trainer's exact progress coordinate is serialized in the
                # selected phase/weight.  Recompute expected from that phase to
                # avoid floating boundary ambiguity in this final audit.
                observed_phase = int(row["source_anchor_phase"])
                observed_kind = str(row["source_anchor_kind"])
                if observed_kind != kind:
                    raise ValueError("Source-anchor kind drift in trainer history")
                expected_by_kind = {
                    "diagnosis_eedi": eedi_schedule,
                    "diagnosis_malrule": malrule_schedule,
                    "diagnosis_other": diagnosis_other_schedule,
                    "persistence": persistence_schedule,
                }
                expected_weight = (
                    expected_by_kind[observed_kind][observed_phase]
                    if observed_kind in expected_by_kind
                    else other_weight
                )
                if float(row["source_anchor_effective_weight"]) != float(
                    expected_weight
                ):
                    raise ValueError("A row used the wrong source/task anchor")
                if float(row["source_anchor_cosine_weight"]) != cosine_weight:
                    raise ValueError("Cosine component audit drift")
                if (
                    float(row["source_anchor_relative_mse_weight"])
                    != relative_mse_weight
                ):
                    raise ValueError("Relative-MSE component audit drift")
                kind_weight_counts[observed_kind][expected_weight] += 1
            causal_teacher_audit = aggregate_causal_teacher_binding(
                history, causal_teacher_weight
            )
            if kind_counts["diagnosis_eedi"] <= 0 or kind_counts["diagnosis_malrule"] <= 0:
                raise ValueError(f"Source-aware diagnosis support is incomplete: {kind_counts}")
            if kind_counts["persistence"] <= 0 or kind_counts["other"] <= 0:
                raise ValueError(f"Unified retention support is incomplete: {kind_counts}")
            if diagnosis_receipt.get("field_order") != diagnosis_field_order:
                raise ValueError("Configured diagnosis field order drift")
            routed_rows = sum(
                int(audit.get("current_query_routed_rows", 0))
                for audit in routing_audits
            )
            routed_teacher_rows = sum(
                int(audit.get("current_query_teacher_rows", 0))
                for audit in routing_audits
            )
            if current_query_routing and (
                routed_rows <= 0 or routed_teacher_rows != routed_rows
            ):
                raise ValueError(
                    "Current-query training did not route every diagnosis row "
                    "through aligned teacher and student queries"
                )
            receipt = {
                "stage": "state8_diagnosis_first_source_anchored_repair_v1",
                "status": "complete",
                "state_token_count": 8,
                "eedi_anchor_schedule": list(eedi_schedule),
                "malrule_anchor_schedule": list(malrule_schedule),
                "other_diagnosis_anchor_schedule": list(
                    diagnosis_other_schedule
                ),
                "persistence_anchor_schedule": list(persistence_schedule),
                "other_task_anchor_weight": other_weight,
                "schedule_boundaries": [0.40, 0.70],
                "trajectory_cosine_weight": cosine_weight,
                "trajectory_relative_mse_weight": relative_mse_weight,
                "causal_teacher_trajectory_weight": causal_teacher_weight,
                "causal_teacher_trajectory_cosine_weight": (
                    causal_teacher_cosine_weight
                ),
                "causal_teacher_trajectory_relative_mse_weight": (
                    causal_teacher_relative_mse_weight
                ),
                "causal_teacher_trajectory_scope": (
                    "native_and_donor_diagnosis_pairs_train_only"
                    if causal_teacher_weight > 0
                    else "disabled"
                ),
                "causal_teacher_binding": causal_teacher_audit,
                "kind_counts": dict(sorted(kind_counts.items())),
                "phase_counts": dict(sorted(phase_counts.items())),
                "kind_weight_counts": {
                    kind: {str(weight): count for weight, count in sorted(counts.items())}
                    for kind, counts in sorted(kind_weight_counts.items())
                },
                "source_anchor_train_only": True,
                "source_visible_at_inference": False,
                "teacher_trainable": False,
                "raw_history_visible_to_decoder": False,
                "final_inference_route": "strict_state_only",
                "current_query_routing": (
                    f"current_item_visible_response_state_only_v1:{routing_mode}"
                    if current_query_routing
                    else "legacy_generic_query"
                ),
                "current_query_sources": (
                    sorted(routing_sources) if routing_sources is not None else ["all"]
                ) if current_query_routing else [],
                "current_item_visible_to_decoder": current_query_routing,
                "student_response_visible_to_decoder": False,
                "current_query_audits": routing_audits,
                "field_order": diagnosis_field_order,
                "canonical_label_role": (
                    "state_identity_first_then_semantic_explanation"
                    if diagnosis_field_order[0] == "label"
                    else "auxiliary_ordinary_weight"
                ),
                "entrypoint_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
            }
            (output_dir / "diagnosis_first_source_anchored_repair.json").write_text(
                json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    finally:
        argparse.ArgumentParser.parse_args = original_parse
        trainer.backward_training_objective = original_objective
        taxonomy.apply_state8_natural_curriculum_view = original_natural_view


if __name__ == "__main__":
    main()
