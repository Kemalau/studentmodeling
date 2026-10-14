from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from train_recurrent_student_m4_promoted import (  # noqa: E402
    apply_two_stage_consolidation_transition,
    apply_promoted_learning_rate_transition,
    validate_two_stage_promotion,
    validate_promoted_resume_recipe,
)
import torch


def two_stage_manifest(profile: str = "state_kd") -> dict:
    return {
        **manifest(profile),
        "branch": "equal_t3_t6_open_t3_two_stage",
        "selected_candidate_id": f"equal_t3_t6_open_t3_two_stage__{profile}",
        "curriculum_mode": "open_t3_formation_then_causal_task_consolidation",
        "curriculum_phases": [
            {"name": "open_t3_formation"},
            {"name": "causal_task_consolidation"},
        ],
        "completed_prefix_rows": 12000,
        "promoted_rows": 2000,
        "formation_requested_rows": 1000,
        "formation_aligned_rows": 1008,
        "formation_transition_global_position": 13008,
        "formation_loss_profile": "state_kd",
        "formation_learning_rate": 5e-6,
        "consolidation_loss_profile": profile,
        "consolidation_learning_rate": 5e-7,
    }


def test_two_stage_manifest_requires_synchronized_boundary() -> None:
    value = validate_two_stage_promotion(two_stage_manifest(), world_size=16)
    assert value["transition_round"] == 813
    broken = two_stage_manifest()
    broken["formation_transition_global_position"] = 13000
    try:
        validate_two_stage_promotion(broken, world_size=16)
    except ValueError as error:
        assert "boundary" in str(error)
    else:
        raise AssertionError("unaligned two-stage boundary was accepted")


def test_two_stage_transition_rescales_once_and_can_disable_kd() -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=5e-6)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    config = {
        "promoted_two_stage_enabled": True,
        "promoted_two_stage_transition_round": 813,
        "promoted_two_stage_phase": "formation",
        "promoted_two_stage_formation_learning_rate": 5e-6,
        "promoted_two_stage_consolidation_learning_rate": 5e-7,
        "promoted_two_stage_consolidation_loss_profile": "state_path",
    }
    args = type("Args", (), {})()
    args.learning_rate = 5e-6
    args.behavior_kd_weight = 0.5
    args.direct_replay_weight = 0.0
    assert apply_two_stage_consolidation_transition(
        optimizer, scheduler, config, args, round_index=812
    ) is False
    assert apply_two_stage_consolidation_transition(
        optimizer, scheduler, config, args, round_index=813
    ) is True
    assert config["promoted_two_stage_phase"] == "consolidation"
    assert args.learning_rate == 5e-7
    assert args.behavior_kd_weight == 0.0
    assert abs(optimizer.param_groups[0]["lr"] - 5e-7) < 1e-15
    assert apply_two_stage_consolidation_transition(
        optimizer, scheduler, config, args, round_index=814
    ) is False


def two_stage_config(profile: str, phase: str) -> dict:
    target_weights = {
        "state_kd": (0.5, 0.0),
        "state_path": (0.0, 0.0),
    }[profile]
    active_lr = 5e-6 if phase == "formation" else 5e-7
    active_weights = (0.5, 0.0) if phase == "formation" else target_weights
    value = promoted_config(profile)
    value.update(
        learning_rate=active_lr,
        behavior_kd_weight=active_weights[0],
        direct_replay_weight=active_weights[1],
        promoted_selected_candidate_id=(
            f"equal_t3_t6_open_t3_two_stage__{profile}"
        ),
        promoted_selected_learning_rate=5e-7,
        promoted_lr_transition_applied=True,
        promoted_lr_transition_target=5e-6,
        training_view="open_t3_selective_kd",
        promoted_training_view="open_t3_selective_kd",
        promoted_two_stage_enabled=True,
        promoted_two_stage_transition_global_position=13008,
        promoted_two_stage_transition_round=813,
        promoted_two_stage_formation_learning_rate=5e-6,
        promoted_two_stage_consolidation_learning_rate=5e-7,
        promoted_two_stage_formation_loss_profile="state_kd",
        promoted_two_stage_consolidation_loss_profile=profile,
        promoted_two_stage_phase=phase,
        promoted_two_stage_active_learning_rate=active_lr,
        promoted_two_stage_transition_applied=phase == "consolidation",
        promoted_objective_transition_target_weights=list(active_weights),
    )
    return value


def test_two_stage_first_application_and_both_recovery_phases() -> None:
    selected = two_stage_manifest("state_path")
    formation = two_stage_config("state_path", "formation")
    validate_promoted_resume_recipe(
        source_config(), formation, source_payload(), selected
    )
    formation_checkpoint = dict(formation)
    validate_promoted_resume_recipe(
        formation_checkpoint,
        dict(formation),
        {**source_payload(), "round": 800},
        selected,
    )
    consolidation = two_stage_config("state_path", "consolidation")
    validate_promoted_resume_recipe(
        consolidation,
        dict(consolidation),
        {**source_payload(), "round": 900},
        selected,
    )


def test_two_stage_recovery_cannot_revert_phase_or_active_lr() -> None:
    selected = two_stage_manifest("state_path")
    checkpoint = two_stage_config("state_path", "consolidation")
    reverted = two_stage_config("state_path", "formation")
    try:
        validate_promoted_resume_recipe(
            checkpoint,
            reverted,
            {**source_payload(), "round": 900},
            selected,
        )
    except ValueError as error:
        assert "drift" in str(error) or "phase" in str(error)
    else:
        raise AssertionError("two-stage recovery reverted to formation")


def manifest(profile: str = "state_path") -> dict:
    return {
        "stage": "layerwise_task_retention_promoted_order_v1",
        "branch": "staged_t3_t6",
        "selected_loss_profile": profile,
        "selected_candidate_id": f"staged_t3_t6__{profile}",
        "selected_learning_rate": 5e-7,
        "selected_causal_weight": 2.0,
        "source_checkpoint_round": 750,
    }


def source_config() -> dict:
    return {
        "behavior_kd_weight": 0.5,
        "direct_replay_weight": 0.25,
        "causal_weight": 1.0,
        "state_token_alias": "<|vision_pad|>",
        "train_sha256": "abc",
        "learning_rate": 5e-7,
    }


def promoted_config(profile: str = "state_path") -> dict:
    weights = {
        "standard": (0.5, 0.25),
        "state_kd": (0.5, 0.0),
        "state_path": (0.0, 0.0),
    }[profile]
    return {
        **source_config(),
        "behavior_kd_weight": weights[0],
        "direct_replay_weight": weights[1],
        "promoted_selected_candidate_id": f"staged_t3_t6__{profile}",
        "promoted_selected_loss_profile": profile,
        "promoted_objective_transition_applied": profile != "standard",
        "promoted_objective_transition_source_round": 750,
        "promoted_objective_transition_source_weights": [0.5, 0.25],
        "promoted_objective_transition_target_weights": list(weights),
        "promoted_selected_learning_rate": 5e-7,
        "promoted_selected_causal_weight": 2.0,
        "promoted_causal_weight_transition_applied": True,
        "promoted_causal_weight_transition_source": 1.0,
        "promoted_causal_weight_transition_target": 2.0,
        "promoted_lr_transition_applied": False,
        "promoted_lr_transition_source": 5e-7,
        "promoted_lr_transition_target": 5e-7,
        "training_view": "standard",
        "promoted_training_view": "standard",
        "causal_weight": 2.0,
    }


def source_payload() -> dict:
    return {"distributed": True, "world_size": 16, "round": 750}


def test_allows_only_selected_state_path_transition_at_round_750() -> None:
    validate_promoted_resume_recipe(
        source_config(), promoted_config(), source_payload(), manifest()
    )


def test_standard_profile_remains_an_exact_resume() -> None:
    validate_promoted_resume_recipe(
        source_config(), promoted_config("standard"), source_payload(), manifest("standard")
    )


def test_state_kd_removes_only_direct_replay_at_promotion_boundary() -> None:
    validate_promoted_resume_recipe(
        source_config(), promoted_config("state_kd"), source_payload(), manifest("state_kd")
    )


def test_rejects_non_loss_scientific_drift_during_transition() -> None:
    current = promoted_config()
    current["causal_weight"] = 3.0
    try:
        validate_promoted_resume_recipe(source_config(), current, source_payload(), manifest())
    except ValueError as error:
        assert "causal weight" in str(error)
    else:
        raise AssertionError("non-loss drift was accepted")


def test_promoted_recovery_is_strict_and_cannot_switch_again() -> None:
    checkpoint = promoted_config()
    current = promoted_config()
    validate_promoted_resume_recipe(checkpoint, current, {**source_payload(), "round": 900}, manifest())
    current["behavior_kd_weight"] = 0.5
    try:
        validate_promoted_resume_recipe(checkpoint, current, {**source_payload(), "round": 900}, manifest())
    except ValueError as error:
        assert "target weights drift" in str(error)
    else:
        raise AssertionError("second objective transition was accepted")


def test_wrong_source_round_or_world_size_fails_closed() -> None:
    for key, value in (("round", 751), ("world_size", 8)):
        payload = source_payload()
        payload[key] = value
        try:
            validate_promoted_resume_recipe(source_config(), promoted_config(), payload, manifest())
        except ValueError as error:
            assert "drift" in str(error)
        else:
            raise AssertionError(f"source {key} drift was accepted")


def test_full_eedi_candidate_allows_only_declared_lr_transition() -> None:
    selected = manifest()
    selected.update(
        branch="eedi_full",
        selected_candidate_id="eedi_full__state_path__lr2e-06",
        selected_learning_rate=2e-6,
    )
    current = promoted_config()
    current.update(
        learning_rate=2e-6,
        promoted_selected_candidate_id="eedi_full__state_path__lr2e-06",
        promoted_selected_learning_rate=2e-6,
        promoted_lr_transition_applied=True,
        promoted_lr_transition_target=2e-6,
    )
    validate_promoted_resume_recipe(source_config(), current, source_payload(), selected)
    current["learning_rate"] = 5e-6
    try:
        validate_promoted_resume_recipe(source_config(), current, source_payload(), selected)
    except ValueError as error:
        assert "learning rate" in str(error)
    else:
        raise AssertionError("undeclared learning-rate transition was accepted")


def test_full_eedi_state_kd_candidate_uses_actual_loss_profile_in_id() -> None:
    selected = manifest("state_kd")
    selected.update(
        branch="eedi_full",
        selected_candidate_id="eedi_full__state_kd__lr2e-06",
        selected_learning_rate=2e-6,
    )
    current = promoted_config("state_kd")
    current.update(
        learning_rate=2e-6,
        promoted_selected_candidate_id="eedi_full__state_kd__lr2e-06",
        promoted_selected_learning_rate=2e-6,
        promoted_lr_transition_applied=True,
        promoted_lr_transition_target=2e-6,
    )
    validate_promoted_resume_recipe(source_config(), current, source_payload(), selected)


def test_equal_open_state_kd_candidate_allows_declared_lr_transition() -> None:
    selected = manifest("state_kd")
    selected.update(
        branch="equal_t3_t6_open_t3",
        selected_candidate_id="equal_t3_t6_open_t3__state_kd__lr2e-06",
        selected_learning_rate=2e-6,
    )
    current = promoted_config("state_kd")
    current.update(
        learning_rate=2e-6,
        promoted_selected_candidate_id=(
            "equal_t3_t6_open_t3__state_kd__lr2e-06"
        ),
        promoted_selected_learning_rate=2e-6,
        promoted_lr_transition_applied=True,
        promoted_lr_transition_target=2e-6,
        training_view="open_t3_selective_kd",
        promoted_training_view="open_t3_selective_kd",
    )
    validate_promoted_resume_recipe(source_config(), current, source_payload(), selected)


def test_equal_open_state_kd_midpoint_allows_lr1e6_transition() -> None:
    selected = manifest("state_kd")
    selected.update(
        branch="equal_t3_t6_open_t3",
        selected_candidate_id="equal_t3_t6_open_t3__state_kd__lr1e-06",
        selected_learning_rate=1e-6,
    )
    current = promoted_config("state_kd")
    current.update(
        learning_rate=1e-6,
        promoted_selected_candidate_id=(
            "equal_t3_t6_open_t3__state_kd__lr1e-06"
        ),
        promoted_selected_learning_rate=1e-6,
        promoted_lr_transition_applied=True,
        promoted_lr_transition_target=1e-6,
        training_view="open_t3_selective_kd",
        promoted_training_view="open_t3_selective_kd",
    )
    validate_promoted_resume_recipe(source_config(), current, source_payload(), selected)


def test_full_eedi_open_t3_candidate_has_a_distinct_audited_id() -> None:
    selected = manifest()
    selected.update(
        branch="eedi_full_open_t3",
        selected_candidate_id="eedi_full_open_t3__state_path__lr2e-06",
        selected_learning_rate=2e-6,
    )
    current = promoted_config()
    current.update(
        learning_rate=2e-6,
        promoted_selected_candidate_id="eedi_full_open_t3__state_path__lr2e-06",
        promoted_selected_learning_rate=2e-6,
        promoted_lr_transition_applied=True,
        promoted_lr_transition_target=2e-6,
        training_view="open_t3_no_candidate_enumeration",
        promoted_training_view="open_t3_no_candidate_enumeration",
    )
    validate_promoted_resume_recipe(source_config(), current, source_payload(), selected)


def test_open_t3_candidate_rejects_standard_training_view() -> None:
    selected = manifest()
    selected.update(
        branch="eedi_full_open_t3",
        selected_candidate_id="eedi_full_open_t3__state_path__lr2e-06",
        selected_learning_rate=2e-6,
    )
    current = promoted_config()
    current.update(
        learning_rate=2e-6,
        promoted_selected_candidate_id="eedi_full_open_t3__state_path__lr2e-06",
        promoted_selected_learning_rate=2e-6,
        promoted_lr_transition_applied=True,
        promoted_lr_transition_target=2e-6,
    )
    try:
        validate_promoted_resume_recipe(source_config(), current, source_payload(), selected)
    except ValueError as error:
        assert "training-view" in str(error)
    else:
        raise AssertionError("open-T3 candidate accepted the standard training view")


def test_open_t3_selective_kd_candidate_uses_distinct_training_view() -> None:
    selected = manifest("state_kd")
    selected.update(
        branch="eedi_full_open_t3",
        selected_candidate_id="eedi_full_open_t3__state_kd__lr2e-06",
        selected_learning_rate=2e-6,
    )
    current = promoted_config("state_kd")
    current.update(
        learning_rate=2e-6,
        promoted_selected_candidate_id="eedi_full_open_t3__state_kd__lr2e-06",
        promoted_selected_learning_rate=2e-6,
        promoted_lr_transition_applied=True,
        promoted_lr_transition_target=2e-6,
        training_view="open_t3_selective_kd",
        promoted_training_view="open_t3_selective_kd",
    )
    validate_promoted_resume_recipe(source_config(), current, source_payload(), selected)


def test_lr_transition_preserves_moments_and_rescales_scheduler() -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=5e-7)
    parameter.grad = torch.tensor(2.0)
    optimizer.step()
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 0.5)
    before = optimizer.state[parameter]["exp_avg"].clone()
    apply_promoted_learning_rate_transition(
        optimizer, scheduler, source_learning_rate=5e-7, target_learning_rate=2e-6
    )
    assert torch.equal(optimizer.state[parameter]["exp_avg"], before)
    assert abs(float(optimizer.param_groups[0]["lr"]) - 1e-6) < 1e-15
    assert abs(float(scheduler.base_lrs[0]) - 2e-6) < 1e-15
