import json
from pathlib import Path

import yaml

from scripts.finalize_state8_eedi_same_question_promotion import finalize
from scripts.materialize_state8_eedi_same_question_answer_c4 import materialize
from scripts.validate_state8_eedi_same_question_answer_c4_amlt import validate


def selection():
    selected = {
        "arm": "updater", "qualified": True,
        "artifact_sha256": {"adapter_model": "a" * 64, "recurrent_state": "b" * 64},
        "finalfit_variant": "same_question_updater_mse1_seed3072",
        "remote_root": "/mnt/amulet/studentmodeling/diagnostics/state8_stateonly_repair_seed3072_x",
    }
    return {
        "stage": "state8_eedi_same_question_joint_diagnostic_selection_v1",
        "status": "complete", "pass": True, "selected_arm": "updater", "selected": selected,
    }


def answer(passed=True):
    clean = {
        "swap": {"flip_to_A_rate": 0.6}, "random": {"flip_to_A_rate": 0.1},
        "samemal": {"flip_to_A_rate": 0.05}, "noop": {"flip_to_A_rate": 0.0},
    }
    return {
        "state_shape": [64, 8, 5120], "raw_history_visible_to_decoder": False,
        "persistent_history_kv_cache": False, "pair_seed": 42,
        "artifact_sha256": {"adapter_model": "a" * 64, "recurrent_state": "b" * 64},
        "summary": {"all_pairs_n": 150, "clean_n": 100, "D_specific": 0.5,
                    "scientific_gate_pass": passed, "clean": clean},
    }


def test_finalizer_requires_frozen_answer_c4():
    assert finalize(selection(), answer())["pass"] is True
    assert finalize(selection(), answer(False))["pass"] is False


def test_materialized_answer_c4_config_is_valid(tmp_path: Path):
    source = Path("amlt/feeds_state8_eedi_same_question_answer_c4_20260811.yaml")
    selection_path = tmp_path / "selection.json"
    output = tmp_path / "bound.yaml"
    selection_path.write_text(json.dumps(selection()))
    materialize(source, selection_path, output)
    validate(output)
    config = yaml.safe_load(output.read_text())
    assert config["jobs"][0]["submit_args"]["env"]["SAME_QUESTION_ARM"] == "updater"
    assert config["jobs"][0]["submit_args"]["env"]["SAME_QUESTION_REMOTE_ROOT"].startswith("/mnt/amulet")


def test_materialized_answer_c4_accepts_hash_bound_repair_candidate(tmp_path: Path):
    value = selection()
    value["selected_arm"] = "repair_semantic_binding_seed3201"
    value["selected"].update(
        arm="repair_semantic_binding_seed3201",
        finalfit_variant="same_question_repair_promoted",
        remote_root=(
            "/mnt/amulet/studentmodeling/diagnostics/"
            "state8_stateonly_repair_seed3201_taskanchor_source_anchor_"
            "full_mse1_eedi_same_question_v1_semantic_binding_seed3201"
        ),
    )
    source = Path("amlt/feeds_state8_eedi_same_question_answer_c4_20260811.yaml")
    selection_path = tmp_path / "repair-selection.json"
    output = tmp_path / "repair-bound.yaml"
    selection_path.write_text(json.dumps(value))
    materialize(source, selection_path, output)
    validate(output)
    env = yaml.safe_load(output.read_text())["jobs"][0]["submit_args"]["env"]
    assert env["SAME_QUESTION_ARM"].startswith("repair_")
    assert env["SAME_QUESTION_REMOTE_ROOT"] == value["selected"]["remote_root"]


def test_materialized_answer_c4_accepts_curriculum_snapshot(tmp_path: Path):
    value = selection()
    value["stage"] = "state8_curriculum_checkpoint_joint_selection_v1"
    value["selected_arm"] = "curriculum_current_query_cp3000"
    value["selected"].update(
        arm="curriculum_current_query_cp3000",
        finalfit_variant="same_question_curriculum_promoted",
        remote_root=(
            "/mnt/amulet/studentmodeling/checkpoint_snapshots/"
            "state8_diagnosis_first_eedi_current_query_curriculum_20260812/"
            "checkpoint-3000"
        ),
    )
    source = Path("amlt/feeds_state8_eedi_same_question_answer_c4_20260811.yaml")
    selection_path = tmp_path / "curriculum-selection.json"
    output = tmp_path / "curriculum-bound.yaml"
    selection_path.write_text(json.dumps(value))
    materialize(source, selection_path, output)
    validate(output)
    env = yaml.safe_load(output.read_text())["jobs"][0]["submit_args"]["env"]
    assert env["SAME_QUESTION_ARTIFACT_LAYOUT"] == "curriculum_snapshot"


def test_materialized_answer_c4_accepts_repair_checkpoint_snapshot(tmp_path: Path):
    value = selection()
    value["stage"] = "state8_global_joint_diagnostic_selection_v1"
    value["selected_arm"] = "repair_checkpoint_semantic_binding_seed6201_cp500"
    value["selected"].update(
        arm=value["selected_arm"],
        artifact_layout="repair_checkpoint_snapshot",
        finalfit_variant="same_question_repair_checkpoint_promoted",
        remote_root=(
            "/mnt/amulet/studentmodeling/checkpoint_snapshots/"
            "state8_curriculum_repairs_20260812/"
            "seed6201_semantic_binding/checkpoint-500"
        ),
    )
    source = Path("amlt/feeds_state8_eedi_same_question_answer_c4_20260811.yaml")
    selection_path = tmp_path / "repair-checkpoint-selection.json"
    output = tmp_path / "repair-checkpoint-bound.yaml"
    selection_path.write_text(json.dumps(value))
    materialize(source, selection_path, output)
    validate(output)
    env = yaml.safe_load(output.read_text())["jobs"][0]["submit_args"]["env"]
    assert env["SAME_QUESTION_ARTIFACT_LAYOUT"] == "repair_checkpoint_snapshot"


def test_early_label_checkpoint_flows_to_answer_c4_and_finalfit(tmp_path: Path):
    value = selection()
    value["stage"] = "state8_global_joint_diagnostic_selection_v1"
    value["selected_arm"] = "repair_checkpoint_early_label_low_seed8301_cp20"
    value["selected"].update(
        arm=value["selected_arm"],
        artifact_layout="repair_checkpoint_snapshot",
        finalfit_variant="same_question_repair_checkpoint_promoted",
        remote_root=(
            "/mnt/amulet/studentmodeling/checkpoint_snapshots/"
            "state8_curriculum_repairs_20260812/"
            "seed8301_early_label_low/checkpoint-20"
        ),
        repair_seed=8301,
        repair_branch="early_label_low",
        checkpoint_step=20,
        source_checkpoint_step=3000,
        source_history_mode=(
            "five_stage_curriculum_then_strict_state_only_early_label_binding"
        ),
    )
    source = Path("amlt/feeds_state8_eedi_same_question_answer_c4_20260811.yaml")
    selection_path = tmp_path / "early-label-selection.json"
    output = tmp_path / "early-label-answer-c4.yaml"
    selection_path.write_text(json.dumps(value))
    materialize(source, selection_path, output)
    validate(output)
    env = yaml.safe_load(output.read_text())["jobs"][0]["submit_args"]["env"]
    assert env["SAME_QUESTION_ARM"] == value["selected_arm"]
    assert env["SAME_QUESTION_ARTIFACT_LAYOUT"] == "repair_checkpoint_snapshot"


def test_global_selection_flows_through_answer_c4_and_promotion(tmp_path: Path):
    watcher = Path("scripts/watch_state8_global_selection_and_promotion.sh").read_text()
    assert 'touch "$promotion/diagnostic_gate_failed" "$promotion/complete"' in watcher
    assert 'current_query_control_robust "$control_robust/selection.json"' in watcher
    assert 'early_label_binding "$early_label/selection.json"' in watcher
    value = selection()
    value["stage"] = "state8_global_joint_diagnostic_selection_v1"
    value["selected_arm"] = "repair_semantic_binding_seed4201"
    value["selected"].update(
        arm="repair_semantic_binding_seed4201",
        finalfit_variant="same_question_repair_promoted",
        source_selection="proactive_cp4000",
        remote_root=(
            "/mnt/amulet/studentmodeling/diagnostics/"
            "state8_stateonly_repair_seed4201_taskanchor_source_anchor_"
            "full_mse1_eedi_same_question_v1_semantic_binding_seed4201"
        ),
        checks={
            "malrule_diagnosis_C4_pass": True,
            "eedi_same_question_C4_pass": True,
        },
    )
    selection_path = tmp_path / "global-selection.json"
    selection_path.write_text(json.dumps(value))
    output = tmp_path / "global-answer-c4.yaml"
    materialize(
        Path("amlt/feeds_state8_eedi_same_question_answer_c4_20260811.yaml"),
        selection_path,
        output,
    )
    validate(output)
    promoted = finalize(value, answer())
    assert promoted["pass"] is True
    assert promoted["selected"]["source_selection"] == "proactive_cp4000"
