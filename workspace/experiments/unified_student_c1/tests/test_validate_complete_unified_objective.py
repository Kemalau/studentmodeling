from __future__ import annotations

import json
from pathlib import Path

from scripts.validate_complete_unified_objective import validate


def write(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value))
    return path


def fixtures(tmp_path: Path):
    hashes = {"adapter_model": "a" * 64, "recurrent_state": "b" * 64}
    selection = {
        "stage": "state8_exact_diagnosis_curriculum_joint_promotion_v1",
        "pass": True,
        "selected": {
            "qualified": True,
            "slug": "exact_diagnosis_curriculum_seed65",
            "finalfit_variant": "strong1_exact_diagnosis_curriculum_seed65",
        },
    }
    final = {
        "stage": "state8_complete_unified_model_final_gate_v1",
        "pass": True,
        "checks": {"all": True, "history_closed": True},
        "performance": {
            key: {
                "pass": True,
                **(
                    {"positive_recall": 0.25}
                    if key == "predict_error_persistence"
                    else {}
                ),
            }
            for key in (
                "predict_correctness",
                "predict_response",
                "predict_prospective_error",
                "predict_error_persistence",
            )
        },
        "T3_semantic": {"pass": True, "support": 1494},
        "T3_exact_match_role": "auxiliary_only_not_used_for_gate",
        "C4": {"replication_gate_pass": True},
        "diagnosis_C4": {"pass": True},
        "artifact_sha256": hashes,
        "final_fit_receipt": {"one_epoch_exact": True, "real_examples": 259845},
    }
    publish = {
        "stage": "state8_canonical_final_checkpoint_publish_v1",
        "status": "complete",
        "selected_replay_variant": "strong1_exact_diagnosis_curriculum_seed65",
        "canonical_run_dir": "/mnt/amulet/studentmodeling/canonical/state8",
        "artifact_sha256": hashes,
    }
    arm = {
        "checks": {"train_support": True, "closed_state": True},
        "non_T3_preservation_pass": True,
        "T3_semantic": {"support": 75, "judge_backend": "trapi"},
        "C4": {"clean_n": 80, "scientific_gate_pass": True},
    }
    c3 = {
        "stage": "C3_exposure_matched_auxiliary_transfer_v1",
        "status": "complete",
        "controlled_bundle_pass": True,
        "arms": {
            "malrule_correctness": arm,
            "malrule_response": arm,
            "malrule_misconception": arm,
        },
        "conclusion": {
            "C3_transfer_hypothesis_supported": False,
            "recipe_promotion_allowed": False,
        },
    }
    return tuple(
        write(tmp_path / name, value)
        for name, value in (
            ("selection.json", selection),
            ("final.json", final),
            ("publish.json", publish),
            ("c3.json", c3),
        )
    )


def test_complete_objective_accepts_negative_but_complete_c3(tmp_path: Path) -> None:
    paths = fixtures(tmp_path)
    c3 = json.loads(paths[3].read_text())
    for arm in c3["arms"].values():
        arm["non_T3_preservation_pass"] = False
    c3["scientific_gate_pass"] = False
    paths[3].write_text(json.dumps(c3))
    result = validate(*paths)
    assert result["pass"] is True
    assert result["C3_transfer_hypothesis_supported"] is False
    assert result["C1_C4_claim_matrix"]["C1"]["pass"] is True
    assert result["C1_C4_claim_matrix"]["C2"]["pass"] is True
    assert result["C1_C4_claim_matrix"]["C2"]["checks"][
        "natural_language_misconception_diagnosis"
    ] is True
    assert result["C1_C4_claim_matrix"]["C3"]["experiment_complete"] is True
    assert result["C1_C4_claim_matrix"]["C3"]["hypothesis_supported"] is False
    assert result["C1_C4_claim_matrix"]["C3"]["claim_status"] == "not_supported"
    assert result["C1_C4_claim_matrix"]["C4"]["pass"] is True
    assert result["C1_C4_claim_matrix"]["C4"]["checks"][
        "predictive_sufficiency"
    ] is True
    assert all(result["checks"].values())


def test_c2_requires_nonzero_strict_persistence_recall(tmp_path: Path) -> None:
    paths = fixtures(tmp_path)
    final = json.loads(paths[1].read_text())
    final["performance"]["predict_error_persistence"]["positive_recall"] = 0.0
    paths[1].write_text(json.dumps(final))
    result = validate(*paths)
    assert result["C1_C4_claim_matrix"]["C2"]["pass"] is False
    assert result["pass"] is False
    assert result["checks"]["C2_claim"] is False
    assert result["checks"]["T1_T2_T4_T5_retention"] is True
    assert result["task_taxonomy"]["legacy_to_public"]["T4"] == "T6"


def test_artifact_mismatch_fails(tmp_path: Path) -> None:
    paths = fixtures(tmp_path)
    value = json.loads(paths[2].read_text())
    value["artifact_sha256"]["adapter_model"] = "c" * 64
    paths[2].write_text(json.dumps(value))
    result = validate(*paths)
    assert result["pass"] is False
    assert result["checks"]["artifact_binding"] is False


def test_watcher_requires_both_model_and_c3_terminal_evidence() -> None:
    source = Path(
        "/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1/"
        "scripts/watch_complete_unified_objective.sh"
    ).read_text()
    for fragment in (
        '"$final_eval_dir/complete"',
        '"$c3/complete"',
        "canonical_publish_receipt.json",
        "validate_complete_unified_objective.py",
        "scientific_gate_pass",
        "OBJECTIVE_FINAL_EVAL_DIR",
        "OBJECTIVE_FINAL_FIT_DIR",
        "OBJECTIVE_SELECTION",
    ):
        assert fragment in source


def test_same_question_objective_requires_confirmatory_identical_query_c4(
    tmp_path: Path,
) -> None:
    paths = fixtures(tmp_path)
    selection = json.loads(paths[0].read_text())
    selection.update(
        stage="state8_eedi_same_question_joint_promotion_v1",
        selected={
            "qualified": True,
            "arm": "updater",
            "finalfit_variant": "same_question_updater_mse1_seed3072",
        },
    )
    paths[0].write_text(json.dumps(selection))
    publish = json.loads(paths[2].read_text())
    publish["selected_replay_variant"] = "same_question_updater_mse1_seed3072"
    paths[2].write_text(json.dumps(publish))
    final = json.loads(paths[1].read_text())
    final["eedi_same_question_C4"] = {
        "pass": True,
        "confirmatory": True,
        "generation": {
            "same_question_query_identical": True,
            "student_response_visible_to_decoder": False,
        },
    }
    paths[1].write_text(json.dumps(final))
    value = validate(*paths)
    assert value["checks"]["C4_answer_and_diagnosis"] is True
    assert value["C1_C4_claim_matrix"]["C4"]["checks"][
        "identical_query_eedi_swap"
    ] is True
    assert value["pass"] is True


def test_same_question_objective_rejects_missing_identical_query_c4(
    tmp_path: Path,
) -> None:
    paths = fixtures(tmp_path)
    selection = json.loads(paths[0].read_text())
    selection.update(
        stage="state8_eedi_same_question_joint_promotion_v1",
        selected={
            "qualified": True,
            "arm": "full",
            "finalfit_variant": "same_question_full_mse1_seed3071",
        },
    )
    paths[0].write_text(json.dumps(selection))
    publish = json.loads(paths[2].read_text())
    publish["selected_replay_variant"] = "same_question_full_mse1_seed3071"
    paths[2].write_text(json.dumps(publish))
    value = validate(*paths)
    assert value["checks"]["C4_answer_and_diagnosis"] is False
    assert value["pass"] is False
