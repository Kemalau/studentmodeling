import json
from pathlib import Path

import yaml

from scripts.materialize_state8_repair_dual_c4_quickscreen import materialize
from scripts.validate_state8_repair_dual_c4_quickscreen_amlt import validate
from scripts.summarize_state8_repair_dual_c4_quickscreen import summarize


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "amlt/feeds_state8_repair_dual_c4_quickscreen_20260813.yaml"


def test_materializes_hash_bound_two_job_quickscreen(tmp_path: Path) -> None:
    receipts = tmp_path / "receipts"
    receipts.mkdir()
    for seed, branch in ((8201, "causal_teacher_low"), (8202, "causal_teacher_high")):
        (receipts / f"seed{seed}.{branch}.checkpoint-20.completion.json").write_text(
            json.dumps({
                "status": "complete", "step": 20,
                "sha256": {"adapter_model": "a" * 64, "recurrent_state": "b" * 64},
            })
        )
    output = tmp_path / "bound.yaml"
    materialize(TEMPLATE, receipts, output)
    validate(output)
    config = yaml.safe_load(output.read_text())
    assert len(config["jobs"]) == 2
    assert {job["submit_args"]["env"]["REPAIR_SEED"] for job in config["jobs"]} == {"8201", "8202"}
    assert all(job["sku"] == "80G1-A100" for job in config["jobs"])


def test_quickscreen_is_dual_semantic_c4_and_read_only() -> None:
    text = (ROOT / "scripts/run_state8_repair_dual_c4_quickscreen.sh").read_text()
    assert "evaluate_recurrent_diagnosis_c4_diagnosis_first.py" in text
    assert "evaluate_state8_eedi_same_question_c4.py" in text
    assert text.count("build_pairwise_diagnosis_c4_judge_panel.py") == 2
    assert "audit_state8_curriculum_repair_checkpoint.py" in text
    assert "optimizer.step" not in text
    assert "--init-adapter" not in text


def test_joint_summary_keeps_gold_and_pairwise_semantics_separate() -> None:
    receipt = {
        "stage": "state8_repair_dual_c4_quickscreen_generation_v1",
        "status": "complete", "diagnostic_only": True,
        "repair_seed": 8201, "repair_branch": "causal_teacher_low",
        "checkpoint_step": 20, "raw_history_visible_to_decoder": False,
        "artifact_sha256": {"adapter_model": "a" * 64, "recurrent_state": "b" * 64},
    }
    def gold():
        cells = {
            f"{condition}_to_{role}": {"semantic_equivalence_rate": 0.8 if role == "recipient" else 0.2}
            for condition in ("native", "noop", "samemal", "random", "swap")
            for role in ("donor", "recipient")
        }
        cells["swap_to_donor"]["semantic_equivalence_rate"] = 0.7
        return {
            "semantic_judge_authoritative": True,
            "exact_label_role": "auxiliary_only_not_used_for_gate",
            "generation": {"artifact_sha256": receipt["artifact_sha256"]},
            "pass": True, "semantic_D_specific": 0.5,
            "random_matches_either_stable_target_rate": 0.1,
            "noop_recipient_retention": 0.8, "same_rule_recipient_retention": 0.8,
            "cells": cells,
        }
    pairwise = {
        "stage": "state8_diagnosis_c4_pairwise_summary_v1",
        "semantic_judge_authoritative": True,
        "noop_native_output_invariance": 1.0,
        "same_rule_native_output_invariance": 0.9,
        "random_native_output_invariance": 0.1,
        "swap_native_output_invariance": 0.2,
        "same_rule_vs_swap_separation": 0.7,
    }
    result = summarize(receipt, gold(), gold(), pairwise, pairwise)
    assert result["dual_c4_pass"] is True
    assert result["eedi_same_question"]["swap_to_donor"] == 0.7
    assert result["eedi_same_question"]["noop_native_output_invariance"] == 1.0
