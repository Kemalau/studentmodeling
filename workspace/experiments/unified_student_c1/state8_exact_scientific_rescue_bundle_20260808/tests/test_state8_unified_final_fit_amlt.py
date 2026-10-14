from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from validate_state8_unified_final_fit_amlt import validate  # noqa: E402


class State8UnifiedFinalFitAmltTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = yaml.safe_load(
            (ROOT / "amlt/feeds_state8_unified_final_fit_20260805.yaml").read_text()
        )

    def test_frozen_config_passes(self) -> None:
        validate(copy.deepcopy(self.config))
        self.assertEqual(len(self.config["jobs"]), 13)

    def test_promoted_config_includes_task_anchor_variants(self) -> None:
        config = yaml.safe_load(
            (ROOT / "amlt/feeds_state8_highcausal_final_fit_20260806.yaml").read_text()
        )
        validate(config)
        names = {job["name"] for job in config["jobs"]}
        self.assertIn("state8-task-anchor-base-unified-finalfit-seed42", names)
        self.assertIn("state8-task-anchor-c4strong-unified-finalfit-seed42", names)

    def test_strong1_widen8_finalfit_variants_are_mutually_selectable(self) -> None:
        config = yaml.safe_load(
            (ROOT / "amlt/feeds_state8_strong1_widen8_final_fit_20260808.yaml").read_text()
        )
        self.assertEqual(len(config["jobs"]), 7)
        validate(config)
        self.assertTrue(
            all(job["submit_args"]["max_run_duration_seconds"] == 43200 for job in config["jobs"])
        )
        self.assertTrue(all(job["sku"] == "80G4-A100" for job in config["jobs"]))
        self.assertTrue(
            all(
                "run_state8_unified_final_fit_world4.sh" in str(job["command"])
                for job in config["jobs"]
            )
        )
        launcher = (ROOT / "scripts/run_state8_unified_final_fit.sh").read_text()
        for fragment in (
            "strong1_widen8_lr5e7_seed58|strong1_widen8_lr1e6_seed59|strong1_widen8_single_lr5e7_seed60",
            "state8_strong1_widen8_joint_promotion_v1",
            "strong1_rescue_taxonomy_joint_seed61|strong1_rescue_taskanchor_c4strong_seed62",
            "state8_strong1_rescue_joint_promotion_v1",
            "strong1_midpoint_diagnosis_first_seed63",
            "state8_midpoint_diagnosis_first_joint_promotion_v1",
            "strong1_exact_diagnosis_curriculum_seed65",
            "state8_exact_diagnosis_curriculum_joint_promotion_v1",
            "EXPECTED_HISTORY_MODE=strict_state_only",
            "STATE8_CANONICAL_RUN_DIR",
        ):
            self.assertIn(fragment, launcher)

    def test_rejects_larger_world_size(self) -> None:
        value = copy.deepcopy(self.config)
        value["jobs"][0]["sku"] = "4x80G8-A100-IB-NvLink"
        with self.assertRaisesRegex(ValueError, "Topology drift"):
            validate(value)

    def test_filtered_highkd_job_passes(self) -> None:
        value = copy.deepcopy(self.config)
        value["jobs"] = [job for job in value["jobs"] if "highkd" in job["name"]]
        validate(value)

    def test_filtered_strict_repair_job_passes(self) -> None:
        value = copy.deepcopy(self.config)
        value["jobs"] = [
            job for job in value["jobs"] if "repair-logits" in job["name"]
        ]
        validate(value)

    def test_filtered_semantic_eedi_job_passes(self) -> None:
        value = copy.deepcopy(self.config)
        value["jobs"] = [
            job for job in value["jobs"] if "semantic-eedi-g010" in job["name"]
        ]
        validate(value)

    def test_filtered_single_seed46_semantic_eedi_job_passes(self) -> None:
        value = copy.deepcopy(self.config)
        value["jobs"] = [
            job
            for job in value["jobs"]
            if "semantic-eedi-g025-single-seed46" in job["name"]
        ]
        validate(value)

    def test_rejects_unapproved_identity(self) -> None:
        value = copy.deepcopy(self.config)
        value["jobs"][0]["submit_args"]["env"]["_AZUREML_SINGULARITY_JOB_UAI"] = "other"
        with self.assertRaisesRegex(ValueError, "Environment/UAI/variant drift"):
            validate(value)

    def test_full_fit_uses_preemption_resilient_checkpoint_cadence(self) -> None:
        launcher = (ROOT / "scripts/run_state8_unified_final_fit.sh").read_text()
        self.assertIn("--save-steps 50 --recovery-save-steps 25", launcher)
        self.assertIn('--expected-planned-optimizer-steps "$TOTAL_ROUNDS"', launcher)
        self.assertIn("--save-steps 50 --checkpoint-retention 2", launcher)
        self.assertIn("state8_qualified_unified_full259845_seed42_${FINALFIT_VARIANT}_v1_20260805", launcher)

    def test_full_fit_offloads_every_direct_teacher_forward(self) -> None:
        launcher = (ROOT / "scripts/run_state8_unified_final_fit.sh").read_text()
        self.assertIn("--direct-offload-min-tokens 1", launcher)
        self.assertIn("config['direct_offload_min_tokens']==1", launcher)

    def test_full_fit_atomically_publishes_the_evaluator_canonical_path(self) -> None:
        producer = (ROOT / "scripts/run_state8_unified_final_fit.sh").read_text()
        formal = (ROOT / "scripts/run_state8_final_formal_shard.sh").read_text()
        causal = (ROOT / "scripts/run_state8_final_c4.sh").read_text()
        canonical = "state8_qualified_unified_full259845_seed42_v1_20260805"
        self.assertIn(f"CANONICAL_RUN_DIR=/mnt/amulet/studentmodeling/checkpoints/{canonical}", producer)
        self.assertIn('--destination "$CANONICAL_RUN_DIR"', producer)
        self.assertIn(f"RUN_DIR=/mnt/amulet/studentmodeling/checkpoints/{canonical}", formal)
        self.assertIn(f"RUN_DIR=/mnt/amulet/studentmodeling/checkpoints/{canonical}", causal)
        self.assertIn("canonical_publish_receipt.json", formal)
        self.assertIn("canonical_publish_receipt.json", causal)
        self.assertIn("STATE8_FINAL_EVAL_RUN_DIR", formal)
        self.assertIn("STATE8_FINAL_EVAL_RUN_DIR", causal)

    def test_publisher_accepts_and_binds_all_promoted_strict_variants(self) -> None:
        publisher = (ROOT / "scripts/publish_state8_final_checkpoint.py").read_text()
        for fragment in (
            "taxonomy_curriculum_t3heavy",
            "taxonomy_curriculum_balanced",
            "taxonomy_curriculum_t3heavy_diag",
            "taxonomy_curriculum_balanced_diag",
            "task_anchor_base",
            "task_anchor_c4strong",
            "taxonomy_causal_repair.json",
            "task_anchored_causal_repair.json",
        ):
            self.assertIn(fragment, publisher)

    def test_full_fit_trains_answer_and_natural_diagnosis_causal_twins(self) -> None:
        launcher = (ROOT / "scripts/run_state8_unified_final_fit.sh").read_text()
        self.assertIn("augment_recurrent_causal_diagnosis.py", launcher)
        self.assertIn("causal_answer_diagnosis_augmentation.json", launcher)
        self.assertIn("diagnosis_natural_schema", launcher)
        self.assertIn("canonical_label_auxiliary_supervision", launcher)
        self.assertIn("STATE8_CAUSAL_DIAGNOSIS_ONLY=0", launcher)
        self.assertIn("causal_target_kind_counts", launcher)
        self.assertIn("'prospective_answer':2102", launcher)
        self.assertIn("'diagnosis_natural_schema':2102", launcher)

    def test_repair_final_fit_keeps_teacher_on_with_strict_state_only_decoder(self) -> None:
        launcher = (ROOT / "scripts/run_state8_unified_final_fit.sh").read_text()
        self.assertIn("EXPECTED_HISTORY_MODE=strict_state_only", launcher)
        self.assertIn("train_state8_stateonly_teacher_repair.py", launcher)
        self.assertIn("STATE8_REPAIR_LABEL_WEIGHT=4.0", launcher)
        self.assertIn("STATE8_REPAIR_TRAJECTORY_WEIGHT=0.1", launcher)
        self.assertIn("assert counts=={1.0:259845}", launcher)
        self.assertIn("online_kd_phase_weight", launcher)

    def test_semantic_final_fit_preserves_c4_and_eedi_geometry(self) -> None:
        launcher = (ROOT / "scripts/run_state8_unified_final_fit.sh").read_text()
        self.assertIn("train_state8_semantic_causal_repair.py", launcher)
        self.assertIn("STATE8_RANDOM_NEUTRAL=1", launcher)
        self.assertIn("STATE8_EEDI_GEOMETRY_WEIGHT=0.10", launcher)
        self.assertIn("augment_state8_eedi_semantic_pairs.py", launcher)
        self.assertIn("canonical_labels_visible_to_decoder", launcher)
        self.assertIn("semantic_random_target", launcher)

    def test_highcausal_eedi_can_pass_the_promotion_binding(self) -> None:
        launcher = (ROOT / "scripts/run_state8_unified_final_fit.sh").read_text()
        self.assertIn("'highcausal_eedi':'eedi'", launcher)

    def test_highcausal_taxonomy_can_pass_the_promotion_binding(self) -> None:
        launcher = (ROOT / "scripts/run_state8_unified_final_fit.sh").read_text()
        self.assertIn("'highcausal_taxonomy':'taxonomy'", launcher)
        self.assertIn("train_state8_taxonomy_causal_repair.py", launcher)
        self.assertIn(
            '|| "$FINALFIT_VARIANT" == highcausal_taxonomy', launcher
        )
        self.assertIn('export FINAL_TRAIN_JSONL="$EEDI_PAIR_JSONL"', launcher)

    def test_final_fit_binds_the_current_atomic_publisher(self) -> None:
        import hashlib

        launcher = (ROOT / "scripts/run_state8_unified_final_fit.sh").read_text()
        publisher = ROOT / "scripts/publish_state8_final_checkpoint.py"
        digest = hashlib.sha256(publisher.read_bytes()).hexdigest()
        self.assertIn(digest, launcher)

    def test_taxonomy_final_fit_binds_selected_artifact_hashes(self) -> None:
        launcher = (ROOT / "scripts/run_state8_unified_final_fit.sh").read_text()
        self.assertIn("x['selected']['artifact_sha256']==observed", launcher)
        self.assertIn("INITIAL_ADAPTER_SHA256", launcher)
        self.assertIn("INITIAL_RECURRENT_STATE_SHA256", launcher)

    def test_task_anchor_final_fit_is_a_bound_strict_state_only_variant(self) -> None:
        config = yaml.safe_load(
            (ROOT / "amlt/feeds_state8_highcausal_final_fit_20260806.yaml").read_text()
        )
        config["jobs"] = [
            job for job in config["jobs"] if "task-anchor" in job["name"]
        ]
        self.assertEqual(len(config["jobs"]), 2)
        validate(config)
        launcher = (ROOT / "scripts/run_state8_unified_final_fit.sh").read_text()
        for fragment in (
            "task_anchor_base|task_anchor_c4strong",
            "state8_unified_candidate_selection_v1",
            "train_state8_task_anchored_causal_repair.py",
            "task_anchored_causal_repair.json",
        ):
            self.assertIn(fragment, launcher)

    def test_controller_verifies_the_downloaded_canonical_receipt(self) -> None:
        controller = (ROOT / "scripts/watch_state8_unified_final_fit.sh").read_text()
        self.assertIn("find \"$state_dir/download\" -name canonical_publish_receipt.json", controller)
        self.assertIn("state8_canonical_final_checkpoint_publish_v1", controller)
        self.assertIn("p['artifact_sha256']=={'adapter_model':x['adapter_sha256']", controller)


if __name__ == "__main__":
    unittest.main()
