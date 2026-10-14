from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from validate_layerwise_promoted_unified_pipeline import validate  # noqa: E402


class PromotedOverallAuditTest(unittest.TestCase):
    def fixture(self, root: Path) -> None:
        fit = root / "task_retention_promoted_final_fit"
        evaluation = root / "task_retention_promoted_final_eval"
        semantic = root / "task_retention_promoted_t3_semantic_judge"
        for directory in (fit, evaluation, semantic):
            directory.mkdir(parents=True)
            (directory / "complete").touch()
            (directory / "scientific_gate_pass").touch()
        metrics = fit / "download/job/metrics"
        model = fit / "download/job/model"
        metrics.mkdir(parents=True)
        model.mkdir(parents=True)
        receipt = {
            "status": "final_fit_complete",
            "stage": "layerwise_task_retention_promoted_final_fit",
            "branch": "equal_t3_t6",
            "loss_profile": "standard",
            "candidate_id": "equal_t3_t6__standard",
            "learning_rate": 5e-7,
            "causal_weight": 2.0,
            "promoted_source_round": 750,
            "promoted_order_sha256": "a" * 64,
            "one_epoch_exact": True,
        }
        (metrics / "promoted_final_fit_receipt.json").write_text(json.dumps(receipt))
        (metrics / "run_config.json").write_text(json.dumps({
            "promoted_real_order_applied": True,
            "promoted_real_order_source_round": 750,
            "promoted_real_order_manifest_sha256": "b" * 64,
            "promoted_selected_causal_weight": 2.0,
            "causal_weight": 2.0,
        }))
        (metrics / "promoted_real_order_manifest.json").write_text(json.dumps({
            "stage": "layerwise_task_retention_promoted_order_v1",
            "branch": "equal_t3_t6",
            "selected_loss_profile": "standard",
            "selected_candidate_id": "equal_t3_t6__standard",
            "selected_learning_rate": 5e-7,
            "selected_causal_weight": 2.0,
            "source_checkpoint_round": 750,
            "source_world_size": 16,
            "completed_prefix_rows": 12000,
            "epoch_rows": 259845,
            "promoted_rows": 1000,
            "one_epoch_permutation": True,
            "completed_prefix_unchanged": True,
            "formal_example_id_overlap": 0,
            "real_order_sha256": "a" * 64,
        }))
        (metrics / "source_checkpoint_audit.json").write_text(json.dumps({
            "status": "passed",
            "exact_resume_ready": True,
            "checkpoint_round": 750,
            "world_size": 16,
            "history_rows": 12000,
        }))
        (model / "adapter_model.safetensors").write_bytes(b"adapter")
        (model / "recurrent_state.pt").write_bytes(b"state")
        (fit / "artifact_manifest.json").write_text("{}")
        (evaluation / "summary.json").write_text("{}")
        (semantic / "t3_semantic_gate.json").write_text("{}")

    def test_valid_promotion_reuses_strict_base_audit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.fixture(root)
            with patch(
                "validate_layerwise_promoted_unified_pipeline.validate_base",
                return_value={"checks": {"strict_base": True}},
            ) as base:
                result = validate(root)
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["selected_branch"], "equal_t3_t6")
            self.assertEqual(result["selected_loss_profile"], "standard")
            self.assertEqual(result["selected_candidate"], "equal_t3_t6__standard")
            self.assertTrue(all(result["promoted_checks"].values()))
            base.assert_called_once()

    def test_promotion_order_drift_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.fixture(root)
            path = next(root.rglob("promoted_real_order_manifest.json"))
            value = json.loads(path.read_text())
            value["completed_prefix_unchanged"] = False
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "order"):
                validate(root)

    def test_valid_eedi_full_state_kd_promotion_uses_actual_loss_profile(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.fixture(root)
            receipt_path = next(root.rglob("promoted_final_fit_receipt.json"))
            receipt = json.loads(receipt_path.read_text())
            receipt.update({
                "branch": "eedi_full",
                "loss_profile": "state_kd",
                "candidate_id": "eedi_full__state_kd__lr2e-06",
                "learning_rate": 2e-6,
            })
            receipt_path.write_text(json.dumps(receipt))
            promotion_path = next(root.rglob("promoted_real_order_manifest.json"))
            promotion = json.loads(promotion_path.read_text())
            promotion.update({
                "branch": "eedi_full",
                "selected_loss_profile": "state_kd",
                "selected_candidate_id": "eedi_full__state_kd__lr2e-06",
                "selected_learning_rate": 2e-6,
                "promoted_rows": 5000,
            })
            promotion_path.write_text(json.dumps(promotion))
            with patch(
                "validate_layerwise_promoted_unified_pipeline.validate_base",
                return_value={"checks": {"strict_base": True}},
            ):
                result = validate(root)
            self.assertEqual(result["selected_branch"], "eedi_full")
            self.assertEqual(result["selected_loss_profile"], "state_kd")
            self.assertEqual(
                result["selected_candidate"], "eedi_full__state_kd__lr2e-06"
            )

    def test_open_t3_promotion_requires_its_train_view_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.fixture(root)
            receipt_path = next(root.rglob("promoted_final_fit_receipt.json"))
            receipt = json.loads(receipt_path.read_text())
            receipt.update({
                "branch": "eedi_full_open_t3",
                "loss_profile": "state_path",
                "candidate_id": "eedi_full_open_t3__state_path__lr2e-06",
                "learning_rate": 2e-6,
            })
            receipt_path.write_text(json.dumps(receipt))
            promotion_path = next(root.rglob("promoted_real_order_manifest.json"))
            promotion = json.loads(promotion_path.read_text())
            promotion.update({
                "branch": "eedi_full_open_t3",
                "selected_loss_profile": "state_path",
                "selected_candidate_id": "eedi_full_open_t3__state_path__lr2e-06",
                "selected_learning_rate": 2e-6,
                "promoted_rows": 5000,
            })
            promotion_path.write_text(json.dumps(promotion))
            open_view = promotion_path.parent / "promoted_open_t3_training_view.json"
            open_view.write_text(json.dumps({
                "stage": "promoted_open_t3_training_view_v1",
                "scope": "unseen_exact_epoch_suffix_only",
                "behavior_kd_weight": 0.0,
                "direct_replay_weight": 0.0,
                "causal_rows_modified": False,
                "validation_rows_modified": False,
                "changed_malrule_t3_rows": 1,
            }))
            with patch(
                "validate_layerwise_promoted_unified_pipeline.validate_base",
                return_value={"checks": {"strict_base": True}},
            ):
                result = validate(root)
            self.assertTrue(result["promoted_checks"]["open_t3_training_view"])
            open_view.unlink()
            with self.assertRaisesRegex(ValueError, "promoted_open_t3_training_view"):
                validate(root)

    def test_equal_open_t3_state_kd_promotion_requires_selective_kd_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.fixture(root)
            receipt_path = next(root.rglob("promoted_final_fit_receipt.json"))
            receipt = json.loads(receipt_path.read_text())
            receipt.update({
                "branch": "equal_t3_t6_open_t3",
                "loss_profile": "state_kd",
                "candidate_id": "equal_t3_t6_open_t3__state_kd",
                "learning_rate": 5e-7,
            })
            receipt_path.write_text(json.dumps(receipt))
            promotion_path = next(root.rglob("promoted_real_order_manifest.json"))
            promotion = json.loads(promotion_path.read_text())
            promotion.update({
                "branch": "equal_t3_t6_open_t3",
                "selected_loss_profile": "state_kd",
                "selected_candidate_id": "equal_t3_t6_open_t3__state_kd",
                "selected_learning_rate": 5e-7,
                "promoted_rows": 1000,
            })
            promotion_path.write_text(json.dumps(promotion))
            open_view = promotion_path.parent / "promoted_open_t3_selective_kd_training_view.json"
            open_view.write_text(json.dumps({
                "stage": "promoted_open_t3_selective_kd_training_view_v1",
                "scope": "unseen_exact_epoch_suffix_only",
                "global_behavior_kd_weight": 0.5,
                "excluded_behavior_kd_weight": 0.0,
                "direct_replay_weight": 0.0,
                "excluded_history_rows": 1,
                "retained_history_rows": 1,
                "causal_rows_modified": False,
                "validation_rows_modified": False,
            }))
            with patch(
                "validate_layerwise_promoted_unified_pipeline.validate_base",
                return_value={"checks": {"strict_base": True}},
            ):
                result = validate(root)
            self.assertTrue(result["promoted_checks"]["open_t3_training_view"])

    def test_equal_open_t3_state_kd_lr_promotion_keeps_equal_manifest_size(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.fixture(root)
            receipt_path = next(root.rglob("promoted_final_fit_receipt.json"))
            receipt = json.loads(receipt_path.read_text())
            receipt.update({
                "branch": "equal_t3_t6_open_t3",
                "loss_profile": "state_kd",
                "candidate_id": "equal_t3_t6_open_t3__state_kd__lr2e-06",
                "learning_rate": 2e-6,
            })
            receipt_path.write_text(json.dumps(receipt))
            promotion_path = next(root.rglob("promoted_real_order_manifest.json"))
            promotion = json.loads(promotion_path.read_text())
            promotion.update({
                "branch": "equal_t3_t6_open_t3",
                "selected_loss_profile": "state_kd",
                "selected_candidate_id": (
                    "equal_t3_t6_open_t3__state_kd__lr2e-06"
                ),
                "selected_learning_rate": 2e-6,
                "promoted_rows": 1000,
            })
            promotion_path.write_text(json.dumps(promotion))
            open_view = (
                promotion_path.parent
                / "promoted_open_t3_selective_kd_training_view.json"
            )
            open_view.write_text(json.dumps({
                "stage": "promoted_open_t3_selective_kd_training_view_v1",
                "scope": "unseen_exact_epoch_suffix_only",
                "global_behavior_kd_weight": 0.5,
                "excluded_behavior_kd_weight": 0.0,
                "direct_replay_weight": 0.0,
                "excluded_history_rows": 1,
                "retained_history_rows": 1,
                "causal_rows_modified": False,
                "validation_rows_modified": False,
            }))
            with patch(
                "validate_layerwise_promoted_unified_pipeline.validate_base",
                return_value={"checks": {"strict_base": True}},
            ):
                result = validate(root)
            self.assertEqual(
                result["selected_candidate"],
                "equal_t3_t6_open_t3__state_kd__lr2e-06",
            )

    def test_equal_open_t3_state_kd_lr1e6_promotion_is_audited(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.fixture(root)
            receipt_path = next(root.rglob("promoted_final_fit_receipt.json"))
            receipt = json.loads(receipt_path.read_text())
            receipt.update({
                "branch": "equal_t3_t6_open_t3",
                "loss_profile": "state_kd",
                "candidate_id": "equal_t3_t6_open_t3__state_kd__lr1e-06",
                "learning_rate": 1e-6,
            })
            receipt_path.write_text(json.dumps(receipt))
            promotion_path = next(root.rglob("promoted_real_order_manifest.json"))
            promotion = json.loads(promotion_path.read_text())
            promotion.update({
                "branch": "equal_t3_t6_open_t3",
                "selected_loss_profile": "state_kd",
                "selected_candidate_id": (
                    "equal_t3_t6_open_t3__state_kd__lr1e-06"
                ),
                "selected_learning_rate": 1e-6,
                "promoted_rows": 1000,
            })
            promotion_path.write_text(json.dumps(promotion))
            open_view = (
                promotion_path.parent
                / "promoted_open_t3_selective_kd_training_view.json"
            )
            open_view.write_text(json.dumps({
                "stage": "promoted_open_t3_selective_kd_training_view_v1",
                "scope": "unseen_exact_epoch_suffix_only",
                "global_behavior_kd_weight": 0.5,
                "excluded_behavior_kd_weight": 0.0,
                "direct_replay_weight": 0.0,
                "excluded_history_rows": 1,
                "retained_history_rows": 1,
                "causal_rows_modified": False,
                "validation_rows_modified": False,
            }))
            with patch(
                "validate_layerwise_promoted_unified_pipeline.validate_base",
                return_value={"checks": {"strict_base": True}},
            ):
                result = validate(root)
            self.assertEqual(
                result["selected_candidate"],
                "equal_t3_t6_open_t3__state_kd__lr1e-06",
            )

    def test_two_stage_state_path_requires_transition_and_selective_view(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.fixture(root)
            receipt_path = next(root.rglob("promoted_final_fit_receipt.json"))
            receipt = json.loads(receipt_path.read_text())
            receipt.update({
                "branch": "equal_t3_t6_open_t3_two_stage",
                "loss_profile": "state_path",
                "candidate_id": "equal_t3_t6_open_t3_two_stage__state_path",
                "learning_rate": 5e-7,
                "two_stage_curriculum": True,
            })
            receipt_path.write_text(json.dumps(receipt))
            promotion_path = next(root.rglob("promoted_real_order_manifest.json"))
            promotion = json.loads(promotion_path.read_text())
            promotion.update({
                "branch": "equal_t3_t6_open_t3_two_stage",
                "selected_loss_profile": "state_path",
                "selected_candidate_id": (
                    "equal_t3_t6_open_t3_two_stage__state_path"
                ),
                "selected_learning_rate": 5e-7,
                "promoted_rows": 2000,
                "curriculum_mode": (
                    "open_t3_formation_then_causal_task_consolidation"
                ),
                "formation_transition_global_position": 13008,
            })
            promotion_path.write_text(json.dumps(promotion))
            config_path = next(root.rglob("run_config.json"))
            config = json.loads(config_path.read_text())
            config.update({
                "promoted_two_stage_phase": "consolidation",
                "promoted_two_stage_transition_applied": True,
            })
            config_path.write_text(json.dumps(config))
            metrics = promotion_path.parent
            (metrics / "promoted_open_t3_selective_kd_training_view.json").write_text(
                json.dumps({
                    "stage": "promoted_open_t3_selective_kd_training_view_v1",
                    "scope": "unseen_exact_epoch_suffix_only",
                    "global_behavior_kd_weight": 0.5,
                    "excluded_behavior_kd_weight": 0.0,
                    "direct_replay_weight": 0.0,
                    "excluded_history_rows": 1,
                    "retained_history_rows": 1,
                    "causal_rows_modified": False,
                    "validation_rows_modified": False,
                })
            )
            transition = metrics / "two_stage_transition_receipt.json"
            transition.write_text(json.dumps({
                "stage": "promoted_two_stage_transition_v1",
                "global_position": 13008,
                "optimizer_moments_preserved": True,
            }))
            with patch(
                "validate_layerwise_promoted_unified_pipeline.validate_base",
                return_value={"checks": {"strict_base": True}},
            ):
                result = validate(root)
            self.assertTrue(result["promoted_checks"]["two_stage_curriculum"])
            self.assertTrue(result["promoted_checks"]["open_t3_training_view"])
            transition.unlink()
            with self.assertRaisesRegex(ValueError, "two_stage_transition_receipt"):
                validate(root)


if __name__ == "__main__":
    unittest.main()
