from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from validate_final_fit_result import validate  # noqa: E402


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, dict):
        path.write_text(json.dumps(value, sort_keys=True) + "\n")
    else:
        path.write_bytes(value)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FinalFitResultTest(unittest.TestCase):
    def fixture(self) -> tuple[tempfile.TemporaryDirectory, Path]:
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name) / "experiment/job"
        model = root / "model"
        metrics = root / "metrics"
        write(model / "adapter_model.safetensors", b"adapter")
        write(model / "recurrent_state.pt", b"state")
        write(metrics / "validation_metrics.json", {"n": 300})
        write(metrics / "causal_validation_metrics.json", {"n": 165})
        write(
            metrics / "run_config.json",
            {
                "distributed": True,
                "distributed_world_size": 8,
                "gradient_accumulation_steps": 2,
                "planned_optimizer_steps": 16241,
                "real_train_rows": 259845,
                "full_micro_steps": 259845,
                "causal_repeat_fraction": 0.05,
                "planned_causal_exposures": 12992,
                "causal_weight": 4.0,
                "causal_objective_mode": "symmetric",
                "causal_train_sha256": "edb1bda6cc6aceb4507ac599377284c533014cef34b80e1063e062a74752267e",
                "causal_valid_sha256": "3e6b5fbcab3d232d5b3a65a005fbf9ba5719a8a4f8251d6ea85b861a7c1c3e22",
                "normalize_causal_exposure": False,
                "final_fit_causal_weight": 4.0,
                "final_fit_normalize_causal_exposure": False,
                "final_fit_causal_repeat_fraction": 1726.0 / 12000.0,
                "init_run_dir": "/remote/m4-pilot",
                "effective_init_adapter": "/remote/m4-pilot/final_adapter",
                "decoder_view": "state_plus_current_query_strict_late_fusion",
                "real_data_exposure": "exactly_one_epoch_without_replacement",
                "trainable_projections": True,
                "behavior_kd_weight": 0.5,
            },
        )
        write(metrics / "gradient_audit.json", {"frozen_tensors_with_grad": []})
        write(metrics / "training_metrics.json", {"loss": 1.0})
        write(metrics / "data_audit.json", {"rows": 259845})
        write(
            metrics / "pilot_receipt.json",
            {
                "status": "candidate_complete",
                "arm": "generation_kd",
                "projection_mode": "trainable",
                "mask_protocol": "answer_anchor_visible_v2",
                "run_dir": "/remote/m4-pilot",
                "causal_rescue_branch": "raw4",
                "causal_weight": 4.0,
                "normalize_causal_exposure": False,
                "C4_D_specific": 0.2,
            },
        )
        write(
            metrics / "final_fit_manifest.json",
            {
                "unique_examples": 259845,
                "counts": {"rows": 259845},
                "purpose": "post_selection_final_fit_train_plus_validation",
            },
        )
        files = {
            "adapter_model": model / "adapter_model.safetensors",
            "recurrent_state": model / "recurrent_state.pt",
            "validation_metrics": metrics / "validation_metrics.json",
            "causal_validation_metrics": metrics / "causal_validation_metrics.json",
            "run_config": metrics / "run_config.json",
            "gradient_audit": metrics / "gradient_audit.json",
        }
        write(
            metrics / "completion.json",
            {
                "status": "complete",
                "diagnostic": False,
                "micro_steps": 259845,
                "optimizer_steps": 16241,
                "real_examples_seen": 259845,
                "real_examples_seen_once": True,
                "causal_pairs_seen": 12992,
                "causal_exposure_count_matches_plan": True,
                "all_causal_pairs_covered": True,
                "sha256": {key: digest(path) for key, path in files.items()},
            },
        )
        config_path = next(root.rglob("metrics/run_config.json"))
        config = json.loads(config_path.read_text())
        config["causal_repeat_fraction"] = 1726.0 / 12000.0
        config["planned_causal_exposures"] = 37374
        write(config_path, config)
        completion_path = next(root.rglob("metrics/completion.json"))
        completion = json.loads(completion_path.read_text())
        completion["causal_pairs_seen"] = 37374
        completion["sha256"]["run_config"] = digest(config_path)
        write(completion_path, completion)
        return temporary, Path(temporary.name)

    def test_complete_artifact_passes(self) -> None:
        temporary, root = self.fixture()
        self.addCleanup(temporary.cleanup)
        self.assertTrue(validate(root, "generation_kd", "trainable")["pass"])

    def test_equivalent_sixteen_rank_artifact_passes(self) -> None:
        temporary, root = self.fixture()
        self.addCleanup(temporary.cleanup)
        config_path = next(root.rglob("metrics/run_config.json"))
        config = json.loads(config_path.read_text())
        config["distributed_world_size"] = 16
        config["gradient_accumulation_steps"] = 1
        write(config_path, config)
        completion_path = next(root.rglob("metrics/completion.json"))
        completion = json.loads(completion_path.read_text())
        completion["sha256"]["run_config"] = digest(config_path)
        write(completion_path, completion)
        self.assertTrue(validate(root, "generation_kd", "trainable")["pass"])

    def test_non_equivalent_global_batch_is_rejected(self) -> None:
        temporary, root = self.fixture()
        self.addCleanup(temporary.cleanup)
        config_path = next(root.rglob("metrics/run_config.json"))
        config = json.loads(config_path.read_text())
        config["gradient_accumulation_steps"] = 1
        write(config_path, config)
        completion_path = next(root.rglob("metrics/completion.json"))
        completion = json.loads(completion_path.read_text())
        completion["sha256"]["run_config"] = digest(config_path)
        write(completion_path, completion)
        with self.assertRaisesRegex(ValueError, "distributed_exact_global_batch_16"):
            validate(root, "generation_kd", "trainable")

    def test_exposure_or_hash_failure_is_rejected(self) -> None:
        temporary, root = self.fixture()
        self.addCleanup(temporary.cleanup)
        path = next(root.rglob("metrics/completion.json"))
        payload = json.loads(path.read_text())
        payload["real_examples_seen_once"] = False
        write(path, payload)
        with self.assertRaisesRegex(ValueError, "real_once"):
            validate(root, "generation_kd", "trainable")

    def test_manifest_field_drift_is_rejected(self) -> None:
        temporary, root = self.fixture()
        self.addCleanup(temporary.cleanup)
        path = next(root.rglob("metrics/final_fit_manifest.json"))
        payload = json.loads(path.read_text())
        payload["counts"]["rows"] = 259844
        write(path, payload)
        with self.assertRaisesRegex(ValueError, "manifest_rows"):
            validate(root, "generation_kd", "trainable")

    def test_unregistered_causal_policy_is_rejected(self) -> None:
        temporary, root = self.fixture()
        self.addCleanup(temporary.cleanup)
        config_path = next(root.rglob("metrics/run_config.json"))
        config = json.loads(config_path.read_text())
        config["causal_weight"] = 3.0
        write(config_path, config)
        completion_path = next(root.rglob("metrics/completion.json"))
        completion = json.loads(completion_path.read_text())
        completion["sha256"]["run_config"] = digest(config_path)
        write(completion_path, completion)
        with self.assertRaisesRegex(ValueError, "causal_policy_inherited"):
            validate(root, "generation_kd", "trainable")

    def test_dense_pilot_uses_declared_full_fit_policy(self) -> None:
        temporary, root = self.fixture()
        self.addCleanup(temporary.cleanup)
        receipt_path = next(root.rglob("metrics/pilot_receipt.json"))
        receipt = json.loads(receipt_path.read_text())
        receipt.update(
            {
                "causal_rescue_branch": "repeat100_w1",
                "causal_weight": 1.0,
                "normalize_causal_exposure": False,
                "causal_repeat_fraction": 1.0,
                "final_fit_causal_weight": 1.0,
                "final_fit_normalize_causal_exposure": True,
                "final_fit_causal_repeat_fraction": 0.05,
            }
        )
        write(receipt_path, receipt)
        config_path = next(root.rglob("metrics/run_config.json"))
        config = json.loads(config_path.read_text())
        config["causal_weight"] = 1.0
        config["normalize_causal_exposure"] = True
        config["causal_repeat_fraction"] = 0.05
        config["planned_causal_exposures"] = 12992
        write(config_path, config)
        completion_path = next(root.rglob("metrics/completion.json"))
        completion = json.loads(completion_path.read_text())
        completion["causal_pairs_seen"] = 12992
        completion["sha256"]["run_config"] = digest(config_path)
        write(completion_path, completion)
        self.assertTrue(validate(root, "generation_kd", "trainable")["pass"])

    def test_sum_aligned_pilot_preserves_sequence_preference(self) -> None:
        temporary, root = self.fixture()
        self.addCleanup(temporary.cleanup)
        receipt_path = next(root.rglob("metrics/pilot_receipt.json"))
        receipt = json.loads(receipt_path.read_text())
        receipt.update(
            {
                "causal_rescue_branch": "repeat100_sum_w1",
                "causal_weight": 1.0,
                "normalize_causal_exposure": False,
                "causal_repeat_fraction": 1.0,
                "causal_preference_reduction": "sum",
                "final_fit_causal_weight": 1.0,
                "final_fit_normalize_causal_exposure": True,
                "final_fit_causal_repeat_fraction": 0.05,
                "final_fit_causal_preference_reduction": "sum",
            }
        )
        write(receipt_path, receipt)
        config_path = next(root.rglob("metrics/run_config.json"))
        config = json.loads(config_path.read_text())
        config.update(
            {
                "causal_weight": 1.0,
                "normalize_causal_exposure": True,
                "causal_repeat_fraction": 0.05,
                "causal_preference_reduction": "sum",
                "planned_causal_exposures": 12992,
            }
        )
        write(config_path, config)
        completion_path = next(root.rglob("metrics/completion.json"))
        completion = json.loads(completion_path.read_text())
        completion["causal_pairs_seen"] = 12992
        completion["sha256"]["run_config"] = digest(config_path)
        write(completion_path, completion)
        self.assertTrue(validate(root, "generation_kd", "trainable")["pass"])

        config["causal_preference_reduction"] = "mean"
        write(config_path, config)
        completion["sha256"]["run_config"] = digest(config_path)
        write(completion_path, completion)
        with self.assertRaisesRegex(
            ValueError, "causal_preference_reduction_inherited"
        ):
            validate(root, "generation_kd", "trainable")

    def test_bootstrap_pilot_inherits_symmetric_sequence_sum_policy(self) -> None:
        temporary, root = self.fixture()
        self.addCleanup(temporary.cleanup)
        receipt_path = next(root.rglob("metrics/pilot_receipt.json"))
        receipt = json.loads(receipt_path.read_text())
        receipt.update(
            {
                "causal_rescue_branch": "bootstrap_sum_w1",
                "causal_weight": 1.0,
                "normalize_causal_exposure": False,
                "causal_repeat_fraction": 1.0,
                "causal_preference_reduction": "sum",
                "final_fit_causal_weight": 1.0,
                "final_fit_normalize_causal_exposure": True,
                "final_fit_causal_repeat_fraction": 0.05,
                "final_fit_causal_preference_reduction": "sum",
            }
        )
        write(receipt_path, receipt)
        config_path = next(root.rglob("metrics/run_config.json"))
        config = json.loads(config_path.read_text())
        config.update(
            {
                "causal_weight": 1.0,
                "normalize_causal_exposure": True,
                "causal_repeat_fraction": 0.05,
                "causal_preference_reduction": "sum",
                "planned_causal_exposures": 12992,
            }
        )
        write(config_path, config)
        completion_path = next(root.rglob("metrics/completion.json"))
        completion = json.loads(completion_path.read_text())
        completion["causal_pairs_seen"] = 12992
        completion["sha256"]["run_config"] = digest(config_path)
        write(completion_path, completion)
        self.assertTrue(validate(root, "generation_kd", "trainable")["pass"])

    def test_cf4_pilot_requires_cf4_balanced_data_provenance(self) -> None:
        temporary, root = self.fixture()
        self.addCleanup(temporary.cleanup)
        receipt_path = next(root.rglob("metrics/pilot_receipt.json"))
        receipt = json.loads(receipt_path.read_text())
        receipt.update(
            {
                "causal_rescue_branch": "cf4_donor_directional_w1",
                "causal_data_mode": "cf4_balanced",
                "causal_weight": 1.0,
                "normalize_causal_exposure": False,
                "causal_repeat_fraction": 1.0,
                "causal_preference_reduction": "sum",
                "causal_objective_mode": "donor_directional",
                "final_fit_causal_weight": 1.0,
                "final_fit_normalize_causal_exposure": True,
                "final_fit_causal_repeat_fraction": 0.05,
                "final_fit_causal_preference_reduction": "sum",
                "final_fit_causal_objective_mode": "donor_directional",
            }
        )
        write(receipt_path, receipt)
        config_path = next(root.rglob("metrics/run_config.json"))
        config = json.loads(config_path.read_text())
        config.update(
            {
                "causal_weight": 1.0,
                "normalize_causal_exposure": True,
                "causal_repeat_fraction": 0.05,
                "causal_preference_reduction": "sum",
                "causal_objective_mode": "donor_directional",
                "causal_train_sha256": "30f68f54120e5a9ada0a37b1dba2b6480de0f8eaeac98d0b7ed7c627429f9766",
                "planned_causal_exposures": 12992,
            }
        )
        write(config_path, config)
        completion_path = next(root.rglob("metrics/completion.json"))
        completion = json.loads(completion_path.read_text())
        completion["causal_pairs_seen"] = 12992
        completion["sha256"]["run_config"] = digest(config_path)
        write(completion_path, completion)
        self.assertTrue(validate(root, "generation_kd", "trainable")["pass"])

        config["causal_train_sha256"] = "0" * 64
        write(config_path, config)
        completion["sha256"]["run_config"] = digest(config_path)
        write(completion_path, completion)
        with self.assertRaisesRegex(ValueError, "causal_data_provenance_inherited"):
            validate(root, "generation_kd", "trainable")


if __name__ == "__main__":
    unittest.main()
