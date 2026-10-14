from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from validate_layerwise_unified_final_fit_amlt import load, validate  # noqa: E402


class LayerwiseUnifiedFinalFitAmltTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load(ROOT / "amlt/feeds_layerwise_unified_final_fit_20260731.yaml")

    def test_config_passes(self) -> None:
        validate(self.config)

    def test_frozen_training_runtime_is_required(self) -> None:
        value = copy.deepcopy(self.config)
        value["jobs"][0]["command"] = value["jobs"][0]["command"].replace(
            "cp scripts/frozen_layerwise_finalfit/layerwise_recurrent_student_model.py scripts/layerwise_recurrent_student_model.py",
            ":",
        )
        with self.assertRaisesRegex(ValueError, "command drift"):
            validate(value)

    def test_scalar_state_is_rejected(self) -> None:
        value = copy.deepcopy(self.config)
        value["jobs"][0]["command"] += "\n--init-recurrent-state x.pt\n"
        with self.assertRaisesRegex(ValueError, "scalar"):
            validate(value)

    def test_single_node_drift_is_rejected(self) -> None:
        value = copy.deepcopy(self.config)
        value["jobs"][0]["sku"] = "80G8-A100-IB-NvLink"
        with self.assertRaisesRegex(ValueError, "GPU"):
            validate(value)

    def test_duration_above_vc_policy_is_rejected(self) -> None:
        value = copy.deepcopy(self.config)
        value["jobs"][0]["submit_args"]["max_run_duration_seconds"] = 604800
        with self.assertRaisesRegex(ValueError, "Retry/UAI"):
            validate(value)

    def test_selected_pilot_warm_start_is_required(self) -> None:
        value = copy.deepcopy(self.config)
        value["jobs"][0]["command"] = value["jobs"][0]["command"].replace(
            '--init-run-dir "$$SELECTED_RUN_DIR" \\\n', ""
        )
        with self.assertRaisesRegex(ValueError, "command drift"):
            validate(value)

    def test_rank_local_gpu_isolation_is_required(self) -> None:
        value = copy.deepcopy(self.config)
        value["jobs"][0]["command"] = value["jobs"][0]["command"].replace(
            'export LOCAL_RANK=0\n', ""
        )
        with self.assertRaisesRegex(ValueError, "command drift"):
            validate(value)

    def test_resilient_checkpoint_policy_is_required(self) -> None:
        value = copy.deepcopy(self.config)
        value["jobs"][0]["command"] = value["jobs"][0]["command"].replace(
            "--save-steps 250 --checkpoint-retention 2", "--save-steps 1000"
        )
        with self.assertRaisesRegex(ValueError, "command drift"):
            validate(value)

    def test_resume_selection_requires_hash_valid_fallback(self) -> None:
        value = copy.deepcopy(self.config)
        value["jobs"][0]["command"] = value["jobs"][0]["command"].replace(
            "select_latest_valid_distributed_checkpoint.py",
            "select_latest_checkpoint_by_existence.py",
        )
        with self.assertRaisesRegex(ValueError, "command drift"):
            validate(value)

    def test_explicit_nccl_device_binding_revision_is_required(self) -> None:
        value = copy.deepcopy(self.config)
        value["jobs"][0]["command"] = value["jobs"][0]["command"].replace(
            "20f85e37512965d560761de26d99d3329b15c16de679d34ad460a1407bad2f9a",
            "0" * 64,
        )
        with self.assertRaisesRegex(ValueError, "command drift"):
            validate(value)

    def test_hash_valid_checkpoint_selector_revision_is_required(self) -> None:
        value = copy.deepcopy(self.config)
        value["jobs"][0]["command"] = value["jobs"][0]["command"].replace(
            "7456f167e16c79025ca029d6dccd18612ae3a92c33caede06da995fe6e8be2b1",
            "0" * 64,
        )
        with self.assertRaisesRegex(ValueError, "command drift"):
            validate(value)


if __name__ == "__main__":
    unittest.main()
