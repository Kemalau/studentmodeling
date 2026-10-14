from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from validate_layerwise_staged_task_retention_amlt import load, validate  # noqa: E402


class StagedTaskRetentionAmltTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load(ROOT / "amlt/feeds_layerwise_staged_task_retention_20260802.yaml")

    def test_config_passes(self) -> None:
        validate(self.config)

    def test_validator_tracks_current_launcher_branch_registry(self) -> None:
        source = (
            ROOT / "scripts/validate_layerwise_staged_task_retention_amlt.py"
        ).read_text(encoding="utf-8")
        self.assertIn("eedi_full_open_t3", source)
        self.assertIn("equal_t3_t6_open_t3", source)

    def test_recipe_and_manifest_are_frozen(self) -> None:
        env = self.config["jobs"][0]["submit_args"]["env"]
        self.assertEqual(env["BRANCH"], "staged_t3_t6")
        self.assertEqual(env["LOSS_PROFILE"], "standard")
        self.assertEqual(
            env["MANIFEST_SHA256"],
            "a4eb41b2130f7e0810eeff36b5f3d16b8dd2c8631c50dbd25216f0793af23557",
        )

    def test_gpu_or_uai_drift_is_rejected(self) -> None:
        value = copy.deepcopy(self.config)
        value["jobs"][0]["sku"] = "80G1-A100"
        with self.assertRaisesRegex(ValueError, "GPU"):
            validate(value)
        value = copy.deepcopy(self.config)
        value["jobs"][0]["submit_args"]["env"]["_AZUREML_SINGULARITY_JOB_UAI"] = "other"
        with self.assertRaisesRegex(ValueError, "environment"):
            validate(value)


if __name__ == "__main__":
    unittest.main()
