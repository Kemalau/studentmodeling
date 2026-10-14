from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from validate_layerwise_task_retention_rehearsal_amlt import load, validate  # noqa: E402


class TaskRetentionAmltTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load(ROOT / "amlt/feeds_layerwise_task_retention_rehearsal_20260801.yaml")

    def test_config_passes(self) -> None:
        validate(self.config)

    def test_arms_share_one_command(self) -> None:
        self.assertEqual(self.config["jobs"][0]["command"], self.config["jobs"][1]["command"])

    def test_launcher_forces_the_job_venv_interpreter(self) -> None:
        launcher = (ROOT / "scripts/run_layerwise_task_retention_rehearsal.sh").read_text()
        for job in self.config["jobs"]:
            self.assertEqual(
                job["submit_args"]["env"]["VENV_ROOT"],
                "/tmp/studentmodeling-layerwise-task-retention-venv",
            )
        self.assertIn('python() { "$VENV_ROOT/bin/python" "$@"; }', launcher)
        self.assertIn("import accelerate,bitsandbytes,peft,safetensors,transformers", launcher)

    def test_manifest_drift_is_rejected(self) -> None:
        value = copy.deepcopy(self.config)
        value["jobs"][0]["submit_args"]["env"]["MANIFEST_SHA256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "arm drift"):
            validate(value)

    def test_gpu_or_target_drift_is_rejected(self) -> None:
        value = copy.deepcopy(self.config)
        value["jobs"][0]["sku"] = "80G1-A100"
        with self.assertRaisesRegex(ValueError, "GPU"):
            validate(value)


if __name__ == "__main__":
    unittest.main()
