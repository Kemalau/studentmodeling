from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from validate_layerwise_task_retention_promoted_final_eval_amlt import (  # noqa: E402
    load,
    validate,
)


class PromotedFinalEvalAmltTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load(
            ROOT / "amlt/feeds_layerwise_task_retention_promoted_final_eval_20260802.yaml"
        )

    def test_config_passes(self) -> None:
        validate(self.config)

    def test_every_job_forces_the_promoted_eval_venv(self) -> None:
        for job in self.config["jobs"]:
            self.assertEqual(
                job["submit_args"]["env"]["VENV_ROOT"],
                "/tmp/studentmodeling-promoted-final-eval-venv",
            )
            self.assertIn('python() { "$$VENV_ROOT/bin/python" "$$@"; }', job["command"])

    def test_receipt_drift_is_rejected(self) -> None:
        value = copy.deepcopy(self.config)
        value["jobs"][0]["command"] = value["jobs"][0]["command"].replace(
            "task_retention_promotion/final_fit_receipt.json", "final_fit/seed42.json"
        )
        with self.assertRaisesRegex(ValueError, "receipt"):
            validate(value)

    def test_single_recovery_job_passes(self) -> None:
        value = copy.deepcopy(self.config)
        value["jobs"] = [value["jobs"][2]]
        validate(value, {"layerwise-promoted-final-eval-t3-seed42"})

    def test_amlt_dump_command_list_passes(self) -> None:
        value = copy.deepcopy(self.config)
        for job in value["jobs"]:
            job["command"] = [job["command"].replace("$$", "$")]
        validate(value)

    def test_t6_threshold_application_is_required(self) -> None:
        value = copy.deepcopy(self.config)
        job = value["jobs"][0]
        job["command"] = job["command"].replace(
            "python scripts/apply_frozen_t6_threshold.py",
            "python scripts/apply_unfrozen_t6_threshold.py",
        )
        with self.assertRaisesRegex(ValueError, "frozen-calibration"):
            validate(value)


if __name__ == "__main__":
    unittest.main()
