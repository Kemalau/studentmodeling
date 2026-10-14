from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from validate_promoted_round1250_same_rule_repair_ab_amlt import load, validate  # noqa: E402


class PromotedRound1250SameRuleRepairAmltTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load(ROOT / "amlt/feeds_promoted_round1250_same_rule_repair_ab_20260804.yaml")

    def test_config_passes(self) -> None:
        validate(self.config)

    def test_only_controlled_factor_differs(self) -> None:
        left, right = [job["submit_args"]["env"] for job in self.config["jobs"]]
        self.assertEqual({left["CAUSAL_VARIANT"], right["CAUSAL_VARIANT"]}, {"base", "same_rule"})
        left = {k: v for k, v in left.items() if k != "CAUSAL_VARIANT"}
        right = {k: v for k, v in right.items() if k != "CAUSAL_VARIANT"}
        self.assertEqual(left, right)

    def test_uses_frozen_warm_start_and_failure_receipt(self) -> None:
        for job in self.config["jobs"]:
            env = job["submit_args"]["env"]
            self.assertEqual(
                env["WARM_START_CHECKPOINT_DIR"],
                "/mnt/amulet/studentmodeling/eval_artifacts/"
                "promoted_round1250_frozen_v1/checkpoint-round-1250",
            )
            self.assertIn("FAILED line=%s rc=%s command=%s", job["command"])

    def test_gpu_or_arm_drift_is_rejected(self) -> None:
        value = copy.deepcopy(self.config)
        value["jobs"][0]["sku"] = "80G1-A100"
        with self.assertRaisesRegex(ValueError, "GPU"):
            validate(value)
        value = copy.deepcopy(self.config)
        value["jobs"][0]["submit_args"]["env"]["LEARNING_RATE"] = "1e-6"
        with self.assertRaisesRegex(ValueError, "Controlled arm drift"):
            validate(value)

    def test_single_failed_sibling_dump_is_validated(self) -> None:
        value = copy.deepcopy(self.config)
        value["jobs"] = [value["jobs"][1]]
        validate(
            value,
            selected_jobs={"promoted-round1250-repair-same-rule-seed42"},
        )

    def test_selected_job_name_must_match_dump(self) -> None:
        value = copy.deepcopy(self.config)
        value["jobs"] = [value["jobs"][0]]
        with self.assertRaisesRegex(ValueError, "Job-set drift"):
            validate(
                value,
                selected_jobs={"promoted-round1250-repair-same-rule-seed42"},
            )


if __name__ == "__main__":
    unittest.main()
