from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from validate_c3_exposure_matched_amlt import ARMS, load, validate


class C3AmltTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load(ROOT / "amlt/feeds_c3_exposure_matched_round1500_20260804.yaml")

    def test_full_config_passes(self) -> None:
        validate(self.config)

    def test_selected_sibling_passes(self) -> None:
        value = copy.deepcopy(self.config)
        value["jobs"] = [value["jobs"][1]]
        validate(value, selected_jobs={value["jobs"][0]["name"]})

    def test_uncontrolled_lr_change_fails(self) -> None:
        value = copy.deepcopy(self.config)
        value["jobs"][1]["submit_args"]["env"]["LEARNING_RATE"] = "1e-6"
        with self.assertRaisesRegex(ValueError, "controlled environment"):
            validate(value)

    def test_amlt_scientific_notation_rewrite_is_equivalent(self) -> None:
        value = copy.deepcopy(self.config)
        for job in value["jobs"]:
            job["submit_args"]["env"]["LEARNING_RATE"] = 5.0e-07
        validate(value)

    def test_job_set_is_exact(self) -> None:
        value = copy.deepcopy(self.config)
        value["jobs"] = value["jobs"][:2]
        with self.assertRaisesRegex(ValueError, "job-set"):
            validate(value)

    def test_runner_and_trainer_provenance_are_current(self) -> None:
        runner = ROOT / "scripts/run_layerwise_task_retention_rehearsal.sh"
        import hashlib

        self.assertEqual(
            hashlib.sha256(runner.read_bytes()).hexdigest(),
            "93ff8cd9bac2a06d84ff5f1d6089ed4161fd2e4b387b18b259a83a9e6d9501ac",
        )
        source = runner.read_text(encoding="utf-8")
        self.assertIn("TASK_RETENTION_FAILED", source)
        self.assertIn("TASK_RETENTION_STAGE", source)
        self.assertIn("materialize_legacy_layerwise_state_count_compat.py", source)
        self.assertIn(
            "a67662448408a587f891d32d4c5170fdca0435adac47554268edf5a3972330b5",
            source,
        )
        self.assertIn(
            "69fd4f2f4943a78a0afe9a701533d9ab034f247f87c7a09327f7523e310b7a52",
            source,
        )
        self.assertNotIn(
            "46f1ebf5ce216c76f92216729247bbaef990f1accda82b10ee72b683423d40e6",
            source,
        )
        c3_runner = ROOT / "scripts/run_c3_exposure_matched_promoted_order.sh"
        self.assertIn(
            "round1500_promoted_order_plus_controlled_segment_v1",
            (ROOT / "scripts/train_recurrent_student_m4_open_t3_c3_promoted_order.py").read_text(),
        )
        self.assertIn("93ff8cd9bac2a06d84ff5f1d6089ed4161fd2e4b387b18b259a83a9e6d9501ac", c3_runner.read_text())
        import yaml

        config = yaml.safe_load(
            (ROOT / "amlt/feeds_c3_exposure_matched_round1500_20260804.yaml").read_text()
        )
        self.assertTrue(
            all(
                job["submit_args"]["env"]["SAME_RULE_AUGMENTER_SHA256"]
                == "1ae82541e97353d23b674af80dfb0e2d50cfd92e3a6ee99c2b55f3b0caf1d785"
                for job in config["jobs"]
            )
        )
        self.assertTrue(
            all(
                job["submit_args"]["env"]["TRAIN_ENTRYPOINT"]
                == "scripts/train_recurrent_student_m4_open_t3_c3_promoted_order.py"
                for job in config["jobs"]
            )
        )


if __name__ == "__main__":
    unittest.main()
