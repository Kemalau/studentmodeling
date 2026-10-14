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
            "a6e9ac56f263caa1982403979a6b4ff70750ea2f95086cdd6087edeca4488bc8",
        )
        source = runner.read_text(encoding="utf-8")
        self.assertIn("TASK_RETENTION_FAILED", source)
        self.assertIn("TASK_RETENTION_STAGE", source)
        self.assertIn(
            "ae502968872ff6597e73bcc12cb3ff9a5bbb0fe9d7a763383f6e76bc392de42c",
            source,
        )
        self.assertNotIn(
            "46f1ebf5ce216c76f92216729247bbaef990f1accda82b10ee72b683423d40e6",
            source,
        )


if __name__ == "__main__":
    unittest.main()
