from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from validate_user_candidate_parallel_eval_amlt import load, validate  # noqa: E402


class UserCandidateParallelEvalAmltTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load(ROOT / "amlt/feeds_user_candidate_lr2e6_parallel_eval_20260804.yaml")

    def test_full_config_passes(self) -> None:
        validate(self.config)

    def test_failed_sibling_subset_passes(self) -> None:
        config = copy.deepcopy(self.config)
        config["jobs"] = [config["jobs"][1]]
        validate(config, {"user-candidate-lr2e6-t3-seed42"})

    def test_task_support_drift_is_rejected(self) -> None:
        config = copy.deepcopy(self.config)
        config["jobs"][0]["submit_args"]["env"]["EXPECTED_N"] = "10"
        with self.assertRaisesRegex(ValueError, "Task/support drift"):
            validate(config)

    def test_unknown_subset_is_rejected(self) -> None:
        config = copy.deepcopy(self.config)
        config["jobs"] = [config["jobs"][0]]
        with self.assertRaisesRegex(ValueError, "Job-set drift"):
            validate(config, {"unknown-job"})


if __name__ == "__main__":
    unittest.main()
