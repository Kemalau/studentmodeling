from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from validate_candidate_lr5e6_statepath_full_t3_amlt import load, validate  # noqa: E402


class CandidateLr5e6StatepathFullT3AmltTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load(
            ROOT / "amlt/feeds_candidate_lr5e6_statepath_full_t3_20260804.yaml"
        )

    def test_config_passes(self) -> None:
        validate(self.config)

    def test_artifact_hash_drift_is_rejected(self) -> None:
        config = copy.deepcopy(self.config)
        config["jobs"][0]["command"] = config["jobs"][0]["command"].replace(
            "53369a7befa5cdf771de9df979e2785346d9ad7002470502c5670ad3674329c8",
            "0" * 64,
        )
        with self.assertRaisesRegex(ValueError, "command drift"):
            validate(config)

    def test_non_t3_task_is_rejected(self) -> None:
        config = copy.deepcopy(self.config)
        config["jobs"][0]["command"] += "\necho predict_response\n"
        with self.assertRaisesRegex(ValueError, "non-T3"):
            validate(config)


if __name__ == "__main__":
    unittest.main()
