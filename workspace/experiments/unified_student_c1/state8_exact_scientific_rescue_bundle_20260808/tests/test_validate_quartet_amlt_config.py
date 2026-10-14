from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from validate_quartet_amlt_config import load, validate  # noqa: E402


class QuartetAMLTConfigTest(unittest.TestCase):
    def setUp(self) -> None:
        self.path = ROOT / "amlt/feeds_paired_state_quartet_v2_seed42_20260730.yaml"
        self.config = load(self.path)

    def test_frozen_config_passes(self) -> None:
        validate(self.config)

    def test_rejects_retry_and_uai_drift(self) -> None:
        value = copy.deepcopy(self.config)
        value["jobs"][0]["submit_args"]["max_attempts"] = 1
        with self.assertRaisesRegex(ValueError, "Retry drift"):
            validate(value)
        value = copy.deepcopy(self.config)
        value["jobs"][0]["submit_args"]["env"]["_AZUREML_SINGULARITY_JOB_UAI"] = "wrong"
        with self.assertRaisesRegex(ValueError, "UAI drift"):
            validate(value)


if __name__ == "__main__":
    unittest.main()
