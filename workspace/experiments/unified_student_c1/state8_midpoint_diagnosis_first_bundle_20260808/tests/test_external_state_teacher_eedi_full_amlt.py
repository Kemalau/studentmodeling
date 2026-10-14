import copy
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from validate_external_state_teacher_eedi_full_amlt import load, validate


class ExternalStateTeacherAmltTest(unittest.TestCase):
    def setUp(self):
        self.config = load(
            ROOT / "amlt/feeds_external_state_teacher_eedi_full_diagnostic_20260804.yaml"
        )

    def test_config_passes(self):
        validate(self.config)

    def test_teacher_hash_and_weight_are_frozen(self):
        drift = copy.deepcopy(self.config)
        drift["jobs"][0]["submit_args"]["env"]["STUDENTMODELING_EXTERNAL_STATE_TEACHER_SHA256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "Controlled arm drift"):
            validate(drift)
        drift = copy.deepcopy(self.config)
        drift["jobs"][1]["submit_args"]["env"]["CONSISTENCY_WEIGHT"] = "0.25"
        with self.assertRaisesRegex(ValueError, "Controlled arm drift"):
            validate(drift)


if __name__ == "__main__":
    unittest.main()
