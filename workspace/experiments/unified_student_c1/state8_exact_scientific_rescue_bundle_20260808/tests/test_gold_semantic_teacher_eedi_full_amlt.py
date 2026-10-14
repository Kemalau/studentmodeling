import copy
import unittest
from pathlib import Path

from scripts.validate_gold_semantic_teacher_eedi_full_amlt import load, validate


ROOT = Path(__file__).resolve().parents[1]


class GoldSemanticTeacherAmltTest(unittest.TestCase):
    def setUp(self):
        self.config = load(
            ROOT / "amlt/feeds_gold_semantic_teacher_eedi_full_diagnostic_20260804.yaml"
        )

    def test_config_passes(self):
        validate(self.config)

    def test_objective_and_teacher_are_frozen(self):
        drift = copy.deepcopy(self.config)
        drift["jobs"][0]["submit_args"]["env"]["TRAIN_ENTRYPOINT"] = "wrong.py"
        with self.assertRaisesRegex(ValueError, "Controlled arm drift"):
            validate(drift)
        drift = copy.deepcopy(self.config)
        drift["jobs"][1]["submit_args"]["env"]["CONSISTENCY_WEIGHT"] = "0.25"
        with self.assertRaisesRegex(ValueError, "Controlled arm drift"):
            validate(drift)


if __name__ == "__main__":
    unittest.main()
