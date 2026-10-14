from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from validate_current_lr2e6_direct_teacher_t3_amlt import load, validate  # noqa: E402


class DirectTeacherT3AmltTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load(ROOT / "amlt/feeds_current_lr2e6_direct_teacher_t3_20260804.yaml")

    def test_config_passes(self) -> None:
        validate(self.config)

    def test_recurrent_route_cannot_be_substituted(self) -> None:
        config = copy.deepcopy(self.config)
        config["jobs"][0]["command"] = config["jobs"][0]["command"].replace(
            "evaluate_direct_teacher_t3.py", "evaluate_recurrent_formal_protocols.py"
        )
        with self.assertRaisesRegex(ValueError, "command drift"):
            validate(config)

    def test_final_gate_eligibility_cannot_be_claimed(self) -> None:
        config = copy.deepcopy(self.config)
        config["jobs"][0]["command"] = config["jobs"][0]["command"].replace(
            "eligible_for_final_gate=false", "eligible_for_final_gate=true"
        )
        with self.assertRaisesRegex(ValueError, "command drift"):
            validate(config)


if __name__ == "__main__":
    unittest.main()
