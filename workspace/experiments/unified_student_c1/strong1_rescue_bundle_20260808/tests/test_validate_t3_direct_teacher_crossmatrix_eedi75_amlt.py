from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from validate_t3_direct_teacher_crossmatrix_eedi75_amlt import validate


class DirectTeacherCrossmatrixValidationTest(unittest.TestCase):
    def setUp(self) -> None:
        path = ROOT / "amlt" / "feeds_t3_direct_teacher_crossmatrix_eedi75_20260804.yaml"
        self.config = yaml.safe_load(path.read_text(encoding="utf-8"))

    def test_accepts_frozen_config(self) -> None:
        validate(copy.deepcopy(self.config))

    def test_rejects_capture_drift(self) -> None:
        value = copy.deepcopy(self.config)
        value["jobs"][0]["submit_args"]["env"]["CAPTURE_SHA256"] = "0" * 64
        with self.assertRaises(ValueError):
            validate(value)

    def test_accepts_single_recovery_selector(self) -> None:
        value = copy.deepcopy(self.config)
        value["jobs"] = value["jobs"][:1]
        validate(value, {"current-teacher-full-decoder-eedi75-seed42"})


if __name__ == "__main__":
    unittest.main()
