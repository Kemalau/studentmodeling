from __future__ import annotations

import sys
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_recurrent_t3_constrained import CLOSE_TAG, OPEN_TAG  # noqa: E402


class RecurrentT3ConstrainedTest(unittest.TestCase):
    def test_xml_envelope_is_literal_and_semantics_remain_generated(self) -> None:
        self.assertEqual(OPEN_TAG, "<misconception>")
        self.assertEqual(CLOSE_TAG, "</misconception>")
        source = (SCRIPT_DIR / "evaluate_recurrent_t3_constrained.py").read_text()
        self.assertIn("semantic_content_is_model_generated", source)
        self.assertIn("wrapper.decode_from_state", source)
        self.assertIn("wrapper.backbone", source)
        self.assertIn("raw_history_visible_to_decoder", source)
        self.assertNotIn("row[\"target\"] +", source)

    def test_diagnostic_cannot_be_confused_with_formal_evaluation(self) -> None:
        source = (SCRIPT_DIR / "evaluate_recurrent_t3_constrained.py").read_text()
        self.assertIn('"diagnostic_only": True', source)
        self.assertIn('"frozen_formal_evaluator_unchanged": True', source)
        self.assertIn('row.get("task") == "diagnose_misconception"', source)


if __name__ == "__main__":
    unittest.main()
