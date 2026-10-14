"""Tests for the separate, explicitly post-hoc numeric-format audit."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audit_c4_freegen_results import format_sensitivity
from c4_freegen_protocol import parse_output


class FormatAuditTest(unittest.TestCase):
    def test_keeps_frozen_parser_unchanged(self):
        text = "<predicted_response>1 1/4</predicted_response>"
        self.assertEqual(parse_output(text), (None, "invalid"))
        self.assertEqual(format_sensitivity(text), ("5/4", "posthoc_mixed_number"))

    def test_signed_mixed_number(self):
        self.assertEqual(format_sensitivity("<predicted_response>-2 1/3</predicted_response>")[0], "-7/3")

    def test_single_unit_only(self):
        self.assertEqual(format_sensitivity("<predicted_response>0.119 kg</predicted_response>"),
                         ("119/1000", "posthoc_kg_suffix"))
        self.assertEqual(format_sensitivity("<predicted_response>56/30</predicted_response>"),
                         ("28/15", "tagged"))

    def test_rejects_equations_guesses_and_bad_fractions(self):
        for body in ("245 - 181 = 64", "1 or 2", "1/0", "1 1/0", "1 5/4", "1 kg or 2 kg", "1 kg kg"):
            self.assertIsNone(format_sensitivity(f"<predicted_response>{body}</predicted_response>")[0])


if __name__ == "__main__":
    unittest.main()
