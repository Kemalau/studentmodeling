from __future__ import annotations

import unittest

from scripts.filter_state8_causal_diagnosis import filter_rows


class FilterState8CausalDiagnosisTest(unittest.TestCase):
    def test_keeps_balanced_diagnosis_twins_only(self) -> None:
        rows = [
            {
                "pair_id": "a",
                "causal_target_kind": "prospective_answer",
                "native_control_role": "recipient_native",
            },
            {
                "pair_id": "a:diagnosis",
                "causal_target_kind": "diagnosis_natural_schema",
                "native_control_role": "recipient_native",
                "raw_history_visible_to_decoder": False,
            },
            {
                "pair_id": "b:diagnosis",
                "causal_target_kind": "diagnosis_natural_schema",
                "native_control_role": "same_rule_native",
                "raw_history_visible_to_decoder": False,
            },
        ]
        selected, audit = filter_rows(rows)
        self.assertEqual([row["pair_id"] for row in selected], ["a:diagnosis", "b:diagnosis"])
        self.assertEqual(audit["rows"], 2)
        self.assertEqual(
            audit["role_counts"], {"recipient_native": 1, "same_rule_native": 1}
        )

    def test_fails_closed_on_role_imbalance(self) -> None:
        with self.assertRaisesRegex(ValueError, "imbalance"):
            filter_rows(
                [
                    {
                        "pair_id": "a:diagnosis",
                        "causal_target_kind": "diagnosis_natural_schema",
                        "native_control_role": "recipient_native",
                        "raw_history_visible_to_decoder": False,
                    }
                ]
            )


if __name__ == "__main__":
    unittest.main()
