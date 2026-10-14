from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from augment_recurrent_causal_same_rule import augment  # noqa: E402


def source(example_id: str, malrule: str, answer: str) -> dict:
    return {
        "example_id": example_id,
        "metadata": {"family": "toy", "malrule_id": malrule},
        "history": [
            {
                "problem_text": "Observed problem",
                "student_answer": answer,
                "correct_answer": "0",
                "is_correct": False,
            }
        ],
    }


class AugmentRecurrentCausalSameRuleTest(unittest.TestCase):
    def test_doubles_pairs_with_same_rule_native_control(self) -> None:
        sources = [
            source("donor", "rule_a", "A"),
            source("recipient", "rule_b", "B"),
            source("control", "rule_b", "B2"),
        ]
        row = {
            "pair_id": "recurrent-causal:0000000",
            "donor_id": "donor",
            "recipient_id": "recipient",
            "donor_label": "rule_a",
            "recipient_label": "rule_b",
            "family": "toy",
            "donor_update_texts": ["donor"],
            "native_update_texts": ["recipient"],
            "decode_text": "<STUDENT_STATE> query",
            "donor_target": "A",
            "native_target": "B",
        }
        rows, stats = augment(sources, [row], seed=7, max_updates=1)
        self.assertEqual(len(rows), 2)
        self.assertEqual(stats["same_rule_native_pairs"], 1)
        self.assertEqual(rows[0]["native_state_source_id"], "recipient")
        self.assertEqual(rows[1]["native_state_source_id"], "control")
        self.assertEqual(rows[1]["native_control_role"], "same_rule_native")
        self.assertEqual(rows[1]["native_target"], "B")
        self.assertEqual(rows[1]["decode_text"], row["decode_text"])
        self.assertNotIn("<student_history>", rows[1]["decode_text"].lower())
        self.assertEqual(rows[1]["native_update_texts"][0].count("<STUDENT_STATE>"), 2)
        for item in rows:
            self.assertEqual(item["recipient_native_update_texts"], ["recipient"])
            self.assertEqual(
                item["same_rule_control_update_texts"], item["native_update_texts"]
                if item["native_control_role"] == "same_rule_native"
                else rows[1]["native_update_texts"],
            )

    def test_is_deterministic_and_rejects_missing_control(self) -> None:
        sources = [
            source("donor", "rule_a", "A"),
            source("recipient", "rule_b", "B"),
            source("control1", "rule_b", "B1"),
            source("control2", "rule_b", "B2"),
        ]
        row = {
            "pair_id": "p",
            "donor_id": "donor",
            "recipient_id": "recipient",
            "recipient_label": "rule_b",
            "family": "toy",
        }
        first, _ = augment(sources, [row], seed=11, max_updates=1)
        second, _ = augment(list(reversed(sources)), [row], seed=11, max_updates=1)
        self.assertEqual(first, second)
        with self.assertRaisesRegex(ValueError, "No same-rule control"):
            augment(sources[:2], [row], seed=11, max_updates=1)


if __name__ == "__main__":
    unittest.main()
