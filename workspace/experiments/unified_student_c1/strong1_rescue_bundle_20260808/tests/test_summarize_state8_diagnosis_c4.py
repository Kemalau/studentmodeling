import unittest

from scripts.summarize_state8_diagnosis_c4 import summarize


class DiagnosisC4SemanticSummaryTest(unittest.TestCase):
    def test_semantic_contrasts(self):
        rows = []
        truth = {
            "native": {"donor": False, "recipient": True},
            "noop": {"donor": False, "recipient": True},
            "samemal": {"donor": False, "recipient": True},
            "random": {"donor": False, "recipient": False},
            "swap": {"donor": True, "recipient": False},
        }
        for condition, roles in truth.items():
            for role, correct in roles.items():
                rows.append(
                    {
                        "pair_idx": 1,
                        "condition": condition,
                        "reference_role": role,
                        "llm_judge_correct": correct,
                        "llm_judge_confidence": 0.9,
                    }
                )
        result = summarize(rows)
        self.assertEqual(result["semantic_D_specific"], 1.0)
        self.assertEqual(result["random_matches_either_stable_target_rate"], 0.0)
        self.assertEqual(result["same_rule_recipient_retention"], 1.0)

    def test_thresholds_are_semantic_not_exact_label(self):
        source = __import__(
            "pathlib"
        ).Path("scripts/summarize_state8_diagnosis_c4.py").read_text()
        self.assertIn("semantic_judge_authoritative", source)
        self.assertIn("exact_label_role", source)
        self.assertIn("auxiliary_only_not_used_for_gate", source)


if __name__ == "__main__":
    unittest.main()
