import unittest
from pathlib import Path

from scripts.evaluate_recurrent_diagnosis_c4 import (
    category,
    diagnosis_prompt,
    diagnosis_target,
    summarize_exact,
)


class DiagnosisC4Test(unittest.TestCase):
    def test_prompt_is_closed_and_natural(self):
        prompt = diagnosis_prompt(
            {"query": {"problem_text": "Compare 1/2 and 1/3", "correct_answer": "1/2"}}
        )
        self.assertIn("<STUDENT_STATE>", prompt)
        self.assertIn("<diagnosis>", prompt)
        self.assertNotIn("<student_history>", prompt)

    def test_taxonomy_wrapper_disambiguates_label_from_math_answer(self):
        from scripts.evaluate_recurrent_diagnosis_c4_taxonomy import taxonomy_prompt

        prompt = taxonomy_prompt(
            {"query": {"problem_text": "Compare 1/2 and 1/3", "correct_answer": "1/2"}}
        )
        self.assertIn("domain.rule_name", prompt)
        self.assertIn("not a prediction of the current problem's answer", prompt)
        self.assertNotIn("<student_history>", prompt)

    def test_target_keeps_label_and_natural_diagnosis(self):
        target = diagnosis_target("fractions.denominator_comparison_error")
        self.assertIn("<label>fractions.denominator_comparison_error</label>", target)
        self.assertIn("larger denominator", target)

    def test_exact_summary(self):
        records = []
        outputs = {
            "native": ("recipient", "n"),
            "noop": ("recipient", "n"),
            "samemal": ("recipient", "s"),
            "random": ("other", "r"),
            "swap": ("donor", "d"),
        }
        for condition, (value, output) in outputs.items():
            records.append(
                {
                    "pair_idx": 1,
                    "condition": condition,
                    "category": value,
                    "natural_schema_valid": True,
                    "raw_output": output,
                }
            )
        summary = summarize_exact(records)
        self.assertEqual(summary["exact_label_D_specific"], 1.0)
        self.assertEqual(summary["noop_output_identical_to_native_rate"], 1.0)

    def test_category(self):
        self.assertEqual(category("a", "a", "b"), "donor")
        self.assertEqual(category("b", "a", "b"), "recipient")
        self.assertEqual(category("c", "a", "b"), "other")

    def test_evaluator_can_bind_immutable_artifact_hashes(self):
        source = Path("scripts/evaluate_recurrent_diagnosis_c4.py").read_text()
        self.assertIn("--expected-adapter-sha256", source)
        self.assertIn("--expected-recurrent-state-sha256", source)
        self.assertIn("Diagnosis-C4 {key} hash drift", source)

    def test_evaluator_supports_stable_pair_ranges(self):
        source = Path("scripts/evaluate_recurrent_diagnosis_c4.py").read_text()
        self.assertIn('"--pair-start"', source)
        self.assertIn('"--pair-count"', source)
        self.assertIn("start=args.pair_start + 1", source)
        self.assertIn('"pair_stop": pair_stop', source)


if __name__ == "__main__":
    unittest.main()
