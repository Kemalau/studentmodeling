import unittest

from scripts.augment_recurrent_causal_diagnosis import augment, clean_query
from scripts.t3_judge_utils import extract_tag, natural_schema_valid


def update(student: str, correct: str) -> str:
    return (
        "<student_state_update><STUDENT_STATE>"
        f'<observed_interaction>{{"student_answer":"{student}",'
        f'"correct_answer":"{correct}"}}</observed_interaction>'
        "<STUDENT_STATE></student_state_update>"
    )


class CausalDiagnosisAugmentationTest(unittest.TestCase):
    def row(self):
        return {
            "pair_id": "pair:1",
            "recipient_label": "fractions.denominator_comparison_error",
            "donor_label": "fractions.ignore_denominators",
            "native_update_texts": [update("1/8", "1/2")],
            "donor_update_texts": [update("3", "3/8")],
            "decode_text": (
                "Predict response.\n<STUDENT_STATE>\n<query>\nProblem: Compare 1/8 and 1/2\n"
                "Correct answer: 1/2\nValid outputs:\n<predicted_response>1/8</predicted_response>\n"
                "</query>\nAnswer:"
            ),
            "native_target": "<predicted_response>1/8</predicted_response>",
            "donor_target": "<predicted_response>3</predicted_response>",
            "native_target_label": "1/8",
            "donor_target_label": "3",
            "candidate_labels": ["1/8", "3"],
            "candidate_targets": {
                "1/8": "<predicted_response>1/8</predicted_response>",
                "3": "<predicted_response>3</predicted_response>",
            },
        }

    def test_answer_and_diagnosis_twins(self):
        output, stats = augment([self.row()])
        self.assertEqual(len(output), 2)
        answer, diagnosis = output
        self.assertEqual(answer["causal_target_kind"], "prospective_answer")
        self.assertEqual(diagnosis["causal_target_kind"], "diagnosis_natural_schema")
        self.assertEqual(diagnosis["pair_id"], "pair:1:diagnosis")
        self.assertTrue(natural_schema_valid(diagnosis["native_target"]))
        self.assertTrue(natural_schema_valid(diagnosis["donor_target"]))
        self.assertEqual(
            extract_tag(diagnosis["native_target"], "label"),
            "fractions.denominator_comparison_error",
        )
        self.assertIn("larger denominator", diagnosis["native_target"])
        self.assertNotIn("Valid outputs", diagnosis["decode_text"])
        self.assertNotIn("student_history", diagnosis["decode_text"])
        self.assertEqual(stats["output_pairs"], 2)

    def test_clean_query_rejects_missing_query(self):
        with self.assertRaises(ValueError):
            clean_query("no query")


if __name__ == "__main__":
    unittest.main()
