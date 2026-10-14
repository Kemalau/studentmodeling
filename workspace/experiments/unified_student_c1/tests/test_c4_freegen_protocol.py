import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from c4_freegen_protocol import canonical, parse_output, render_prompt, signatures, summarize


class ProtocolTest(unittest.TestCase):
    def test_equivalent_answers_and_reject_extra_guesses(self):
        self.assertEqual(parse_output("<predicted_response>56/30</predicted_response>"), ("28/15", "tagged"))
        self.assertEqual(parse_output("0.5"), ("1/2", "bare"))
        for bad in ("1 or 2", "<predicted_response>1</predicted_response> 2", "1/0", "<think>1</think>2"):
            self.assertIsNone(parse_output(bad)[0])

    def test_no_query_answer_leak(self):
        row = {"seg_instruction": "instruction\n", "seg_history": "<student_history>old answers</student_history>\n"}
        prompt = render_prompt(row, "Compute: 7/5 ÷ 4/3")
        self.assertNotIn("28/15", prompt)
        self.assertNotIn("21/20", prompt)
        self.assertNotIn("Valid outputs", prompt)
        self.assertTrue(prompt.endswith("</query>\nAnswer:"))

    def test_seen_operands_cannot_be_reworded(self):
        a = {"operands": ["2", "5"], "problem_text": "What is 5 minus 2?"}
        b = {"operands": ["5", "2.0"], "problem_text": "Subtract 2 from 5."}
        self.assertTrue(signatures(a) & signatures(b))

    def test_completeness_and_donor_clusters(self):
        manifest = {"groups": [{"group_idx": g, "queries": [{"query_idx": q} for q in range(3)]} for g in range(2)]}
        records = []
        for g in range(2):
            for q in range(3):
                for c in ("none", "noop", "swap", "random_real", "samemal"):
                    records.append({"group_idx": g, "query_idx": q, "condition": c, "donor_id": str(g),
                                    "matches_A": c == "swap", "matches_B": c == "none", "parsed": "1", "text": "1"})
        summary = summarize(manifest, records, bootstrap=100)
        self.assertEqual(summary["all_queries"]["n_donors"], 2)
        self.assertEqual(summary["all_queries"]["directed_minus_real_random"], 1)
        self.assertEqual(summary["all_three_match_A"]["swap"], 2)
        with self.assertRaises(ValueError):
            summarize(manifest, records[:-1], bootstrap=100)
        with self.assertRaises(ValueError):
            summarize(manifest, records + records[:1], bootstrap=100)


if __name__ == "__main__":
    unittest.main()
