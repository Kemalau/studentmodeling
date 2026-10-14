import copy
import unittest

from scripts.open_t3_training_view import (
    apply_open_t3_training_view,
    is_open_t3_row,
    remove_valid_outputs,
    selective_behavior_kd_weight,
)


class OpenT3TrainingViewTest(unittest.TestCase):
    def test_removes_only_candidate_block(self):
        text = (
            "Instruction\n<query>\nProblem: 1/2 + 1/3\nCorrect answer: 5/6\n"
            "Valid outputs:\n<misconception>A</misconception>\n"
            "<misconception>B</misconception>\n</query>\nAnswer:"
        )
        self.assertEqual(
            remove_valid_outputs(text),
            "Instruction\n<query>\nProblem: 1/2 + 1/3\nCorrect answer: 5/6\n</query>\nAnswer:",
        )

    def test_view_preserves_state_target_and_candidate_metadata(self):
        malrule = {
            "example_id": "m1",
            "task": "diagnose_misconception",
            "source": "malrule",
            "target": "<misconception>B</misconception>",
            "target_label": "B",
            "candidate_labels": ["A", "B"],
            "candidate_targets": {
                "A": "<misconception>A</misconception>",
                "B": "<misconception>B</misconception>",
            },
            "decode_text": (
                "<query>\nProblem: x\nValid outputs:\n"
                "<misconception>A</misconception>\n"
                "<misconception>B</misconception>\n</query>\nAnswer:"
            ),
            "direct_text": "DIRECT_WITH_CANDIDATES",
            "update_texts": ["STATE_UPDATE"],
        }
        other = {
            "example_id": "e1",
            "task": "diagnose_misconception",
            "source": "eedi_mining",
            "target": "<misconception>E</misconception>",
            "decode_text": "EEDI_OPEN_PROMPT",
        }
        rows = [copy.deepcopy(malrule), copy.deepcopy(other)]
        audit = apply_open_t3_training_view(rows)
        self.assertEqual(audit["changed_malrule_t3_rows"], 1)
        self.assertNotIn("Valid outputs:", rows[0]["decode_text"])
        self.assertNotIn(rows[0]["target"], rows[0]["decode_text"])
        for key in (
            "example_id", "target", "target_label", "candidate_labels",
            "candidate_targets", "direct_text", "update_texts",
        ):
            self.assertEqual(rows[0][key], malrule[key])
        self.assertEqual(rows[1], other)

    def test_malformed_candidate_block_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "without </query>"):
            remove_valid_outputs("Valid outputs:\n<misconception>A</misconception>")

    def test_already_open_prompt_is_idempotent(self):
        self.assertEqual(remove_valid_outputs("<query>Open</query>"), "<query>Open</query>")

    def test_selective_kd_excludes_only_rewritten_malrule_t3(self):
        excluded = {"task": "diagnose_misconception", "source": "malrule"}
        retained = [
            {"task": "predict_correctness", "source": "malrule"},
            {"task": "diagnose_misconception", "source": "eedi_mining"},
            {"task": "predict_error_persistence", "source": "foundationalassist"},
        ]
        self.assertTrue(is_open_t3_row(excluded))
        self.assertEqual(selective_behavior_kd_weight(excluded, 0.5), 0.0)
        for row in retained:
            self.assertFalse(is_open_t3_row(row))
            self.assertEqual(selective_behavior_kd_weight(row, 0.5), 0.5)

    def test_selective_kd_rejects_negative_global_weight(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            selective_behavior_kd_weight({}, -0.5)


if __name__ == "__main__":
    unittest.main()
