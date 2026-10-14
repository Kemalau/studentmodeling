import unittest

import torch

from scripts.train_recurrent_student_m4_gold_semantic_teacher import (
    gold_semantic_carrier_batch,
    is_gold_semantic_row,
)


class GoldSemanticTeacherTest(unittest.TestCase):
    def test_only_eedi_t3_receives_gold_semantic_teacher(self):
        self.assertTrue(
            is_gold_semantic_row(
                {"task": "diagnose_misconception", "source": "eedi_mining"}
            )
        )
        self.assertFalse(
            is_gold_semantic_row(
                {"task": "diagnose_misconception", "source": "malrule"}
            )
        )
        self.assertFalse(
            is_gold_semantic_row(
                {"task": "predict_correctness", "source": "eedi_mining"}
            )
        )

    def test_carrier_moves_after_answer_without_changing_tokens(self):
        original = torch.tensor([[10, 99, 11, 12, 13]])
        ids, attention = gold_semantic_carrier_batch(original, 99)
        self.assertEqual(ids.tolist(), [[10, 11, 12, 13, 99]])
        self.assertEqual(sorted(ids.tolist()[0]), sorted(original.tolist()[0]))
        self.assertEqual(attention.tolist(), [[1, 1, 1, 1, 1]])

    def test_carrier_transform_rejects_missing_or_duplicate_address(self):
        with self.assertRaisesRegex(ValueError, "0 state carriers"):
            gold_semantic_carrier_batch(torch.tensor([[1, 2, 3]]), 99)
        with self.assertRaisesRegex(ValueError, "2 state carriers"):
            gold_semantic_carrier_batch(torch.tensor([[99, 2, 99]]), 99)

    def test_multiple_carriers_move_together_after_answer(self):
        ids, attention = gold_semantic_carrier_batch(
            torch.tensor([[10, 99, 99, 99, 11, 12]]), 99, 3
        )
        self.assertEqual(ids.tolist(), [[10, 11, 12, 99, 99, 99]])
        self.assertEqual(attention.tolist(), [[1, 1, 1, 1, 1, 1]])


if __name__ == "__main__":
    unittest.main()
