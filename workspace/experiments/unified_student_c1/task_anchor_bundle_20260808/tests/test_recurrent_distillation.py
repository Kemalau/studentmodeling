from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from recurrent_distillation import (  # noqa: E402
    supervised_token_kl,
    topk_teacher_cross_entropy,
)


class RecurrentDistillationTest(unittest.TestCase):
    def test_uses_only_supervised_positions(self) -> None:
        teacher = torch.randn(1, 5, 11)
        student = teacher.clone()
        labels = torch.tensor([[-100, -100, -100, 4, 5]])
        self.assertLess(float(supervised_token_kl(student, teacher, labels)), 1e-7)
        changed_unsupervised = student.clone()
        changed_unsupervised[:, 0] += 20
        self.assertLess(
            float(supervised_token_kl(changed_unsupervised, teacher, labels)), 1e-7
        )
        changed_supervised = student.clone()
        changed_supervised[:, 2, 0] += 20
        self.assertGreater(
            float(supervised_token_kl(changed_supervised, teacher, labels)), 0.1
        )

    def test_aligns_different_prompt_lengths_by_target_tokens(self) -> None:
        teacher = torch.randn(1, 7, 11)
        student = teacher[:, 2:].clone()
        student_labels = torch.tensor([[-100, -100, -100, 4, 5]])
        teacher_labels = torch.tensor([[-100, -100, -100, -100, -100, 4, 5]])
        self.assertLess(
            float(
                supervised_token_kl(
                    student, teacher, student_labels, teacher_labels
                )
            ),
            1e-7,
        )
        bad_teacher_labels = teacher_labels.clone()
        bad_teacher_labels[0, -1] = 6
        with self.assertRaisesRegex(ValueError, "not aligned"):
            supervised_token_kl(
                student, teacher, student_labels, bad_teacher_labels
            )

    def test_rejects_invalid_temperature_and_empty_labels(self) -> None:
        logits = torch.randn(1, 3, 5)
        labels = torch.full((1, 3), -100)
        with self.assertRaisesRegex(ValueError, "temperature"):
            supervised_token_kl(logits, logits, labels, temperature=0)
        with self.assertRaisesRegex(ValueError, "No supervised"):
            supervised_token_kl(logits, logits, labels)

    def test_topk_cache_loss_aligns_targets_and_prefers_teacher_modes(self) -> None:
        labels = torch.tensor([[-100, -100, 4, 5]])
        targets = torch.tensor([4, 5])
        ids = torch.tensor([[1, 2], [3, 4]])
        teacher = torch.log(torch.tensor([[0.9, 0.1], [0.8, 0.2]]))
        aligned = torch.zeros(1, 4, 7)
        aligned[0, 1, 1] = 5
        aligned[0, 1, 2] = 0
        aligned[0, 2, 3] = 5
        aligned[0, 2, 4] = 0
        reversed_logits = aligned.clone()
        reversed_logits[0, 1, 1], reversed_logits[0, 1, 2] = (
            aligned[0, 1, 2], aligned[0, 1, 1]
        )
        reversed_logits[0, 2, 3], reversed_logits[0, 2, 4] = (
            aligned[0, 2, 4], aligned[0, 2, 3]
        )
        good = topk_teacher_cross_entropy(aligned, labels, targets, ids, teacher)
        bad = topk_teacher_cross_entropy(
            reversed_logits, labels, targets, ids, teacher
        )
        self.assertLess(float(good), float(bad))


if __name__ == "__main__":
    unittest.main()
