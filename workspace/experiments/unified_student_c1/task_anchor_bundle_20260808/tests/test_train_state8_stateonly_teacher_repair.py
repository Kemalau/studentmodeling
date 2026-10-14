from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from train_state8_stateonly_teacher_repair import (  # noqa: E402
    find_subsequence,
    schema_weighted_loss_factory,
)


class TinyTokenizer:
    mapping = {"<label>": [10, 11], "</label>": [12, 13]}

    def __call__(self, value, add_special_tokens=False):
        return {"input_ids": self.mapping[value]}


class State8StateonlyRepairTest(unittest.TestCase):
    def test_subsequence(self) -> None:
        self.assertEqual(find_subsequence([1, 2, 3, 2, 3], [2, 3]), 1)
        self.assertEqual(find_subsequence([1, 2], [3]), -1)

    def test_label_value_receives_extra_weight(self) -> None:
        loss = schema_weighted_loss_factory(TinyTokenizer(), 4.0)
        # Shifted supervised target is <label> 99 </label> 42.
        labels = torch.tensor([[-100, 10, 11, 99, 12, 13, 42]])
        logits = torch.zeros(1, 7, 128)
        logits[0, 2, 99] = -4.0
        logits.requires_grad_(True)
        weighted = loss(logits, labels)
        plain = torch.nn.functional.cross_entropy(
            logits[:, :-1][labels[:, 1:].ne(-100)], labels[:, 1:][labels[:, 1:].ne(-100)]
        )
        self.assertGreater(float(weighted), float(plain))
        weighted.backward()
        self.assertIsNotNone(logits.grad)

    def test_source_declares_strict_stateonly_schedule(self) -> None:
        source = (ROOT / "scripts/train_state8_stateonly_teacher_repair.py").read_text()
        self.assertIn("lambda _args, _progress: (0.0, 1.0, 1.0)", source)
        self.assertIn('namespace.final_inference_route = "strict_state_only"', source)


if __name__ == "__main__":
    unittest.main()
