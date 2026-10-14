from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

import torch


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from paired_state_quartet import build_quartet, patch_state_residual  # noqa: E402
from train_c1_lora import (  # noqa: E402
    assert_bottleneck_mask,
    build_bottleneck_attention_mask,
)


def row(name: str, history: str, target: str) -> dict:
    candidates = {"A": "<answer>A</answer>", "B": "<answer>B</answer>"}
    instruction = "instruction\n"
    seg_history = f"<student_history>{history}</student_history>\n"
    question = "<query>same question</query>\n"
    return {
        "example_id": name,
        "prompt": f"{instruction}{seg_history}<STUDENT_STATE>\n{question}",
        "seg_instruction": instruction,
        "seg_history": seg_history,
        "seg_question": question,
        "query": {"question": "same question"},
        "candidate_labels": ["A", "B"],
        "candidate_targets": candidates,
        "target_label": target,
        "target": candidates[target],
    }


class QuartetTest(unittest.TestCase):
    def setUp(self) -> None:
        self.donor = row("donor", "history A", "A")
        self.recipient = row("recipient", "history B", "B")
        self.normal = row("normal", "history normal", "B")

    def test_treatment_identifies_state_as_only_input_difference(self) -> None:
        rows, swaps, targets = build_quartet(
            self.donor, self.recipient, self.normal, "A", 7, "paired_swap"
        )
        self.assertEqual(len(rows), 4)
        self.assertEqual(swaps, [(1, 3)])
        self.assertEqual(rows[2]["prompt"], rows[3]["prompt"])
        self.assertEqual(rows[2]["target_label"], "B")
        self.assertEqual(rows[3]["target_label"], "A")
        self.assertEqual(targets, {"donor_on_recipient": "A", "native_recipient": "B"})

    def test_late_fusion_assertion_allows_only_answer_anchor_and_targets(self) -> None:
        attention = torch.ones(1, 9, dtype=torch.long)
        mask = build_bottleneck_attention_mask(
            attention,
            torch.tensor([1]),
            torch.tensor([4]),
            torch.tensor([7]),
            style="late_fusion",
        )
        assert_bottleneck_mask(
            mask,
            history_start=1,
            state_pos=4,
            valid_len=9,
            prompt_end=7,
            style="late_fusion",
        )
        self.assertLess(float(mask[0, 0, 5, 4]), -1e20)
        self.assertEqual(float(mask[0, 0, 6, 4]), 0.0)
        self.assertEqual(float(mask[0, 0, 7, 4]), 0.0)

    def test_control_is_budget_matched_without_contradiction(self) -> None:
        rows, swaps, _ = build_quartet(
            self.donor, self.recipient, self.normal, "A", 8, "control"
        )
        self.assertEqual(len(rows), 4)
        self.assertEqual(swaps, [])
        self.assertEqual(rows[1]["prompt"], rows[3]["prompt"])
        self.assertEqual(rows[1]["target"], rows[3]["target"])
        self.assertEqual(rows[2]["target_label"], "B")

    def test_inputs_are_not_mutated(self) -> None:
        originals = copy.deepcopy((self.donor, self.recipient, self.normal))
        build_quartet(self.donor, self.recipient, self.normal, "A", 9, "paired_swap")
        self.assertEqual((self.donor, self.recipient, self.normal), originals)

    def test_rejects_no_contrast(self) -> None:
        with self.assertRaisesRegex(ValueError, "A/B contrast"):
            build_quartet(self.donor, self.recipient, self.normal, "B", 10, "paired_swap")

    def test_residual_patch_is_local_and_differentiable(self) -> None:
        hidden = torch.arange(4 * 5 * 3, dtype=torch.float32).reshape(4, 5, 3).requires_grad_()
        positions = torch.tensor([1, 2, 3, 2])
        patched = patch_state_residual(hidden, positions, torch.tensor([1]), torch.tensor([3]))
        expected = hidden.detach().clone()
        expected[3, 2] = expected[1, 2]
        torch.testing.assert_close(patched.detach(), expected)
        mask = torch.ones_like(hidden, dtype=torch.bool)
        mask[3, 2] = False
        torch.testing.assert_close(patched.detach()[mask], hidden.detach()[mask])
        patched[3, 2].sum().backward()
        self.assertEqual(float(hidden.grad[1, 2].sum()), 3.0)
        self.assertEqual(float(hidden.grad[3, 2].sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
