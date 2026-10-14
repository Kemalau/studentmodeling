from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_layerwise_state_bridge import (  # noqa: E402
    activate_decode_adapter,
    closed_position_ids,
    summarize_records,
)


def record(pair: int, condition: str, category: str, prediction: str) -> dict:
    return {
        "pair_idx": pair,
        "condition": condition,
        "category": category,
        "prediction": prediction,
        "margin_A_minus_B": 1.0 if category == "flip_to_A" else -1.0,
    }


class LayerwiseStateBridgeTest(unittest.TestCase):
    def test_cross_adapter_switch_is_explicit(self) -> None:
        class FakeModel:
            def __init__(self):
                self.loaded = []
                self.active = None
                self.training = True

            def load_adapter(self, path, adapter_name, is_trainable):
                self.loaded.append((path, adapter_name, is_trainable))

            def set_adapter(self, name):
                self.active = name

            def eval(self):
                self.training = False

        model = FakeModel()
        name = activate_decode_adapter(model, "capture", "decode")
        self.assertEqual(name, "closed_state_decoder")
        self.assertEqual(
            model.loaded, [("decode", "closed_state_decoder", False)]
        )
        self.assertEqual(model.active, name)
        self.assertFalse(model.training)

    def test_position_alignment_preserves_prefix_and_offsets_suffix(self) -> None:
        compact = closed_position_ids(7, 2, 5, "compact", torch.device("cpu"))
        aligned = closed_position_ids(
            7, 2, 5, "one_shot_aligned", torch.device("cpu")
        )
        torch.testing.assert_close(compact, torch.tensor([[0, 1, 2, 3, 4, 5, 6]]))
        torch.testing.assert_close(aligned, torch.tensor([[0, 1, 5, 6, 7, 8, 9]]))

    def test_bridge_gate_uses_matched_clean_subset(self) -> None:
        records = []
        for pair in range(1, 51):
            records.extend(
                [
                    record(pair, "native", "stay_B", "B"),
                    record(pair, "noop", "stay_B", "B"),
                    record(pair, "samemal", "stay_B", "B"),
                    record(pair, "random", "stay_B", "B"),
                    record(pair, "swap", "flip_to_A", "A"),
                ]
            )
        result = summarize_records(records)
        self.assertEqual(result["clean_n"], 50)
        self.assertEqual(result["D_specific"], 1.0)
        self.assertTrue(result["scientific_gate_pass"])

    def test_position_alignment_rejects_backwards_jump(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot precede"):
            closed_position_ids(5, 3, 2, "one_shot_aligned", torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
