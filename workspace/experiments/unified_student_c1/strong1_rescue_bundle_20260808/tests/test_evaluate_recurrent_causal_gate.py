from __future__ import annotations

import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

import torch


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_recurrent_causal_gate import (  # noqa: E402
    atomic_json,
    atomic_torch_save,
    flush_progress,
    install_termination_flush,
    random_state_like,
    progress_fingerprint,
    summarize_records,
    validate_resumable_records,
)


def record(pair: int, condition: str, category: str, prediction: str) -> dict:
    return {
        "pair_idx": pair,
        "condition": condition,
        "category": category,
        "prediction": prediction,
        "p_A": 0.8 if category == "flip_to_A" else 0.1,
        "p_B": 0.8 if category == "stay_B" else 0.1,
        "margin_A_minus_B": 1.0 if category == "flip_to_A" else -1.0,
    }


class RecurrentCausalGateTest(unittest.TestCase):
    def test_atomic_progress_fsyncs_file_and_parent_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with mock.patch("evaluate_recurrent_causal_gate.os.fsync", wraps=sys.modules["os"].fsync) as synced:
                json_path = root / "records.json"
                atomic_json(json_path, {"records": [1]})
                self.assertEqual(json_path.read_text(encoding="utf-8").strip(), '{\n  "records": [\n    1\n  ]\n}')
                self.assertGreaterEqual(synced.call_count, 2)
            with mock.patch("evaluate_recurrent_causal_gate.os.fsync", wraps=sys.modules["os"].fsync) as synced:
                tensor_path = root / "state_cache.pt"
                atomic_torch_save(tensor_path, {"state": torch.tensor([1.0])})
                self.assertEqual(float(torch.load(tensor_path, weights_only=False)["state"].item()), 1.0)
                self.assertGreaterEqual(synced.call_count, 2)

    def test_cluster_termination_installs_a_progress_flush(self) -> None:
        with mock.patch("evaluate_recurrent_causal_gate.signal.signal") as registered:
            install_termination_flush(True)
        self.assertEqual(registered.call_count, 2)
        signals = {call.args[0] for call in registered.call_args_list}
        self.assertEqual(signals, {2, 15})

    def test_explicit_progress_flush_reaches_the_filesystem(self) -> None:
        with mock.patch("evaluate_recurrent_causal_gate.os.sync") as synced:
            flush_progress()
        synced.assert_called_once_with()

    def test_random_state_is_reproducible_and_norm_matched(self) -> None:
        reference = torch.tensor([[3.0, 4.0]])
        first = random_state_like(reference, 7)
        second = random_state_like(reference, 7)
        torch.testing.assert_close(first, second)
        torch.testing.assert_close(
            torch.linalg.vector_norm(first), torch.linalg.vector_norm(reference)
        )

    def test_specific_gate_uses_one_clean_subset(self) -> None:
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
        self.assertEqual(result["D_damage"], 0.0)
        self.assertTrue(result["scientific_gate_pass"])

    def test_incomplete_conditions_fail_closed(self) -> None:
        records = [record(1, "native", "stay_B", "B")]
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            summarize_records(records)

    def test_large_random_target_migration_fails_even_with_specificity(self) -> None:
        records = []
        for pair in range(1, 101):
            random_category = "flip_to_A" if pair <= 30 else "stay_B"
            swap_category = "flip_to_A" if pair <= 60 else "stay_B"
            records.extend(
                [
                    record(pair, "native", "stay_B", "B"),
                    record(pair, "noop", "stay_B", "B"),
                    record(pair, "samemal", "stay_B", "B"),
                    record(pair, "random", random_category, "A" if random_category == "flip_to_A" else "B"),
                    record(pair, "swap", swap_category, "A" if swap_category == "flip_to_A" else "B"),
                ]
            )
        result = summarize_records(records)
        self.assertAlmostEqual(result["D_specific"], 0.30)
        self.assertFalse(result["checks"]["random_control"])
        self.assertFalse(result["scientific_gate_pass"])

    def test_resumable_records_require_complete_atomic_pairs(self) -> None:
        complete = [record(1, condition, "stay_B", "B") for condition in (
            "native", "noop", "samemal", "random", "swap"
        )]
        self.assertEqual(validate_resumable_records(complete, 2), {1})
        with self.assertRaisesRegex(ValueError, "Incomplete resumable"):
            validate_resumable_records(complete[:-1], 2)

    def test_progress_fingerprint_binds_data_manifest_and_recipe(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data.jsonl"
            manifest = root / "pairs.json"
            data.write_text("{}\n", encoding="utf-8")
            manifest.write_text("[]\n", encoding="utf-8")
            args = Namespace(
                progress_key="checkpoint-sha",
                jsonl=data,
                pair_manifest=manifest,
                state_architecture="layerwise",
                state_token_alias="<|vision_pad|>",
                max_length=1536,
                num_pairs=150,
                seed=42,
            )
            first = progress_fingerprint(args)
            data.write_text('{"changed":true}\n', encoding="utf-8")
            second = progress_fingerprint(args)
            self.assertNotEqual(first["jsonl_sha256"], second["jsonl_sha256"])


if __name__ == "__main__":
    unittest.main()
