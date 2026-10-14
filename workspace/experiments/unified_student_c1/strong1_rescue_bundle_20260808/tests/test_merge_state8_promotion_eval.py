from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from merge_state8_promotion_eval import merge_c4  # noqa: E402


class MergeState8PromotionEvalTest(unittest.TestCase):
    def test_c4_requires_exact_seed_set_and_computes_replication(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = []
            for seed, effect, passed in ((13, 0.2, True), (42, 0.18, True), (97, 0.01, False)):
                path = Path(temporary) / f"{seed}.json"
                path.write_text(json.dumps({
                    "pair_seed": seed,
                    "state_shape": [64, 8, 5120],
                    "raw_history_visible_to_decoder": False,
                    "persistent_history_kv_cache": False,
                    "summary": {
                        "clean_n": 80,
                        "D_specific": effect,
                        "scientific_gate_pass": passed,
                        "clean": {name: {"flip_to_A_rate": value} for name, value in {
                            "swap": .4, "random": .2, "samemal": .05, "noop": 0.0
                        }.items()},
                    },
                }))
                paths.append(path)
            result = merge_c4(paths)
            self.assertTrue(result["replication_gate_pass"])
            self.assertEqual(result["passing_seeds"], 2)

    def test_c4_rejects_wrong_state_capacity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = []
            for seed in (13, 42, 97):
                path = Path(temporary) / f"{seed}.json"
                path.write_text(json.dumps({
                    "pair_seed": seed,
                    "state_shape": [64, 5120],
                    "raw_history_visible_to_decoder": False,
                    "persistent_history_kv_cache": False,
                    "summary": {"clean_n": 1, "D_specific": 0.2,
                                "scientific_gate_pass": True,
                                "clean": {name: {"flip_to_A_rate": 0.0} for name in
                                          ("swap", "random", "samemal", "noop")}},
                }))
                paths.append(path)
            with self.assertRaisesRegex(ValueError, "state shape"):
                merge_c4(paths)

    def test_c4_rejects_persistent_history_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = []
            for seed in (13, 42, 97):
                path = Path(temporary) / f"{seed}.json"
                path.write_text(json.dumps({
                    "pair_seed": seed,
                    "state_shape": [64, 8, 5120],
                    "raw_history_visible_to_decoder": False,
                    "persistent_history_kv_cache": seed == 42,
                    "summary": {"clean_n": 1, "D_specific": 0.2,
                                "scientific_gate_pass": True,
                                "clean": {name: {"flip_to_A_rate": 0.0} for name in
                                          ("swap", "random", "samemal", "noop")}},
                }))
                paths.append(path)
            with self.assertRaisesRegex(ValueError, "persistent history KV cache"):
                merge_c4(paths)


if __name__ == "__main__":
    unittest.main()
