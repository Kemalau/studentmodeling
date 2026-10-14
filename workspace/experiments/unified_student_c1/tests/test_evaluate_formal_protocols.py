from __future__ import annotations

import sys
import unittest
import inspect
from pathlib import Path

import torch


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_formal_protocols import (  # noqa: E402
    classify_rows_bottleneck,
    evaluation_shard,
    generation_metrics,
    incremental_bottleneck_attention_mask,
    prepare_bottleneck_prompt,
)


class FormalBottleneckProtocolTest(unittest.TestCase):
    def test_natural_t3_metrics_do_not_treat_missing_legacy_tags_as_exact(self) -> None:
        records = [
            {
                "raw_output": (
                    "<label>wrong.label</label>\n"
                    "<diagnosis>A different diagnosis</diagnosis>\n"
                    "<evidence>Different evidence</evidence>"
                ),
                "target": (
                    "<label>gold.label</label>\n"
                    "<diagnosis>The gold diagnosis</diagnosis>\n"
                    "<evidence>Gold evidence</evidence>"
                ),
            }
        ]
        metrics = generation_metrics(records, "diagnose_misconception")
        self.assertEqual(metrics["format_valid_rate"], 1.0)
        self.assertEqual(metrics["natural_schema_valid_rate"], 1.0)
        self.assertEqual(metrics["normalized_exact_match"], 0.0)
        self.assertEqual(metrics["canonical_label_exact_match"], 0.0)
        self.assertEqual(
            metrics["exact_match_role"], "auxiliary_only_not_authoritative"
        )

    def test_missing_generation_tags_are_not_exact_matches(self) -> None:
        metrics = generation_metrics(
            [{"raw_output": "malformed", "target": "also malformed"}],
            "diagnose_misconception",
        )
        self.assertEqual(metrics["format_valid_rate"], 0.0)
        self.assertEqual(metrics["normalized_exact_match"], 0.0)

    def test_evaluation_shards_are_stable_disjoint_and_complete(self) -> None:
        rows = [{"example_id": f"row-{index}"} for index in range(17)]
        shards = [evaluation_shard(rows, index, 4) for index in range(4)]
        reordered = [evaluation_shard(list(reversed(rows)), index, 4) for index in range(4)]
        self.assertEqual(
            [[row["example_id"] for row in shard] for shard in shards],
            [[row["example_id"] for row in shard] for shard in reordered],
        )
        flattened = [row["example_id"] for shard in shards for row in shard]
        self.assertEqual(len(flattened), len(set(flattened)))
        self.assertEqual(set(flattened), {row["example_id"] for row in rows})

    def test_evaluation_shard_rejects_partial_coordinates(self) -> None:
        with self.assertRaisesRegex(ValueError, "provided together"):
            evaluation_shard([{"example_id": "x"}], 0, None)

    def test_generation_declares_task_specific_closing_tag_stops(self) -> None:
        source = (SCRIPT_DIR / "evaluate_formal_protocols.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('"predict_response": ("</predicted_response>",)', source)
        self.assertIn('"diagnose_misconception": ("</misconception>",)', source)

    def test_masked_classifier_is_not_the_open_cached_scorer(self) -> None:
        source = inspect.unwrap(classify_rows_bottleneck).__code__.co_names
        self.assertIn("build_bottleneck_attention_mask", source)
        self.assertNotIn("score_rows", source)

    def test_incremental_mask_blocks_only_raw_history(self) -> None:
        mask = incremental_bottleneck_attention_mask(
            key_value_length=9,
            history_start=2,
            state_pos=5,
            device=torch.device("cpu"),
        )
        self.assertEqual(tuple(mask.shape), (1, 1, 1, 9))
        blocked = torch.finfo(torch.float32).min
        self.assertTrue(torch.all(mask[..., 2:5] == blocked))
        self.assertTrue(torch.all(mask[..., :2] == 0))
        self.assertTrue(torch.all(mask[..., 5:] == 0))

    def test_prompt_truncation_rejects_removed_state_coordinates(self) -> None:
        class CharacterTokenizer:
            def __call__(self, text, add_special_tokens=False):
                del add_special_tokens
                return {"input_ids": list(range(len(text)))}

        prompt = "prefix<student_history>secret</student_history><STUDENT_STATE>query"
        with self.assertRaisesRegex(ValueError, "removed bottleneck coordinates"):
            prepare_bottleneck_prompt(
                CharacterTokenizer(), prompt, "S", max_length=5
            )


if __name__ == "__main__":
    unittest.main()
