from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_layerwise_task_retention_promotion import build, promote_order, retention_stratum  # noqa: E402


def row(task: str, source: str, target: str = "") -> dict[str, str]:
    return {"task": task, "source": source, "target": target}


class TaskRetentionPromotionTest(unittest.TestCase):
    def test_seen_rows_are_replaced_without_changing_curriculum_support(self) -> None:
        rows = [
            row("t1", "a"),
            row("t3", "eedi"),
            row("t1", "a"),
            row("t1", "a"),
            row("t1", "a"),
            row("t6", "fa", "persistent"),
            row("t3", "malrule"),
            row("t3", "eedi"),
            row("t6", "fa", "transient"),
        ]
        result, promoted, replacements = promote_order(
            list(range(len(rows))), rows, [1, 5, 6], prefix_count=4
        )
        self.assertEqual(promoted, [7, 5, 6])
        self.assertEqual(result[:4], [0, 1, 2, 3])
        self.assertEqual(result[4:7], promoted)
        self.assertEqual(sorted(result), list(range(len(rows))))
        self.assertEqual(len(replacements), 1)
        self.assertEqual(
            [retention_stratum(rows[index]) for index in promoted],
            [retention_stratum(rows[index]) for index in [1, 5, 6]],
        )

    def test_missing_unseen_match_fails_closed(self) -> None:
        rows = [row("t3", "eedi"), row("t1", "a"), row("t1", "a")]
        with self.assertRaisesRegex(ValueError, "No unseen replacement"):
            promote_order([0, 1, 2], rows, [0], prefix_count=1)

    def test_source_order_must_be_a_complete_permutation(self) -> None:
        rows = [row("t1", "a"), row("t1", "a")]
        with self.assertRaisesRegex(ValueError, "complete permutation"):
            promote_order([0, 0], rows, [1], prefix_count=1)

    def test_equal_open_t3_override_is_allowed_but_unrelated_override_fails(self) -> None:
        source = ROOT / "manifests/layerwise_task_retention_equal_t3_t6_seed42_v1.json"
        checkpoint = ROOT / "results/layerwise_unified_final_pipeline_20260731/checkpoint750_local_audit/recurrent_state.pt"
        train = ROOT / "data/processed/unified_recurrent_full259845_u4_v1/final_fit_trainvalid.jsonl"
        formal = ROOT / "data/processed/recurrent_formal_evaluation_fixedu4_v2/test.jsonl"
        if not checkpoint.is_file():
            self.skipTest("local round-750 audit artifact is unavailable")
        value = build(
            train_jsonl=train,
            formal_eval_jsonl=formal,
            checkpoint=checkpoint,
            diagnostic_manifest=source,
            branch_override="equal_t3_t6_open_t3",
        )
        self.assertEqual(value["branch"], "equal_t3_t6_open_t3")
        self.assertEqual(value["training_view"], "open_t3_selective_kd")
        self.assertEqual(value["promoted_rows"], 1000)
        with self.assertRaisesRegex(ValueError, "Unsupported promotion branch override"):
            build(
                train_jsonl=train,
                formal_eval_jsonl=formal,
                checkpoint=checkpoint,
                diagnostic_manifest=source,
                branch_override="unrelated",
            )

    def test_two_stage_curriculum_metadata_is_preserved(self) -> None:
        source = ROOT / "manifests/layerwise_open_t3_two_stage_equal_t3_t6_seed42_v1.json"
        checkpoint = ROOT / "results/layerwise_unified_final_pipeline_20260731/checkpoint750_local_audit/recurrent_state.pt"
        train = ROOT / "data/processed/unified_recurrent_full259845_u4_v1/final_fit_trainvalid.jsonl"
        formal = ROOT / "data/processed/recurrent_formal_evaluation_fixedu4_v2/test.jsonl"
        if not checkpoint.is_file():
            self.skipTest("local round-750 audit artifact is unavailable")
        value = build(
            train_jsonl=train,
            formal_eval_jsonl=formal,
            checkpoint=checkpoint,
            diagnostic_manifest=source,
            branch_override="equal_t3_t6_open_t3",
        )
        self.assertEqual(value["promoted_rows"], 2000)
        self.assertEqual(
            value["curriculum_mode"],
            "open_t3_formation_then_causal_task_consolidation",
        )
        self.assertEqual(
            [phase["name"] for phase in value["curriculum_phases"]],
            ["open_t3_formation", "causal_task_consolidation"],
        )
        self.assertTrue(value["curriculum_source_manifest"].endswith(source.name))

    def test_two_stage_branch_override_is_distinct_and_exact_epoch(self) -> None:
        source = ROOT / "manifests/layerwise_open_t3_two_stage_equal_t3_t6_seed42_v1.json"
        checkpoint = ROOT / "results/layerwise_unified_final_pipeline_20260731/checkpoint750_local_audit/recurrent_state.pt"
        train = ROOT / "data/processed/unified_recurrent_full259845_u4_v1/final_fit_trainvalid.jsonl"
        formal = ROOT / "data/processed/recurrent_formal_evaluation_fixedu4_v2/test.jsonl"
        if not checkpoint.is_file():
            self.skipTest("local round-750 audit artifact is unavailable")
        value = build(
            train_jsonl=train,
            formal_eval_jsonl=formal,
            checkpoint=checkpoint,
            diagnostic_manifest=source,
            branch_override="equal_t3_t6_open_t3_two_stage",
        )
        self.assertEqual(value["branch"], "equal_t3_t6_open_t3_two_stage")
        self.assertEqual(value["promoted_rows"], 2000)
        self.assertEqual(len(value["real_order"]), 259845)
        self.assertEqual(len(set(value["real_order"])), 259845)
        self.assertEqual(value["training_view"], "open_t3_selective_kd")


if __name__ == "__main__":
    unittest.main()
