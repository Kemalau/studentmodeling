from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from validate_checkpoint_t3_semantic_result import (  # noqa: E402
    EXPECTED_SUPPORT,
    FORMAL_PANEL_SUPPORT,
    validate,
)


class CheckpointT3SemanticResultTest(unittest.TestCase):
    def fixture(self, accuracy: float = 0.5) -> dict:
        return {
            "generation_rows": 75,
            "judged_rows": 75,
            "judge_errors": 0,
            "judge_backend": "trapi",
            "judge_rubric": "t3-semantic-equivalence-v2-substantive-schema",
            "judge_rubric_sha256": "a" * 64,
            "metrics_by_group": {
                name: {
                    "n": support,
                    "llm_judge_n": support,
                    "llm_judge_accuracy": accuracy,
                    "llm_judge_raw_semantic_accuracy": accuracy,
                    "llm_judge_schema_valid_n": support,
                    "llm_judge_mean_confidence": 0.9,
                }
                for name, support in EXPECTED_SUPPORT.items()
            },
        }

    def write(self, root: Path, value: dict) -> Path:
        path = root / "metrics.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def test_complete_fixed_panel_passes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = validate(self.write(Path(directory), self.fixture()))
            self.assertTrue(result["pass"])
            self.assertEqual(result["support"], 75)
            self.assertEqual(result["checkpoint_round"], 1000)
            self.assertEqual(result["formal_panel_support"], 1494)
            self.assertEqual(
                result["weighting_scheme"], "formal_panel_group_prevalence_v1"
            )
            self.assertTrue(result["group_floor_pass"])

    def test_formal_prevalence_prevents_synthetic_groups_hiding_eedi_collapse(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            value = self.fixture()
            groups = value["metrics_by_group"]
            groups[
                "diagnose_misconception/eedi_mining/open_taxonomy_generation"
            ]["llm_judge_accuracy"] = 0.0
            groups[
                "diagnose_misconception/malrule/synthetic_cross_template_open_generation"
            ]["llm_judge_accuracy"] = 0.8
            groups[
                "diagnose_misconception/malrule/synthetic_same_template_open_generation"
            ]["llm_judge_accuracy"] = 0.8
            result = validate(self.write(Path(directory), value))
            self.assertFalse(result["pass"])
            self.assertFalse(result["group_floor_pass"])
            expected = (0.8 * 300 + 0.8 * 300) / sum(FORMAL_PANEL_SUPPORT.values())
            self.assertAlmostEqual(result["semantic_accuracy"], expected)

    def test_semantic_collapse_is_scientific_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = validate(self.write(Path(directory), self.fixture(0.1)))
            self.assertFalse(result["pass"])

    def test_missing_group_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            value = self.fixture()
            value["metrics_by_group"].pop(next(iter(EXPECTED_SUPPORT)))
            with self.assertRaisesRegex(ValueError, "group mismatch"):
                validate(self.write(Path(directory), value))

    def test_watcher_is_local_judging_only(self) -> None:
        text = (ROOT / "scripts/watch_layerwise_checkpoint1000_t3_semantic_judge.sh").read_text()
        self.assertIn("predictions.jsonl", text)
        self.assertIn("full75", text)
        self.assertIn("validate_checkpoint_t3_semantic_result.py", text)
        self.assertNotIn("amlt run", text)
        self.assertNotIn("amlt cancel", text)

    def test_formal_owner_preserves_unredacted_predictions_for_local_judge(self) -> None:
        text = (ROOT / "scripts/watch_layerwise_checkpoint250_formal_screen.sh").read_text()
        self.assertIn("--include predictions.jsonl", text)
        self.assertIn("--include predictions_redacted.jsonl", text)


if __name__ == "__main__":
    unittest.main()
