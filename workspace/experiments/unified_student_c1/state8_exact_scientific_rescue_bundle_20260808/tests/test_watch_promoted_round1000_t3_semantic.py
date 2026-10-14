from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class PromotedRound1000T3SemanticWatcherTest(unittest.TestCase):
    def test_local_independent_semantic_judging_is_fail_closed(self) -> None:
        text = (ROOT / "scripts/watch_promoted_round1000_t3_semantic.sh").read_text()
        for required in (
            "local_round1000_formal_screen",
            "t1_t6_t3/predictions.jsonl",
            "judge_formal_t3_predictions.py",
            "--judge-backend trapi",
            "--max-records 16",
            "full75",
            "validate_checkpoint_t3_semantic_result.py",
            "--checkpoint-round 1000",
            "scientific_gate_failed",
        ):
            self.assertIn(required, text)
        self.assertNotIn("amlt run", text)
        self.assertNotIn("amlt cancel", text)
        self.assertIn('test "$(sha256sum "$eval_jsonl"', text)
        self.assertNotIn('$$(sha256sum', text)


if __name__ == "__main__":
    unittest.main()
