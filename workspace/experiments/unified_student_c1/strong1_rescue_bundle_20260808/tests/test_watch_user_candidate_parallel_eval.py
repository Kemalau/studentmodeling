from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class WatchUserCandidateParallelEvalTest(unittest.TestCase):
    def test_full_support_and_semantic_gate_are_required(self) -> None:
        source = (ROOT / "scripts/watch_user_candidate_parallel_eval.sh").read_text()
        self.assertIn('"T1":10913,"T2":3693,"T3":1494,"T5":4600,"T6":976', source)
        self.assertIn("judge_formal_t3_predictions.py", source)
        self.assertIn("validate_t3_semantic_result.py", source)
        self.assertIn("full1494", source)
        self.assertIn("t3_semantic_gate.json", source)
        self.assertIn("scientific_gate_failed", source)
        self.assertIn("a368cefb0c49cb6a305ad1a892808361fba2d44d555a0a28ea64e56babe32525", source)
        self.assertIn("terminal_nonpass_job_names", source)
        self.assertIn("preserve_passed", source)
        self.assertIn("--include '*.jsonl'", source)
        self.assertIn("failed-sibling recovery", source)
        self.assertIn("validate_user_candidate_parallel_eval_amlt.py", source)
        self.assertIn("feeds_submission_global.lock", source)
        self.assertIn('semantic_dir/judge.lock', source)

    def test_early_semantic_judge_reuses_the_same_lock_and_gate(self) -> None:
        source = (ROOT / "scripts/watch_user_candidate_t3_semantic_early.sh").read_text()
        self.assertIn('semantic_dir/judge.lock', source)
        self.assertIn("partial_pass_download", source)
        self.assertIn("judge_formal_t3_predictions.py", source)
        self.assertIn("validate_t3_semantic_result.py", source)
        self.assertIn("full1494", source)
        self.assertIn("scientific_gate_pass", source)
        self.assertIn("scientific_gate_failed", source)
        self.assertIn("a368cefb0c49cb6a305ad1a892808361fba2d44d555a0a28ea64e56babe32525", source)


if __name__ == "__main__":
    unittest.main()
