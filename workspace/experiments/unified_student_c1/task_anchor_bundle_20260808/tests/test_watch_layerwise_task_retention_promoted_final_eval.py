from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class PromotedFinalEvalWatcherTest(unittest.TestCase):
    def test_full_recoverable_evaluation_and_semantic_gate(self) -> None:
        text = (
            ROOT / "scripts/watch_layerwise_task_retention_promoted_final_eval.sh"
        ).read_text(encoding="utf-8")
        for required in (
            "validate_layerwise_task_retention_promoted_final_eval_amlt.py",
            "all_expected_jobs_pass",
            "terminal_nonpass_job_names",
            "preserve_passed",
            "judge_formal_t3_predictions.py",
            "validate_t3_semantic_result.py",
            "summarize_layerwise_unified_final_eval.py",
            "validate_layerwise_promoted_unified_pipeline.py",
            "overall_completion.json",
            "overall_complete",
        ):
            self.assertIn(required, text)

    def test_stage_markers_are_transactional_around_strict_completion_audit(self) -> None:
        text = (
            ROOT / "scripts/watch_layerwise_task_retention_promoted_final_eval.sh"
        ).read_text(encoding="utf-8")
        marker = 'touch "$state_dir/scientific_gate_pass" "$state_dir/complete"'
        audit = "if ! python scripts/validate_layerwise_promoted_unified_pipeline.py"
        rollback = 'rm -f "$state_dir/scientific_gate_pass" "$state_dir/complete"'
        overall = 'touch "$pipeline/overall_complete"'
        self.assertLess(text.index(marker), text.index(audit))
        self.assertLess(text.index(audit), text.index(rollback))
        self.assertLess(text.index(rollback), text.index(overall))


if __name__ == "__main__":
    unittest.main()
