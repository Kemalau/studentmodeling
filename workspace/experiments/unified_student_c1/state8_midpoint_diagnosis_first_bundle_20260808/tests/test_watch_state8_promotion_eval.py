from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class WatchState8PromotionEvalTest(unittest.TestCase):
    def test_controller_is_hash_bound_sharded_and_recoverable(self) -> None:
        text = (ROOT / "scripts/watch_state8_promotion_eval.sh").read_text()
        for fragment in (
            "state8-qualified-teacher-curriculum-20260805",
            "state8-qualified-promotion-eval-20260805",
            "state8_qualified_teacher_pipeline_20260805",
            "PROMOTION_VARIANT:-t3_heavy",
            "state8_qualified_balanced_promotion_eval_20260805",
            "state8_qualified_balanced_teacher_pipeline_20260805",
            'train_artifact_dir="$train_dir/stage_b_download"',
            'train_artifact_dir="$train_dir/download"',
            "state8_qualified_balanced_promotion_gate_v1",
            "state8_qualified_highkd_promotion_eval_20260805",
            "state8_qualified_highkd_teacher_pipeline_20260805",
            "state8_qualified_highkd_promotion_gate_v1",
            "state8-qualified-highkd-promotion-eval-20260805",
            "validate_state8_promotion_eval_amlt.py",
            "terminal_nonpass_job_names",
            "preserve_passed",
            "wave=$((wave+1))",
            "merge_state8_promotion_eval.py",
            "--adapter-sha256",
            "--recurrent-state-sha256",
            "judge_formal_t3_predictions.py",
            "--judge-backend trapi",
            "validate_t3_semantic_result.py",
            "summarize_layerwise_checkpoint250_formal_screen.py",
            "T1_T2_T5_T6_small_screen",
            "replication_gate_pass",
            "scientific_gate_pass",
            "scientific_gate_failed",
            'active="$(cat "$train_dir/active_experiment.txt"',
            'train_active_dir="$train_dir/primary_curriculum"',
            'active="$(cat "$train_active_dir/active_experiment.txt"',
            "upstream controller owns failed-job recovery",
            "has_forward_progress_job",
            "paused_job_names",
            "amlt resume",
            "sys.stdout.write(''.join(value+'\\n' for value in values))",
            "MAX_RECOVERY_WAVES:-24",
        ):
            if fragment == 'active="$(cat "$train_dir/active_experiment.txt"':
                self.assertNotIn(fragment, text)
            else:
                self.assertIn(fragment, text)
        self.assertNotIn("amlt cancel", text)
        self.assertNotIn('fail "qualified state8 training did not pass"', text)
        self.assertLess(
            text.index('exec 8>"$project_dir/results/feeds_submission_global.lock"'),
            text.index('wait_quota "${#names[@]}"'),
        )


if __name__ == "__main__":
    unittest.main()
