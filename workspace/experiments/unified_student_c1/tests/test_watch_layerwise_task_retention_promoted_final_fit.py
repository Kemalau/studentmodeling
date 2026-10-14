from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class PromotedFinalFitWatcherTest(unittest.TestCase):
    def test_watcher_is_gate_bound_and_recovers(self) -> None:
        text = (
            ROOT / "scripts/watch_layerwise_task_retention_promoted_final_fit.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("x['any_all_gates_pass'] is True", text)
        self.assertIn("row['all_gates_pass'] is True", text)
        self.assertIn("az storage blob upload", text)
        self.assertIn("cmp -s", text)
        self.assertIn("validate_layerwise_task_retention_promoted_final_fit_amlt.py", text)
        self.assertIn("amlt run", text)
        self.assertIn("wave=$((wave+1))", text)
        self.assertIn('max_recovery_waves="${MAX_RECOVERY_WAVES:-32}"', text)
        self.assertIn("wave < max_recovery_waves", text)
        self.assertIn("all_expected_jobs_pass", text)
        self.assertIn("promoted_final_fit_receipt.json", text)
        self.assertIn("final_fit_receipt.json", text)
        self.assertIn("receipt upload failed", text)
        self.assertIn("remote promoted receipt differs", text)
        self.assertIn("artifact_manifest.json", text)
        self.assertIn("--state-kd", text)
        self.assertIn("'standard','state_kd','state_path'", text)
        self.assertIn("--eedi-full-state-kd", text)
        self.assertIn("total_candidate_count']==24", text)
        self.assertIn("--equal-state-kd", text)
        self.assertIn("--available", text)
        self.assertIn("completed_candidates_only", text)
        self.assertIn("all candidate diagnostics terminated", text)
        self.assertIn('! -f "$required_dir/deferred"', text)
        self.assertIn('! -f "$required_dir/superseded"', text)
        self.assertIn("--eedi-full-open-t3", text)
        self.assertIn("--eedi-full-open-t3-selective-kd", text)
        self.assertIn("--equal-state-path-open-t3", text)
        self.assertIn("--equal-state-kd-open-t3-lr-scan", text)
        self.assertIn("--equal-state-kd-open-t3-lr1e6", text)
        self.assertIn("--open-t3-consolidation", text)
        self.assertIn("equal_t3_t6_open_t3_two_stage", text)
        self.assertIn("promotion_barrier_dirs", text)
        self.assertIn("promotion_barrier_complete", text)
        self.assertIn("open_t3_selective_kd", text)
        self.assertIn("wait_local_result_space", text)
        self.assertIn("12 * 1024 * 1024 * 1024", text)
        self.assertIn("local_result_space.receipt", text)
        launcher = (
            ROOT / "scripts/run_layerwise_task_retention_promoted_final_fit.sh"
        ).read_text(encoding="utf-8")
        self.assertIn(
            'f"{x[\'branch\']}__{x[\'selected_loss_profile\']}__lr{selected_lr:.0e}"',
            launcher,
        )
        self.assertIn("train_recurrent_student_m4_promoted_open_t3.py", launcher)
        self.assertIn(
            "train_recurrent_student_m4_promoted_open_t3_selective_kd.py",
            launcher,
        )
        self.assertIn(
            "train_recurrent_student_m4_promoted_open_t3_two_stage.py",
            launcher,
        )


if __name__ == "__main__":
    unittest.main()
