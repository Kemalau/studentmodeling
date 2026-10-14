from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class TaskRetentionWatcherTest(unittest.TestCase):
    def test_launcher_uses_unique_node_local_diagnostic_workspace(self) -> None:
        text = (ROOT / "scripts/run_layerwise_task_retention_rehearsal.sh").read_text()
        self.assertIn("WANDB_RUN_ID is required for an isolated diagnostic workspace", text)
        self.assertIn("${TMPDIR:-/tmp}/layerwise_task_retention_${BRANCH}_${RUN_ID}", text)
        self.assertIn('test ! -e "$RUN_DIR"', text)
        self.assertNotIn('rm -rf "$RUN_DIR"', text)

    def test_failed_siblings_use_isolated_validated_experiments(self) -> None:
        text = (ROOT / "scripts/watch_layerwise_task_retention_rehearsal.sh").read_text()
        self.assertIn("terminal_nonpass_job_names", text)
        self.assertNotIn("amlt rerun", text)
        self.assertIn('amlt run "$config" "${selectors[@]}" --dump', text)
        self.assertIn("validate_layerwise_task_retention_rehearsal_amlt.py", text)
        self.assertIn("active_jobs.txt", text)
        self.assertIn('all_expected_jobs_pass(sys.argv[1],int(sys.argv[2]))', text)
        self.assertIn('cp -a "$state_dir/preserved/."', text)
        self.assertNotIn("--upload-code", text)
        self.assertIn("--status pass", text)
        self.assertIn("judge_formal_t3_predictions.py", text)
        self.assertIn("summarize_layerwise_task_retention_rehearsal.py", text)


if __name__ == "__main__":
    unittest.main()
