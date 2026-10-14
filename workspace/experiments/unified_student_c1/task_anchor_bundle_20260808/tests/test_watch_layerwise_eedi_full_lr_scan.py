from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class LayerwiseEediFullLrScanWatcherTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.text = (ROOT / "scripts" / "watch_layerwise_eedi_full_lr_scan.sh").read_text()

    def test_tracks_and_persists_each_lr_arm_independently(self) -> None:
        self.assertIn('arm_map="$state_dir/arm_experiments.tsv"', self.text)
        self.assertIn('declare -A arm_experiment=()', self.text)
        self.assertIn('save_arm_map', self.text)
        self.assertIn('passed_receipts/$job.complete', self.text)

    def test_queries_one_job_and_recovers_it_without_waiting_for_sibling(self) -> None:
        self.assertIn('amlt status "$experiment" ":$job"', self.text)
        self.assertIn('submit_wave "$wave" "$job"', self.text)
        self.assertIn('has_active_job', self.text)
        self.assertNotIn("grep -Eqi 'running|queued|preparing", self.text)

    def test_preserves_fail_closed_and_idempotent_submission_gates(self) -> None:
        for required in (
            "wait_quota",
            "--dump -t Feeds",
            'validator="${VALIDATOR:-scripts/validate_layerwise_eedi_full_lr_scan_amlt.py}"',
            "classify_experiment_status",
            'exec 8>"$project_dir/results/feeds_submission_global.lock"',
            "terminal_nonpass_job_names",
        ):
            self.assertIn(required, self.text)

    def test_records_seen_and_unseen_eedi_taxonomy_quality(self) -> None:
        self.assertIn("summarize_t3_seen_unseen.py", self.text)
        self.assertIn('predictions_redacted.jsonl', self.text)
        self.assertIn('seen_unseen.json', self.text)


if __name__ == "__main__":
    unittest.main()
