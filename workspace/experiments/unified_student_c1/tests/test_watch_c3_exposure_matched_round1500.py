from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class WatchC3ExposureMatchedTest(unittest.TestCase):
    def test_controller_recovers_siblings_and_separates_completion_from_effect(self) -> None:
        source = (ROOT / "scripts/watch_c3_exposure_matched_round1500.sh").read_text()
        self.assertIn("c3-exposure-matched-round1500-seed20260804", source)
        self.assertIn("c3_exposure_matched_round1500_hashfix_v2", source)
        self.assertIn("c3-hashfix-v2-round1500", source)
        self.assertIn("has_active_job", source)
        self.assertIn("all_expected_jobs_pass", source)
        self.assertIn("validate_c3_exposure_matched_amlt.py", source)
        self.assertIn("feeds_submission_global.lock", source)
        self.assertIn("failed-sibling recovery", source)
        self.assertIn("select_azure_profile", source)
        self.assertIn("auth_required", source)
        self.assertIn("C3_MAX_RECOVERY_WAVES:-12", source)
        self.assertIn("C3_REQUIRED_GPUS:-4", source)
        self.assertIn("C3_EXPECTED_TOPOLOGY:-world4", source)
        self.assertIn("classify_experiment_status", source)
        self.assertIn("waiting without submitting", source)
        self.assertIn("count_active_a100_requests.py", source)
        self.assertIn("available_user_slots", source)
        self.assertIn("C3_MAX_ACTIVE_ARMS:-3", source)
        self.assertIn("active_owned_arms", source)
        self.assertIn("C3_EXPERIMENT_MAP", source)
        self.assertIn("C3 active-arm cap reached", source)
        self.assertIn("c3_quota_gate_v1", source)
        self.assertIn("insufficient_feeds_standard_quota", source)
        self.assertIn("insufficient_user_a100_slots", source)
        self.assertIn("quota_gate.json", source)
        self.assertIn("Cannot find an experiment", source)
        self.assertIn("Missing therefore consumes no owned slot", source)
        self.assertIn("accepted_after_ambiguous_submit", source)
        self.assertIn("for attempt in 1 2 3 4 5 6", source)
        self.assertIn("Query only this exact immutable name", source)
        self.assertIn("active_a100_requests.json", source)
        self.assertIn("C3_PRIORITY_READY_MARKER", source)
        self.assertIn('sleep_unlocked(){ sleep "$1" 9>&-; }', source)
        self.assertIn('waves["$job"] <= max_recovery_waves', source)
        self.assertIn("deterministic C3 provenance failure", source)
        self.assertIn("source pin must be repaired before resubmission", source)
        self.assertIn('destination="$state_dir/preserved/$1"', source)
        self.assertIn('staging="/tmp/c3-preserve-$1-$$"', source)
        self.assertIn("Diagnostic warm-start real-data order drift", source)
        self.assertIn("/home/v-yuejiali/.azure", source)
        self.assertIn("/home/v-yuejiali/.azure-studentmodeling-feeds", source)
        self.assertIn("judge_formal_t3_predictions.py", source)
        self.assertIn("summarize_c3_exposure_matched.py", source)
        self.assertIn("scientific_gate_failed", source)
        self.assertIn('touch "$state_dir/complete"', source)

    def test_world2_launcher_pipelines_after_priority_eval_submission(self) -> None:
        source = (ROOT / "scripts/watch_c3_exposure_matched_world2_20260809.sh").read_text()
        self.assertIn(
            "submitted.state8-exact-diagnosis-checkpoint5000-eval-20260809",
            source,
        )
        self.assertNotIn(
            'C3_PRIORITY_READY_MARKER="$project/results/'
            'state8_exact_diagnosis_checkpoint5000_eval_20260809/remote_passed"',
            source,
        )
        self.assertIn("C3_REQUIRED_GPUS=2", source)
        self.assertIn("C3_MAX_ACTIVE_ARMS=2", source)
        self.assertIn("experiments.promoted-order-v1.tsv", source)
        self.assertIn("c3-world2-promoted-order-v1", source)


if __name__ == "__main__":
    unittest.main()
