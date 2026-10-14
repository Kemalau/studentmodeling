from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from render_m4_causal_bootstrap_rehearsal import SPECS, load, render  # noqa: E402
from validate_m4_causal_bootstrap_rehearsal_amlt_config import validate  # noqa: E402
from render_m4_donor_preemptible import render as render_preemptible  # noqa: E402
from validate_m4_donor_preemptible_amlt_config import (  # noqa: E402
    validate as validate_preemptible,
)


class CausalBootstrapRehearsalTest(unittest.TestCase):
    def setUp(self) -> None:
        self.value = render(load(ROOT / "amlt/feeds_m4_causal_rehearsal_sum_20260731.yaml"))

    def test_rendered_jobs_pass_fail_closed_validator(self) -> None:
        validate(self.value)
        self.assertEqual({job["name"] for job in self.value["jobs"]}, {
            SPECS["bootstrap_sum_w1"]["job"],
            SPECS["bootstrap_sum_w4"]["job"],
        })
        for job in self.value["jobs"]:
            command = str(job["command"])
            self.assertIn("checkpoint-round-*", command)
            self.assertIn("latest_checkpoint", command)
            self.assertIn("sort -V", command)
            self.assertNotIn(
                "branch in {'repeat100_sum_w1', 'repeat100_sum_w4'}", command
            )

    def test_bootstrap_or_gpu_drift_fails(self) -> None:
        drift = copy.deepcopy(self.value)
        drift["jobs"][0]["command"] = str(drift["jobs"][0]["command"]).replace(
            "--causal-bootstrap-native-weight 0.0",
            "--causal-bootstrap-native-weight 1.0",
        )
        with self.assertRaisesRegex(ValueError, "Command drift"):
            validate(drift)

    def test_native_anchor_dense_jobs_are_registered(self) -> None:
        value = render(
            load(ROOT / "amlt/feeds_m4_causal_rehearsal_sum_20260731.yaml"),
            ["bootstrap_native25_sum_w1", "bootstrap_native25_sum_w4"],
        )
        validate(value, {job["name"] for job in value["jobs"]})
        for job in value["jobs"]:
            self.assertIn(
                "--causal-bootstrap-native-weight 0.25", job["command"]
            )

    def test_persistent_donor_directional_jobs_are_registered(self) -> None:
        value = render(
            load(ROOT / "amlt/feeds_m4_causal_rehearsal_sum_20260731.yaml"),
            ["donor_directional_sum_w1", "donor_directional_sum_w4"],
        )
        validate(value, {job["name"] for job in value["jobs"]})
        for job in value["jobs"]:
            command = str(job["command"])
            self.assertIn("--causal-objective-mode donor_directional", command)
            self.assertIn("--causal-bootstrap-native-weight 0.0", command)

    def test_preemptible_donor_only_changes_checkpoint_interval(self) -> None:
        donor = render(
            load(ROOT / "amlt/feeds_m4_causal_rehearsal_sum_20260731.yaml"),
            ["donor_directional_sum_w1", "donor_directional_sum_w4"],
        )
        value = render_preemptible(donor)
        validate_preemptible(value)
        for job in value["jobs"]:
            command = str(job["command"])
            self.assertIn("--save-steps 50", command)
            self.assertIn("int(receipt['round']) % 50 == 0", command)
            self.assertIn("--diagnostic-stop-after-round 500", command)
            self.assertIn("--causal-objective-mode donor_directional", command)

    def test_preemptible_validator_accepts_explicit_one_job_recovery(self) -> None:
        donor = render(
            load(ROOT / "amlt/feeds_m4_causal_rehearsal_sum_20260731.yaml"),
            ["donor_directional_sum_w1", "donor_directional_sum_w4"],
        )
        value = render_preemptible(donor)
        selected = value["jobs"][1]["name"]
        value["jobs"] = [value["jobs"][1]]
        validate_preemptible(value, {selected})

    def test_donor_devicefix_restart_reuses_submitted_recovery(self) -> None:
        text = (ROOT / "scripts/watch_m4_donor_round500_c4_devicefix.sh").read_text()
        self.assertIn('candidate_experiment', text)
        self.assertIn('.submission.txt', text)
        self.assertIn('training_experiment="$candidate_experiment"', text)

    def test_watcher_compares_both_pair8_curricula_before_promotion(self) -> None:
        text = (ROOT / "scripts/watch_m4_causal_bootstrap_rehearsal.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("causal_bootstrap_pair8_native25", text)
        self.assertIn("select_m4_causal_bootstrap_pair8.py", text)
        self.assertIn("bootstrap_native25_sum_w1", text)
        self.assertIn("bootstrap_native25_sum_w4", text)
        self.assertIn("causal_bootstrap_pair8_balanced", text)
        self.assertIn("causal_bootstrap_pair8_native25_balanced", text)
        self.assertIn("feeds_m4_causal_bootstrap_rehearsal_selected_20260731.yaml", text)
        self.assertIn('--branches "${expected_branches[@]}"', text)
        self.assertIn('early_stopped.json', text)
        self.assertIn('scientifically early-stopped', text)

    def test_directional_tail_watcher_recovers_and_publishes_comparison(self) -> None:
        text = (
            ROOT / "scripts/watch_m4_causal_bootstrap_directional_tail.sh"
        ).read_text(encoding="utf-8")
        for required in (
            "feeds_submission_global.lock",
            "terminal_nonpass_job_names",
            "for recovery_wave in 0 1 2 3",
            "bootstrap_balanced_sum_w1",
            "bootstrap_balanced_sum_w4",
            "summarize_m4_causal_strength_branches.py",
            'touch "$state_dir/complete"',
        ):
            self.assertIn(required, text)

    def test_donor_directional_watcher_is_parallel_and_recoverable(self) -> None:
        text = (
            ROOT / "scripts/watch_m4_causal_donor_directional.sh"
        ).read_text(encoding="utf-8")
        for required in (
            "donor_directional_sum_w1",
            "donor_directional_sum_w4",
            "feeds_submission_global.lock",
            "for recovery_wave in 0 1 2 3",
            "terminal_nonpass_job_names",
            "summarize_m4_causal_strength_branches.py",
            'touch "$state_dir/complete"',
        ):
            self.assertIn(required, text)


if __name__ == "__main__":
    unittest.main()
