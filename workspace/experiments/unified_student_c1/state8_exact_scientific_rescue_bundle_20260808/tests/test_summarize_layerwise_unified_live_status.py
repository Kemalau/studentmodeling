from __future__ import annotations

import json
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from summarize_layerwise_unified_live_status import CONTROLLERS, summarize  # noqa: E402


class LiveStatusTest(unittest.TestCase):
    def test_round500_c4_controller_is_part_of_completion_visibility(self) -> None:
        self.assertIn("checkpoint500_c4", CONTROLLERS)
        self.assertIn("promoted_progress", CONTROLLERS)
        self.assertIn("checkpoint500_causal_rescue_recovery", CONTROLLERS)
        self.assertIn("known_oom_row_alignedmask", CONTROLLERS)
        self.assertIn("checkpoint1000_t3_semantic_judge", CONTROLLERS)
        self.assertIn("task_retention_rehearsal", CONTROLLERS)
        self.assertIn("eedi_state_path_ab", CONTROLLERS)
        self.assertIn("state_kd", CONTROLLERS)
        self.assertIn("equal_state_path", CONTROLLERS)
        self.assertIn("equal_state_kd", CONTROLLERS)
        self.assertIn("equal_state_path_open_t3", CONTROLLERS)
        self.assertIn("equal_state_kd_open_t3_lr_scan", CONTROLLERS)
        self.assertIn("equal_state_kd_open_t3_lr1e6", CONTROLLERS)
        self.assertIn("open_t3_consolidation", CONTROLLERS)
        self.assertIn("staged_task_retention", CONTROLLERS)
        self.assertIn("staged_state_path", CONTROLLERS)
        self.assertIn("eedi_full_lr_scan", CONTROLLERS)
        self.assertIn("eedi_full_state_kd_lr_scan", CONTROLLERS)
        self.assertIn("eedi_full_open_t3_lr_scan", CONTROLLERS)
        self.assertIn("eedi_full_open_t3_selective_kd_lr_scan", CONTROLLERS)
        self.assertIn("eedi_open_t3_t6_repair", CONTROLLERS)
        self.assertIn(
            "eedi_full_open_t3_selective_kd_lr_scan/local_lr2e6_round3800_preflight",
            CONTROLLERS,
        )
        self.assertIn(
            "eedi_full_open_t3_selective_kd_lr_scan/local_lr2e6_round3950_c4",
            CONTROLLERS,
        )
        self.assertIn("task_retention_promoted_final_fit", CONTROLLERS)
        self.assertIn(
            "task_retention_promoted_final_fit/local_round1000_c4", CONTROLLERS
        )
        self.assertIn(
            "task_retention_promoted_final_fit/round1000_c4_n150_multiseed",
            CONTROLLERS,
        )
        self.assertIn(
            "task_retention_promoted_final_fit/local_round1000_t3_semantic",
            CONTROLLERS,
        )
        self.assertIn("task_retention_promoted_final_eval", CONTROLLERS)
        self.assertIn("task_retention_promoted_t3_semantic_judge", CONTROLLERS)
        self.assertIn("t3_full_candidate_screen_20260804/lr5e6_statepath", CONTROLLERS)
        self.assertIn("cf1_fulltask_crossadapter_full_t3_20260804", CONTROLLERS)
        self.assertIn("selective_kd_lr2e6_full_t3_20260804", CONTROLLERS)
        self.assertIn("current_lr2e6_fulltask_crossadapter_full_t3_20260804", CONTROLLERS)
        self.assertIn("current_lr2e6_direct_teacher_t3_20260804", CONTROLLERS)
        self.assertIn("t3_direct_teacher_crossmatrix_eedi75_20260804", CONTROLLERS)
        self.assertIn("external_state_teacher_eedi_full_20260804", CONTROLLERS)
        self.assertIn("gold_semantic_teacher_eedi_full_20260804", CONTROLLERS)
        self.assertIn("checkpoint750_local_audit/formal_full", CONTROLLERS)

    def fixture(self, base: Path, now: datetime) -> Path:
        root = base / "layerwise_unified_final_pipeline_20260731"
        for name in CONTROLLERS:
            directory = root / name
            directory.mkdir(parents=True)
            (directory / "controller.heartbeat").write_text(now.isoformat())
        (root / "final_fit/active_experiment.txt").write_text("exp")
        (root / "progress/progress.json").write_text(json.dumps({
            "experiment": "exp",
            "round": 300,
            "total_rounds": 16241,
            "amlt_status": "running",
            "observed_at": now.isoformat(),
            "health": {"healthy": True},
        }))
        return root

    def test_previous_experiment_progress_is_not_attributed_to_active_run(self) -> None:
        now = datetime(2026, 8, 1, 6, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as temporary:
            root = self.fixture(Path(temporary), now)
            progress = root / "progress/progress.json"
            value = json.loads(progress.read_text())
            value["experiment"] = "previous-exp"
            value["round"] = 890
            progress.write_text(json.dumps(value))
            (root / "progress/latest_rank0.log").write_text(
                '{"round":890,"optimizer_step":890,"loss":1.0}\n'
            )

            status = summarize(root, now=now)

            self.assertEqual(status["training"]["experiment"], "exp")
            self.assertEqual(
                status["training"]["last_progress_experiment"], "previous-exp"
            )
            self.assertFalse(
                status["training"]["progress_matches_active_experiment"]
            )
            self.assertIsNone(status["training"]["last_observed_round"])
            self.assertIsNone(status["training"]["last_remote_observation"])
            self.assertFalse(status["training"]["observation_fresh"])
            self.assertIsNone(status["training"]["health_at_last_observation"])
            self.assertEqual(
                status["training"]["state_claim"],
                "awaiting_current_experiment_observation",
            )

    def test_current_run_waiting_for_first_record_is_not_reported_as_progress(self) -> None:
        now = datetime(2026, 8, 1, 6, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as temporary:
            root = self.fixture(Path(temporary), now)
            (root / "progress/progress.json").write_text(json.dumps({
                "experiment": "exp",
                "phase": "awaiting_first_training_record",
                "round": None,
                "optimizer_step": None,
                "amlt_status": "running",
                "observed_at": now.isoformat(),
                "total_rounds": 16241,
                "health": None,
            }))
            (root / "progress/latest_rank0.log").write_text(
                '{"round":890,"optimizer_step":890,"loss":1.0,'
                '"causal_pair_id":"stale","peak_cuda_allocated_gib":60,'
                '"peak_cuda_reserved_gib":70}\n'
            )
            status = summarize(root, now=now)
            self.assertTrue(
                status["training"]["progress_matches_active_experiment"]
            )
            self.assertIsNone(status["training"]["last_observed_round"])
            self.assertEqual(
                status["training"]["state_claim"],
                "awaiting_first_training_record",
            )

    def test_scientifically_deferred_fit_reports_candidate_repair_stage(self) -> None:
        now = datetime(2026, 8, 4, 10, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as temporary:
            root = self.fixture(Path(temporary), now)
            promoted = root / "task_retention_promoted_final_fit"
            (promoted / "active_experiment.txt").write_text("promoted-exp")
            (promoted / "deferred").touch()
            status = summarize(root, now=now)
            self.assertEqual(
                status["current_pipeline_stage"], "candidate_repair_diagnostics"
            )
            self.assertTrue(
                status["experiments"]["promoted_exact_epoch_fit"]["deferred"]
            )
            self.assertIsNone(status["training"]["health_at_last_observation"])

    def test_auth_expiry_makes_old_round_explicitly_last_known(self) -> None:
        now = datetime(2026, 8, 1, 6, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as temporary:
            root = self.fixture(Path(temporary), now - timedelta(hours=1))
            supervisor = root.parent / "recurrent_unified_supervisor_20260731"
            supervisor.mkdir()
            (supervisor / "auth_required").touch()
            value = summarize(root, now=now)
            self.assertEqual(value["training"]["last_observed_round"], 300)
            self.assertEqual(value["training"]["state_claim"], "last_known_remote_observation_only")
            self.assertFalse(value["auth"]["remote_state_readable"])
            self.assertFalse(value["training"]["observation_fresh"])

    def test_attention_and_completion_are_not_inferred(self) -> None:
        now = datetime(2026, 8, 1, 6, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as temporary:
            root = self.fixture(Path(temporary), now)
            (root / "final_fit/needs_attention").touch()
            value = summarize(root, now=now)
            self.assertEqual(value["attention"], ["final_fit"])
            self.assertFalse(value["overall_complete"])

    def test_superseded_attention_is_not_actionable(self) -> None:
        now = datetime(2026, 8, 1, 6, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as temporary:
            root = self.fixture(Path(temporary), now)
            (root / "final_fit/needs_attention").touch()
            (root / "final_fit/superseded").touch()
            value = summarize(root, now=now)
            self.assertEqual(value["attention"], [])
            self.assertTrue(value["controllers"]["final_fit"]["superseded"])

    def test_terminal_controller_does_not_fail_active_heartbeat_summary(self) -> None:
        now = datetime(2026, 8, 1, 6, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as temporary:
            root = self.fixture(Path(temporary), now)
            stale = root / "checkpoint500_c4/controller.heartbeat"
            stale.write_text((now - timedelta(hours=1)).isoformat())
            (root / "checkpoint500_c4/complete").touch()

            value = summarize(root, now=now)

            row = value["controllers"]["checkpoint500_c4"]
            self.assertFalse(row["alive_locally"])
            self.assertFalse(row["required_now"])
            self.assertNotIn("checkpoint500_c4", value["required_controllers"])
            self.assertTrue(value["all_controllers_alive_locally"])

    def test_stale_nonterminal_controller_fails_active_heartbeat_summary(self) -> None:
        now = datetime(2026, 8, 1, 6, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as temporary:
            root = self.fixture(Path(temporary), now)
            stale = root / "task_retention_promoted_final_fit/controller.heartbeat"
            stale.write_text((now - timedelta(hours=1)).isoformat())

            value = summarize(root, now=now)

            row = value["controllers"]["task_retention_promoted_final_fit"]
            self.assertTrue(row["required_now"])
            self.assertIn(
                "task_retention_promoted_final_fit", value["required_controllers"]
            )
            self.assertFalse(value["all_controllers_alive_locally"])

    def test_deferred_controller_is_visible_without_claiming_completion(self) -> None:
        now = datetime(2026, 8, 1, 6, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as temporary:
            root = self.fixture(Path(temporary), now)
            (root / "staged_task_retention/deferred").touch()
            value = summarize(root, now=now)
            self.assertTrue(value["controllers"]["staged_task_retention"]["deferred"])
            self.assertFalse(value["controllers"]["staged_task_retention"]["complete"])
            self.assertFalse(
                value["controllers"]["staged_task_retention"]["required_now"]
            )

    def test_superseded_legacy_fit_is_not_reported_as_active_training(self) -> None:
        now = datetime(2026, 8, 1, 6, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as temporary:
            root = self.fixture(Path(temporary), now)
            (root / "final_fit/superseded").touch()
            (root / "task_retention_rehearsal/active_experiment.txt").write_text(
                "rehearsal-exp"
            )
            (root / "eedi_state_path_ab/active_experiment.txt").write_text("ab-exp")
            value = summarize(root, now=now)
            self.assertIsNone(value["training"]["experiment"])
            self.assertEqual(value["training"]["legacy_superseded_experiment"], "exp")
            self.assertEqual(
                value["current_pipeline_stage"],
                "task_retention_candidate_evaluation",
            )
            self.assertEqual(
                value["experiments"]["task_retention_rehearsal"]["experiment"],
                "rehearsal-exp",
            )
            self.assertEqual(
                value["experiments"]["eedi_state_path_ab"]["experiment"], "ab-exp"
            )

    def test_promoted_fit_replaces_legacy_as_active_training(self) -> None:
        now = datetime(2026, 8, 1, 6, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as temporary:
            root = self.fixture(Path(temporary), now)
            (root / "final_fit/superseded").touch()
            (root / "task_retention_rehearsal/complete").touch()
            (root / "task_retention_promoted_final_fit/active_experiment.txt").write_text(
                "promoted-exp"
            )
            (root / "promoted_progress/progress.json").write_text(json.dumps({
                "experiment": "promoted-exp",
                "phase": "awaiting_first_training_record",
                "round": None,
                "optimizer_step": None,
                "amlt_status": "queued",
                "observed_at": now.isoformat(),
                "total_rounds": 16241,
                "health": None,
            }))
            value = summarize(root, now=now)
            self.assertEqual(value["training"]["experiment"], "promoted-exp")
            self.assertEqual(value["current_pipeline_stage"], "promoted_exact_epoch_fit")
            self.assertTrue(
                value["training"]["progress_matches_active_experiment"]
            )
            self.assertEqual(value["training"]["last_observed_status"], "queued")
            self.assertEqual(
                value["training"]["state_claim"],
                "awaiting_first_training_record",
            )

    def test_round1000_multiseed_c4_is_visible_but_not_completion_evidence(self) -> None:
        now = datetime(2026, 8, 4, 3, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as temporary:
            root = self.fixture(Path(temporary), now)
            directory = (
                root
                / "task_retention_promoted_final_fit/round1000_c4_n150_multiseed"
            )
            (directory / "complete").touch()
            (directory / "summary.json").write_text(
                json.dumps({"two_of_three_gate_pass": True})
            )
            value = summarize(root, now=now)
            row = value["experiments"]["promoted_round1000_c4_n150_multiseed"]
            self.assertTrue(row["complete"])
            self.assertTrue(row["scientific_gate_pass"])
            self.assertEqual(
                row["role"], "diagnostic_pair_seed_stability_not_final_gate"
            )
            self.assertFalse(value["overall_complete"])

            summary = json.loads((directory / "summary.json").read_text())
            summary["two_of_three_gate_pass"] = False
            (directory / "summary.json").write_text(json.dumps(summary))
            value = summarize(root, now=now)
            self.assertFalse(
                value["experiments"]["promoted_round1000_c4_n150_multiseed"][
                    "scientific_gate_pass"
                ]
            )

    def test_completed_failed_rehearsal_does_not_claim_final_fit_started(self) -> None:
        now = datetime(2026, 8, 1, 6, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as temporary:
            root = self.fixture(Path(temporary), now)
            (root / "final_fit/superseded").touch()
            (root / "task_retention_rehearsal/complete").touch()
            for name, experiment in (
                ("equal_state_path", "equal-exp"),
                ("staged_task_retention", "staged-standard-exp"),
                ("staged_state_path", "staged-state-path-exp"),
            ):
                (root / name / "active_experiment.txt").write_text(experiment)
            value = summarize(root, now=now)
            self.assertEqual(
                value["current_pipeline_stage"],
                "task_retention_candidate_evaluation",
            )
            self.assertEqual(
                value["experiments"]["staged_state_path"]["experiment"],
                "staged-state-path-exp",
            )

    def test_health_backfill_does_not_refresh_remote_observation(self) -> None:
        now = datetime(2026, 8, 1, 6, tzinfo=timezone.utc)
        observed = now - timedelta(hours=1)
        with tempfile.TemporaryDirectory() as temporary:
            root = self.fixture(Path(temporary), observed)
            progress = root / "progress/progress.json"
            value = json.loads(progress.read_text())
            value.pop("health")
            progress.write_text(json.dumps(value))
            (root / "progress/latest_rank0.log").write_text(
                '{"round":1,"optimizer_step":1,"loss":1.0,"causal_pair_id":"pair",'
                '"peak_cuda_allocated_gib":60,"peak_cuda_reserved_gib":70}\n'
            )
            status = summarize(root, now=now)
            self.assertTrue(status["training"]["health_at_last_observation"])
            self.assertEqual(
                status["training"]["health_source"], "last_downloaded_rank0_log"
            )
            self.assertFalse(status["training"]["observation_fresh"])
            self.assertEqual(
                status["training"]["last_remote_observation"], observed.isoformat()
            )


if __name__ == "__main__":
    unittest.main()
