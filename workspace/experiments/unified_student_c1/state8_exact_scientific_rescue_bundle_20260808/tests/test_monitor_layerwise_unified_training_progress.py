from __future__ import annotations

import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from monitor_layerwise_unified_training_progress import (
    attach_fatal_log_health,
    awaiting_first_record,
    health_summary,
    latest_record,
    preserve_progress_during_log_gap,
    summarize,
)


class TrainingProgressMonitorTest(unittest.TestCase):
    def test_waiting_receipt_is_explicit_and_has_no_invented_round(self) -> None:
        now = datetime(2026, 8, 1, 3, 0, tzinfo=timezone.utc)
        value = awaiting_first_record(
            experiment="exp",
            log_file="rank0",
            status="running",
            now=now,
            total_rounds=16241,
        )
        self.assertEqual(value["phase"], "awaiting_first_training_record")
        self.assertIsNone(value["round"])
        self.assertIsNone(value["optimizer_step"])
        self.assertIsNone(value["health"])
        self.assertEqual(value["observed_at"], now.isoformat())
        self.assertEqual(value["total_rounds"], 16241)
        self.assertEqual(value["attempt_started_at"], now.isoformat())
        self.assertEqual(value["running_started_at"], now.isoformat())
        self.assertEqual(value["seconds_since_progress"], 0.0)

    def test_waiting_receipt_preserves_first_record_stall_clock(self) -> None:
        started = datetime(2026, 8, 1, 2, 0, tzinfo=timezone.utc)
        previous = awaiting_first_record(
            experiment="exp",
            log_file="rank0",
            status="running",
            now=started,
            total_rounds=16241,
        )
        value = awaiting_first_record(
            experiment="exp",
            log_file="rank0",
            status="running",
            previous=previous,
            now=datetime(2026, 8, 1, 3, 0, tzinfo=timezone.utc),
            total_rounds=16241,
        )
        self.assertEqual(value["running_started_at"], started.isoformat())
        self.assertEqual(value["seconds_since_progress"], 3600.0)
        self.assertEqual(value["attempt_elapsed_seconds"], 3600.0)

    def test_queue_time_does_not_count_as_first_record_stall(self) -> None:
        queued = awaiting_first_record(
            experiment="exp",
            log_file="",
            status="queued",
            now=datetime(2026, 8, 1, 1, 0, tzinfo=timezone.utc),
        )
        running = awaiting_first_record(
            experiment="exp",
            log_file="rank0",
            status="running",
            previous=queued,
            now=datetime(2026, 8, 1, 3, 0, tzinfo=timezone.utc),
        )
        self.assertEqual(
            running["running_started_at"],
            datetime(2026, 8, 1, 3, 0, tzinfo=timezone.utc).isoformat(),
        )
        self.assertEqual(running["seconds_since_progress"], 0.0)

    def test_latest_structured_round_wins(self) -> None:
        text = "\n".join(
            [
                "noise",
                '{"round": 1, "optimizer_step": 1, "rank": 0, "loss": 2.0}',
                "{broken",
                '{"round": 10, "optimizer_step": 10, "rank": 0, "loss": 0.5}',
            ]
        )
        self.assertEqual(latest_record(text)["round"], 10)
        value = summarize(text, experiment="exp", log_file="rank0", status="running")
        self.assertEqual(value["optimizer_step"], 10)
        self.assertEqual(value["amlt_status"], "running")

    def test_single_gpu_step_schema_is_normalized_to_progress_round(self) -> None:
        text = "\n".join(
            [
                '{"step": 1, "optimizer_step": 1, "loss": 2.0}',
                '{"step": 10, "optimizer_step": 3, "loss": 1.5, '
                '"causal_pair_id": null, "peak_cuda_reserved_gib": 54.2}',
            ]
        )
        record = latest_record(text)
        self.assertEqual(record["round"], 10)
        self.assertEqual(record["progress_unit"], "step")
        value = summarize(
            text,
            experiment="single",
            log_file="rank0",
            status="running",
            total_rounds=1000,
        )
        self.assertEqual(value["round"], 10)
        self.assertEqual(value["optimizer_step"], 3)
        self.assertEqual(value["progress_unit"], "step")
        self.assertAlmostEqual(value["progress_fraction"], 0.01)

    def test_missing_record_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "No structured"):
            latest_record("plain log only")

    def test_unchanged_round_preserves_last_progress_time(self) -> None:
        previous = {
            "experiment": "exp",
            "round": 10,
            "observed_at": "2026-08-01T02:00:00+00:00",
            "last_progress_at": "2026-08-01T02:00:00+00:00",
        }
        value = summarize(
            '{"round": 10, "optimizer_step": 10, "rank": 0}',
            experiment="exp",
            log_file="rank0",
            status="running",
            previous=previous,
            now=datetime(2026, 8, 1, 3, 0, tzinfo=timezone.utc),
        )
        self.assertFalse(value["advanced"])
        self.assertEqual(value["seconds_since_progress"], 3600.0)

    def test_new_round_resets_last_progress_time(self) -> None:
        previous = {
            "experiment": "exp",
            "round": 10,
            "last_progress_at": "2026-08-01T02:00:00+00:00",
        }
        now = datetime(2026, 8, 1, 3, 0, tzinfo=timezone.utc)
        value = summarize(
            '{"round": 20, "optimizer_step": 20, "rank": 0}',
            experiment="exp",
            log_file="rank0",
            status="running",
            previous=previous,
            now=now,
        )
        self.assertTrue(value["advanced"])
        self.assertEqual(value["last_progress_at"], now.isoformat())

    def test_first_round_after_waiting_receipt_starts_rate_baseline(self) -> None:
        previous = awaiting_first_record(
            experiment="exp",
            log_file="rank0",
            status="running",
            now=datetime(2026, 8, 1, 2, 0, tzinfo=timezone.utc),
            total_rounds=16241,
        )
        now = datetime(2026, 8, 1, 3, 0, tzinfo=timezone.utc)
        value = summarize(
            '{"round": 900, "optimizer_step": 900, "rank": 0}',
            experiment="exp",
            log_file="rank0",
            status="running",
            previous=previous,
            now=now,
            total_rounds=16241,
        )
        self.assertTrue(value["advanced"])
        self.assertEqual(value["last_progress_at"], now.isoformat())
        self.assertEqual(value["rate_baseline_round"], 900)
        self.assertEqual(value["rate_baseline_at"], now.isoformat())

    def test_empty_log_gap_preserves_real_round_and_stall_clock(self) -> None:
        previous = {
            "experiment": "exp",
            "round": 80,
            "optimizer_step": 80,
            "observed_at": "2026-08-01T02:10:00+00:00",
            "last_progress_at": "2026-08-01T02:00:00+00:00",
            "attempt_started_at": "2026-08-01T01:00:00+00:00",
            "max_run_seconds": 96 * 3600,
            "remaining_attempt_seconds": 95 * 3600,
            "health": {"healthy": True},
        }
        value = preserve_progress_during_log_gap(
            previous,
            experiment="exp",
            log_file="rank0",
            status="running",
            now=datetime(2026, 8, 1, 3, 0, tzinfo=timezone.utc),
        )
        self.assertEqual(value["round"], 80)
        self.assertEqual(value["optimizer_step"], 80)
        self.assertFalse(value["advanced"])
        self.assertTrue(value["log_gap"])
        self.assertEqual(value["phase"], "temporary_log_gap")
        self.assertEqual(value["seconds_since_progress"], 3600.0)
        self.assertEqual(value["attempt_elapsed_seconds"], 7200.0)
        self.assertEqual(value["remaining_attempt_seconds"], 94 * 3600)
        self.assertTrue(value["health"]["healthy"])

    def test_log_gap_without_prior_round_cannot_invent_progress(self) -> None:
        previous = awaiting_first_record(
            experiment="exp",
            log_file="rank0",
            status="running",
        )
        with self.assertRaisesRegex(ValueError, "No prior structured"):
            preserve_progress_during_log_gap(
                previous,
                experiment="exp",
                log_file="rank0",
                status="running",
            )

    def test_recipe_drift_without_new_record_is_immediately_unhealthy(self) -> None:
        previous = {
            "experiment": "exp",
            "round": 130,
            "optimizer_step": 130,
            "observed_at": "2026-08-01T02:10:00+00:00",
            "last_progress_at": "2026-08-01T02:00:00+00:00",
            "health": {
                "healthy": True,
                "checks": {"no_runtime_error_signature": True},
                "runtime_error_signatures": [],
            },
        }
        value = preserve_progress_during_log_gap(
            previous,
            experiment="exp",
            log_file="retry_001/rank0",
            status="running",
            now=datetime(2026, 8, 1, 3, 0, tzinfo=timezone.utc),
        )
        value = attach_fatal_log_health(
            value,
            "ValueError: Distributed resume scientific recipe drift\n"
            "STATE8_CURRICULUM_FAILED line=24 rc=1",
        )
        self.assertEqual(value["round"], 130)
        self.assertEqual(value["phase"], "fatal_runtime_error")
        self.assertFalse(value["health"]["healthy"])
        self.assertFalse(
            value["health"]["checks"]["no_runtime_error_signature"]
        )
        self.assertIn(
            "distributed resume scientific recipe drift",
            value["health"]["runtime_error_signatures"],
        )

    def test_rate_eta_and_single_attempt_risk(self) -> None:
        previous = {
            "experiment": "exp",
            "round": 20,
            "observed_at": "2026-08-01T02:00:00+00:00",
            "rate_baseline_round": 20,
            "rate_baseline_at": "2026-08-01T02:00:00+00:00",
        }
        value = summarize(
            '{"round": 40, "optimizer_step": 40, "rank": 0}',
            experiment="exp",
            log_file="rank0",
            status="running",
            previous=previous,
            now=datetime(2026, 8, 1, 3, 0, tzinfo=timezone.utc),
            total_rounds=16_240,
            max_run_seconds=96 * 3600,
            validation_reserve_seconds=6 * 3600,
        )
        self.assertAlmostEqual(value["rounds_per_hour"], 20.0)
        self.assertEqual(value["eta_seconds"], 810 * 3600)
        self.assertTrue(value["single_attempt_duration_risk"])
        self.assertAlmostEqual(value["progress_fraction"], 40 / 16_240)
        self.assertEqual(value["attempt_elapsed_seconds"], 3600)
        self.assertEqual(value["remaining_attempt_seconds"], 95 * 3600)
        self.assertEqual(value["validation_reserve_seconds"], 6 * 3600)

    def test_invalid_total_rounds_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "total round"):
            summarize(
                '{"round": 10, "optimizer_step": 10, "rank": 0}',
                experiment="exp",
                log_file="rank0",
                status="running",
                total_rounds=9,
            )

    def test_health_summary_tracks_causal_path_and_memory(self) -> None:
        text = "\n".join(
            [
                '{"round":1,"optimizer_step":1,"loss":1.0,"causal_loss":0.5,"causal_pair_id":"pair-1","peak_cuda_allocated_gib":68.0,"peak_cuda_reserved_gib":70.0}',
                '{"round":10,"optimizer_step":10,"loss":0.8,"causal_loss":0.0,"causal_pair_id":null,"peak_cuda_allocated_gib":50.0,"peak_cuda_reserved_gib":70.0}',
            ]
        )
        value = health_summary(text)
        self.assertTrue(value["healthy"])
        self.assertEqual(value["causal_pair_ids"], ["pair-1"])
        self.assertEqual(
            value["causal_observation_scope"], "bounded_rank_zero_log_tail"
        )
        self.assertFalse(value["causal_absence_is_schedule_failure"])
        self.assertEqual(
            value["causal_schedule_authority"], "distributed_checkpoint_audit"
        )
        self.assertEqual(value["peak_cuda_reserved_gib"], 70.0)
        self.assertTrue(value["advisory_only"])

    def test_nonfinite_or_oom_produces_advisory_alert(self) -> None:
        value = health_summary(
            '{"round":1,"optimizer_step":1,"loss":NaN,"causal_pair_id":null,'
            '"peak_cuda_allocated_gib":79.5,"peak_cuda_reserved_gib":79.5}\n'
            'CUDA out of memory'
        )
        self.assertFalse(value["healthy"])
        self.assertFalse(value["checks"]["finite_losses"])
        self.assertFalse(value["checks"]["no_runtime_error_signature"])
        self.assertFalse(value["checks"]["memory_below_a100_80gb"])
        self.assertFalse(value["checks"]["causal_path_observed_when_round1_present"])


if __name__ == "__main__":
    unittest.main()
