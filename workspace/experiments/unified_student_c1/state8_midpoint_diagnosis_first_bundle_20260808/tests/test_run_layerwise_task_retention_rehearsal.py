#!/usr/bin/env python3
"""Static invariants for the task-retention Feeds launcher."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_layerwise_task_retention_rehearsal.sh"


class TaskRetentionLauncherTest(unittest.TestCase):
    def test_unique_run_directory_is_created_only_under_the_rank_lock(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        guard = 'if [[ ! -f "/tmp/layerwise-task-retention-${BRANCH}-${RUN_ID}-run.ready" ]]'
        before_guard, after_guard = source.split(guard, 1)

        # Every MPI rank shares /tmp.  Creating RUN_DIR before the guarded
        # test, directly or through C4_PROGRESS_DIR, makes
        # `test ! -e "$RUN_DIR"` fail deterministically.
        self.assertNotIn('mkdir -p "$RUN_DIR"', before_guard)
        self.assertNotIn(' "$C4_PROGRESS_DIR"', before_guard)
        self.assertIn('test ! -e "$RUN_DIR"', after_guard)
        self.assertIn('mkdir -p "$RUN_DIR" "$C4_PROGRESS_DIR"', after_guard)
        self.assertIn('test -d "$C4_PROGRESS_DIR"', after_guard)

    def test_state_path_profile_removes_direct_replay_and_kd_regularization(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn('LOSS_PROFILE="${LOSS_PROFILE:-standard}"', source)
        self.assertIn('BEHAVIOR_KD_WEIGHT=0.0', source)
        self.assertIn('DIRECT_REPLAY_WEIGHT=0.0', source)
        self.assertIn('--generation-weight 1.0 --consistency-weight 0.25', source)
        self.assertIn('--causal-objective-mode specificity_triplet', source)
        self.assertIn('STUDENTMODELING_TEACHER_CACHE_UNUSED=1', source)
        self.assertIn('unset STUDENTMODELING_TEACHER_CACHE_UNUSED', source)

    def test_state_kd_profile_removes_only_the_bypass(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("state_kd)", source)
        self.assertIn("'state_kd':(0.5,0.0)", source)

    def test_extended_formation_scan_is_parameterized_without_changing_defaults(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn('SEGMENT_ROWS="${SEGMENT_ROWS:-1000}"', source)
        self.assertIn('LEARNING_RATE="${LEARNING_RATE:-5e-7}"', source)
        self.assertIn('EXPECTED_CAUSAL_PAIRS="${EXPECTED_CAUSAL_PAIRS:-$((SEGMENT_ROWS / 20))}"', source)
        self.assertIn('--learning-rate "$LEARNING_RATE"', source)
        self.assertIn('--diagnostic-stop-after-round "$DIAGNOSTIC_STOP_AFTER_ROUND"', source)

    def test_service_retries_use_hash_verified_shared_checkpoints(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("select_latest_distributed_checkpoint.py", source)
        self.assertIn('TRAINER_SHA256="$(sha256sum scripts/train_recurrent_student_m4.py', source)
        self.assertIn('ENTRYPOINT_SHA256="$(sha256sum "$TRAIN_ENTRYPOINT"', source)
        self.assertIn('test "$ENTRYPOINT_SHA256" = "$EXPECTED_ENTRYPOINT_SHA256"', source)
        self.assertIn("train_recurrent_student_m4_open_t3_selective_kd.py", source)
        self.assertIn('CHECKPOINT_MIRROR_DIR="/mnt/amulet/', source)
        self.assertIn('--checkpoint-mirror-dir "$CHECKPOINT_MIRROR_DIR"', source)
        self.assertIn('--resume-from-checkpoint "$RESUME_CHECKPOINT"', source)
        self.assertIn(
            '--diagnostic-warm-start-checkpoint "$CHECKPOINT/recurrent_state.pt"',
            source,
        )

    def test_warm_start_boundary_is_parameterized_and_audited(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn('WARM_START_CHECKPOINT_DIR', source)
        self.assertIn('WARM_START_EXPECTED_ROUND', source)
        self.assertIn('WARM_START_EXPECTED_WORLD_SIZE', source)
        self.assertIn('audit_layerwise_diagnostic_warm_start.py', source)
        self.assertIn("expected_round*expected_world==expected_start", source)

    def test_service_retries_checkpoint_before_observed_preemptions(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("--save-steps 50 --checkpoint-retention 2", source)
        self.assertNotIn("--save-steps 250 --checkpoint-retention 1", source)

    def test_rank_zero_emits_machine_readable_evaluation_stage_transitions(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        training_done = 'if [[ "${OMPI_COMM_WORLD_RANK:-0}" != 0 ]]; then exit 0; fi'
        self.assertLess(source.index(training_done), source.index('"stage":"c4"'))
        for stage in ("c4", "formal_screen", "t6_full", "publish"):
            self.assertIn(f'"stage":"{stage}"', source)
        self.assertIn('"branch":"%s"', source)
        self.assertIn('"loss_profile":"%s"', source)

    def test_same_rule_variant_does_not_copy_base_files_onto_themselves(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn('CAUSAL_VARIANT="${CAUSAL_VARIANT:-base}"', source)
        self.assertIn('if [[ "$CAUSAL_VARIANT" == same_rule ]]', source)
        self.assertNotIn('cp "$CAUSAL_DATA_BASE/train.jsonl" "$CAUSAL_DATA/train.jsonl"', source)
        self.assertIn("augment_recurrent_causal_same_rule.py", source)


if __name__ == "__main__":
    unittest.main()
