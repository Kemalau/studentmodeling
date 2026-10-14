from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEXT = (ROOT / "scripts/watch_state8_qualified_teacher_pipeline.sh").read_text()


def test_watcher_is_fail_closed_and_orders_gates_before_stage_b() -> None:
    assert TEXT.index('run_recoverable_stage "$train_experiment"') < TEXT.index("teacher_published")
    assert TEXT.index("validate_state8_teacher_gate.py") < TEXT.index(
        'run_recoverable_stage "$stage_b_experiment"'
    )
    launcher = (ROOT / "scripts/run_state8_qualified_teacher_curriculum.sh").read_text()
    assert "exact_match_is_authoritative'] is False" in launcher
    assert "teacher_gate_pass" in TEXT
    assert "Stage-B training exhausted recovery waves" in TEXT
    assert "MAX_RECOVERY_WAVES:-12" in TEXT
    assert "state8-full-history-natural-teacher-resumable-recovery" in TEXT
    assert "state8-full-history-teacher-eval-recovery" in TEXT
    assert "state8-qualified-teacher-curriculum-recovery" in TEXT
    assert "stage_b_download" in TEXT
    accelerator = (ROOT / "scripts/watch_state8_teacher_sharded_accelerator.sh").read_text()
    assert "state8_clean_prompt_teacher_provenance_gate_v2" in accelerator
    assert "teacher_provenance_gate.json" in accelerator
    assert "clean_v2_16way" in accelerator
    assert 'shard_count=16' in accelerator
    assert accelerator.index('exec 6>"$project/results/feeds_submission_global.lock"') < accelerator.index(
        'amlt status --no-update-db "$curriculum_experiment"'
    )
    assert 'stage_b_finalize.lock' in accelerator
    assert accelerator.index('stage_b_finalize.lock') < accelerator.index(
        'rm -rf "$teacher_state/stage_b_download"'
    )
    assert 'teacher_semantic_judge.lock' in accelerator
    assert 'semantic_reused' in accelerator
    assert 'published teacher gate identity drift' in accelerator
    assert accelerator.index('if [[ "$class" == missing ]]') < accelerator.index(
        'wait_quota "${#names[@]}"'
    )
    assert 'has_forward_progress_job' in accelerator
    assert 'paused_job_names' in accelerator
    assert 'converting stalled paused teacher shard to explicit recovery' in accelerator
    assert "sys.stdout.write(''.join(value+'\\n' for value in values))" in accelerator
    launcher = (ROOT / "scripts/run_state8_qualified_teacher_curriculum.sh").read_text()
    assert "state8_clean_prompt_teacher_provenance_gate_v2" in launcher
    assert "legacy_answer_adjacent_contracts" in launcher
    assert "qualified_teacher_receipt.json" in TEXT
    assert "eval/formal_screen/metrics.json" in TEXT
    assert "needs_attention" in TEXT
    assert "amlt cancel" not in TEXT


def test_watcher_uses_same_feeds_identity_and_remote_paths() -> None:
    assert "AZURE_CONFIG_DIR=/home/v-yuejiali/.azure" in TEXT
    assert "state8_full_history_natural_teacher_v1" in TEXT
    assert "rankfun_aml" not in TEXT  # UAI remains in validated AMLT configs only.
    assert "select_a100_80gb" in TEXT
    assert 'wait_quota "$required_units"' in TEXT
    assert TEXT.index('if [[ "$class" == missing ]]') < TEXT.index(
        'wait_quota "$required_units"'
    )
    assert "user_available" in TEXT
