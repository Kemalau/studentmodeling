from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_pipeline_is_recoverable_and_covers_full_goal() -> None:
    source = (ROOT / "scripts/watch_state8_taxonomy_unified_pipeline.sh").read_text()
    for fragment in (
        "submit_arm", "recovery${wave}", "select_state8_taxonomy_consolidation.py",
        "state8_taxonomy_curriculum_v1/selection.json", "validate_state8_unified_final_fit_amlt.py",
        "259845-row", "watch_state8_final_eval.sh", "scientific_gate_failed",
        "recover_source", "recovery.lock", "validate_state8_taxonomy_teacher_curriculum_amlt.py",
        "Automatic exact-resume recovery",
        "monitor_layerwise_unified_training_progress.py",
        "STATE8_PIPELINE_POSTTRAIN_STALL_SECONDS:-21600",
        "threshold=float(sys.argv[3] if training_complete else sys.argv[2])",
        "cancelled stalled/unhealthy job for exact recovery",
        "remote_experiment_class", "classify_experiment_status",
        "preexisting_not_resubmitted",
        "restarted local postprocess without retraining",
        "observe_progress \"$experiment\" \"$job\" \"$arm\" 250 2700 172800",
        "observe_progress \"$experiment\" \"$final_job\" \"$final_state\" 16241 7200 345600",
    ):
        assert fragment in source


def test_pipeline_parallelizes_the_four_bounded_screens() -> None:
    source = (ROOT / "scripts/watch_state8_taxonomy_unified_pipeline.sh").read_text()
    assert "worker_pids" in source
    assert "for index in 0 1 2 3" in source
    assert "(submit_arm \"$index\")" in source


def test_pipeline_prefers_a_profile_with_a_live_token() -> None:
    source = (ROOT / "scripts/watch_state8_taxonomy_unified_pipeline.sh").read_text()
    assert ".azure az account get-access-token" in source
    assert ".azure-studentmodeling-feeds" in source


def test_selected_final_fit_persists_recovery_experiment_across_controller_restart() -> None:
    source = (ROOT / "scripts/watch_state8_taxonomy_unified_pipeline.sh").read_text()
    assert 'final_state/active_experiment.txt' in source
    assert 'experiment="$(<"$final_state/active_experiment.txt")"' in source
    assert '[[ "$experiment" =~ recovery([0-9]+)-20260807 ]]' in source
    intent = 'printf \'%s\\n\' "$experiment" > "$final_state/active_experiment.txt"'
    assert source.count(intent) >= 3
    assert source.index(intent) < source.index('amlt run "$final_config"')


def test_all_cost_bearing_submissions_are_guarded_by_remote_existence_checks() -> None:
    source = (ROOT / "scripts/watch_state8_taxonomy_unified_pipeline.sh").read_text()
    assert source.count('class="$(remote_experiment_class') == 3
    assert source.count('[[ "$class" == missing ]]') == 3
    assert source.count('[[ "$class" == exists ]]') == 3
    assert source.count('status=preexisting_not_resubmitted') == 3


def test_remote_pass_only_restarts_local_postprocessing() -> None:
    source = (ROOT / "scripts/watch_state8_taxonomy_unified_pipeline.sh").read_text()
    branch = source[source.index('if [[ -f "$root/remote_passed"'):source.index("wave=2", source.index('if [[ -f "$root/remote_passed"'))]
    assert "watch_state8_taxonomy_teacher_curriculum.sh" in branch
    assert "amlt run" not in branch
    assert "return 0" in branch


def test_supervisor_restarts_infrastructure_only_and_stops_for_science() -> None:
    source = (ROOT / "scripts/supervise_state8_taxonomy_unified_pipeline.sh").read_text()
    assert "watch_state8_taxonomy_unified_pipeline.sh" in source
    assert "STATE8_PIPELINE_RESTART_SECONDS:-120" in source
    assert source.count('scientific_gate_failed') >= 2
    assert '[[ ! -f "$state/complete" ]]' in source
