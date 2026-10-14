from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_pipeline_is_recoverable_and_covers_full_goal() -> None:
    source = (ROOT / "scripts/watch_state8_taxonomy_unified_pipeline.sh").read_text()
    for fragment in (
        "submit_arm", "recovery${wave}", "select_state8_taxonomy_consolidation.py",
        "state8_unified_candidate_v1/selection.json", "validate_state8_unified_final_fit_amlt.py",
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
        'task_anchor_state="$project/results/state8_task_anchor_screen_20260808"',
        "--candidate task_anchor_base",
        "--candidate task_anchor_c4strong",
    ):
        assert fragment in source


def test_pipeline_parallelizes_the_configured_bounded_screens() -> None:
    source = (ROOT / "scripts/watch_state8_taxonomy_unified_pipeline.sh").read_text()
    assert "worker_pids" in source
    assert 'STATE8_PIPELINE_ACTIVE_SLUGS:-${slugs[*]}' in source
    assert 'for index in "${active_indices[@]}"' in source
    assert 'active_candidate_slugs.txt' in source
    assert "(supervise_arm \"$index\")" in source


def test_fast_subset_falls_back_to_all_arms_before_scientific_failure() -> None:
    source = (ROOT / "scripts/watch_state8_taxonomy_unified_pipeline.sh").read_text()
    fast_failure = source.index("selection.fast_subset_failed.json")
    fallback = source.index("expanding to all bounded arms", fast_failure)
    full_set = source.index('active_indices=("${!slugs[@]}")', fallback)
    terminal_failure = source.index('touch "$state/scientific_gate_failed"', full_set)
    assert fast_failure < fallback < full_set < terminal_failure
    assert "fallback consolidation arm supervisor failed" in source


def test_each_screen_has_an_independent_infrastructure_recovery_loop() -> None:
    source = (ROOT / "scripts/watch_state8_taxonomy_unified_pipeline.sh").read_text()
    start = source.index("supervise_arm(){")
    end = source.index("# Start an arm", start)
    supervisor = source[start:end]
    assert '( submit_arm "$index" )' in supervisor
    assert 'STATE8_PIPELINE_ARM_RESTART_SECONDS:-120' in supervisor
    assert '[[ -f "$arm/complete" ]]' in supervisor
    assert '[[ -f "$state/scientific_gate_failed" ]]' in supervisor
    assert 'recoverable arm worker exit' in supervisor


def test_controller_cleanup_stops_only_local_worker_supervisors() -> None:
    source = (ROOT / "scripts/watch_state8_taxonomy_unified_pipeline.sh").read_text()
    cleanup = source[source.index("cleanup_controller(){"):source.index("trap cleanup_controller")]
    assert '[[ "$BASHPID" == "$controller_pid" ]]' in cleanup
    assert 'pkill -TERM -P "$pid"' in cleanup
    assert 'kill -TERM "$pid"' in cleanup
    assert "amlt cancel" not in cleanup


def test_arm_workers_do_not_inherit_the_controller_lock() -> None:
    source = (ROOT / "scripts/watch_state8_taxonomy_unified_pipeline.sh").read_text()
    start = source.index("supervise_arm(){")
    end = source.index("# Start an arm", start)
    assert "exec 9>&-" in source[start:end]


def test_pipeline_prefers_a_profile_with_a_live_token() -> None:
    source = (ROOT / "scripts/watch_state8_taxonomy_unified_pipeline.sh").read_text()
    assert ".azure az account get-access-token" in source
    assert ".azure-studentmodeling-feeds" in source


def test_selected_final_fit_persists_recovery_experiment_across_controller_restart() -> None:
    source = (ROOT / "scripts/watch_state8_taxonomy_unified_pipeline.sh").read_text()
    assert 'final_state/active_experiment.txt' in source
    assert 'experiment="$(<"$final_state/active_experiment.txt")"' in source
    assert '[[ "$experiment" =~ recovery([0-9]+)-20260808 ]]' in source
    intent = 'printf \'%s\\n\' "$experiment" > "$final_state/active_experiment.txt"'
    assert source.count(intent) >= 3
    assert source.index(intent) < source.index('amlt run "$final_config"')


def test_all_cost_bearing_submissions_are_guarded_by_remote_existence_checks() -> None:
    source = (ROOT / "scripts/watch_state8_taxonomy_unified_pipeline.sh").read_text()
    assert source.count('class="$(remote_experiment_class') == 4
    assert source.count('[[ "$class" == missing ]]') == 4
    assert source.count('[[ "$class" == exists ]]') == 4
    assert source.count('status=preexisting_not_resubmitted') == 4


def test_remote_pass_only_restarts_local_postprocessing() -> None:
    source = (ROOT / "scripts/watch_state8_taxonomy_unified_pipeline.sh").read_text()
    branch = source[source.index('if [[ -f "$root/remote_passed"'):source.index("wave=2", source.index('if [[ -f "$root/remote_passed"'))]
    assert "watch_state8_taxonomy_teacher_curriculum.sh" in branch
    assert "amlt run" not in branch
    assert "return 0" in branch


def test_balanced_recovery_reuses_pinned_code_after_local_hash_drift() -> None:
    source = (ROOT / "scripts/watch_state8_taxonomy_unified_pipeline.sh").read_text()
    for fragment in (
        "state8-balanced-curriculum-recovery20-20260807",
        "ae502968872ff6597e73bcc12cb3ff9a5bbb0fe9d7a763383f6e76bc392de42c",
        'amlt code download "$pinned_source" --only code',
        'amlt code download "$pinned_source" --only yaml',
        'amlt run "$config" ":$job" "$next" -t Feeds -y --code-dir "$source_root"',
        '--expected-save-steps 50',
        '--expected-save-steps 5',
        'source_save_steps=50',
        'recovery_save_steps=5',
        "pinned recovery trainer identity drifted",
    ):
        assert fragment in source
    start = source.index('if [[ -n "$pinned_source" ]')
    pinned = source[start:source.index("  else\n  amlt run", start)]
    assert "amlt rerun" not in pinned


def test_balanced_pinned_source_validator_accepts_archived_amlt_target() -> None:
    source = (ROOT / "scripts/amlt_validation_utils.py").read_text()
    for fragment in (
        '"cluster": "Feeds"',
        '"subscription_id": "b6dc87f3-c479-49c8-8cb5-7896da3ff895"',
        '"workspace_subscription_id": "b6dc87f3-c479-49c8-8cb5-7896da3ff895"',
    ):
        assert fragment in source


def test_consolidation_reuses_the_immutable_curriculum_code_and_staged_manifest() -> None:
    source = (ROOT / "scripts/watch_state8_taxonomy_unified_pipeline.sh").read_text()
    for fragment in (
        "state8-t3heavy-curriculum-recovery11-20260807",
        "ae502968872ff6597e73bcc12cb3ff9a5bbb0fe9d7a763383f6e76bc392de42c",
        "a4eb41b2130f7e0810eeff36b5f3d16b8dd2c8631c50dbd25216f0793af23557",
        "ce56a97b71aaf703cabd29dcd19f34d987da71102124ea77ea0d0ab50c915068",
        "prepare_pinned_consolidation_code",
        'amlt code download "$pinned_consolidation_source" --only code',
        '--code-dir "$pinned_consolidation_code"',
        "save_every_five_recover_any_atomic_step",
    ):
        assert fragment in source
    config = (ROOT / "amlt/feeds_state8_taxonomy_curriculum_consolidation_20260807.yaml").read_text()
    assert "layerwise_task_retention_staged_t3_t6_seed42_v1.json" in config
    assert "layerwise_task_retention_equal_t3_t6_seed42_v1.json" not in config


def test_supervisor_restarts_infrastructure_only_and_stops_for_science() -> None:
    source = (ROOT / "scripts/supervise_state8_taxonomy_unified_pipeline.sh").read_text()
    assert "watch_state8_taxonomy_unified_pipeline.sh" in source
    assert "STATE8_PIPELINE_RESTART_SECONDS:-120" in source
    assert "STATE8_PIPELINE_ACTIVE_SLUGS:-taxonomy_t3heavy taxonomy_t3heavy_diag" in source
    assert source.count('scientific_gate_failed') >= 2
    assert '[[ ! -f "$state/complete" ]]' in source
