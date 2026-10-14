from pathlib import Path


SOURCE = (
    Path(__file__).resolve().parents[1]
    / "scripts/watch_state8_taxonomy_teacher_curriculum.sh"
).read_text()


def test_postprocess_uses_semantic_judge_without_exact_match_authority() -> None:
    assert "--judge-backend trapi" in SOURCE
    assert "llm_judge_exact_shortcuts" in SOURCE
    assert "'semantic_judge_authoritative':True" in SOURCE
    assert "'exact_match_authoritative':False" in SOURCE
    assert "taxonomy_teacher_curriculum.json" in SOURCE
    assert "STATE8_CURRICULUM_EXPERIMENT" in SOURCE
    assert "STATE8_CURRICULUM_JOB" in SOURCE
    assert "STATE8_CURRICULUM_SLUG" in SOURCE


def test_postprocess_uses_live_feeds_profile() -> None:
    assert "unset AZURE_CONFIG_DIR" not in SOURCE
    assert ".azure-studentmodeling-feeds" in SOURCE
    assert ".azure az account get-access-token" in SOURCE
    assert "subscription b6dc87f3-c479-49c8-8cb5-7896da3ff895" in SOURCE


def test_curriculum_watcher_recovers_soft_stalls_fail_closed() -> None:
    for fragment in (
        "STATE8_CURRICULUM_STALL_SECONDS:-2700",
        "STATE8_CURRICULUM_STARTUP_STALL_SECONDS:-7200",
        "STATE8_CURRICULUM_POSTTRAIN_STALL_SECONDS:-21600",
        "monitor_layerwise_unified_training_progress.py",
        "seconds_since_progress",
        "training_complete",
        "awaiting_first=x.get('phase')=='awaiting_first_training_record'",
        "threshold=float(sys.argv[5] if training_complete else sys.argv[4] if awaiting_first else sys.argv[3])",
        "health_alert",
        "stalled.json",
        'amlt cancel "$experiment" ":$job" -y',
        "cancelled stalled/unhealthy curriculum for exact recovery",
    ):
        assert fragment in SOURCE
    assert SOURCE.index("stalled.log") < SOURCE.index("stalled.cancel.txt")


def test_completed_t3heavy_checkpoint_uses_posttrain_stall_budget() -> None:
    for fragment in (
        "state8-taxonomy-t3heavy-finalizer-seed42",
        "'\"round\": 1250' in log",
        "'\"status\": \"complete\"' in log",
        "'STATE8_CURRICULUM_STAGE=training' in log",
    ):
        assert fragment in SOURCE
