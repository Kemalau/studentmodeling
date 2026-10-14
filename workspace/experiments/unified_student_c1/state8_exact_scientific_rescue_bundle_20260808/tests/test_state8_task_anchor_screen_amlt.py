from pathlib import Path

import yaml

from scripts.validate_state8_task_anchor_screen_amlt import validate


ROOT = Path(__file__).resolve().parents[1]


def test_task_anchor_screen_is_fail_closed() -> None:
    validate(
        ROOT / "amlt/feeds_state8_task_anchor_screen_20260808.yaml",
        bundle=ROOT / "task_anchor_bundle_20260808",
    )


def test_task_anchor_recovery_can_validate_one_job(tmp_path: Path) -> None:
    source = ROOT / "amlt/feeds_state8_task_anchor_screen_20260808.yaml"
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    config["jobs"] = [
        job
        for job in config["jobs"]
        if job["name"] == "state8-task-anchor-base-seed56"
    ]
    rendered = tmp_path / "single-job-recovery.yaml"
    rendered.write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )
    validate(
        rendered,
        bundle=ROOT / "task_anchor_bundle_20260808",
        expected_jobs={"state8-task-anchor-base-seed56"},
    )


def test_task_anchor_watcher_resumes_scheduler_pauses() -> None:
    watcher = (ROOT / "scripts/watch_state8_task_anchor_screen.sh").read_text()
    assert "paused_job_names" in watcher
    assert 'amlt resume "$experiment" ":$job"' in watcher
    assert "requested resume for paused task-anchor jobs" in watcher


def test_task_anchor_watcher_detects_stalls_after_a_long_loading_grace() -> None:
    watcher = (ROOT / "scripts/watch_state8_task_anchor_screen.sh").read_text()
    assert "monitor_layerwise_unified_training_progress.py" in watcher
    assert "--total-rounds 250" in watcher
    assert "threshold=3600" in watcher
    assert "threshold=2700" in watcher
    assert "stalled_or_unhealthy.json" in watcher
    assert "cancelled stalled/unhealthy task-anchor" in watcher


def test_task_anchor_watcher_uses_a_long_posttraining_grace() -> None:
    watcher = (ROOT / "scripts/watch_state8_task_anchor_screen.sh").read_text()
    assert "STATE8_TASK_ANCHOR_POSTTRAIN_STALL_SECONDS:-21600" in watcher
    assert "training_complete=(float(x.get('progress_fraction') or 0)>=1.0" in watcher
    assert "stall_threshold=float(sys.argv[3] if training_complete else sys.argv[2])" in watcher


def test_task_anchor_watcher_allows_zero_passed_siblings_during_recovery() -> None:
    watcher = (ROOT / "scripts/watch_state8_task_anchor_screen.sh").read_text()
    preserve = watcher[watcher.index("preserve_passed(){"):watcher.index("submit_jobs(){")]
    assert "pass_count" in preserve
    assert "(( pass_count > 0 )) || return 0" in preserve
    assert "amlt results download" in preserve


def test_task_anchor_supervisor_restarts_infrastructure_only() -> None:
    supervisor = (
        ROOT / "scripts/supervise_state8_task_anchor_screen.sh"
    ).read_text()
    assert "watch_state8_task_anchor_screen.sh" in supervisor
    assert "STATE8_TASK_ANCHOR_RESTART_SECONDS:-120" in supervisor
    assert supervisor.count("scientific_gate_failed") >= 2
    assert '[[ ! -f "$state/complete" ]]' in supervisor
    assert "amlt run" not in supervisor
    assert "amlt cancel" not in supervisor
