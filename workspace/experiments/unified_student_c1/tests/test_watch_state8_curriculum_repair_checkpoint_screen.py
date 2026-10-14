from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_repair_screen_watcher_is_resumable_and_semantically_authoritative():
    source = (ROOT / "scripts/watch_state8_curriculum_repair_checkpoint_screen.sh").read_text()
    for fragment in (
        "materialize_state8_curriculum_repair_checkpoint_screen.py",
        "validate_state8_curriculum_repair_checkpoint_screen_amlt.py",
        "amlt resume",
        "max_recovery=8",
        "count_active_a100_requests.py",
        "available_user_slots",
        "--judge-backend trapi",
        "summarize_state8_curriculum_checkpoint_screen.py",
        "remote_passed",
        "needs_attention",
    ):
        assert fragment in source
    assert "optimizer.step" not in source


def test_repair_screen_download_waits_for_auth_and_retries_transport():
    source = (ROOT / "scripts/watch_state8_curriculum_repair_checkpoint_screen.sh").read_text()
    remote_passed = source.index('while [[ ! -f "$state/remote_passed" ]]')
    download_loop = source.index("download_failures=0")
    assert download_loop > remote_passed
    section = source[download_loop:]
    assert 'source "$project/scripts/use_studentmodeling_feeds_auth.sh"' in section
    assert "STUDENTMODELING_FEEDS_AUTH_HEALTH" in section
    assert "result download waiting for Feeds authentication" in section
    assert "download_failures < 8" in section
    assert "screen result download failed after authenticated retries" in section
    assert 'timeout --foreground 600 amlt results download' in section
