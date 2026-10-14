from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WATCHER = ROOT / "scripts" / "watch_state8_matched_checkpoint1200_screen.sh"


def test_empty_status_lists_do_not_create_phantom_job_selectors() -> None:
    text = WATCHER.read_text(encoding="utf-8")
    assert text.count("sys.stdout.write(''.join(value+'\\n' for value in") == 3
    assert "print(*passed_job_names" not in text
    assert "print(*paused_job_names" not in text
    assert "print(*terminal_nonpass_job_names" not in text


def test_targeted_recovery_and_cross_wave_pass_markers_are_present() -> None:
    text = WATCHER.read_text(encoding="utf-8")
    assert 'selectors+=(":$job")' in text
    assert 'touch "$state/passed.$job"' in text
    assert 'passed.state8-matched-checkpoint1200-t3' in text
    assert 'passed.state8-matched-checkpoint1200-c4' in text

