from pathlib import Path


def test_milestone_watcher_is_receipt_last_and_covers_curriculum_boundaries() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "scripts/watch_state8_eedi_curriculum_milestones.sh"
    ).read_text(encoding="utf-8")
    for fragment in (
        "milestones=(1000 2000 3000 4000 5000)",
        "checkpoint_completion.json",
        '"checkpoint-$step" "checkpoint-round-$step"',
        "ambiguous checkpoint-$step source formats",
        "snapshot_state8_checkpoint_blob.py",
        "receipt-last snapshot complete",
        "--auth-mode login",
        "state8_diagnosis_first_eedi_curriculum_20260812",
        "follow_parent_recovery",
        'next="$(<"$experiment_file")"',
        '[[ -d "/proc/$parent_pid" ]]',
        "following parent exact-resume recovery",
        "complete=$((complete + 1))",
        "final receipt can be snapshotted successfully",
    ):
        assert fragment in source
    assert "kill " not in source
    assert "amlt run" not in source
