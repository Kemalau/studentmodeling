from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_lr_scan_can_defer_one_arm_without_marking_it_passed() -> None:
    source = (ROOT / "scripts/watch_layerwise_eedi_full_lr_scan.sh").read_text()
    start = source.index('receipt="$state_dir/passed_receipts/$job.complete"')
    section = source[start:source.index('experiment="${arm_experiment[$job]}"', start)]
    assert 'all_passed=false' in section
    assert 'grep -Fxq "$job" "$state_dir/deferred_jobs.txt"' in section
    assert section.index('all_passed=false') < section.index('grep -Fxq "$job"')
    assert 'continue' in section
