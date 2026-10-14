from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "scripts/watch_state8_matched_checkpoint1200_diagnosis_c4_shards.sh").read_text()


def test_watcher_has_exact_shard_merge_and_trapi_judge() -> None:
    for fragment in (
        "passed_job_names",
        "terminal_nonpass_job_names",
        "merge_state8_diagnosis_c4_shards.py",
        "--expected-pairs 50",
        "--judge-backend trapi",
        "summarize_state8_diagnosis_c4.py",
    ):
        assert fragment in SOURCE
    assert "sys.stdout.write(''.join(value+'\\n' for value in" in SOURCE
