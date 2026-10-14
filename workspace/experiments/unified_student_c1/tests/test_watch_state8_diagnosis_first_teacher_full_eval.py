from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "scripts/watch_state8_diagnosis_first_teacher_full_eval.sh").read_text()


def test_full_teacher_controller_is_sharded_resumable_and_semantic() -> None:
    for fragment in (
        "seq -w 0 15", "count_active_a100_requests.py", "available_user_slots",
        "preserve_passed", "feeds_submission_global.lock", "max_waves=32",
        "merge_state8_teacher_eval_shards.py", "--judge-backend trapi",
        "validate_state8_teacher_gate.py", "teacher_full_gate.json",
        "teacher_full_provenance_gate.json", "'evaluation_shards':16",
        "'exact_match_authoritative':False",
        "sys.stdout.write(''.join(value+'\\n' for value in values))",
        '[[ -n "$job" ]] || continue',
        'if [[ "$job" == "$expected" ]]',
        'if (( ${#selectors[@]} > 0 )); then',
        "ignored empty or unknown paused-shard parser output",
    ):
        assert fragment in SOURCE
    assert SOURCE.index("paused_job_names") < SOURCE.index("has_forward_progress_job")
    assert 'amlt resume "$experiment" "${selectors[@]}"' in SOURCE
    assert 'selectors+=(":$job")' in SOURCE
