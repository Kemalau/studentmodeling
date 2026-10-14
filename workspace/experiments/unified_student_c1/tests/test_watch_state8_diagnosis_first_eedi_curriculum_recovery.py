from pathlib import Path


def test_supervisor_has_fail_closed_exact_resume_recovery_and_semantic_judge() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "scripts/watch_state8_diagnosis_first_eedi_curriculum.sh"
    ).read_text(encoding="utf-8")
    for fragment in (
        "max_recovery_waves=8",
        "validate_state8_diagnosis_first_eedi_curriculum_amlt.py",
        "count_active_a100_requests.py",
        "select_a100_80gb",
        "amlt run \"$config\" \"$name\" -t Feeds -y",
        "amlt resume",
        "judge_formal_t3_predictions.py",
        "--judge-backend trapi",
        "exact_match_authoritative\":False",
    ):
        assert fragment in source
    assert "80G1-A100" not in source or "select_a100_80gb" in source
