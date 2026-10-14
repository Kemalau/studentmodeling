from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_repair_controller_is_conditional_bounded_and_quota_gated():
    text = (ROOT / "scripts/watch_state8_eedi_same_question_repair_branches.sh").read_text()
    assert "repair_required" in text and "extension_required" in text
    assert "materialize_state8_eedi_same_question_repair_branches.py" in text
    assert "validate_state8_eedi_same_question_repair_branches_amlt.py" in text
    assert "select_a100_80gb" in text
    assert "limit']-quota['used']>=requested" in text
    assert "1<=requested<=3" in text
    assert "80G1-A100" not in text  # topology is bound and validated in YAML
    assert "feeds_submission_global.lock" in text
