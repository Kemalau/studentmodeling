from pathlib import Path
import subprocess


SCRIPT = Path("scripts/watch_state8_eedi_same_question_promotion.sh")


def test_promotion_watcher_is_valid_and_fail_closed():
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
    text = SCRIPT.read_text()
    assert "select_state8_eedi_same_question_candidate.py" in text
    assert "validate_state8_eedi_same_question_answer_c4_amlt.py" in text
    assert "finalize_state8_eedi_same_question_promotion.py" in text
    assert "state8_eedi_same_question_v1/promotion.json" in text
    assert "promotion gate upload failed" in text
    assert "states<={'','unknown'}" in text
    assert "feeds_submission_global.lock" in text
