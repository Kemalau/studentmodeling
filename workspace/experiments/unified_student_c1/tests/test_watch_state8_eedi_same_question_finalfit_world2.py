from pathlib import Path
import subprocess


SCRIPT = Path("scripts/watch_state8_eedi_same_question_finalfit_world2.sh")


def test_finalfit_controller_is_promotion_quota_resume_and_receipt_gated():
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
    text = SCRIPT.read_text()
    assert "scientific_gate_pass" in text and "promotion_failed" in text
    assert "materialize_state8_eedi_same_question_finalfit_world2_amlt.py" in text
    assert "validate_state8_eedi_same_question_finalfit_world2_amlt.py" in text
    assert "limit']-quota['used']>=2" in text
    assert "count_active_a100_requests.py" in text
    assert "available_user_slots" in text
    assert "x['auditable'] is True" in text
    assert "int(user['limit'])>=2" in text
    assert "amlt resume" in text and "recovery" in text
    assert "real_examples']==259845" in text
    assert "one_epoch_exact" in text
    assert "raw_history_visible_to_decoder" in text
    assert "feeds_submission_global.lock" in text
