from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/watch_state8_eedi_same_question_causal_repair.sh"


def test_watcher_covers_t3_and_both_c4_panels():
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
    text = SCRIPT.read_text(encoding="utf-8")
    assert "state8-eedi-same-question-causal-repair-20260811" in text
    assert "eedi_same_question_c4_25" in text
    assert "diagnosis_c4_50" in text
    assert text.count("--judge-backend trapi") == 1
    assert "exact_match_authoritative':False" in text
    assert "states <= {'', 'unknown'}: print('unknown')" in text
    assert "preserving remote job and retrying" in text
    assert 'MAX_RECOVERY_WAVES:-8' in text
    assert 'AUTO_RECOVERY:-1' in text
    assert "automatic recovery disabled" in text
    assert 'STATE_DIR:-' in text
    assert 'EXPERIMENT_NAME:-' in text
    assert "submit_recovery" in text
    assert "validate_state8_current_query_causal_repair_amlt.py" in text
    assert "count_active_a100_requests.py" in text
    assert "available_user_slots" in text
    assert "feeds_submission_global.lock" in text
    assert "exact-resume recovery" in text
    assert "amlt run \"$config\" \"$name\"" in text
    assert "results_downloaded" in text
    assert "preserving Passed job and retrying" in text
    assert 'fail "result download failed"' not in text
