from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_repair_watcher_recovers_failed_siblings_and_semantically_selects():
    source = (ROOT / "scripts/watch_state8_stateonly_repair_ab.sh").read_text()
    assert "AZURE_CONFIG_DIR=/home/v-yuejiali/.azure-keyflip-feeds" in source
    assert "terminal_nonpass_job_names" in source
    assert "for name in paused_job_names" in source
    assert "for name in terminal_nonpass_job_names" in source
    assert "print('\\n'.join(paused_job_names" not in source
    assert "print('\\n'.join(terminal_nonpass_job_names" not in source
    assert "preserved_results" in source
    assert "--judge-backend trapi" in source
    assert "summarize_state8_diagnosis_c4.py" in source
    assert "select_state8_stateonly_repair_ab.py" in source
