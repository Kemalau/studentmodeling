from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_watcher_is_receipt_bound_and_fail_closed() -> None:
    source = (ROOT / "scripts/watch_taskonly_qwen_formal_eval.sh").read_text(encoding="utf-8")
    assert "completion_blob=" in source
    assert "adapter completion receipt invalid" in source
    assert "validate_taskonly_qwen_formal_eval_amlt.py" in source
    assert "submission returned nonzero and acceptance is unproven" in source
    assert "--judge-backend trapi" in source
    assert "generation_rows']==1494" in source


def test_watcher_waits_for_auth_during_receipt_and_result_downloads() -> None:
    source = (ROOT / "scripts/watch_taskonly_qwen_formal_eval.sh").read_text(encoding="utf-8")
    assert source.count("STUDENTMODELING_FEEDS_AUTH_HEALTH") >= 2
    assert "transient adapter completion receipt download failure; retrying" in source
    assert "download_failures=0" in source
    assert "formal result download failed after authenticated retries" in source
    assert 'timeout --foreground 600 amlt results download' in source
