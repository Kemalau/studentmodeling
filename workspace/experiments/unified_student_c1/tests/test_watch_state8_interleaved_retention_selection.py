from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_interleaved_result_enters_shared_global_selection() -> None:
    watcher = (ROOT / "scripts/watch_state8_interleaved_retention_selection.sh").read_text(encoding="utf-8")
    global_watcher = (ROOT / "scripts/watch_state8_global_selection_and_promotion.sh").read_text(encoding="utf-8")
    assert "state8_interleaved_retention_recovery3_20260813" in watcher
    assert "--judge-backend trapi" in watcher
    assert "repair_interleaved_retention_seed8401" in watcher
    assert "same_question_repair_promoted" in watcher
    assert 'interleaved="$results/state8_interleaved_retention_selection_20260813"' in global_watcher
    assert '--selection interleaved_retention' in global_watcher
