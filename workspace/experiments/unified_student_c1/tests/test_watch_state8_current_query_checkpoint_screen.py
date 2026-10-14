from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/watch_state8_current_query_checkpoint_screen.sh"


def test_milestone_watcher_is_valid_and_authoritative() -> None:
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
    source = SCRIPT.read_text(encoding="utf-8")
    assert "CHECKPOINT_STEP must be 20 or 100" in source
    assert "materialize_state8_current_query_checkpoint_screen.py" in source
    assert "validate_state8_current_query_checkpoint_screen_amlt.py" in source
    assert "--judge-backend trapi" in source
    assert "semantic_judge_authoritative" in source
