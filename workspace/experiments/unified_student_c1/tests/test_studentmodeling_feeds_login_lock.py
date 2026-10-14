from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def test_interactive_and_background_login_share_one_lock() -> None:
    interactive = ROOT / "scripts/login_studentmodeling_feeds_interactive.sh"
    background = ROOT / "scripts/reauth_layerwise_unified_pipeline.sh"
    for path in (interactive, background):
        subprocess.run(["bash", "-n", str(path)], check=True)
        text = path.read_text()
        assert "device_login.lock" in text
        assert "flock" in text
    assert "restart_state8_eedi_same_question_controllers.sh" in background.read_text()
