from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/watch_state8_eedi_current_query_causal_repair.sh"


def test_eedi_watcher_wrapper_is_valid_and_hash_bound() -> None:
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
    source = SCRIPT.read_text(encoding="utf-8")
    assert "expected=" in source
    assert "state8-eedi-current-query-causal-repair-20260811" in source
    assert "bash -n" in source
