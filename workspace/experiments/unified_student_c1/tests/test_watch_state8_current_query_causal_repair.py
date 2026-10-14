from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/watch_state8_current_query_causal_repair.sh"


def test_watcher_is_valid_and_semantic_authority_is_explicit() -> None:
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
    source = SCRIPT.read_text(encoding="utf-8")
    assert "judge_formal_t3_predictions.py" in source
    assert "--judge-backend trapi" in source
    assert "summarize_state8_diagnosis_c4.py" in source
    assert "semantic_judge_authoritative" in source
    assert "exact_match_authoritative" in source
    assert "states <= {'', 'unknown'}: print('unknown')" in source
    assert "preserving remote job and retrying" in source
