import subprocess
from pathlib import Path

from scripts.materialize_state8_matched_checkpoint2000_rescue_launcher import materialize


ROOT = Path(__file__).resolve().parents[1]


def test_materialized_launcher_is_shell_valid_and_schema_matched(tmp_path: Path) -> None:
    output = tmp_path / "launcher.sh"
    materialize(
        ROOT / "scripts/run_state8_stateonly_repair_ab.sh",
        output,
        "/mnt/amulet/studentmodeling/diagnostics/state8_diagnosis_first_full_history_teacher_v1",
    )
    subprocess.run(["bash", "-n", str(output)], check=True)
    text = output.read_text()
    assert "\n+      ||" not in text
    assert "train_state8_matched_checkpoint2000_causal_rescue.py" in text
    assert "diagnosis_evidence_label_v1" in text
    assert "si.get('step')==2000" in text
    assert "STATE8_MATCHED_RESCUE_ENTRYPOINT_SHA256" in text
    assert "STATE8_RESCUE_MODE" in text
    assert "state8_diagnosis_first_full_history_teacher_screen_v1" in text
    assert "tg['support']==75" in text
    assert "tg['support']==1494" not in text
    assert "tg['exact_match_authoritative'] is False" in text
