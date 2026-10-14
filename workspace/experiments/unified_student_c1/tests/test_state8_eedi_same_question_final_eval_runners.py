from pathlib import Path
import subprocess


def test_all_same_question_final_eval_runners_are_shell_valid():
    for name in (
        "run_state8_eedi_same_question_final_formal_shard.sh",
        "run_state8_eedi_same_question_final_malrule_c4.sh",
        "run_state8_eedi_same_question_final_eedi_c4.sh",
    ):
        path=Path("scripts")/name
        subprocess.run(["bash","-n",str(path)],check=True)


def test_final_eedi_panel_is_frozen_heldout_and_state_only():
    text=Path("scripts/run_state8_eedi_same_question_final_eedi_c4.sh").read_text()
    assert "offset125_n150" in text
    assert "6743c2a29a1c2b576a96e64cd73a1ffccaa6b0364d04fe2e8c692ad315d97d4e" in text
    assert "student_response_visible_to_decoder'] is False" in text
    assert "raw_history_visible_to_decoder'] is False" in text
    assert "persistent_history_kv_cache'] is False" in text
    assert "--num-pairs 150" in text
