from pathlib import Path
import subprocess

from scripts.render_state8_eedi_same_question_final_formal_shard import render as render_formal
from scripts.render_state8_eedi_same_question_final_c4 import render as render_c4


ROOT=Path(__file__).resolve().parents[1]


def test_same_question_final_formal_uses_eedi_current_query(tmp_path: Path):
    out=tmp_path/"formal.sh"; render_formal(ROOT/"scripts/run_state8_final_formal_shard.sh",out)
    subprocess.run(["bash","-n",str(out)],check=True)
    text=out.read_text()
    assert "same_question_*" in text
    assert "evaluate_recurrent_formal_protocols_diagnosis_first_eedi_current_query.py" in text
    assert "967711843054673eab67e43a8d5392ac7329a31bf5005cc1ec5be8a60507244f" in text


def test_same_question_final_c4_uses_diagnosis_first(tmp_path: Path):
    out=tmp_path/"c4.sh"; render_c4(ROOT/"scripts/run_state8_final_c4.sh",out)
    subprocess.run(["bash","-n",str(out)],check=True)
    text=out.read_text()
    assert "same_question_*" in text
    assert "evaluate_recurrent_diagnosis_c4_diagnosis_first.py" in text
    assert "diagnosis_evidence_label_v1" in text
    assert "same_question_repair_checkpoint_promoted" in text
    assert "same_question_curriculum_promoted" in text
