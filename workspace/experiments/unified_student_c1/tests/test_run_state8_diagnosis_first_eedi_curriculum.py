from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GENERIC = ROOT / "scripts/run_state8_diagnosis_first_eedi_curriculum.sh"
CURRENT = ROOT / "scripts/run_state8_diagnosis_first_eedi_current_query_curriculum.sh"


def render(script: Path, tmp_path: Path) -> str:
    env = os.environ | {"STATE8_RENDER_ONLY": "1", "TMPDIR": str(tmp_path)}
    subprocess.run(["bash", str(script)], cwd=ROOT, env=env, check=True)
    rendered = Path("/tmp") / script.name.replace("run_", "run_").replace(
        ".sh", ".rendered.sh"
    )
    return rendered.read_text(encoding="utf-8")


def test_generic_tail_reconstructs_run_dir_and_is_resumable(tmp_path: Path) -> None:
    text = render(GENERIC, tmp_path)
    assert 'RUN_DIR="${STATE8_EEDI_RUN_DIR:?}"' in text
    assert 'completion[\'all_causal_pairs_covered\'] is True' in text
    assert 'receipt[\'sha256\'] == observed' in text
    assert 'stable / \'eval/formal_screen/predictions.jsonl\'' in text
    assert 'MODEL_REMOTE=/mnt/amulet/studentmodeling/models/Qwen3-32B' in text
    assert 'if [[ ! -f "$MODEL_LOCAL/model.safetensors.index.json" ]]' in text
    assert 'scripts/copy_model_parallel.py' in text
    assert '--model "$MODEL_LOCAL"' in text
    subprocess.run(["bash", "-n", "/tmp/run_state8_diagnosis_first_eedi_curriculum.rendered.sh"], check=True)


def test_current_query_inherits_the_resumable_tail(tmp_path: Path) -> None:
    text = render(CURRENT, tmp_path)
    assert "diagnosis_first_eedi_current_query_curriculum_seed83_single_ga4" in text
    assert 'RUN_DIR="${STATE8_EEDI_RUN_DIR:?}"' in text
    assert "eedi_current_query_curriculum.json" in text
    assert 'if [[ ! -f "$MODEL_LOCAL/model.safetensors.index.json" ]]' in text
    assert '--model "$MODEL_LOCAL"' in text
    subprocess.run(["bash", "-n", "/tmp/run_state8_diagnosis_first_eedi_current_query_curriculum.rendered.sh"], check=True)
