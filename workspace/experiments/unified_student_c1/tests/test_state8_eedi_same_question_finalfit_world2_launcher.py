from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path

from scripts.render_state8_eedi_same_question_finalfit_world2 import (
    SOURCE_SHA256,
    render,
)


ROOT = Path(__file__).resolve().parents[1]


def test_same_question_world2_renderer_is_executable_and_full_epoch(tmp_path: Path):
    source = ROOT / "scripts/run_state8_unified_final_fit.sh"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == SOURCE_SHA256
    output = tmp_path / "same-question-world2.sh"
    rendered_sha = render(source, output)
    assert hashlib.sha256(output.read_bytes()).hexdigest() == rendered_sha
    subprocess.run(["bash", "-n", str(output)], check=True)
    text = output.read_text(encoding="utf-8")
    assert "same_question_updater_mse1_seed3072" in text
    assert "same_question_full_mse1_seed3071" in text
    assert "same_question_curriculum_promoted" in text
    assert "state8_curriculum_checkpoint_promoted_v1" in text
    assert "{5201,5202,5203,6201,6202,6203,8201,8202,8301,8302}" in text
    assert "step in {20,100,240,500}" in text
    assert "source_step==(4000 if seed//1000==5 else 3000)" in text
    assert "export TOTAL_ROUNDS=129923" in text
    assert "export TOTAL_OPTIMIZER_STEPS=16241" in text
    assert 'test "$WORLD_SIZE" = 2' in text
    assert "--gradient-accumulation-steps 8" in text
    assert "--distributed-world-size 2" in text
    assert "--twins 100 --seed 20260811 --offset 0" in text
    assert "--twins 25 --seed 20260811 --offset 100" in text
    assert "STATE8_FINALFIT_ENTRYPOINT=scripts/train_state8_eedi_same_question_finalfit.py" in text
    assert "raw_history_visible_to_decoder" in text
    assert "state8_eedi_same_question_joint_promotion_v1" in text
    assert "if variant.startswith('same_question_'): expected_trajectory=1.0" in text
    assert "export STATE8_CONTROL_RANDOM_SAMPLES=4" in text
    assert "export STATE8_CONTROL_RANDOM_WEIGHT=0.25" in text
    assert "export STATE8_CONTROL_SAME_RULE_WEIGHT=0.50" in text
    assert "8caf665d2e3e0ff3523179289603cba4c3c9f6f20401447137ccf87c8c928598" in text
    assert "f9b96fe182641b9563911463ef6eacd8ffef7131ad613ec6c6865043c2b964ff" in text
    assert "2bec746e79f2a09b3a38dd98228f4bc4a57d4169aff75c507857982f0c42a001" in text
    assert "a2b236d2bab2a351fc7bdff15367de0c3e5ec1ee30a86cfcbc22a173b647ee6b" in text
    assert "STATE8_EARLY_LABEL_BINDING_WEIGHT" in text
    assert "binding_constraints_inherited_from_promotion" in text
    assert "semantic['control_robust_objective'] is True" in text
    assert "'control_random_samples':int(config.get('control_random_samples',1))" in text


def test_same_question_renderer_has_no_patch_marker_artifacts():
    source = (ROOT / "scripts/render_state8_eedi_same_question_finalfit_world2.py").read_text()
    assert '\\n+      || "$FINALFIT_VARIANT"' not in source


def test_same_question_rendered_embedded_python_is_syntax_valid(tmp_path: Path):
    output = tmp_path / "same-question-world2.sh"
    render(ROOT / "scripts/run_state8_unified_final_fit.sh", output)
    text = output.read_text(encoding="utf-8")
    blocks = re.findall(
        r"<<'(?P<tag>PY[A-Z0-9_]*)'\n(?P<body>.*?)\n(?P=tag)(?:\n|$)",
        text,
        re.S,
    )
    assert len(blocks) >= 10
    for index, (_tag, body) in enumerate(blocks):
        compile(body, f"rendered-heredoc-{index}.py", "exec")
