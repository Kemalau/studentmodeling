import json
from pathlib import Path

import yaml

from scripts.materialize_state8_curriculum_repair_checkpoint_screen import materialize
from scripts.validate_state8_curriculum_repair_checkpoint_screen_amlt import validate


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "amlt/feeds_state8_curriculum_checkpoint_screen_20260812.yaml"


def receipt(tmp_path: Path, step: int) -> Path:
    path = tmp_path / f"checkpoint-{step}.json"
    path.write_text(
        json.dumps(
            {
                "status": "complete",
                "step": step,
                "sha256": {
                    "adapter_model": "a" * 64,
                    "recurrent_state": "b" * 64,
                },
            }
        )
    )
    return path


def test_materializes_all_repair_coordinates(tmp_path: Path):
    specs = {
        5201: "semantic_binding",
        5202: "specificity_triplet",
        5203: "balanced_continuation",
        6201: "semantic_binding",
        6202: "specificity_triplet",
        6203: "balanced_continuation",
        8201: "causal_teacher_low",
        8202: "causal_teacher_high",
        8301: "early_label_low",
        8302: "early_label_high",
    }
    for seed, branch in specs.items():
        for step in (20, 100, 240, 500):
            output = tmp_path / f"{seed}-{step}.yaml"
            materialize(TEMPLATE, receipt(tmp_path, step), seed, branch, step, output)
            validate(output, seed, branch, step)


def test_repair_screen_is_read_only_and_current_query_routed(tmp_path: Path):
    output = tmp_path / "bound.yaml"
    materialize(
        TEMPLATE,
        receipt(tmp_path, 20),
        5202,
        "specificity_triplet",
        20,
        output,
    )
    config = yaml.safe_load(output.read_text())
    env = config["jobs"][0]["submit_args"]["env"]
    assert env["CHECKPOINT_FAMILY"] == "repair"
    assert env["CURRICULUM_ARM"] == "repair_specificity_triplet_seed5202"
    text = (ROOT / "scripts/run_state8_curriculum_checkpoint_screen.sh").read_text()
    assert "audit_state8_curriculum_repair_checkpoint.py" in text
    assert '"$CHECKPOINT_FAMILY" == repair' in text
    assert "8201:causal_teacher_low|8202:causal_teacher_high" in text
    assert "8301:early_label_low|8302:early_label_high" in text
    assert 'case "$CHECKPOINT_STEP" in 20|100|240|500)' in text
    assert "|250|" not in text
    assert "optimizer.step" not in text
    assert "--init-adapter" not in text
