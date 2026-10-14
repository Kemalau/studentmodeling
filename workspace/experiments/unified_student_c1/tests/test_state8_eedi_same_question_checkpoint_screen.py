import json
from pathlib import Path

from scripts.materialize_state8_eedi_same_question_checkpoint_screen import materialize
from scripts.validate_state8_eedi_same_question_checkpoint_screen_amlt import validate


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "amlt/feeds_state8_eedi_same_question_checkpoint_screen_20260811.yaml"


def receipt(tmp_path: Path, step: int) -> Path:
    path = tmp_path / f"checkpoint-{step}.json"
    path.write_text(json.dumps({"status":"complete","step":step,"sha256":{
        "adapter_model":"a"*64,"recurrent_state":"b"*64}}), encoding="utf-8")
    return path


def test_materializes_all_four_checkpoint_screens(tmp_path: Path):
    for arm in ("updater", "full"):
        for step in (20, 100):
            output = tmp_path / f"{arm}-{step}.yaml"
            materialize(TEMPLATE, receipt(tmp_path, step), arm, step, output)
            validate(output, arm, step)
            assert str(ROOT) in output.read_text(encoding="utf-8")
