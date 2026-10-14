import json
from pathlib import Path

from scripts.materialize_state8_current_query_checkpoint_screen import materialize
from scripts.validate_state8_current_query_checkpoint_screen_amlt import validate


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "amlt/feeds_state8_current_query_checkpoint_screen_20260811.yaml"


def receipt(tmp_path: Path, step: int) -> Path:
    path = tmp_path / "checkpoint_completion.json"
    path.write_text(json.dumps({
        "status": "complete", "step": step,
        "sha256": {"adapter_model": "a" * 64, "recurrent_state": "b" * 64},
    }), encoding="utf-8")
    return path


def test_materializes_each_bound_milestone(tmp_path: Path) -> None:
    for step in (20, 100):
        output = tmp_path / f"cp{step}.yaml"
        materialize(TEMPLATE, receipt(tmp_path, step), step, output)
        validate(output, step)
        assert str(ROOT) in output.read_text(encoding="utf-8")


def test_rejects_receipt_step_drift(tmp_path: Path) -> None:
    try:
        materialize(TEMPLATE, receipt(tmp_path, 20), 100, tmp_path / "bad.yaml")
    except ValueError as error:
        assert "step drift" in str(error)
    else:
        raise AssertionError("Receipt step drift was accepted")
