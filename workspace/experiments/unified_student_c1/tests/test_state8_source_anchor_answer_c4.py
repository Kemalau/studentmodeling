import json
import subprocess
from pathlib import Path

from scripts.materialize_state8_source_anchor_answer_c4 import main as materialize_main
from scripts.validate_state8_source_anchor_answer_c4_amlt import validate


ROOT = Path(__file__).resolve().parents[1]


def test_answer_c4_runner_is_strict_closed_state() -> None:
    runner = ROOT / "scripts/run_state8_source_anchor_answer_c4.sh"
    subprocess.run(["bash", "-n", str(runner)], check=True)
    text = runner.read_text()
    for fragment in (
        "--num-pairs 150", "--seed 42", "raw_history_visible_to_decoder",
        "persistent_history_kv_cache", "<|vision_pad|>", "SOURCE_ANCHOR_ADAPTER_SHA256",
    ):
        assert fragment in text


def test_materialized_answer_c4_config_validates(tmp_path, monkeypatch) -> None:
    selection = {
        "stage": "state8_source_anchor_diagnostic_selection_v1",
        "diagnostic_gate_pass": True,
        "selected_for_answer_c4_confirmation": "full_mse1",
        "candidates": {
            "full_mse1": {
                "job": "state8-source-anchor-full-mse1-seed3071",
                "artifact_sha256": {"adapter_model": "a" * 64, "recurrent_state": "b" * 64},
            }
        },
    }
    selection_path = tmp_path / "selection.json"
    output = tmp_path / "rendered.yaml"
    selection_path.write_text(json.dumps(selection))
    monkeypatch.setattr(
        "sys.argv",
        ["materialize", "--template", str(ROOT / "amlt/feeds_state8_source_anchor_answer_c4_20260811.yaml"),
         "--selection", str(selection_path), "--output", str(output)],
    )
    materialize_main()
    validate(output)
