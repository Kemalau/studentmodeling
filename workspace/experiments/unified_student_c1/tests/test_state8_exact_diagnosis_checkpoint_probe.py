from __future__ import annotations

from pathlib import Path

import yaml

from scripts.materialize_state8_exact_diagnosis_checkpoint_eval_amlt import materialize
from scripts.validate_state8_exact_diagnosis_midpoint_eval_amlt import validate


ROOT = Path(__file__).resolve().parents[1]


def test_materializes_and_validates_checkpoint3000(tmp_path: Path) -> None:
    output = tmp_path / "checkpoint3000.yaml"
    materialize(
        ROOT / "amlt/feeds_state8_exact_diagnosis_midpoint_eval_20260809.yaml",
        output,
        3000,
    )
    validate(output)
    config = yaml.safe_load(output.read_text())
    assert {job["name"] for job in config["jobs"]} == {
        "state8-exact-diagnosis-midpoint-step3000-t3",
        "state8-exact-diagnosis-midpoint-step3000-c4",
    }
    for job in config["jobs"]:
        env = job["submit_args"]["env"]
        assert env["STATE8_EXACT_CHECKPOINT_STEP"] == "3000"
        assert env["STATE8_EXACT_CHECKPOINT_ROOT"].endswith("/checkpoint-3000")


def test_watcher_waits_for_immutable_later_checkpoint() -> None:
    text = (ROOT / "scripts/watch_state8_exact_diagnosis_midpoint_eval.sh").read_text()
    for fragment in (
        "STATE8_EXACT_PROBE_STEP",
        "checkpoint_completion.json",
        "az storage blob exists",
        "checkpoint${probe_step}",
        "history_dropout_counts':counts",
    ):
        assert fragment in text
