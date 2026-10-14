from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from summarize_state8_large_evaluation_live_status import summarize  # noqa: E402


def test_large_evaluation_status_fails_closed(tmp_path: Path) -> None:
    value = summarize(tmp_path)
    assert value["evaluation_flow_complete"] is False
    assert value["scientific_gate_pass"] is False
    assert value["c3"]["claim_supported"] is False
    assert len(value["repair_screens"]) == 16


def test_flow_completion_and_scientific_success_are_distinct(tmp_path: Path) -> None:
    root = tmp_path / "results/state8_curriculum_complete_objective_20260812"
    root.mkdir(parents=True)
    (root / "complete").touch()
    (root / "final_objective.json").write_text(
        json.dumps({"scientific_gate_pass": False}), encoding="utf-8"
    )
    value = summarize(tmp_path)
    assert value["evaluation_flow_complete"] is True
    assert value["scientific_gate_pass"] is False


def test_last_known_remote_screen_status_is_provenance_labeled(tmp_path: Path) -> None:
    root = tmp_path / "results/state8_curriculum_repair_screens_20260812/seed8301_early_label_low/checkpoint-100"
    root.mkdir(parents=True)
    (root / "latest_status.json").write_text(
        json.dumps([{"status": "running"}]), encoding="utf-8"
    )
    value = summarize(tmp_path)
    assert value["repair_screens"]["seed8301_early_label_low_cp100"]["last_known_remote_status"] == "running"


def test_status_watcher_is_read_only_and_persistent() -> None:
    source = (ROOT / "scripts/watch_state8_large_evaluation_live_status.sh").read_text(
        encoding="utf-8"
    )
    assert "summarize_state8_large_evaluation_live_status.py" in source
    assert "controller.active.pid" in source
    assert "evaluation_flow_complete" in source
    assert "amlt run" not in source
    assert "amlt cancel" not in source
