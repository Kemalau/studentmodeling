from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from scripts.summarize_state8_highcausal_pipeline import STAGES, summarize


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_highcausal_summary_requires_publish_and_final_gate(tmp_path: Path) -> None:
    for _, directory in STAGES:
        root = tmp_path / directory
        root.mkdir()
        (root / "complete").touch()
        (root / "scientific_gate_pass").touch()
        (root / "controller.heartbeat").write_text("2026-08-06T23:00:00Z\n")
    fit = tmp_path / STAGES[1][1]
    write_json(
        fit / "canonical_publish_receipt.json",
        {"stage": "state8_canonical_final_checkpoint_publish_v1", "status": "complete"},
    )
    final = tmp_path / STAGES[2][1]
    write_json(
        final / "final_gate.json",
        {"stage": "state8_complete_unified_model_final_gate_v1", "pass": True},
    )
    value = summarize(tmp_path, datetime(2026, 8, 6, 23, 1, tzinfo=timezone.utc))
    assert value["overall_complete"] is True
    assert value["current_stage"] == "complete"


def test_highcausal_summary_reports_pending_binding_progress(tmp_path: Path) -> None:
    for _, directory in STAGES:
        (tmp_path / directory).mkdir()
    write_json(
        tmp_path / "state8_highcausal_training_progress_20260806/joint/progress.json",
        {"round": 10, "total_rounds": 250},
    )
    value = summarize(tmp_path)
    assert value["overall_complete"] is False
    assert value["current_stage"] == "binding_screen"
    assert value["binding_progress"]["joint"]["round"] == 10
    assert value["binding_progress"]["mid"] is None
    assert value["binding_progress"]["eedi"] is None
    assert value["binding_progress"]["taxonomy"] is None
