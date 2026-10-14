from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from summarize_state8_qualified_pipeline import STAGES, summarize  # noqa: E402


def fixture(tmp_path: Path, now: datetime) -> Path:
    for _name, directory in STAGES:
        root = tmp_path / directory
        root.mkdir(parents=True)
        (root / "controller.heartbeat").write_text(now.isoformat())
    return tmp_path


def test_pending_pipeline_is_not_reported_complete(tmp_path: Path) -> None:
    now = datetime(2026, 8, 5, 12, tzinfo=timezone.utc)
    value = summarize(fixture(tmp_path, now), now)
    assert value["current_stage"] == "teacher_curriculum"
    assert value["overall_complete"] is False
    assert all(row["controller_alive"] for row in value["stages"].values())


def test_semantic_screen_is_on_the_critical_path_after_teacher(tmp_path: Path) -> None:
    now = datetime(2026, 8, 5, 12, tzinfo=timezone.utc)
    root = fixture(tmp_path, now)
    (root / "state8_qualified_teacher_pipeline_20260805/complete").touch()
    value = summarize(root, now)
    assert value["current_stage"] == "semantic_candidate_screen"


def test_teacher_status_prefers_clean_v2_controller(tmp_path: Path) -> None:
    now = datetime(2026, 8, 5, 12, tzinfo=timezone.utc)
    root = fixture(tmp_path, now)
    teacher = root / "state8_qualified_teacher_pipeline_20260805"
    clean = teacher / "sharded_clean_v2"
    clean.mkdir()
    (clean / "controller.heartbeat").write_text(now.isoformat())
    (clean / "active_experiment.txt").write_text(
        "state8-full-history-teacher-eval-clean-v2-sharded-20260805\n"
    )
    value = summarize(root, now)
    stage = value["stages"]["teacher_curriculum"]
    assert stage["controller_directory"].endswith("sharded_clean_v2")
    assert stage["active_experiment"].endswith("clean-v2-sharded-20260805")


def test_teacher_status_prefers_live_16way_controller(tmp_path: Path) -> None:
    now = datetime(2026, 8, 5, 12, tzinfo=timezone.utc)
    root = fixture(tmp_path, now)
    teacher = root / "state8_qualified_teacher_pipeline_20260805"
    for name in ("sharded_clean_v2", "sharded_clean_v2_16way"):
        accelerator = teacher / name
        accelerator.mkdir()
        (accelerator / "controller.heartbeat").write_text(now.isoformat())
        (accelerator / "active_experiment.txt").write_text(f"{name}\n")
    value = summarize(root, now)
    stage = value["stages"]["teacher_curriculum"]
    assert stage["controller_directory"].endswith("sharded_clean_v2_16way")
    assert stage["active_experiment"] == "sharded_clean_v2_16way"


def test_teacher_status_ignores_retired_clean_v2_controller(tmp_path: Path) -> None:
    now = datetime(2026, 8, 5, 12, tzinfo=timezone.utc)
    root = fixture(tmp_path, now)
    teacher = root / "state8_qualified_teacher_pipeline_20260805"
    clean = teacher / "sharded_clean_v2"
    clean.mkdir()
    (clean / "controller.heartbeat").write_text(now.isoformat())
    (clean / "active_experiment.txt").write_text("retired-experiment\n")
    (clean / "retired_in_favor_of_16way.txt").write_text("retired\n")
    value = summarize(root, now)
    stage = value["stages"]["teacher_curriculum"]
    assert stage["controller_directory"] == str(teacher)
    assert stage["active_experiment"] is None


def test_teacher_status_switches_to_primary_curriculum_after_gate(tmp_path: Path) -> None:
    now = datetime(2026, 8, 5, 12, tzinfo=timezone.utc)
    root = fixture(tmp_path, now)
    teacher = root / "state8_qualified_teacher_pipeline_20260805"
    (teacher / "teacher_gate_pass").touch()
    primary = teacher / "primary_curriculum"
    primary.mkdir()
    (primary / "controller.heartbeat").write_text(now.isoformat())
    (primary / "active_experiment.txt").write_text("primary-recovery8\n")
    value = summarize(root, now)
    stage = value["stages"]["teacher_curriculum"]
    assert stage["controller_directory"].endswith("primary_curriculum")
    assert stage["controller_alive"] is True
    assert stage["active_experiment"] == "primary-recovery8"


def test_status_includes_highkd_training_and_promotion(tmp_path: Path) -> None:
    now = datetime(2026, 8, 5, 12, tzinfo=timezone.utc)
    value = summarize(fixture(tmp_path, now), now)
    assert "highkd_curriculum" in value["stages"]
    assert "highkd_promotion" in value["stages"]


def test_training_progress_and_alerts_are_visible(tmp_path: Path) -> None:
    now = datetime(2026, 8, 5, 12, tzinfo=timezone.utc)
    root = fixture(tmp_path, now)
    stage = root / "state8_qualified_balanced_teacher_pipeline_20260805"
    progress = stage / "progress"
    progress.mkdir()
    (progress / "progress.json").write_text(
        json.dumps({"experiment": "balanced", "round": 80, "health": {"healthy": True}})
    )
    (progress / "stalled.json").touch()
    value = summarize(root, now)
    observed = value["stages"]["balanced_curriculum"]
    assert observed["training_progress"]["round"] == 80
    assert observed["progress_stalled"] is True
    assert observed["needs_attention"] is True
    assert "balanced_curriculum" in value["attention"]


def test_complete_requires_final_scientific_gate(tmp_path: Path) -> None:
    now = datetime(2026, 8, 5, 12, tzinfo=timezone.utc)
    root = fixture(tmp_path, now)
    for _name, directory in STAGES:
        (root / directory / "complete").touch()
    final = root / "state8_complete_final_eval_20260805"
    (final / "scientific_gate_pass").touch()
    (final / "final_gate.json").write_text(
        json.dumps({"stage": "state8_complete_unified_model_final_gate_v1", "pass": True})
    )
    value = summarize(root, now)
    assert value["current_stage"] == "complete"
    assert value["overall_complete"] is True


def test_contradictory_scientific_flags_fail_closed(tmp_path: Path) -> None:
    now = datetime(2026, 8, 5, 12, tzinfo=timezone.utc)
    root = fixture(tmp_path, now)
    stage = root / STAGES[2][1]
    (stage / "complete").touch()
    (stage / "scientific_gate_pass").touch()
    (stage / "scientific_gate_failed").touch()
    with pytest.raises(ValueError, match="Contradictory"):
        summarize(root, now)
